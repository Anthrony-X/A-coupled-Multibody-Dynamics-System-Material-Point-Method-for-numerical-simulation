from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chrono_loader import require_chrono_vehicle
from .diagnostics import ConstraintDiagnostics
from .vehicle import MeshBuilder, MultibodyRigidGroundTank, MultibodyState, MultibodyStepMetrics


def _round_list(values, digits: int = 9) -> list[float]:
    return [round(float(value), digits) for value in values]


def _chrono_to_model_vec(value) -> np.ndarray:
    return np.array([float(value.y), float(value.z), -float(value.x)], dtype=np.float64)


def _model_to_chrono_vec(value) -> list[float]:
    arr = np.asarray(value, dtype=np.float64)
    return _round_list([-arr[2], arr[0], arr[1]])


def _model_to_chrono_chvec(chrono, value):
    x, y, z = _model_to_chrono_vec(value)
    return chrono.ChVector3d(float(x), float(y), float(z))


def _chrono_vector(value) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float64)


def _chrono_vec_tuple(value) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))


def _chrono_quat_tuple(value) -> tuple[float, float, float, float]:
    return (float(value.e0), float(value.e1), float(value.e2), float(value.e3))


def _optional_chrono_vec_tuple(obj, method_name: str) -> tuple[float, float, float] | None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return _chrono_vec_tuple(method())
    except Exception:
        return None


def _quat_to_yaw_pitch_model(q) -> tuple[float, float]:
    """Return Chrono-frame yaw and model pitch from a Z-Y-X decomposition."""

    w = float(q.e0)
    x = float(q.e1)
    y = float(q.e2)
    z = float(q.e3)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r20 = 2.0 * (x * z - y * w)
    yaw = math.atan2(r10, r00)
    pitch = math.atan2(-r20, math.hypot(r00, r10))
    return yaw, pitch


def _pitch_rate_from_parent_angular_velocity(chrono_yaw: float, angular_velocity) -> float:
    """Project world angular velocity onto the yaw-rotated pitch axis."""

    pitch_axis_x = -math.sin(float(chrono_yaw))
    pitch_axis_y = math.cos(float(chrono_yaw))
    return pitch_axis_x * float(angular_velocity.x) + pitch_axis_y * float(angular_velocity.y)


def _rotation_matrix_to_quat_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (
        (m[1, 0] - m[0, 1]) / s,
        (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s,
        0.25 * s,
    )


def _model_state_to_chrono_rotation(tank: MultibodyRigidGroundTank, state: MultibodyState):
    axis_x = _model_to_chrono_vec(tank.vector_to_world((0.0, 0.0, -1.0), state=state))
    axis_y = _model_to_chrono_vec(tank.vector_to_world((1.0, 0.0, 0.0), state=state))
    axis_z = _model_to_chrono_vec(tank.vector_to_world((0.0, 1.0, 0.0), state=state))
    matrix = np.column_stack([axis_x, axis_y, axis_z])
    return _rotation_matrix_to_quat_wxyz(matrix)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class ProjectChronoVehicleInitializationSummary:
    bodies: int = 0
    contacts: int = 0
    left_shoes: int = 0
    right_shoes: int = 0
    json_root: str = ""
    vehicle_file: str = ""


class ProjectChronoVehicleBackend:
    """Adapter for Chrono's built-in tracked-vehicle package."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        tank: MultibodyRigidGroundTank,
        obstacles: list[Any],
    ) -> None:
        self.args = args
        self.tank = tank
        self.obstacles = obstacles
        self.chrono, self.veh = require_chrono_vehicle()
        asset_dir = str(getattr(args, "chrono_vehicle_asset_dir", "") or "").strip()
        if asset_dir:
            self.asset_root = Path(asset_dir).resolve()
        else:
            self.asset_root = Path(tempfile.gettempdir()).resolve() / "mpm_chrono_vehicle"
        self.vehicle_config = self.tank.data.get("chrono_vehicle", {})
        raw_asset_name = str(self.vehicle_config.get("asset_name") or self.tank.data.get("model_name") or "TrackedVehicle")
        self.asset_name = re.sub(r"[^A-Za-z0-9_-]+", "", raw_asset_name) or "TrackedVehicle"
        self.vehicle_file = self.asset_root / self.asset_name / "vehicle" / f"{self.asset_name}_Vehicle_SinglePin.json"
        self.vehicle = None
        self.system = None
        self.terrain = None
        self.driver_inputs = None
        self.obstacle_bodies: list[Any] = []
        self.external_shoe_forces: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
        self.external_shoe_torques: dict[tuple[int, int], list[np.ndarray]] = {}
        self.planar_chassis_constraint: dict[str, float] | None = None
        self.summary = ProjectChronoVehicleInitializationSummary()

    def _config_section(self, name: str) -> dict[str, Any]:
        section = self.vehicle_config.get(name, {})
        return section if isinstance(section, dict) else {}

    def _config_value(self, section: str, key: str, fallback: Any) -> Any:
        value = self._config_section(section).get(key)
        return fallback if value is None else value

    def _chrono_track_pitch(self) -> float:
        geometric_pitch = float(np.mean([self.tank.actual_pitch["left"], self.tank.actual_pitch["right"]]))
        override = getattr(self.args, "chrono_track_shoe_pitch", None)
        if override is not None and float(override) > 0.0:
            return float(override)
        return float(self._config_value("track", "chrono_shoe_pitch_m", geometric_pitch))

    def initialize(self) -> ProjectChronoVehicleInitializationSummary:
        self._generate_vehicle_json()
        self.veh.SetDataPath(str(self.asset_root) + "/")
        self.vehicle = self.veh.TrackedVehicle(str(self.vehicle_file), self.chrono.ChContactMethod_SMC)
        initial_pos = np.asarray(
            [
                self.tank.state.lateral_pos,
                self.tank.state.heave,
                self.tank.state.forward_pos,
            ],
            dtype=np.float64,
        )
        q0 = _model_state_to_chrono_rotation(self.tank, self.tank.state)
        init = self.chrono.ChCoordsysd(
            _model_to_chrono_chvec(self.chrono, initial_pos),
            self.chrono.ChQuaterniond(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])),
        )
        self.vehicle.Initialize(init, 0.0)
        self.system = self.vehicle.GetSystem()
        self.system.SetCollisionSystemType(self.chrono.ChCollisionSystem.Type_BULLET)
        self.system.SetSolverType(self.chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.system.SetMaxPenetrationRecoverySpeed(float(getattr(self.args, "chrono_max_penetration_recovery_speed", 0.25)))
        self.vehicle.SetChassisVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSprocketVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetIdlerVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetIdlerWheelVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSuspensionVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetRoadWheelVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetRollerVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetTrackShoeVisualizationType(self.veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSprocketCollide(True)
        self.vehicle.SetIdlerCollide(True)
        self.vehicle.SetRoadWheelCollide(True)
        self.vehicle.SetRollerCollide(True)
        self.vehicle.SetTrackShoeCollide(True)
        if bool(getattr(self.args, "chrono_use_rigid_terrain", True)):
            self.terrain = self._make_rigid_terrain()
            self._add_obstacle_bodies()
        self.driver_inputs = self.veh.DriverInputs()
        self.driver_inputs.m_steering = 0.0
        self.driver_inputs.m_throttle = 0.0
        self.driver_inputs.m_braking = 0.0
        self._sync_tank_state_from_chrono()
        self._set_initial_metrics()
        self.summary = ProjectChronoVehicleInitializationSummary(
            bodies=self._system_body_count(),
            contacts=int(self.system.GetNumContacts()),
            left_shoes=int(self.vehicle.GetNumTrackShoes(self.veh.LEFT)),
            right_shoes=int(self.vehicle.GetNumTrackShoes(self.veh.RIGHT)),
            json_root=str(self.asset_root),
            vehicle_file=str(self.vehicle_file),
        )
        return self.summary

    def enable_planar_chassis_constraint(self) -> None:
        self.planar_chassis_constraint = {
            "forward_pos": float(self.tank.state.forward_pos),
            "lateral_pos": float(self.tank.state.lateral_pos),
            "yaw": float(self.tank.state.yaw),
        }
        self._apply_planar_chassis_constraint()

    def disable_planar_chassis_constraint(self) -> None:
        self.planar_chassis_constraint = None

    def _apply_planar_chassis_constraint(self) -> None:
        if self.planar_chassis_constraint is None or self.vehicle is None:
            return
        chassis = self.vehicle.GetChassisBody()
        pos = chassis.GetPos()
        anchor = self.planar_chassis_constraint
        chassis.SetPos(
            self.chrono.ChVector3d(
                -float(anchor["forward_pos"]),
                float(anchor["lateral_pos"]),
                float(pos.z),
            )
        )
        current_chrono_yaw, current_pitch = _quat_to_yaw_pitch_model(chassis.GetRot())
        current_ang = chassis.GetAngVelParent()
        current_pitch_rate = _pitch_rate_from_parent_angular_velocity(current_chrono_yaw, current_ang)
        constrained_state = self.tank.copy_state()
        constrained_state.forward_pos = float(anchor["forward_pos"])
        constrained_state.lateral_pos = float(anchor["lateral_pos"])
        constrained_state.heave = float(pos.z)
        constrained_state.yaw = float(anchor["yaw"])
        constrained_state.pitch = float(current_pitch)
        q = _model_state_to_chrono_rotation(self.tank, constrained_state)
        chassis.SetRot(self.chrono.ChQuaterniond(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        lin = chassis.GetLinVel()
        chassis.SetLinVel(self.chrono.ChVector3d(0.0, 0.0, float(lin.z)))
        constrained_chrono_yaw = -float(anchor["yaw"])
        pitch_axis_x = -math.sin(constrained_chrono_yaw)
        pitch_axis_y = math.cos(constrained_chrono_yaw)
        chassis.SetAngVelParent(
            self.chrono.ChVector3d(
                pitch_axis_x * current_pitch_rate,
                pitch_axis_y * current_pitch_rate,
                0.0,
            )
        )

    def _set_initial_metrics(self) -> None:
        self.tank.last_metrics = MultibodyStepMetrics(
            t=self.tank.state.t,
            forward_pos=self.tank.state.forward_pos,
            lateral_pos=self.tank.state.lateral_pos,
            heave=self.tank.state.heave,
            yaw=self.tank.state.yaw,
            pitch=self.tank.state.pitch,
            heave_rate=self.tank.state.heave_rate,
            pitch_rate=self.tank.state.pitch_rate,
            forward_speed=self.tank.state.forward_speed,
            yaw_rate=self.tank.state.yaw_rate,
            left_drive_torque=0.0,
            right_drive_torque=0.0,
            left_track_speed=0.0,
            right_track_speed=0.0,
            left_sprocket_omega=self.tank.state.left_sprocket_omega,
            right_sprocket_omega=self.tank.state.right_sprocket_omega,
            left_slip_ratio=0.0,
            right_slip_ratio=0.0,
            left_traction=0.0,
            right_traction=0.0,
            left_normal_load=0.0,
            right_normal_load=0.0,
            rolling_resistance=0.0,
            drag_force=0.0,
            acceleration=0.0,
            yaw_acceleration=0.0,
            left_contact_shoes=0,
            right_contact_shoes=0,
            obstacle_contact_shoes=0,
        )

    def _system_body_count(self) -> int:
        try:
            return len(list(self.system.GetBodies()))
        except Exception:
            try:
                return int(self.system.GetNumBodies())
            except Exception:
                return 0

    def capture_state(self) -> dict[str, Any]:
        """Capture the dynamic Chrono state needed for one macro-step replay."""
        if self.system is None:
            raise RuntimeError("ProjectChronoVehicleBackend.initialize() must be called before capture_state().")
        bodies = []
        for body in self.system.GetBodies():
            bodies.append(
                {
                    "body": body,
                    "pos": _chrono_vec_tuple(body.GetPos()),
                    "rot": _chrono_quat_tuple(body.GetRot()),
                    "lin_vel": _chrono_vec_tuple(body.GetLinVel()),
                    "ang_vel": _chrono_vec_tuple(body.GetAngVelParent()),
                    "lin_acc": _optional_chrono_vec_tuple(body, "GetLinAcc"),
                    "ang_acc": _optional_chrono_vec_tuple(body, "GetAngAccParent"),
                }
            )
        return {
            "time": float(self.system.GetChTime()),
            "tank_state": self.tank.copy_state(),
            "planar_chassis_constraint": (
                None if self.planar_chassis_constraint is None else dict(self.planar_chassis_constraint)
            ),
            "bodies": bodies,
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a state previously returned by capture_state()."""
        if self.system is None:
            raise RuntimeError("ProjectChronoVehicleBackend.initialize() must be called before restore_state().")
        set_time = getattr(self.system, "SetChTime", None)
        if callable(set_time):
            set_time(float(snapshot["time"]))
        for body_state in snapshot["bodies"]:
            body = body_state["body"]
            px, py, pz = body_state["pos"]
            q0, q1, q2, q3 = body_state["rot"]
            vx, vy, vz = body_state["lin_vel"]
            wx, wy, wz = body_state["ang_vel"]
            body.SetPos(self.chrono.ChVector3d(float(px), float(py), float(pz)))
            body.SetRot(self.chrono.ChQuaterniond(float(q0), float(q1), float(q2), float(q3)))
            body.SetLinVel(self.chrono.ChVector3d(float(vx), float(vy), float(vz)))
            body.SetAngVelParent(self.chrono.ChVector3d(float(wx), float(wy), float(wz)))
            lin_acc = body_state.get("lin_acc")
            if lin_acc is not None and callable(getattr(body, "SetLinAcc", None)):
                ax, ay, az = lin_acc
                body.SetLinAcc(self.chrono.ChVector3d(float(ax), float(ay), float(az)))
            ang_acc = body_state.get("ang_acc")
            if ang_acc is not None and callable(getattr(body, "SetAngAccParent", None)):
                ax, ay, az = ang_acc
                body.SetAngAccParent(self.chrono.ChVector3d(float(ax), float(ay), float(az)))
            for method_name in ("EmptyAccumulators", "EmptyAccumulator"):
                method = getattr(body, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
                    break
        self.tank.state = self.tank.copy_state(snapshot["tank_state"])
        constraint = snapshot.get("planar_chassis_constraint")
        self.planar_chassis_constraint = None if constraint is None else dict(constraint)
        if self.driver_inputs is not None:
            self.driver_inputs.m_throttle = 0.0
            self.driver_inputs.m_braking = 0.0
            self.driver_inputs.m_steering = 0.0
        self.clear_external_loads()
        self.tank._clear_track_related_caches()

    def step(
        self,
        *,
        left_drive_torque: float,
        right_drive_torque: float,
        dt: float,
        settle_planar_constraint: bool = False,
        gravity_scale: float = 1.0,
    ) -> MultibodyStepMetrics:
        if self.vehicle is None or self.system is None or self.driver_inputs is None:
            raise RuntimeError("ProjectChronoVehicleBackend.initialize() must be called before step().")
        if settle_planar_constraint and self.planar_chassis_constraint is None:
            self.enable_planar_chassis_constraint()
        elif not settle_planar_constraint and self.planar_chassis_constraint is not None:
            self.disable_planar_chassis_constraint()
        self._apply_planar_chassis_constraint()
        previous = self.tank.copy_state()
        gravity = float(self.tank.geom.gravity) * max(0.0, float(gravity_scale))
        self.system.SetGravitationalAcceleration(self.chrono.ChVector3d(0.0, 0.0, -gravity))
        max_dt = max(float(getattr(self.args, "chrono_max_substep_dt", 5.0e-4)), 1.0e-8)
        substeps = max(1, int(math.ceil(float(dt) / max_dt)))
        sub_dt = float(dt) / float(substeps)
        self.driver_inputs.m_throttle = 0.0
        self.driver_inputs.m_braking = 0.0
        self.driver_inputs.m_steering = 0.0
        for _ in range(substeps):
            time = self.system.GetChTime()
            if self.terrain is not None:
                self.terrain.Synchronize(time)
            self.vehicle.Synchronize(time, self.driver_inputs)
            self._apply_sprocket_torque(left_drive_torque, right_drive_torque)
            self._apply_external_shoe_loads()
            if self.terrain is not None:
                self.terrain.Advance(sub_dt)
            self.vehicle.Advance(sub_dt)
            self._apply_planar_chassis_constraint()
        self._sync_tank_state_from_chrono()
        return self._build_metrics(
            previous_state=previous,
            left_drive_torque=left_drive_torque,
            right_drive_torque=right_drive_torque,
            dt=dt,
        )

    def _generate_vehicle_json(self) -> None:
        g = self.tank.geom
        asset = self.asset_name
        track_root = self.asset_root / asset
        masses = self._config_section("masses_kg")
        width = float(g.track_width)
        wheel_width = max(0.28 * width, 0.08)
        wheel_gap = max(width - 2.0 * wheel_width, 0.02)
        wheel_mass = float(masses.get("road_wheel", getattr(self.args, "chrono_wheel_mass", 120.0)))
        roller_mass = float(masses.get("return_roller", getattr(self.args, "chrono_roller_mass", 0.0)))
        if roller_mass <= 0.0:
            roller_mass = max(0.35 * wheel_mass, 25.0)
        sprocket_mass = float(masses.get("drive_sprocket", getattr(self.args, "chrono_sprocket_mass", 0.0)))
        if sprocket_mass <= 0.0:
            sprocket_mass = max(wheel_mass, 120.0)
        idler_mass = float(masses.get("idler_wheel", getattr(self.args, "chrono_idler_mass", 0.0)))
        if idler_mass <= 0.0:
            idler_mass = wheel_mass
        arm_mass = float(masses.get("suspension_arm", getattr(self.args, "chrono_suspension_arm_mass", 75.0)))
        idler_carrier_mass = float(masses.get("idler_carrier", 25.0))
        track_mass = float(self.tank.total_track_shoe_mass)
        road_count = 0.5 * sum(len(self.tank.road_wheel_names_by_side[side]) for side in ("left", "right"))
        roller_count = 0.5 * sum(len(self.tank.return_roller_names_by_side[side]) for side in ("left", "right"))
        wheel_total = 2.0 * (
            road_count * wheel_mass
            + roller_count * roller_mass
            + idler_mass
            + sprocket_mass
            + road_count * arm_mass
            + idler_carrier_mass
        )
        auto_chassis_mass = max(float(g.mass) - track_mass - wheel_total, 0.55 * float(g.mass))
        chassis_mass = float(masses.get("chassis", getattr(self.args, "chrono_chassis_mass", 0.0)))
        if chassis_mass <= 0.0:
            chassis_mass = auto_chassis_mass
        roll_inertia = 0.82 * chassis_mass * (
            max(float(g.hull_width), 2.0 * abs(float(g.left_track_center_x)) + width) ** 2 + float(g.hull_height) ** 2
        ) / 12.0
        pitch_inertia = float(g.pitch_inertia) * chassis_mass / max(float(g.mass), 1.0e-9)
        yaw_inertia = float(g.yaw_inertia) * chassis_mass / max(float(g.mass), 1.0e-9)

        _write_json(
            track_root / "vehicle" / f"{asset}_Vehicle_SinglePin.json",
            {
                "Name": f"{asset} vehicle",
                "Type": "Vehicle",
                "Template": "TrackedVehicle",
                "Chassis": {"Input File": f"{asset}/chassis/{asset}_Chassis.json"},
                "Track Assemblies": [
                    {"Input File": f"{asset}/track_assembly/{asset}_TrackAssemblySinglePin_Left.json", "Offset": float(g.left_track_center_x)},
                    {"Input File": f"{asset}/track_assembly/{asset}_TrackAssemblySinglePin_Right.json", "Offset": float(g.right_track_center_x)},
                ],
                "Driveline": {"Input File": f"{asset}/driveline/{asset}_SimpleTrackDriveline.json"},
            },
        )
        _write_json(
            track_root / "chassis" / f"{asset}_Chassis.json",
            {
                "Name": f"{asset} chassis",
                "Type": "Chassis",
                "Template": "RigidChassis",
                "Components": [
                    {
                        "Centroidal Frame": {"Location": [0, 0, 0], "Orientation": [1, 0, 0, 0]},
                        "Mass": chassis_mass,
                        "Moments of Inertia": _round_list([roll_inertia, pitch_inertia, yaw_inertia]),
                        "Products of Inertia": [0, 0, 0],
                        "Void": False,
                    }
                ],
                "Driver Position": {"Location": [0.8, 0.45, 0.5], "Orientation": [1, 0, 0, 0]},
                "Visualization": {
                    "Primitives": [
                        {
                            "Type": "BOX",
                            "Location": [0, 0, 0],
                            "Orientation": [1, 0, 0, 0],
                            "Dimensions": _round_list([float(g.hull_length), float(g.hull_width), float(g.hull_height)]),
                        }
                    ]
                },
            },
        )
        self._write_track_shoe_json(track_root / "track_shoe" / f"{asset}_TrackShoeSinglePin.json")
        for side, sign in (("Left", -1.0), ("Right", 1.0)):
            self._write_wheel_json(track_root / "road_wheel" / f"{asset}_DoubleRoadWheel_{side}.json", "road wheel", g.road_wheel_radius, wheel_width, wheel_gap, wheel_mass)
            self._write_wheel_json(track_root / "idler_wheel" / f"{asset}_DoubleIdlerWheel_{side}.json", "idler wheel", g.idler_radius, wheel_width, wheel_gap, idler_mass)
            self._write_wheel_json(track_root / "roller" / f"{asset}_ReturnRoller_{side}.json", "return roller", g.return_roller_radius, wheel_width, wheel_gap, roller_mass)
            self._write_sprocket_json(track_root / "sprocket" / f"{asset}_SprocketSinglePin_{side}.json", mass=sprocket_mass)
            self._write_idler_json(
                track_root / "idler" / f"{asset}_Idler_{side}.json",
                side=side,
                sign=sign,
                carrier_mass=idler_carrier_mass,
            )
            station_count = len(self.tank.suspension_station_names_by_side[side.lower()])
            for station_index in range(1, station_count + 1):
                self._write_suspension_json(
                    track_root / "suspension" / f"{asset}_Suspension_{side}_{station_index:02d}.json",
                    side=side,
                    sign=sign,
                    arm_mass=arm_mass,
                    station_index=station_index,
                )
            self._write_track_assembly_json(track_root / "track_assembly" / f"{asset}_TrackAssemblySinglePin_{side}.json", side=side)
        _write_json(
            track_root / "brake" / f"{asset}_TrackBrakeSimple.json",
            {"Name": f"{asset} Disabled Brake", "Type": "TrackBrake", "Template": "TrackBrakeSimple", "Maximum Torque": 0.0},
        )
        _write_json(
            track_root / "driveline" / f"{asset}_SimpleTrackDriveline.json",
            {"Name": f"{asset} Simple Track Driveline", "Type": "TrackDriveline", "Template": "SimpleTrackDriveline", "Differential Max Bias": 1.0},
        )

    def _contact_material(self, *, young: float | None = None, friction: float | None = None) -> dict[str, Any]:
        return {
            "Coefficient of Friction": float(self.tank.ground.friction_mu if friction is None else friction),
            "Coefficient of Restitution": 0.05,
            "Properties": {
                "Young Modulus": float(getattr(self.args, "chrono_contact_young_modulus", 2.0e7) if young is None else young),
                "Poisson Ratio": 0.30,
            },
            "Coefficients": {
                "Normal Stiffness": float(getattr(self.args, "chrono_contact_kn", 3.0e6)),
                "Normal Damping": float(getattr(self.args, "chrono_contact_gn", 4.0e4)),
                "Tangential Stiffness": float(getattr(self.args, "chrono_contact_kt", 1.0e6)),
                "Tangential Damping": float(getattr(self.args, "chrono_contact_gt", 1.0e4)),
            },
        }

    def _write_track_shoe_json(self, path: Path) -> None:
        g = self.tank.geom
        pitch = self._chrono_track_pitch()
        width = float(g.track_width)
        length = float(g.shoe_length)
        thickness = float(g.shoe_thickness)
        pad_width = max(width * 0.84, 0.05)
        horn_width = max(float(g.guide_horn_width), 0.035)
        horn_height = max(float(g.guide_horn_height), 0.06)
        pin_radius = max(0.09 * thickness, 0.008)
        side_pad_width = max(0.16 * width, 0.04)
        material = self._contact_material()
        data = {
            "Name": f"{self.asset_name} SinglePin TrackShoe",
            "Type": "TrackShoe",
            "Template": "TrackShoeSinglePin",
            "Shoe": {
                "Height": thickness,
                "Pitch": pitch,
                "Mass": float(self.tank.track_shoe_mass_value),
                "Inertia": _round_list([
                    float(self.tank.track_shoe_pitch_inertia_value),
                    max(float(self.tank.track_shoe_mass_value) * (width * width + thickness * thickness) / 12.0, 1.0e-6),
                    max(float(self.tank.track_shoe_mass_value) * (width * width + length * length) / 12.0, 1.0e-6),
                ]),
            },
            "Guide Pin Center": _round_list([0.25 * length, 0.0, 0.5 * thickness + 0.5 * horn_height]),
            "Contact": {
                "Cylinder Material": self._contact_material(young=1.0e9, friction=0.5),
                "Cylinder Shape": {"Radius": pin_radius, "Front Offset": 0.42 * pitch, "Rear Offset": -0.42 * pitch},
                "Shoe Materials": [material, material, material],
                "Shoe Shapes": [
                    {
                        "Type": "BOX",
                        "Ground Contact": True,
                        "Location": [0, 0, -0.25 * thickness],
                        "Orientation": [1, 0, 0, 0],
                        "Dimensions": _round_list([0.92 * length, pad_width, 0.50 * thickness]),
                        "Material Index": 0,
                    },
                    {
                        "Type": "BOX",
                        "Location": [0, 0, 0.25 * thickness],
                        "Orientation": [1, 0, 0, 0],
                        "Dimensions": _round_list([0.86 * length, pad_width, 0.50 * thickness]),
                        "Material Index": 1,
                    },
                    {
                        "Type": "BOX",
                        "Location": _round_list([0.25 * length, 0, 0.5 * thickness + 0.5 * horn_height]),
                        "Orientation": [1, 0, 0, 0],
                        "Dimensions": _round_list([0.32 * length, horn_width, horn_height]),
                        "Material Index": 2,
                    },
                    {
                        "Type": "BOX",
                        "Ground Contact": True,
                        "Location": _round_list([0, 0.5 * width - 0.5 * side_pad_width, 0]),
                        "Orientation": [1, 0, 0, 0],
                        "Dimensions": _round_list([0.92 * length, side_pad_width, 0.35 * thickness]),
                        "Material Index": 0,
                    },
                    {
                        "Type": "BOX",
                        "Ground Contact": True,
                        "Location": _round_list([0, -0.5 * width + 0.5 * side_pad_width, 0]),
                        "Orientation": [1, 0, 0, 0],
                        "Dimensions": _round_list([0.92 * length, side_pad_width, 0.35 * thickness]),
                        "Material Index": 0,
                    },
                ],
            },
            "Visualization": {
                "Primitives": [
                    {"Type": "BOX", "Location": [0, 0, 0], "Orientation": [1, 0, 0, 0], "Dimensions": _round_list([length, width, thickness])}
                ]
            },
        }
        _write_json(path, data)

    def _write_wheel_json(self, path: Path, label: str, radius: float, width: float, gap: float, mass: float) -> None:
        inertia_xz = 0.25 * float(mass) * float(radius) * float(radius)
        inertia_y = 0.5 * float(mass) * float(radius) * float(radius)
        data = {
            "Name": f"{self.asset_name} {label}",
            "Type": "TrackWheel",
            "Template": "DoubleTrackWheel",
            "Wheel": {
                "Radius": float(radius),
                "Width": float(width),
                "Gap": float(gap),
                "Mass": float(mass),
                "Inertia": _round_list([inertia_xz, inertia_y, inertia_xz]),
            },
            "Contact Material": self._contact_material(friction=0.4),
        }
        _write_json(path, data)

    def _write_sprocket_json(self, path: Path, *, mass: float) -> None:
        g = self.tank.geom
        pitch = self._chrono_track_pitch()
        radius = float(g.drive_sprocket_radius)
        teeth = int(self._config_value("track", "sprocket_teeth", getattr(self.args, "sprocket_tooth_count", 0)))
        if teeth <= 0:
            teeth = max(8, int(round(2.0 * math.pi * max(radius, 1.0e-9) / max(pitch, 1.0e-9))))
        if mass <= 0.0:
            mass = max(float(getattr(self.args, "chrono_wheel_mass", 120.0)), 120.0)
        inertia_xz = 0.25 * mass * radius * radius
        inertia_y = 0.5 * mass * radius * radius
        data = {
            "Name": f"{self.asset_name} SinglePin Sprocket",
            "Type": "Sprocket",
            "Template": "SprocketSinglePin",
            "Number Teeth": teeth,
            "Gear Mass": mass,
            "Gear Inertia": _round_list([inertia_xz, inertia_y, inertia_xz]),
            "Axle Inertia": float(getattr(self.args, "sprocket_inertia", 220.0)),
            "Gear Separation": max(0.42 * float(g.track_width), 0.05),
            "Lateral Backlash": 0.02,
            "Profile": {
                "Addendum Radius": radius,
                "Arc Radius": max(0.58 * pitch, 0.02),
                "Arc Centers Radius": radius + max(0.25 * pitch, 0.02),
                "Assembly Radius": max(radius - 0.10 * pitch, 0.5 * radius),
            },
            "Contact Material": self._contact_material(young=1.0e9, friction=0.4),
        }
        _write_json(path, data)

    def _write_idler_json(self, path: Path, *, side: str, sign: float, carrier_mass: float) -> None:
        tensioner = self._config_section("tensioner")
        data = {
            "Name": f"{self.asset_name} Idler {side}",
            "Type": "Idler",
            "Template": "TranslationalIdler",
            "Carrier": {
                "Mass": float(carrier_mass),
                "COM": [0, 0.1 * sign, 0],
                "Inertia": [0.2, 0.2, 0.2],
                "Location Wheel": [0, 0, 0],
                "Location Chassis": [0, 0.2 * sign, 0],
                "Visualization Radius": 0.02,
                "Pitch Angle": 0,
            },
            "Tensioner": {
                "Location Carrier": [0, 0.2 * sign, 0],
                "Location Chassis": [0.5, 0.2 * sign, 0],
                "Preload": float(tensioner.get("preload_n", getattr(self.args, "chrono_tensioner_preload", 2.0e4))),
                "Free Length": float(tensioner.get("free_length_m", getattr(self.args, "chrono_tensioner_free_length", 0.75))),
                "Spring Coefficient": float(tensioner.get("stiffness_n_per_m", getattr(self.args, "chrono_tensioner_stiffness", 1.0e6))),
                "Damping Coefficient": float(tensioner.get("damping_n_s_per_m", getattr(self.args, "chrono_tensioner_damping", 1.4e4))),
            },
            "Idler Wheel Input File": f"{self.asset_name}/idler_wheel/{self.asset_name}_DoubleIdlerWheel_{side}.json",
        }
        _write_json(path, data)

    def _write_suspension_json(
        self,
        path: Path,
        *,
        side: str,
        sign: float,
        arm_mass: float,
        station_index: int,
    ) -> None:
        suspension = self._config_section("suspension")
        lateral_offset = float(suspension.get("arm_lateral_offset_m", 0.12))
        arm_x = 0.28
        arm_z = 0.13
        if self.tank.data.get("calibrated_suspension"):
            side_key = side.lower()
            station_name = self.tank.suspension_station_names_by_side[side_key][station_index - 1]
            wheel_name = self.tank.suspension_kinematics[station_name].road_wheel_name
            station_origin = np.asarray(self.tank.body_map[station_name]["joint"]["origin_xyz"], dtype=np.float64)
            wheel_origin = np.asarray(self.tank.body_map[wheel_name]["joint"]["origin_xyz"], dtype=np.float64)
            relative = station_origin - wheel_origin
            arm_x, _, arm_z = _model_to_chrono_vec([0.0, relative[1], relative[2]])

        arm_length = max(math.hypot(float(arm_x), float(arm_z)), 1.0e-6)
        arm_radius = float(suspension.get("arm_radius_m", 0.03))
        transverse_inertia = float(arm_mass) * (arm_length * arm_length + 3.0 * arm_radius * arm_radius) / 12.0
        axial_inertia = 0.5 * float(arm_mass) * arm_radius * arm_radius
        damper_chassis_xz = suspension.get("damper_location_chassis_xz_m", [-0.128, 0.542])
        damper_arm_xz = suspension.get("damper_location_arm_xz_m", [-0.034, 0.152])
        data = {
            "Name": f"{self.asset_name} Suspension {side} {station_index:02d}",
            "Type": "TrackSuspension",
            "Template": "TranslationalDamperSuspension",
            "Suspension Arm": {
                "Mass": float(arm_mass),
                "COM": _round_list([0.5 * arm_x, lateral_offset * sign, 0.5 * arm_z]),
                "Inertia": _round_list([transverse_inertia, axial_inertia, transverse_inertia]),
                "Location Chassis": _round_list([arm_x, lateral_offset * sign, arm_z]),
                "Location Wheel": _round_list([0, lateral_offset * sign, 0]),
                "Radius": arm_radius,
            },
            "Torsional Spring": {
                "Free Angle": float(suspension.get("free_angle_rad", 0.0)),
                "Spring Constant": float(suspension.get("torsion_stiffness_n_m_per_rad", getattr(self.args, "chrono_suspension_spring_constant", 8.0e4))),
                "Damping Coefficient": float(suspension.get("torsion_damping_n_m_s_per_rad", getattr(self.args, "chrono_suspension_damping_coefficient", 2.0e3))),
                "Preload": float(suspension.get("preload_n_m", getattr(self.args, "chrono_suspension_preload", -1.0e4))),
            },
            "Damper": {
                "Location Chassis": _round_list([damper_chassis_xz[0], lateral_offset * sign, damper_chassis_xz[1]]),
                "Location Arm": _round_list([damper_arm_xz[0], lateral_offset * sign, damper_arm_xz[1]]),
                "Damping Coefficient": float(suspension.get("aux_damper_n_s_per_m", getattr(self.args, "chrono_aux_damper_coefficient", 1.0e2))),
            },
            "Road Wheel Input File": f"{self.asset_name}/road_wheel/{self.asset_name}_DoubleRoadWheel_{side}.json",
        }
        _write_json(path, data)

    def _relative_track_location(self, body_name: str) -> list[float]:
        info = self.tank.body_map[body_name]
        center = self.tank.point_for_body_to_world(body_name, info["joint"]["origin_xyz"])
        rel = np.asarray(center, dtype=np.float64) - np.asarray(self.tank.cg_world(), dtype=np.float64)
        return _model_to_chrono_vec([0.0, rel[1], rel[2]])

    def _write_track_assembly_json(self, path: Path, *, side: str) -> None:
        side_key = side.lower()
        drive = f"{side_key}_drive_sprocket"
        idler = f"{side_key}_idler"
        road_wheels = self.tank.road_wheel_names_by_side[side_key]
        rollers = self.tank.return_roller_names_by_side[side_key]
        sprocket_location = self._relative_track_location(drive)
        idler_location = self._relative_track_location(idler)
        assembly_override = getattr(self.args, "chrono_idler_assembly_retraction", None)
        assembly_retraction = float(
            self._config_section("tensioner").get("assembly_retraction_m", 0.0)
            if assembly_override is None
            else assembly_override
        )
        if assembly_retraction != 0.0:
            toward_sprocket = math.copysign(1.0, sprocket_location[0] - idler_location[0])
            idler_location[0] += toward_sprocket * assembly_retraction
        shock_stations = {
            int(value) for value in self._config_section("suspension").get("shock_stations", [1, 2, 5, 6])
        }
        road_entries = []
        for index, name in enumerate(road_wheels):
            station_index = index + 1
            road_entries.append(
                {
                    "Input File": f"{self.asset_name}/suspension/{self.asset_name}_Suspension_{side}_{station_index:02d}.json",
                    "Has Shock": station_index in shock_stations,
                    "Location": self._relative_track_location(name),
                }
            )
        roller_entries = [
            {"Input File": f"{self.asset_name}/roller/{self.asset_name}_ReturnRoller_{side}.json", "Location": self._relative_track_location(name)}
            for name in rollers
        ]
        shoe_count = int(getattr(self.args, "chrono_track_shoe_count", 0))
        if shoe_count <= 0:
            configured_count = self._config_section("track").get("shoe_count_per_side")
            if configured_count is not None:
                shoe_count = int(configured_count)
            else:
                shoe_count = int(self.tank.shoe_count[side_key]) + max(0, int(getattr(self.args, "chrono_track_shoe_count_padding", 2)))
        data = {
            "Name": f"{self.asset_name} SinglePin TrackAssembly {side}",
            "Type": "TrackAssembly",
            "Template": "TrackAssemblySinglePin",
            "Sprocket": {"Input File": f"{self.asset_name}/sprocket/{self.asset_name}_SprocketSinglePin_{side}.json", "Location": sprocket_location},
            "Brake": {"Input File": f"{self.asset_name}/brake/{self.asset_name}_TrackBrakeSimple.json"},
            "Idler": {"Input File": f"{self.asset_name}/idler/{self.asset_name}_Idler_{side}.json", "Location": idler_location},
            "Suspension Subsystems": road_entries,
            "Rollers": roller_entries,
            "Track Shoes": {
                "Input File": f"{self.asset_name}/track_shoe/{self.asset_name}_TrackShoeSinglePin.json",
                "Number Shoes": shoe_count,
            },
        }
        _write_json(path, data)

    def _make_rigid_terrain(self):
        terrain = self.veh.RigidTerrain(self.system)
        mat = self.chrono.ChContactMaterialSMC()
        mat.SetFriction(float(self.tank.ground.friction_mu))
        mat.SetRestitution(0.0)
        mat.SetYoungModulus(float(getattr(self.args, "chrono_contact_young_modulus", 2.0e7)))
        mat.SetPoissonRatio(0.30)
        try:
            mat.SetKn(float(getattr(self.args, "chrono_contact_kn", 3.0e6)))
            mat.SetGn(float(getattr(self.args, "chrono_contact_gn", 4.0e4)))
            mat.SetKt(float(getattr(self.args, "chrono_contact_kt", 1.0e6)))
            mat.SetGt(float(getattr(self.args, "chrono_contact_gt", 1.0e4)))
        except Exception:
            pass
        length = max(40.0, float(self.tank.geom.hull_length) + 30.0)
        width = max(12.0, float(self.tank.geom.hull_width) + 6.0)
        terrain.AddPatch(mat, self.chrono.CSYSNORM, length, width)
        terrain.Initialize()
        return terrain

    def _add_obstacle_bodies(self) -> None:
        if self.system is None:
            return
        for obstacle in self.obstacles:
            if getattr(obstacle, "shape", "") != "semicylinder":
                continue
            cx, cy, cz = [float(v) for v in obstacle.center]
            width, height, depth = [float(v) for v in obstacle.dims]
            mat = self.chrono.ChContactMaterialSMC()
            mat.SetFriction(float(self.tank.ground.friction_mu))
            mat.SetYoungModulus(float(getattr(self.args, "chrono_contact_young_modulus", 2.0e7)))
            body = self.chrono.ChBodyEasyBox(float(depth), float(width), float(height), 1000.0, False, True, mat)
            body.SetName("chrono_obstacle_box_proxy")
            body.SetFixed(True)
            body.SetPos(self.chrono.ChVector3d(float(cz), float(cx), float(cy + 0.5 * height)))
            body.EnableCollision(True)
            self.system.AddBody(body)
            self.obstacle_bodies.append(body)

    def _apply_sprocket_torque(self, left_drive_torque: float, right_drive_torque: float) -> None:
        sign = float(getattr(self.args, "chrono_drive_torque_sign", -1.0))
        for side, torque in ((self.veh.LEFT, left_drive_torque), (self.veh.RIGHT, right_drive_torque)):
            assembly = self.vehicle.GetTrackAssembly(side)
            sprocket = assembly.GetSprocket()
            sprocket.ApplyAxleTorque(sign * float(torque))

    def clear_external_loads(self) -> None:
        self.external_shoe_forces.clear()
        self.external_shoe_torques.clear()

    def _side_to_chrono(self, side: str | int) -> int:
        if side == self.veh.LEFT or str(side).lower() == "left":
            return self.veh.LEFT
        if side == self.veh.RIGHT or str(side).lower() == "right":
            return self.veh.RIGHT
        raise ValueError(f"Unknown track side: {side!r}")

    def add_shoe_force(
        self,
        side: str | int,
        index: int,
        force_world: np.ndarray,
        *,
        application_point_world: np.ndarray | None = None,
    ) -> None:
        chrono_side = self._side_to_chrono(side)
        shoe_index = int(index)
        if application_point_world is None:
            state = self.vehicle.GetTrackShoeState(chrono_side, shoe_index)
            application_point_world = _chrono_to_model_vec(state.pos)
        key = (chrono_side, shoe_index)
        self.external_shoe_forces.setdefault(key, []).append(
            (np.asarray(force_world, dtype=np.float64), np.asarray(application_point_world, dtype=np.float64))
        )

    def add_shoe_torque(self, side: str | int, index: int, torque_world: np.ndarray) -> None:
        key = (self._side_to_chrono(side), int(index))
        self.external_shoe_torques.setdefault(key, []).append(np.asarray(torque_world, dtype=np.float64))

    def add_shoe_force_by_body_id(
        self,
        body_id: int,
        force_world: np.ndarray,
        *,
        application_point_world: np.ndarray | None = None,
    ) -> None:
        side_index = self._shoe_body_id_to_side_index(int(body_id))
        if side_index is None:
            return
        side, index = side_index
        self.add_shoe_force(side, index, force_world, application_point_world=application_point_world)

    def add_shoe_torque_by_body_id(self, body_id: int, torque_world: np.ndarray) -> None:
        side_index = self._shoe_body_id_to_side_index(int(body_id))
        if side_index is None:
            return
        side, index = side_index
        self.add_shoe_torque(side, index, torque_world)

    def _apply_external_shoe_loads(self) -> None:
        for (side, index), loads in self.external_shoe_forces.items():
            if not loads:
                continue
            body = self.vehicle.GetTrackShoe(side, int(index)).GetShoeBody()
            for force_model, point_model in loads:
                body.AccumulateForce(
                    _model_to_chrono_chvec(self.chrono, force_model),
                    _model_to_chrono_chvec(self.chrono, point_model),
                    False,
                )
        for (side, index), torques in self.external_shoe_torques.items():
            if not torques:
                continue
            body = self.vehicle.GetTrackShoe(side, int(index)).GetShoeBody()
            for torque_model in torques:
                body.AccumulateTorque(_model_to_chrono_chvec(self.chrono, torque_model), False)

    def _sync_tank_state_from_chrono(self) -> None:
        if self.vehicle is None or self.system is None:
            return
        chassis = self.vehicle.GetChassisBody()
        pos = chassis.GetPos()
        rot = chassis.GetRot()
        chrono_yaw, pitch = _quat_to_yaw_pitch_model(rot)
        state = self.tank.copy_state()
        state.forward_pos = -float(pos.x)
        state.lateral_pos = float(pos.y)
        state.heave = float(pos.z)
        state.yaw = -chrono_yaw
        state.pitch = pitch
        lin = chassis.GetLinVel()
        ang = chassis.GetAngVelParent()
        state.forward_speed = -float(lin.x)
        state.heave_rate = float(lin.z)
        state.yaw_rate = -float(ang.z)
        state.pitch_rate = _pitch_rate_from_parent_angular_velocity(chrono_yaw, ang)
        state.left_sprocket_omega = float(self.vehicle.GetTrackAssembly(self.veh.LEFT).GetSprocket().GetAxleSpeed())
        state.right_sprocket_omega = float(self.vehicle.GetTrackAssembly(self.veh.RIGHT).GetSprocket().GetAxleSpeed())
        state.t = float(self.system.GetChTime())
        self.tank.state = self.tank.copy_state(state)
        self.tank._clear_track_related_caches()

    def _build_metrics(
        self,
        *,
        previous_state: MultibodyState,
        left_drive_torque: float,
        right_drive_torque: float,
        dt: float,
    ) -> MultibodyStepMetrics:
        g = self.tank.geom
        left_normal, right_normal = self._track_contact_loads()
        left_traction, right_traction = self._track_tractions()
        left_omega = self.tank.state.left_sprocket_omega
        right_omega = self.tank.state.right_sprocket_omega
        left_track_speed = -left_omega * float(g.drive_sprocket_radius)
        right_track_speed = -right_omega * float(g.drive_sprocket_radius)
        left_ground = self.tank.state.forward_speed - self.tank.state.yaw_rate * float(g.left_track_center_x)
        right_ground = self.tank.state.forward_speed - self.tank.state.yaw_rate * float(g.right_track_center_x)
        left_slip = (left_track_speed - left_ground) / max(abs(left_track_speed), abs(left_ground), 0.25)
        right_slip = (right_track_speed - right_ground) / max(abs(right_track_speed), abs(right_ground), 0.25)
        inv_dt = 1.0 / max(float(dt), 1.0e-12)
        metrics = MultibodyStepMetrics(
            t=self.tank.state.t,
            forward_pos=self.tank.state.forward_pos,
            lateral_pos=self.tank.state.lateral_pos,
            heave=self.tank.state.heave,
            yaw=self.tank.state.yaw,
            pitch=self.tank.state.pitch,
            heave_rate=self.tank.state.heave_rate,
            pitch_rate=self.tank.state.pitch_rate,
            forward_speed=self.tank.state.forward_speed,
            yaw_rate=self.tank.state.yaw_rate,
            left_drive_torque=left_drive_torque,
            right_drive_torque=right_drive_torque,
            left_track_speed=left_track_speed,
            right_track_speed=right_track_speed,
            left_sprocket_omega=left_omega,
            right_sprocket_omega=right_omega,
            left_slip_ratio=left_slip,
            right_slip_ratio=right_slip,
            left_traction=left_traction,
            right_traction=right_traction,
            left_normal_load=left_normal,
            right_normal_load=right_normal,
            rolling_resistance=0.0,
            drag_force=0.0,
            acceleration=(self.tank.state.forward_speed - previous_state.forward_speed) * inv_dt,
            yaw_acceleration=(self.tank.state.yaw_rate - previous_state.yaw_rate) * inv_dt,
            left_contact_shoes=self._contact_shoe_count(self.veh.LEFT),
            right_contact_shoes=self._contact_shoe_count(self.veh.RIGHT),
            obstacle_contact_shoes=0,
        )
        self.tank.last_metrics = metrics
        return metrics

    def _shoe_force(self, side: int, index: int) -> np.ndarray:
        try:
            shoe = self.vehicle.GetTrackShoe(side, index)
            return _chrono_vector(shoe.GetShoeBody().GetContactForce())
        except Exception:
            return np.zeros(3, dtype=np.float64)

    def _track_contact_loads(self) -> tuple[float, float]:
        loads = []
        for side in (self.veh.LEFT, self.veh.RIGHT):
            total = 0.0
            for index in range(int(self.vehicle.GetNumTrackShoes(side))):
                total += max(0.0, float(self._shoe_force(side, index)[2]))
            loads.append(total)
        return float(loads[0]), float(loads[1])

    def _track_tractions(self) -> tuple[float, float]:
        values = []
        for side in (self.veh.LEFT, self.veh.RIGHT):
            total = 0.0
            for index in range(int(self.vehicle.GetNumTrackShoes(side))):
                total += -float(self._shoe_force(side, index)[0])
            values.append(total)
        return float(values[0]), float(values[1])

    def _contact_shoe_count(self, side: int) -> int:
        count = 0
        for index in range(int(self.vehicle.GetNumTrackShoes(side))):
            if float(np.linalg.norm(self._shoe_force(side, index))) > 1.0e-6:
                count += 1
        return count

    def track_contact_scalars(self) -> dict[int, dict[str, float]]:
        area = max(float(self.tank.geom.shoe_length) * float(self.tank.geom.track_width), 1.0e-12)
        scalars: dict[int, dict[str, float]] = {}
        for side in (self.veh.LEFT, self.veh.RIGHT):
            count = int(self.vehicle.GetNumTrackShoes(side))
            for index in range(count):
                normal = max(0.0, float(self._shoe_force(side, index)[2]))
                scalars[self._track_shoe_body_id(side, index)] = {
                    "contact_pressure": normal / area if normal > 1.0e-12 else 0.0,
                    "contact_normal_force": normal,
                    "contact_flag": 1.0 if normal > 1.0e-12 else 0.0,
                    "track_shoe_deflection": 0.0,
                    "track_shoe_pitch": 0.0,
                }
        return scalars

    def diagnostics_by_side(self, *, step: int | None = None) -> list[ConstraintDiagnostics]:
        step_value = 0 if step is None else int(step)
        return [
            ConstraintDiagnostics(step_value, "left", 0.0, 0.0, 0.0, 0.0, 0.0),
            ConstraintDiagnostics(step_value, "right", 0.0, 0.0, 0.0, 0.0, 0.0),
        ]

    def _track_shoe_body_id(self, side: int, index: int) -> int:
        side_offset = 0 if side == self.veh.LEFT else 100_000
        return 1_000_000 + side_offset + int(index)

    def _shoe_body_id_to_side_index(self, body_id: int) -> tuple[int, int] | None:
        local = int(body_id) - 1_000_000
        if 0 <= local < 100_000:
            return self.veh.LEFT, local
        local = int(body_id) - 1_100_000
        if 0 <= local < 100_000:
            return self.veh.RIGHT, local
        return None

    def _body_pose_model(self, body) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = body.GetRot()
        return (
            _chrono_to_model_vec(body.GetPos()),
            _chrono_to_model_vec(q.GetAxisX()),
            _chrono_to_model_vec(q.GetAxisY()),
            _chrono_to_model_vec(q.GetAxisZ()),
        )

    def shoe_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for side in (self.veh.LEFT, self.veh.RIGHT):
            side_name = "left" if side == self.veh.LEFT else "right"
            count = int(self.vehicle.GetNumTrackShoes(side))
            for index in range(count):
                state = self.vehicle.GetTrackShoeState(side, index)
                q = state.rot
                axis_normal = _chrono_to_model_vec(q.GetAxisZ())
                center = _chrono_to_model_vec(state.pos)
                records.append(
                    {
                        "side": side_name,
                        "index": int(index),
                        "body_id": int(self._track_shoe_body_id(side, index)),
                        "center": center,
                        "axis_long": _chrono_to_model_vec(q.GetAxisX()),
                        "axis_width": _chrono_to_model_vec(q.GetAxisY()),
                        "axis_normal": axis_normal,
                    }
                )
        return records

    def track_contact_min_model_y(
        self,
        *,
        surface_offset_extra: float = 0.0,
        bottom_run_only: bool = False,
    ) -> float:
        min_y = float("inf")
        surface_offset = 0.5 * float(self.tank.geom.shoe_thickness) + max(0.0, float(surface_offset_extra))
        half_long = 0.5 * float(self.tank.geom.shoe_length)
        half_width = 0.5 * float(self.tank.geom.track_width)
        for record in self.shoe_records():
            center = np.asarray(record["center"], dtype=np.float64)
            axis_long = np.asarray(record["axis_long"], dtype=np.float64)
            axis_width = np.asarray(record["axis_width"], dtype=np.float64)
            normal = np.asarray(record["axis_normal"], dtype=np.float64)
            axis_long = axis_long / max(float(np.linalg.norm(axis_long)), 1.0e-12)
            axis_width = axis_width / max(float(np.linalg.norm(axis_width)), 1.0e-12)
            normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
            if bottom_run_only and normal[1] < 0.85:
                continue
            if normal[1] > 0.0:
                normal = -normal
            contact_center = center + normal * surface_offset
            for s_long in (-1.0, 1.0):
                for s_width in (-1.0, 1.0):
                    point = contact_center + s_long * half_long * axis_long + s_width * half_width * axis_width
                    min_y = min(min_y, float(point[1]))
        return min_y

    def shift_model_position(self, delta_model: np.ndarray) -> None:
        if self.system is None:
            return
        delta_model_arr = np.asarray(delta_model, dtype=np.float64)
        dx, dy, dz = _model_to_chrono_vec(delta_model_arr)
        delta = self.chrono.ChVector3d(float(dx), float(dy), float(dz))
        for body in self.system.GetBodies():
            pos = body.GetPos()
            body.SetPos(self.chrono.ChVector3d(float(pos.x + delta.x), float(pos.y + delta.y), float(pos.z + delta.z)))
        state = self.tank.copy_state()
        state.lateral_pos += float(delta_model_arr[0])
        state.heave += float(delta_model_arr[1])
        state.forward_pos += float(delta_model_arr[2])
        self.tank.state = self.tank.copy_state(state)
        self.tank._clear_track_related_caches()

    def align_track_contact_to_model_y(
        self,
        target_y: float,
        *,
        surface_offset_extra: float = 0.0,
    ) -> float:
        current_y = self.track_contact_min_model_y(
            surface_offset_extra=surface_offset_extra,
            bottom_run_only=True,
        )
        if not math.isfinite(current_y):
            return 0.0
        delta_y = float(target_y) - float(current_y)
        if abs(delta_y) > 1.0e-12:
            self.shift_model_position(np.array([0.0, delta_y, 0.0], dtype=np.float64))
        return delta_y

    def _add_chrono_double_wheel(
        self,
        mesh: MeshBuilder,
        wheel,
        *,
        part_kind: int,
        body_id: int,
    ) -> None:
        body = wheel.GetBody()
        center, axis_forward, axis_width, axis_up = self._body_pose_model(body)
        radius = float(wheel.GetRadius())
        disk_width = max(float(wheel.GetWidth()), 0.02)
        track_width = float(self.tank.geom.track_width)
        gap = max(track_width - 2.0 * disk_width, 0.0)
        if gap > 0.02:
            offsets = (-0.5 * (gap + disk_width), 0.5 * (gap + disk_width))
        else:
            offsets = (0.0,)
            disk_width = min(track_width, max(disk_width, 0.08))
        for offset in offsets:
            mesh.add_cylinder(
                center=center + axis_width * offset,
                axis_length=axis_width,
                radius_axis_a=axis_forward,
                radius_axis_b=axis_up,
                radius=radius,
                width=disk_width,
                part_kind=part_kind,
                body_id=body_id,
                segments=32,
            )

    def _add_chrono_sprocket(self, mesh: MeshBuilder, sprocket, *, body_id: int) -> None:
        body = sprocket.GetGearBody()
        center, axis_forward, axis_width, axis_up = self._body_pose_model(body)
        radius = float(sprocket.GetAddendumRadius())
        track_width = float(self.tank.geom.track_width)
        disk_width = max(0.12 * track_width, 0.055)
        separation = max(0.42 * track_width, disk_width)
        for offset in (-0.5 * separation, 0.5 * separation):
            mesh.add_cylinder(
                center=center + axis_width * offset,
                axis_length=axis_width,
                radius_axis_a=axis_forward,
                radius_axis_b=axis_up,
                radius=radius,
                width=disk_width,
                part_kind=6,
                body_id=body_id,
                segments=32,
            )

    def _add_chrono_track_shoe(
        self,
        mesh: MeshBuilder,
        *,
        side: int,
        index: int,
        body_id: int,
        track_contact_scalars: dict[int, dict[str, float]] | None,
    ) -> None:
        g = self.tank.geom
        state = self.vehicle.GetTrackShoeState(side, index)
        center = _chrono_to_model_vec(state.pos)
        q = state.rot
        axis_forward = _chrono_to_model_vec(q.GetAxisX())
        axis_width = _chrono_to_model_vec(q.GetAxisY())
        axis_up = _chrono_to_model_vec(q.GetAxisZ())
        width = float(g.track_width)
        length = float(g.shoe_length)
        thickness = float(g.shoe_thickness)
        scalars = (track_contact_scalars or {}).get(body_id, {})
        contact_flag = int(float(scalars.get("contact_flag", 0.0)) > 0.5)
        contact_pressure = float(scalars.get("contact_pressure", 0.0))
        contact_normal_force = float(scalars.get("contact_normal_force", 0.0))
        mesh.add_box(
            center=center,
            axis_x=axis_forward,
            axis_y=axis_width,
            axis_z=axis_up,
            dims=(length, width, thickness),
            part_kind=5,
            body_id=body_id,
            contact_flag=contact_flag,
            contact_pressure=contact_pressure,
            contact_normal_force=contact_normal_force,
            track_shoe_deflection=0.0,
            track_shoe_pitch=0.0,
            contact_face_axis_z=-1 if contact_flag else None,
        )

    def _add_chrono_running_gear(self, mesh: MeshBuilder) -> None:
        base_id = 900_000
        for side in (self.veh.LEFT, self.veh.RIGHT):
            assembly = self.vehicle.GetTrackAssembly(side)
            side_offset = 0 if side == self.veh.LEFT else 50_000
            self._add_chrono_sprocket(mesh, assembly.GetSprocket(), body_id=base_id + side_offset + 1)
            self._add_chrono_double_wheel(mesh, assembly.GetIdlerWheel(), part_kind=6, body_id=base_id + side_offset + 2)
            for index in range(int(assembly.GetNumTrackSuspensions())):
                self._add_chrono_double_wheel(
                    mesh,
                    assembly.GetRoadWheel(index),
                    part_kind=4,
                    body_id=base_id + side_offset + 100 + index,
                )
            for index in range(int(assembly.GetNumRollers())):
                self._add_chrono_double_wheel(
                    mesh,
                    assembly.GetRoller(index),
                    part_kind=7,
                    body_id=base_id + side_offset + 200 + index,
                )

    def build_mesh(
        self,
        *,
        include_outer_grousers: bool = False,
        track_contact_scalars: dict[int, dict[str, float]] | None = None,
    ) -> MeshBuilder:
        del include_outer_grousers
        mesh = MeshBuilder()
        self._add_chrono_running_gear(mesh)
        for side in (self.veh.LEFT, self.veh.RIGHT):
            count = int(self.vehicle.GetNumTrackShoes(side))
            for index in range(count):
                self._add_chrono_track_shoe(
                    mesh,
                    side=side,
                    index=index,
                    body_id=self._track_shoe_body_id(side, index),
                    track_contact_scalars=track_contact_scalars,
                )
        return mesh
