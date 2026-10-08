from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .vtk_io import write_polydata_vtk


VISUAL_DETAIL_SIMPLE = "simple"
VISUAL_DETAIL_FULL = "full"
VISUAL_DETAIL_CHOICES = (VISUAL_DETAIL_SIMPLE, VISUAL_DETAIL_FULL)
RUNNING_GEAR_BODY_ROLES = {"drive_sprocket", "idler", "return_roller", "road_wheel", "suspension_station"}


@dataclass
class GroundParams:
    friction_mu: float = 0.78
    rolling_resistance: float = 0.035
    slip_velocity_regularization: float = 0.35
    yaw_damping: float = 0.0
    drag_coefficient_area: float = 0.0
    air_density: float = 1.225


@dataclass
class MultibodyGeometry:
    mass: float
    gravity: float
    hull_length: float
    hull_width: float
    hull_height: float
    left_track_center_x: float
    right_track_center_x: float
    track_width: float
    track_pitch: float
    shoe_length: float
    shoe_thickness: float
    grouser_height: float
    guide_horn_height: float
    guide_horn_width: float
    road_wheel_radius: float
    return_roller_radius: float
    drive_sprocket_radius: float
    idler_radius: float
    left_road_wheel_z: tuple[float, ...]
    right_road_wheel_z: tuple[float, ...]

    @property
    def half_gauge(self) -> float:
        return 0.5 * abs(self.left_track_center_x - self.right_track_center_x)

    @property
    def yaw_inertia(self) -> float:
        width = max(self.hull_width, abs(self.left_track_center_x) + abs(self.right_track_center_x) + self.track_width)
        return 0.82 * self.mass * (self.hull_length**2 + width**2) / 12.0

    @property
    def pitch_inertia(self) -> float:
        return 0.82 * self.mass * (self.hull_length**2 + self.hull_height**2) / 12.0


@dataclass
class MultibodyState:
    t: float = 0.0
    forward_pos: float = 0.0
    lateral_pos: float = 0.0
    heave: float = 0.0
    heave_rate: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0
    forward_speed: float = 0.0
    yaw_rate: float = 0.0
    left_track_phase: float = 0.0
    right_track_phase: float = 0.0
    left_sprocket_omega: float = 0.0
    right_sprocket_omega: float = 0.0
    joint_angles: dict[str, float] = field(default_factory=dict)
    joint_rates: dict[str, float] = field(default_factory=dict)
    track_shoe_normal_deflections: dict[str, list[float]] = field(default_factory=dict)
    track_shoe_normal_rates: dict[str, list[float]] = field(default_factory=dict)
    track_shoe_pitch_deflections: dict[str, list[float]] = field(default_factory=dict)
    track_shoe_pitch_rates: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class MultibodyStepMetrics:
    t: float
    forward_pos: float
    lateral_pos: float
    heave: float
    yaw: float
    pitch: float
    heave_rate: float
    pitch_rate: float
    forward_speed: float
    yaw_rate: float
    left_drive_torque: float
    right_drive_torque: float
    left_track_speed: float
    right_track_speed: float
    left_sprocket_omega: float
    right_sprocket_omega: float
    left_slip_ratio: float
    right_slip_ratio: float
    left_traction: float
    right_traction: float
    left_normal_load: float
    right_normal_load: float
    rolling_resistance: float
    drag_force: float
    acceleration: float
    yaw_acceleration: float
    left_contact_shoes: int
    right_contact_shoes: int
    obstacle_contact_shoes: int


@dataclass(frozen=True)
class SceneObstacle:
    center: tuple[float, float, float]
    dims: tuple[float, float, float]
    shape: str = "box"
    body_id: int = -1
    part_kind: int = 11
    contact_flag: int = 0


@dataclass(frozen=True)
class SuspensionStationKinematics:
    station_name: str
    road_wheel_name: str
    arm_length: float
    vertical_sensitivity: float
    max_angle: float
    support_z: float


@dataclass
class TrackCircle:
    name: str
    center_zy: np.ndarray
    radius: float
    arc_preference: str


@dataclass
class TrackSuperElementSideResult:
    side: str
    node_z: np.ndarray
    node_y: np.ndarray
    rest_lengths: np.ndarray
    terrain_normal: np.ndarray
    terrain_traction: np.ndarray
    wheel_longitudinal_loads: dict[str, float]
    wheel_vertical_loads: dict[str, float]
    wheel_penetrations: dict[str, float]
    wheel_torques: dict[str, float]
    total_normal_load: float
    total_traction: float
    pitch_moment: float
    undeformed_length: float


class TrackEnvelopePath:
    def __init__(self, segments: list[dict[str, Any]]):
        self.segments = segments
        self.length = float(sum(segment["length"] for segment in segments))

    def sample(self, s: float) -> tuple[np.ndarray, np.ndarray, str, str]:
        if not self.segments:
            raise ValueError("track path has no segments")

        s = s % self.length
        accum = 0.0
        for segment in self.segments:
            seg_len = segment["length"]
            if s <= accum + seg_len or segment is self.segments[-1]:
                local_s = s - accum
                if segment["type"] == "line":
                    a = local_s / max(seg_len, 1.0e-12)
                    pos = segment["start"] + a * (segment["end"] - segment["start"])
                    tangent = _unit2(segment["end"] - segment["start"])
                    return pos, tangent, segment["tag"], segment["type"]

                theta = segment["theta0"] + segment["direction"] * local_s / max(segment["radius"], 1.0e-12)
                pos = segment["center"] + segment["radius"] * np.array([math.cos(theta), math.sin(theta)], dtype=float)
                tangent = segment["direction"] * np.array([-math.sin(theta), math.cos(theta)], dtype=float)
                return pos, _unit2(tangent), segment["tag"], segment["type"]
            accum += seg_len

        last = self.segments[-1]
        if last["type"] == "line":
            return np.asarray(last["end"], dtype=float), _unit2(last["end"] - last["start"]), last["tag"], last["type"]

        theta = last["theta0"] + last["direction"] * last["length"] / max(last["radius"], 1.0e-12)
        pos = last["center"] + last["radius"] * np.array([math.cos(theta), math.sin(theta)], dtype=float)
        tangent = last["direction"] * np.array([-math.sin(theta), math.cos(theta)], dtype=float)
        return pos, _unit2(tangent), last["tag"], last["type"]


@dataclass
class OBJVisualPart:
    object_name: str
    body_name: str
    part_kind: int
    faces: list[list[int]]
    unique_vertex_indices: tuple[int, ...]


class OBJVisualModel:
    def __init__(self, vertices: list[np.ndarray], parts: list[OBJVisualPart]):
        self.vertices = vertices
        self.parts = parts

    @classmethod
    def load(cls, obj_path: Path, part_specs: list[dict[str, Any]]) -> "OBJVisualModel":
        spec_map = {spec["object_name"]: spec for spec in part_specs}
        vertices: list[np.ndarray] = []
        parsed_parts: list[OBJVisualPart] = []
        current_name: str | None = None
        current_faces: list[list[int]] | None = None

        def flush_current() -> None:
            nonlocal current_name, current_faces
            if current_name is None or current_faces is None or current_name not in spec_map:
                current_name = None
                current_faces = None
                return

            unique_indices = sorted({index for face in current_faces for index in face})
            spec = spec_map[current_name]
            parsed_parts.append(
                OBJVisualPart(
                    object_name=current_name,
                    body_name=spec["body_name"],
                    part_kind=_part_kind_from_visual_spec(spec),
                    faces=current_faces,
                    unique_vertex_indices=tuple(unique_indices),
                )
            )
            current_name = None
            current_faces = None

        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    _, xs, ys, zs = line.split()[:4]
                    vertices.append(np.array([float(xs), float(ys), float(zs)], dtype=float))
                    continue

                if line.startswith("o "):
                    flush_current()
                    current_name = line[2:].strip()
                    current_faces = [] if current_name in spec_map else None
                    continue

                if current_faces is not None and line.startswith("f "):
                    current_faces.append([_parse_obj_index(token, len(vertices)) for token in line.split()[1:]])

        flush_current()
        return cls(vertices=vertices, parts=parsed_parts)


class MeshBuilder:
    def __init__(self):
        self.points: list[np.ndarray] = []
        self.faces: list[list[int]] = []
        self.part_kind: list[int] = []
        self.body_id: list[int] = []
        self.contact_flag: list[int] = []
        self.contact_pressure: list[float] = []
        self.contact_normal_force: list[float] = []
        self.track_shoe_deflection: list[float] = []
        self.track_shoe_pitch: list[float] = []

    def add_face(
        self,
        vertices: Iterable[Iterable[float]],
        part_kind: int,
        body_id: int,
        contact_flag: int = 0,
        contact_pressure: float = 0.0,
        contact_normal_force: float = 0.0,
        track_shoe_deflection: float = 0.0,
        track_shoe_pitch: float = 0.0,
    ) -> None:
        ids = []
        for vertex in vertices:
            ids.append(len(self.points))
            self.points.append(np.asarray(vertex, dtype=float))
        self.faces.append(ids)
        self.part_kind.append(part_kind)
        self.body_id.append(body_id)
        self.contact_flag.append(contact_flag)
        self.contact_pressure.append(float(contact_pressure))
        self.contact_normal_force.append(float(contact_normal_force))
        self.track_shoe_deflection.append(float(track_shoe_deflection))
        self.track_shoe_pitch.append(float(track_shoe_pitch))

    def add_box(
        self,
        center: np.ndarray,
        axis_x: np.ndarray,
        axis_y: np.ndarray,
        axis_z: np.ndarray,
        dims: tuple[float, float, float],
        part_kind: int,
        body_id: int,
        contact_flag: int = 0,
        contact_pressure: float = 0.0,
        contact_normal_force: float = 0.0,
        track_shoe_deflection: float = 0.0,
        track_shoe_pitch: float = 0.0,
        contact_face_axis_z: int | None = None,
    ) -> None:
        ax = _unit(axis_x) * (0.5 * dims[0])
        ay = _unit(axis_y) * (0.5 * dims[1])
        az = _unit(axis_z) * (0.5 * dims[2])
        corners = [
            center - ax - ay - az,
            center + ax - ay - az,
            center + ax + ay - az,
            center - ax + ay - az,
            center - ax - ay + az,
            center + ax - ay + az,
            center + ax + ay + az,
            center - ax + ay + az,
        ]
        face_ids = [
            [0, 1, 2, 3],
            [4, 7, 6, 5],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ]
        for face_index, face in enumerate(face_ids):
            face_contact_flag = contact_flag
            face_contact_pressure = contact_pressure
            face_contact_normal_force = contact_normal_force
            if contact_face_axis_z is not None:
                is_contact_face = (contact_face_axis_z < 0 and face_index == 0) or (
                    contact_face_axis_z > 0 and face_index == 1
                )
                if not is_contact_face:
                    face_contact_flag = 0
                    face_contact_pressure = 0.0
                    face_contact_normal_force = 0.0
            self.add_face(
                [corners[i] for i in face],
                part_kind=part_kind,
                body_id=body_id,
                contact_flag=face_contact_flag,
                contact_pressure=face_contact_pressure,
                contact_normal_force=face_contact_normal_force,
                track_shoe_deflection=track_shoe_deflection,
                track_shoe_pitch=track_shoe_pitch,
            )

    def add_cylinder(
        self,
        center: np.ndarray,
        axis_length: np.ndarray,
        radius_axis_a: np.ndarray,
        radius_axis_b: np.ndarray,
        radius: float,
        width: float,
        part_kind: int,
        body_id: int,
        segments: int = 24,
    ) -> None:
        axis_l = _unit(axis_length)
        axis_a = _unit(radius_axis_a)
        axis_b = _unit(radius_axis_b)

        centers = [center - axis_l * (0.5 * width), center + axis_l * (0.5 * width)]
        rings: list[list[np.ndarray]] = []
        for cap_center in centers:
            ring: list[np.ndarray] = []
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                ring.append(cap_center + radius * math.cos(angle) * axis_a + radius * math.sin(angle) * axis_b)
            rings.append(ring)

        for i in range(segments):
            j = (i + 1) % segments
            self.add_face([rings[0][i], rings[0][j], rings[1][j], rings[1][i]], part_kind=part_kind, body_id=body_id)

        self.add_face(reversed(rings[0]), part_kind=part_kind, body_id=body_id)
        self.add_face(rings[1], part_kind=part_kind, body_id=body_id)

    def add_half_cylinder(
        self,
        center: np.ndarray,
        axis_width: np.ndarray,
        axis_height: np.ndarray,
        axis_long: np.ndarray,
        radius: float,
        width: float,
        part_kind: int,
        body_id: int,
        contact_flag: int = 0,
        contact_pressure: float = 0.0,
        contact_normal_force: float = 0.0,
        track_shoe_deflection: float = 0.0,
        track_shoe_pitch: float = 0.0,
        segments: int = 20,
    ) -> None:
        axis_w = _unit(axis_width)
        axis_h = _unit(axis_height)
        axis_l = _unit(axis_long)
        cap_centers = [center - axis_w * (0.5 * width), center + axis_w * (0.5 * width)]
        rings: list[list[np.ndarray]] = []
        for cap_center in cap_centers:
            ring: list[np.ndarray] = []
            for i in range(segments + 1):
                theta = math.pi - math.pi * i / segments
                radial = math.cos(theta) * axis_l + math.sin(theta) * axis_h
                ring.append(cap_center + radius * radial)
            rings.append(ring)

        for i in range(segments):
            self.add_face(
                [rings[0][i], rings[0][i + 1], rings[1][i + 1], rings[1][i]],
                part_kind=part_kind,
                body_id=body_id,
                contact_flag=contact_flag,
                contact_pressure=contact_pressure,
                contact_normal_force=contact_normal_force,
                track_shoe_deflection=track_shoe_deflection,
                track_shoe_pitch=track_shoe_pitch,
            )

        base_face = [rings[0][0], rings[1][0], rings[1][-1], rings[0][-1]]
        self.add_face(
            base_face,
            part_kind=part_kind,
            body_id=body_id,
            contact_flag=contact_flag,
            contact_pressure=contact_pressure,
            contact_normal_force=contact_normal_force,
            track_shoe_deflection=track_shoe_deflection,
            track_shoe_pitch=track_shoe_pitch,
        )

        left_cap = [cap_centers[0], *rings[0]]
        right_cap = [cap_centers[1], *reversed(rings[1])]
        self.add_face(
            left_cap,
            part_kind=part_kind,
            body_id=body_id,
            contact_flag=contact_flag,
            contact_pressure=contact_pressure,
            contact_normal_force=contact_normal_force,
            track_shoe_deflection=track_shoe_deflection,
            track_shoe_pitch=track_shoe_pitch,
        )
        self.add_face(
            right_cap,
            part_kind=part_kind,
            body_id=body_id,
            contact_flag=contact_flag,
            contact_pressure=contact_pressure,
            contact_normal_force=contact_normal_force,
            track_shoe_deflection=track_shoe_deflection,
            track_shoe_pitch=track_shoe_pitch,
        )

    def write_vtk(self, path: Path) -> None:
        write_polydata_vtk(
            path,
            "Chrono tracked-vehicle multibody model",
            np.asarray(self.points, dtype=float),
            self.faces,
            {
                "part_kind": ("int", self.part_kind),
                "body_id": ("int", self.body_id),
                "contact_flag": ("int", self.contact_flag),
                "contact_pressure": ("float", self.contact_pressure),
                "contact_normal_force": ("float", self.contact_normal_force),
                "track_shoe_deflection": ("float", self.track_shoe_deflection),
                "track_shoe_pitch": ("float", self.track_shoe_pitch),
            },
        )


class MultibodyRigidGroundTank:
    def __init__(
        self,
        model_path: Path,
        ground: GroundParams,
        mass: float | None = None,
        track_pitch: float | None = None,
        track_width: float | None = None,
        shoe_length: float | None = None,
        track_shoe_mass: float | None = None,
        track_shoe_pitch_inertia: float | None = None,
        shoe_thickness: float = 0.070,
        grouser_height: float = 0.040,
        guide_horn_height: float = 0.130,
        guide_horn_width: float = 0.090,
        suspension_wave_amplitude: float = 0.0,
        suspension_wave_frequency: float = 0.0,
        suspension_stiffness: float = 1800.0,
        suspension_damping: float = 220.0,
        suspension_max_angle: float = 0.35,
        sprocket_inertia: float = 220.0,
        sprocket_damping: float = 450.0,
        track_shoe_dynamics: bool = True,
        track_shoe_normal_stiffness: float = 2.0e6,
        track_shoe_normal_damping: float = 2.5e4,
        track_pin_bending_stiffness: float = 4.0e5,
        track_pin_bending_damping: float = 8.0e3,
        track_pin_pitch_stiffness: float = 2.0e4,
        track_pin_pitch_damping: float = 1.2e3,
        track_shoe_max_deflection: float = 0.080,
        track_shoe_max_pitch: float = 0.35,
    ):
        self.model_path = Path(model_path)
        self.ground = ground
        self.data = json.loads(self.model_path.read_text(encoding="utf-8"))
        self._normalize_symmetric_hardpoints()
        self._apply_suspension_arm_hardpoint_calibration()
        self.body_map = {body["name"]: body for body in self.data["bodies"]}
        self.body_index = {name: index + 1 for index, name in enumerate(sorted(self.body_map))}
        self.visual_model: OBJVisualModel | None = None
        self.active_revolute_roles = {
            "road_wheel",
            "return_roller",
            "drive_sprocket",
            "idler",
            "suspension_station",
        }
        self.revolute_bodies = [
            body["name"]
            for body in self.data["bodies"]
            if body["joint"]["type"] == "revolute" and body["body_role"] in self.active_revolute_roles
        ]
        self.body_chain_cache: dict[str, list[str]] = {}
        self.suspension_wave_amplitude = float(suspension_wave_amplitude)
        self.suspension_wave_frequency = float(suspension_wave_frequency)
        self.suspension_stiffness = float(max(1.0, suspension_stiffness))
        self.suspension_damping = float(max(0.0, suspension_damping))
        self.suspension_max_angle = float(max(0.01, suspension_max_angle))
        self.sprocket_inertia = float(max(1.0e-6, sprocket_inertia))
        self.sprocket_damping = float(max(0.0, sprocket_damping))
        self.track_shoe_dynamics_enabled = bool(track_shoe_dynamics)
        self.track_shoe_normal_stiffness = float(max(1.0, track_shoe_normal_stiffness))
        self.track_shoe_normal_damping = float(max(0.0, track_shoe_normal_damping))
        self.track_pin_bending_stiffness = float(max(0.0, track_pin_bending_stiffness))
        self.track_pin_bending_damping = float(max(0.0, track_pin_bending_damping))
        self.track_pin_pitch_stiffness = float(max(0.0, track_pin_pitch_stiffness))
        self.track_pin_pitch_damping = float(max(0.0, track_pin_pitch_damping))
        self.track_shoe_max_deflection = float(max(0.0, track_shoe_max_deflection))
        self.track_shoe_max_pitch = float(max(0.0, track_shoe_max_pitch))
        self.track_super_element_relaxation = 0.45
        self.track_super_element_iterations = 16
        self.track_super_element_relaxation_min = 0.12
        self.track_super_element_relaxation_max = 1.15
        self.track_vehicle_coupling_iterations = 4
        self.track_vehicle_coupling_tolerance = 5.0e-6
        self.heave_damping = 6.0
        self.pitch_damping = 3.0
        self.scene_obstacles: tuple[SceneObstacle, ...] = ()

        self.geom = self._build_geometry(
            mass=mass,
            track_pitch=track_pitch,
            track_width=track_width,
            shoe_length=shoe_length,
            shoe_thickness=shoe_thickness,
            grouser_height=grouser_height,
            guide_horn_height=guide_horn_height,
            guide_horn_width=guide_horn_width,
        )
        self.track_wheel_visual_clearance = 0.006
        self.state = MultibodyState(
            joint_angles={name: 0.0 for name in self.revolute_bodies},
            joint_rates={name: 0.0 for name in self.revolute_bodies},
        )
        self.role_groups = self._build_role_groups()
        self.suspension_kinematics = self._build_suspension_kinematics()
        self.base_reference = np.array(
            [
                self.body_map["hull"]["joint"]["origin_xyz"][0],
                0.0,
                self.body_map["hull"]["joint"]["origin_xyz"][2],
            ],
            dtype=float,
        )

        initial_track_paths = {
            "left": self._build_track_path("left"),
            "right": self._build_track_path("right"),
        }
        calibrated_geometry = self.data.get("calibrated_geometry", {})
        configured_shoe_count = calibrated_geometry.get("track_shoe_count_per_side")
        self.shoe_count = {}
        for side, path in initial_track_paths.items():
            if isinstance(configured_shoe_count, dict):
                side_count = configured_shoe_count.get(side)
            else:
                side_count = configured_shoe_count
            if side_count is None:
                side_count = round(path.length / max(self.geom.track_pitch, 1.0e-6))
            self.shoe_count[side] = max(12, int(side_count))
        self.actual_pitch = {
            side: initial_track_paths[side].length / self.shoe_count[side]
            for side in initial_track_paths
        }
        self.visual_track_phase_offset = {
            side: 0.32 * self.actual_pitch[side]
            for side in initial_track_paths
        }
        self.track_shoe_names = {
            side: [f"{side}_track_shoe_{index:03d}" for index in range(self.shoe_count[side])]
            for side in initial_track_paths
        }
        self.rotating_bodies_by_side = {
            side: [
                f"{side}_drive_sprocket",
                f"{side}_idler",
                *[name for name in self.role_groups["return_roller"] if name.startswith(side)],
                *[name for name in self.role_groups["road_wheel"] if name.startswith(side)],
            ]
            for side in ("left", "right")
        }
        self.suspension_station_names_by_side = {
            side: sorted(
                [name for name in self.role_groups["suspension_station"] if name.startswith(side)],
                key=lambda item: self.body_map[item]["station"] or 0,
            )
            for side in ("left", "right")
        }
        self.return_roller_names_by_side = {
            side: sorted(
                [name for name in self.role_groups["return_roller"] if name.startswith(side)],
                key=lambda item: self.body_map[item]["joint"]["origin_xyz"][2],
            )
            for side in ("left", "right")
        }
        self.road_wheel_names_by_side = {
            side: [self.suspension_kinematics[name].road_wheel_name for name in self.suspension_station_names_by_side[side]]
            for side in ("left", "right")
        }
        self.suspension_station_pairs = list(
            zip(
                self.suspension_station_names_by_side["left"],
                self.suspension_station_names_by_side["right"],
            )
        )
        self.longitudinal_cg_offset = self._build_balanced_longitudinal_cg_offset()
        self.cg_reference = self._build_cg_reference()
        self.station_static_loads = self._build_static_station_loads()
        self.road_wheel_static_loads = {
            kin.road_wheel_name: self.station_static_loads.get(station_name, 0.0)
            for station_name, kin in self.suspension_kinematics.items()
        }
        self.side_static_normal_loads = {
            side: sum(
                self.station_static_loads.get(station_name, 0.0)
                for station_name in self.suspension_station_names_by_side[side]
            )
            for side in ("left", "right")
        }
        self.station_equivalent_stiffness: dict[str, float] = {}
        self.station_equivalent_damping: dict[str, float] = {}
        self.station_equivalent_inertia: dict[str, float] = {}
        station_mass = self.geom.mass / max(1, len(self.suspension_kinematics))
        for station_name, kin in self.suspension_kinematics.items():
            travel_limit = max(kin.max_angle * abs(kin.vertical_sensitivity), 1.0e-4)
            stiffness_scale = self.suspension_stiffness / 800.0
            damping_scale = self.suspension_damping / 220.0
            base_load = max(self.station_static_loads.get(station_name, 0.0), 1.0)
            wheel_rate = stiffness_scale * base_load / travel_limit
            critical_damping = 2.0 * math.sqrt(wheel_rate * station_mass)
            self.station_equivalent_stiffness[station_name] = wheel_rate
            self.station_equivalent_damping[station_name] = 0.35 * damping_scale * critical_damping
            angular_stiffness = wheel_rate * kin.vertical_sensitivity * kin.vertical_sensitivity
            self.station_equivalent_inertia[station_name] = max(
                angular_stiffness / max(self.suspension_stiffness, 1.0e-6),
                0.05 * station_mass * kin.arm_length * kin.arm_length,
                1.0e-6,
            )
        self.last_wheel_contact_reactions: dict[str, dict[str, Any]] = {}
        next_body_id = max(self.body_index.values(), default=0) + 1
        self.track_shoe_body_index: dict[tuple[str, int], int] = {}
        for side in ("left", "right"):
            for index in range(self.shoe_count[side]):
                self.track_shoe_body_index[(side, index)] = next_body_id
                next_body_id += 1
        self.track_shoe_body_id_to_index = {
            body_id: (side, index)
            for (side, index), body_id in self.track_shoe_body_index.items()
        }
        self._ensure_track_shoe_dynamic_state(self.state)

        self.track_pose_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self.track_path_cache: dict[tuple[Any, ...], TrackEnvelopePath] = {}
        self.track_vertical_offsets = {"left": 0.0, "right": 0.0}
        self.track_vertical_offsets = self._build_track_vertical_offsets()
        self._align_initial_pose_to_ground()
        self.wheel_track_contact_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.bottom_shoe_support_cache: dict[tuple[Any, ...], dict[int, float]] = {}
        self.track_super_element_cache: dict[tuple[Any, ...], TrackSuperElementSideResult] = {}
        self.track_super_element_seed: dict[str, dict[str, np.ndarray | float]] = {}
        self.current_track_side_results: dict[str, TrackSuperElementSideResult] = {}
        track_total_length = 2.0 * (
            max(self.geom.left_road_wheel_z) - min(self.geom.left_road_wheel_z) + 2.5 * self.geom.road_wheel_radius
        )
        shoe_mass, shoe_pitch_inertia, shoe_mass_source, shoe_inertia_source = self._resolve_track_shoe_inertia(
            track_shoe_mass,
            track_shoe_pitch_inertia,
        )
        self.track_shoe_mass_value = shoe_mass
        self.track_shoe_pitch_inertia_value = shoe_pitch_inertia
        self.track_shoe_mass_source = shoe_mass_source
        self.track_shoe_pitch_inertia_source = shoe_inertia_source
        self.track_shoe_mass = {
            side: np.full(self.shoe_count[side], shoe_mass, dtype=float)
            for side in ("left", "right")
        }
        self.track_shoe_pitch_inertia = {
            side: np.full(self.shoe_count[side], shoe_pitch_inertia, dtype=float)
            for side in ("left", "right")
        }
        self.track_mass_by_side = {
            side: float(np.sum(self.track_shoe_mass[side]))
            for side in ("left", "right")
        }
        representative_side_track_mass = 0.5 * (
            self.track_mass_by_side["left"] + self.track_mass_by_side["right"]
        )
        self.last_track_shoe_terrain_forces = {
            side: np.zeros((self.shoe_count[side], 3), dtype=float)
            for side in ("left", "right")
        }
        self.last_track_shoe_terrain_moments = {
            side: np.zeros((self.shoe_count[side], 3), dtype=float)
            for side in ("left", "right")
        }
        self.track_weight_per_length = representative_side_track_mass * self.geom.gravity / max(track_total_length, 1.0e-6)
        self.track_super_element_pretension = 0.65 * representative_side_track_mass * self.geom.gravity
        self.track_super_element_ea = 18.0 * self.track_super_element_pretension
        self.track_super_element_terrain_penalty = 1.0e9
        self.track_super_element_terrain_damping = 8.0e6
        self.track_super_element_wheel_penalty = 1.2e7
        self.track_segment_reference = self._build_track_segment_reference()
        self.station_nominal_track_penetration = self._build_nominal_track_penetrations()
        self.last_station_normal_loads = {name: self.station_static_loads.get(name, 0.0) for name in self.suspension_kinematics}
        self.last_station_compressions = {name: 0.0 for name in self.suspension_kinematics}
        self.last_station_track_penetrations = {
            name: self.station_nominal_track_penetration.get(name, 0.0) for name in self.suspension_kinematics
        }
        self.static_pitch_moment_bias = self._static_pitch_moment_bias()
        self.last_metrics = self._zero_metrics()

    def _normalize_symmetric_hardpoints(self) -> None:
        paired_roles = {
            "suspension_station",
            "road_wheel",
            "return_roller",
            "drive_sprocket",
            "idler",
        }
        bodies = {body.get("name"): body for body in self.data.get("bodies", [])}
        processed: set[str] = set()
        for left_name, left_body in bodies.items():
            if left_name in processed or not left_name.startswith("left_"):
                continue
            right_name = f"right_{left_name[5:]}"
            right_body = bodies.get(right_name)
            if right_body is None:
                continue
            if left_body.get("body_role") != right_body.get("body_role"):
                continue
            if left_body.get("body_role") not in paired_roles:
                continue

            left_joint = left_body.get("joint", {})
            right_joint = right_body.get("joint", {})
            left_origin = np.asarray(left_joint.get("origin_xyz", (0.0, 0.0, 0.0)), dtype=float)
            right_origin = np.asarray(right_joint.get("origin_xyz", (0.0, 0.0, 0.0)), dtype=float)
            if left_origin.size != 3 or right_origin.size != 3:
                continue

            avg_abs_x = 0.5 * (abs(float(left_origin[0])) + abs(float(right_origin[0])))
            avg_y = 0.5 * (float(left_origin[1]) + float(right_origin[1]))
            avg_z = 0.5 * (float(left_origin[2]) + float(right_origin[2]))
            left_joint["origin_xyz"] = [avg_abs_x, avg_y, avg_z]
            right_joint["origin_xyz"] = [-avg_abs_x, avg_y, avg_z]
            processed.add(left_name)
            processed.add(right_name)

        measurements = self.data.get("simplified_measurements")
        if not isinstance(measurements, dict):
            return
        track_center = measurements.get("track_center_x_m")
        if isinstance(track_center, dict) and "left" in track_center and "right" in track_center:
            avg_abs = 0.5 * (abs(float(track_center["left"])) + abs(float(track_center["right"])))
            track_center["left"] = avg_abs
            track_center["right"] = -avg_abs
        track_width = measurements.get("track_width_m")
        if isinstance(track_width, dict) and "left" in track_width and "right" in track_width:
            avg_width = 0.5 * (float(track_width["left"]) + float(track_width["right"]))
            track_width["left"] = avg_width
            track_width["right"] = avg_width

    def _apply_suspension_arm_hardpoint_calibration(self) -> None:
        bodies = {body.get("name"): body for body in self.data.get("bodies", [])}
        calibration = self.data.get("calibrated_suspension", {})
        arm_angle = math.radians(float(calibration.get("arm_angle_deg", 25.0)))
        vertical_drop_by_length = math.sin(arm_angle)
        horizontal_reach_by_length = math.cos(arm_angle)

        lengths = calibration.get("arm_length_m_by_station")
        directions = calibration.get("wheel_longitudinal_direction_by_station")

        def station_value(values: Any, station_index: int, default: float) -> float:
            if isinstance(values, dict):
                return float(values.get(str(station_index), values.get(station_index, default)))
            if isinstance(values, (list, tuple)) and station_index <= len(values):
                return float(values[station_index - 1])
            return float(default)

        for side in ("left", "right"):
            for station_index in range(1, 7):
                station_name = f"{side}_suspension_station_{station_index:02d}"
                wheel_name = f"{side}_road_wheel_{station_index:02d}"
                station_body = bodies.get(station_name)
                wheel_body = bodies.get(wheel_name)
                if station_body is None or wheel_body is None:
                    continue

                old_station_origin = np.asarray(station_body.get("joint", {}).get("origin_xyz", (0.0, 0.0, 0.0)), dtype=float)
                wheel_origin = np.asarray(wheel_body.get("joint", {}).get("origin_xyz", (0.0, 0.0, 0.0)), dtype=float)
                if old_station_origin.shape != (3,) or wheel_origin.shape != (3,):
                    continue

                default_length = 0.34 if station_index in (1, 6) else 0.30
                arm_length = station_value(lengths, station_index, default_length)
                direction_z = station_value(directions, station_index, 1.0 if station_index == 6 else -1.0)
                wheel_from_station = np.array(
                    [
                        wheel_origin[0] - old_station_origin[0],
                        -arm_length * vertical_drop_by_length,
                        direction_z * arm_length * horizontal_reach_by_length,
                    ],
                    dtype=float,
                )
                new_station_origin = np.array(
                    [
                        old_station_origin[0],
                        wheel_origin[1] - wheel_from_station[1],
                        wheel_origin[2] - wheel_from_station[2],
                    ],
                    dtype=float,
                )

                station_body["joint"]["origin_xyz"] = [float(value) for value in new_station_origin]

    def _resolve_source_obj_path(self) -> Path:
        raw = self.data.get("source_obj")
        candidates: list[Path] = []
        if raw:
            raw_path = Path(raw)
            candidates.append(raw_path)
            if not raw_path.is_absolute():
                candidates.append(self.model_path.parent / raw_path)
                candidates.append(self.model_path.parent.parent / raw_path)
            candidates.append(self.model_path.parent / raw_path.name)
            candidates.append(self.model_path.parent.parent / raw_path.name)
        model_name = str(self.data.get("model_name", "")).strip()
        if model_name:
            candidates.append(self.model_path.parent.parent / f"{model_name}.obj")
        candidates.append(self.model_path.parent.parent / "source" / "ZTZ96" / "ZTZ96.obj")
        candidates.append(self.model_path.parent / "source" / "ZTZ96" / "ZTZ96.obj")

        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate.resolve()
            except OSError:
                continue

        joined = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not resolve source OBJ for multibody visual mesh.\nCandidates:\n{joined}")

    def get_visual_model(self) -> OBJVisualModel:
        if self.visual_model is None:
            self.visual_model = OBJVisualModel.load(self._resolve_source_obj_path(), self.data["visual_parts"])
        return self.visual_model

    def _build_geometry(
        self,
        mass: float | None,
        track_pitch: float | None,
        track_width: float | None,
        shoe_length: float | None,
        shoe_thickness: float,
        grouser_height: float,
        guide_horn_height: float,
        guide_horn_width: float,
    ) -> MultibodyGeometry:
        measurements = self.data["simplified_measurements"]
        calibrated = self.data.get("calibrated_geometry", {})
        hull_size = self.body_map["hull"]["size_xyz"]

        left_road = sorted(
            [
                body["joint"]["origin_xyz"][2]
                for body in self.data["bodies"]
                if body["body_role"] == "road_wheel" and body["side"] == "left"
            ],
            reverse=True,
        )
        right_road = sorted(
            [
                body["joint"]["origin_xyz"][2]
                for body in self.data["bodies"]
                if body["body_role"] == "road_wheel" and body["side"] == "right"
            ],
            reverse=True,
        )

        track_width_left = float(measurements["track_width_m"]["left"])
        track_width_right = float(measurements["track_width_m"]["right"])
        track_width_model = 0.5 * (track_width_left + track_width_right)

        def calibrated_value(name: str) -> float | None:
            value = calibrated.get(name)
            if value is None:
                return None
            return float(value)

        calibrated_track_width = calibrated_value("track_width_m")
        calibrated_track_pitch = calibrated_value("track_pitch_m")
        calibrated_shoe_length = calibrated_value("shoe_length_m")

        resolved_shoe_length = (
            float(shoe_length)
            if shoe_length is not None
            else calibrated_shoe_length
        )
        if resolved_shoe_length is None:
            resolved_shoe_length = 0.145
        if resolved_shoe_length <= 0.0:
            raise ValueError("shoe_length must be positive")

        resolved_track_pitch = (
            float(track_pitch)
            if track_pitch is not None
            else calibrated_track_pitch
        )
        if resolved_track_pitch is None:
            resolved_track_pitch = resolved_shoe_length
        if resolved_track_pitch <= 0.0:
            raise ValueError("track_pitch must be positive")

        resolved_track_width = (
            float(track_width)
            if track_width is not None
            else calibrated_track_width
        )
        if resolved_track_width is None:
            resolved_track_width = track_width_model
        if resolved_track_width <= 0.0:
            raise ValueError("track_width must be positive")

        model_mass = self.data.get("mass_properties", {}).get("mass_kg")
        if model_mass is None:
            model_mass = self.data.get("vehicle_specifications", {}).get("mass_kg")

        return MultibodyGeometry(
            mass=float(mass if mass is not None else (model_mass if model_mass is not None else 42000.0)),
            gravity=9.81,
            hull_length=float(hull_size[2]),
            hull_width=float(hull_size[0]),
            hull_height=float(hull_size[1]),
            left_track_center_x=float(measurements["track_center_x_m"]["left"]),
            right_track_center_x=float(measurements["track_center_x_m"]["right"]),
            track_width=resolved_track_width,
            track_pitch=resolved_track_pitch,
            shoe_length=resolved_shoe_length,
            shoe_thickness=float(shoe_thickness),
            grouser_height=float(grouser_height),
            guide_horn_height=float(guide_horn_height),
            guide_horn_width=float(guide_horn_width),
            road_wheel_radius=float(measurements["road_wheel_radius_m"]),
            return_roller_radius=float(measurements["return_roller_radius_m"]),
            drive_sprocket_radius=float(measurements["drive_sprocket_radius_m"]),
            idler_radius=float(measurements["idler_radius_m"]),
            left_road_wheel_z=tuple(float(v) for v in left_road),
            right_road_wheel_z=tuple(float(v) for v in right_road),
        )

    def _resolve_track_shoe_inertia(
        self,
        track_shoe_mass: float | None,
        track_shoe_pitch_inertia: float | None,
    ) -> tuple[float, float, str, str]:
        inertial = self.data.get("calibrated_inertial_properties", {})

        def config_value(name: str) -> float | None:
            value = inertial.get(name)
            if value is None:
                return None
            return float(value)

        mass_source = "cli"
        resolved_mass = None if track_shoe_mass is None else float(track_shoe_mass)
        if resolved_mass is None:
            resolved_mass = config_value("track_shoe_mass_kg")
            mass_source = "model"
        if resolved_mass is None:
            reference_mass = float(inertial.get("legacy_reference_vehicle_mass_kg", 42000.0))
            reference_side_fraction = float(inertial.get("legacy_track_mass_fraction_per_side", 0.075))
            reference_shoes_per_side = max(1, int(inertial.get("legacy_shoes_per_side", self.shoe_count["left"])))
            resolved_mass = reference_side_fraction * reference_mass / reference_shoes_per_side
            mass_source = "legacy-reference"
        if resolved_mass <= 0.0:
            raise ValueError("track_shoe_mass must be positive")

        inertia_source = "cli"
        resolved_pitch_inertia = None if track_shoe_pitch_inertia is None else float(track_shoe_pitch_inertia)
        if resolved_pitch_inertia is None:
            resolved_pitch_inertia = config_value("track_shoe_pitch_inertia_kg_m2")
            inertia_source = "model"
        if resolved_pitch_inertia is None:
            resolved_pitch_inertia = resolved_mass * (self.geom.shoe_length**2 + self.geom.shoe_thickness**2) / 12.0
            inertia_source = "box-estimate"
        if resolved_pitch_inertia <= 0.0:
            raise ValueError("track_shoe_pitch_inertia must be positive")

        return resolved_mass, resolved_pitch_inertia, mass_source, inertia_source

    @property
    def total_track_shoe_mass(self) -> float:
        return float(sum(self.track_mass_by_side.values()))

    def _build_role_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for role in (
            "road_wheel",
            "return_roller",
            "drive_sprocket",
            "idler",
            "suspension_station",
            "track_loop",
            "turret",
            "gun",
            "aa_mg_traverse",
            "aa_mg_elevation",
        ):
            groups[role] = [body["name"] for body in self.data["bodies"] if body["body_role"] == role]
        return groups

    def _build_suspension_kinematics(self) -> dict[str, SuspensionStationKinematics]:
        road_wheels_by_parent = {
            body["parent_body"]: body["name"]
            for body in self.data["bodies"]
            if body["body_role"] == "road_wheel" and body["parent_body"]
        }
        kinematics: dict[str, SuspensionStationKinematics] = {}
        for station_name in self.role_groups["suspension_station"]:
            road_wheel_name = road_wheels_by_parent.get(station_name)
            if road_wheel_name is None:
                continue

            station_origin = np.asarray(self.body_map[station_name]["joint"]["origin_xyz"], dtype=float)
            wheel_origin = np.asarray(self.body_map[road_wheel_name]["joint"]["origin_xyz"], dtype=float)
            rel = wheel_origin - station_origin
            arm_length = float(math.hypot(rel[1], rel[2]))
            vertical_sensitivity = float(-rel[2])
            if arm_length < 1.0e-6 or abs(vertical_sensitivity) < 1.0e-6:
                continue

            kinematics[station_name] = SuspensionStationKinematics(
                station_name=station_name,
                road_wheel_name=road_wheel_name,
                arm_length=arm_length,
                vertical_sensitivity=vertical_sensitivity,
                max_angle=self.suspension_max_angle,
                support_z=float(
                    wheel_origin[2] - self.body_map["hull"]["joint"]["origin_xyz"][2]
                ),
            )
        return kinematics

    def _build_static_station_loads(self) -> dict[str, float]:
        static_loads: dict[str, float] = {}
        total_weight = self.geom.mass * self.geom.gravity
        if not self.suspension_station_pairs:
            return static_loads

        uniform = np.full(len(self.suspension_station_pairs), total_weight / len(self.suspension_station_pairs), dtype=float)
        support_z = np.asarray(
            [
                0.5
                * (
                    self.suspension_kinematics[left_name].support_z
                    + self.suspension_kinematics[right_name].support_z
                )
                for left_name, right_name in self.suspension_station_pairs
            ],
            dtype=float,
        )
        pair_loads = self._solve_supported_loads(
            uniform,
            support_z,
            total_weight,
            self._longitudinal_cg_moment(total_weight),
        )
        for (left_name, right_name), pair_load in zip(self.suspension_station_pairs, pair_loads):
            wheel_load = 0.5 * float(pair_load)
            static_loads[left_name] = wheel_load
            static_loads[right_name] = wheel_load
        return static_loads

    def _build_balanced_longitudinal_cg_offset(self) -> float:
        if not self.suspension_station_pairs:
            return 0.0

        support_z = []
        for left_name, right_name in self.suspension_station_pairs:
            support_z.append(
                0.5
                * (
                    self.suspension_kinematics[left_name].support_z
                    + self.suspension_kinematics[right_name].support_z
                )
            )
        return float(np.mean(np.asarray(support_z, dtype=float)))

    def _longitudinal_cg_moment(self, total_weight: float) -> float:
        return float(total_weight) * float(self.longitudinal_cg_offset)

    def _solve_supported_loads(
        self,
        raw_loads: np.ndarray,
        support_z: np.ndarray,
        target_total: float,
        target_moment: float,
    ) -> np.ndarray:
        active = list(range(len(raw_loads)))
        result = np.zeros(len(raw_loads), dtype=float)
        remaining_total = float(target_total)
        remaining_moment = float(target_moment)

        while active:
            if len(active) == 1:
                result[active[0]] = max(0.0, remaining_total)
                break

            raw_active = raw_loads[active]
            z_active = support_z[active]
            constraint = np.vstack([np.ones(len(active), dtype=float), z_active])
            rhs = np.array([remaining_total, remaining_moment], dtype=float) - constraint @ raw_active
            gram = constraint @ constraint.T
            correction = constraint.T @ np.linalg.pinv(gram) @ rhs
            trial = raw_active + correction

            if np.all(trial >= -1.0e-8):
                result[active] = np.maximum(trial, 0.0)
                break

            drop_local = int(np.argmin(trial))
            drop_index = active.pop(drop_local)
            result[drop_index] = 0.0

        return result

    def _pitch_reaction_moment(self) -> float:
        total_weight = self.geom.mass * self.geom.gravity
        return self._longitudinal_cg_moment(total_weight)

    def _cg_vertical_offset_from_base_reference(self) -> float:
        return self.geom.road_wheel_radius + 0.55 * self.geom.hull_height

    def _build_cg_reference(self) -> np.ndarray:
        reference = np.asarray(self.base_reference, dtype=float).copy()
        reference[1] += self._cg_vertical_offset_from_base_reference()
        reference[2] += float(self.longitudinal_cg_offset)
        return reference

    def cg_world(self, state: MultibodyState | None = None) -> np.ndarray:
        return self.body_to_world(self.cg_reference, state=state)

    def cg_world_z(self, state: MultibodyState | None = None) -> float:
        return float(self.cg_world(state)[2])

    def cg_world_y(self, state: MultibodyState | None = None) -> float:
        return float(self.cg_world(state)[1])

    def _compute_station_normal_loads(self, state: MultibodyState | None = None) -> dict[str, float]:
        active_state = self.state if state is None else state
        total_weight = self.geom.mass * self.geom.gravity
        pitch_moment = self._pitch_reaction_moment()

        loads: dict[str, float] = {}
        compressions: dict[str, float] = {}
        if not self.suspension_station_pairs:
            if active_state is self.state:
                self.last_station_compressions = compressions
            return loads

        raw_pair_loads = []
        pair_support_z = []
        for left_name, right_name in self.suspension_station_pairs:
            pair_raw = 0.0
            pair_support_z.append(
                0.5
                * (
                    self.suspension_kinematics[left_name].support_z
                    + self.suspension_kinematics[right_name].support_z
                )
            )
            for name in (left_name, right_name):
                kin = self.suspension_kinematics[name]
                angle = active_state.joint_angles.get(name, 0.0)
                rate = active_state.joint_rates.get(name, 0.0)
                compression = max(0.0, kin.vertical_sensitivity * angle)
                compression_rate = kin.vertical_sensitivity * rate
                spring_force = self.station_equivalent_stiffness[name] * compression
                damper_force = self.station_equivalent_damping[name] * compression_rate
                pair_raw += self.station_static_loads.get(name, 0.0) + spring_force + damper_force
                compressions[name] = compression
            raw_pair_loads.append(pair_raw)

        projected_pair_loads = self._solve_supported_loads(
            np.asarray(raw_pair_loads, dtype=float),
            np.asarray(pair_support_z, dtype=float),
            total_weight,
            pitch_moment,
        )
        for (left_name, right_name), pair_load in zip(self.suspension_station_pairs, projected_pair_loads):
            wheel_load = 0.5 * float(pair_load)
            loads[left_name] = wheel_load
            loads[right_name] = wheel_load

        if active_state is self.state:
            self.last_station_compressions = compressions
        return loads

    def _build_nominal_track_penetrations(self) -> dict[str, float]:
        nominal: dict[str, float] = {}
        for station_name in self.suspension_kinematics:
            contact = self._road_wheel_track_contact(station_name)
            nominal[station_name] = float(contact["penetration"])
        return nominal

    def _road_wheel_circle(self, body_name: str, state: MultibodyState | None = None) -> TrackCircle:
        joint_origin = np.asarray(self.body_map[body_name]["joint"]["origin_xyz"], dtype=float)
        center_world = self.point_for_body_to_world(body_name, joint_origin, state=state)
        return TrackCircle(
            name=body_name,
            center_zy=np.array([float(center_world[2]), float(center_world[1])], dtype=float),
            radius=self._body_radius(body_name),
            arc_preference="bottom",
        )

    def _track_span_endpoints(
        self,
        side: str,
        state: MultibodyState | None = None,
    ) -> tuple[str, str, np.ndarray, np.ndarray]:
        wheel_names = self.road_wheel_names_by_side[side]
        if len(wheel_names) < 2:
            raise ValueError(f"side {side} does not have enough road wheels for a track super-element span")
        front_name = wheel_names[0]
        rear_name = wheel_names[-1]
        rear_circle = self._road_wheel_circle(rear_name, state=state)
        front_circle = self._road_wheel_circle(front_name, state=state)
        start_zy, end_zy = _circle_external_tangent(rear_circle, front_circle, "lower")
        return rear_name, front_name, np.asarray(start_zy, dtype=float), np.asarray(end_zy, dtype=float)

    def _build_track_segment_reference(self) -> dict[str, dict[str, Any]]:
        references: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            rear_name, front_name, start_zy, end_zy = self._track_span_endpoints(side, state=self.state)
            straight_length = float(np.linalg.norm(end_zy - start_zy))
            pitch = max(self.geom.track_pitch, 1.0e-6)
            element_count = max(2, int(round(straight_length / pitch)))
            reference_length = element_count * pitch
            references[side] = {
                "rear_wheel_name": rear_name,
                "front_wheel_name": front_name,
                "initial_rear_angle": float(self.state.joint_angles.get(rear_name, 0.0)),
                "initial_front_angle": float(self.state.joint_angles.get(front_name, 0.0)),
                "reference_length": reference_length,
                "element_count": element_count,
                "pitch": pitch,
            }
        return references

    def _build_track_vertical_offsets(self) -> dict[str, float]:
        return {"left": 0.0, "right": 0.0}

    def _align_initial_pose_to_ground(self) -> None:
        surface_offset = _track_contact_surface_offset(self.geom)
        min_contact_y = math.inf
        sample_offsets = (0.0, -0.5 * self.geom.shoe_length, 0.5 * self.geom.shoe_length)
        for side in ("left", "right"):
            for pose in self.track_shoe_poses(side):
                center = np.asarray(pose["world_center"], dtype=float)
                axis_normal = np.asarray(pose["axis_normal_world"], dtype=float)
                axis_long = np.asarray(pose["axis_long_world"], dtype=float)
                for long_offset in sample_offsets:
                    contact_point = center + long_offset * axis_long + surface_offset * axis_normal
                    min_contact_y = min(min_contact_y, float(contact_point[1]))

        if not math.isfinite(min_contact_y):
            return
        if abs(min_contact_y) <= 1.0e-9:
            return

        self.state.heave -= min_contact_y
        self.track_pose_cache.clear()

    def _settle_initial_pose_to_static_load(self) -> None:
        target_load = self.geom.mass * self.geom.gravity
        if target_load <= 0.0:
            return

        base_state = self.copy_state(self.state)
        base_state.heave_rate = 0.0
        base_state.pitch_rate = 0.0
        base_state.forward_speed = 0.0
        base_state.yaw_rate = 0.0
        base_state.left_sprocket_omega = 0.0
        base_state.right_sprocket_omega = 0.0

        def static_normal_load(heave: float) -> float:
            trial_state = self.copy_state(base_state)
            trial_state.heave = float(heave)
            left_result, right_result = self._static_track_results_for_state(trial_state)
            return float(left_result.total_normal_load + right_result.total_normal_load)

        h_upper = float(base_state.heave)
        load_upper = static_normal_load(h_upper)
        tolerance = 5.0e-3 * target_load
        if abs(load_upper - target_load) <= tolerance:
            self.state.heave = h_upper
            self.state.heave_rate = 0.0
            self._clear_track_related_caches()
            return

        step = max(0.02, 0.5 * self.geom.shoe_thickness)
        if load_upper > target_load:
            h_lower = h_upper
            load_lower = load_upper
            for _ in range(24):
                h_upper += step
                load_upper = static_normal_load(h_upper)
                if load_upper <= target_load:
                    break
                h_lower = h_upper
                load_lower = load_upper
                step *= 1.35
            else:
                self._clear_track_related_caches()
                return
        else:
            h_lower = h_upper
            load_lower = load_upper
            for _ in range(24):
                h_lower -= step
                load_lower = static_normal_load(h_lower)
                if load_lower >= target_load:
                    break
                step *= 1.35
            else:
                self._clear_track_related_caches()
                return

        for _ in range(32):
            h_mid = 0.5 * (h_lower + h_upper)
            load_mid = static_normal_load(h_mid)
            if abs(load_mid - target_load) <= tolerance:
                h_lower = h_mid
                h_upper = h_mid
                break
            if load_mid >= target_load:
                h_lower = h_mid
                load_lower = load_mid
            else:
                h_upper = h_mid
                load_upper = load_mid

        self.state.heave = 0.5 * (h_lower + h_upper)
        self.state.heave_rate = 0.0
        self.state.pitch_rate = 0.0
        self._clear_track_related_caches()

    def _static_track_results_for_state(
        self,
        state: MultibodyState,
    ) -> tuple[TrackSuperElementSideResult, TrackSuperElementSideResult]:
        self._clear_track_related_caches()
        left_result = self._solve_track_super_element_side("left", 0.0, 0.0, state=state)
        right_result = self._solve_track_super_element_side("right", 0.0, 0.0, state=state)
        return left_result, right_result

    def _static_pitch_moment_bias(self) -> float:
        left_result, right_result = self._static_track_results_for_state(self.state)
        return float(left_result.pitch_moment + right_result.pitch_moment)

    def _road_wheel_names_for_side(self, side: str) -> list[str]:
        return list(self.road_wheel_names_by_side[side])

    def _track_super_element_mesh(
        self,
        side: str,
        state: MultibodyState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        active_state = self.state if state is None else state
        reference = self.track_segment_reference[side]
        rear_name = str(reference["rear_wheel_name"])
        front_name = str(reference["front_wheel_name"])
        pitch = float(reference["pitch"])
        threshold = 0.1 * pitch
        rear_angle = float(active_state.joint_angles.get(rear_name, 0.0))
        front_angle = float(active_state.joint_angles.get(front_name, 0.0))
        delta_rear = rear_angle - float(reference["initial_rear_angle"])
        delta_front = front_angle - float(reference["initial_front_angle"])
        segment_length = float(reference["reference_length"])
        segment_length += self.geom.road_wheel_radius * delta_rear
        segment_length -= self.geom.road_wheel_radius * delta_front
        segment_length = max(0.5 * pitch, segment_length)
        first_length = math.fmod(self.geom.road_wheel_radius * rear_angle, pitch)
        if first_length < 0.0:
            first_length += pitch
        if first_length < threshold:
            first_length = threshold
        if first_length > segment_length:
            first_length = segment_length

        element_lengths: list[float] = [first_length]
        remaining = segment_length - first_length
        while remaining > pitch + threshold:
            element_lengths.append(pitch)
            remaining -= pitch
        if remaining > 1.0e-8:
            if remaining < threshold and element_lengths:
                element_lengths[-1] += remaining
            else:
                element_lengths.append(remaining)

        node_z = [0.0]
        for length in element_lengths:
            node_z.append(node_z[-1] + length)
        z_nodes = np.asarray(node_z, dtype=float)
        rest_lengths = np.diff(z_nodes)
        return z_nodes, rest_lengths

    def _bottom_track_profile_guess(
        self,
        side: str,
        node_s: np.ndarray,
        boundary_start: np.ndarray,
        boundary_end: np.ndarray,
        undeformed_length: float,
        state: MultibodyState | None = None,
    ) -> np.ndarray:
        if node_s.size == 0:
            return np.zeros((0, 2), dtype=float)

        line_alpha = node_s / max(undeformed_length, 1.0e-9)
        line_x = boundary_start[0] + (boundary_end[0] - boundary_start[0]) * line_alpha
        line_y = boundary_start[1] + (boundary_end[1] - boundary_start[1]) * line_alpha

        seed = self.track_super_element_seed.get(side)
        if seed:
            prev_s = np.asarray(seed.get("node_s", np.zeros(0, dtype=float)), dtype=float)
            prev_x = np.asarray(seed.get("node_z", np.zeros(0, dtype=float)), dtype=float)
            prev_y = np.asarray(seed.get("node_y", np.zeros(0, dtype=float)), dtype=float)
            prev_length = float(seed.get("undeformed_length", 0.0))
            if prev_s.size >= 2 and prev_x.size == prev_s.size and prev_y.size == prev_s.size and prev_length > 1.0e-9:
                sample_s = np.clip(node_s / max(undeformed_length, 1.0e-9) * prev_length, prev_s[0], prev_s[-1])
                guess_x = np.interp(sample_s, prev_s, prev_x)
                guess_y = np.interp(sample_s, prev_s, prev_y)
                endpoint_line_x = np.linspace(guess_x[0], guess_x[-1], guess_x.size)
                endpoint_line_y = np.linspace(guess_y[0], guess_y[-1], guess_y.size)
                guess_x += line_x - endpoint_line_x
                guess_y += line_y - endpoint_line_y
                guess_x[0] = boundary_start[0]
                guess_x[-1] = boundary_end[0]
                guess_y[0] = boundary_start[1]
                guess_y[-1] = boundary_end[1]
                return np.column_stack([guess_x, guess_y])

        samples_z: list[float] = []
        samples_y: list[float] = []
        surface_offset = _track_contact_surface_offset(self.geom)
        for pose in self.track_shoe_poses(side, state=state):
            if pose["segment"] != "bottom":
                continue
            center = np.asarray(pose["world_center"], dtype=float)
            axis_normal = np.asarray(pose["axis_normal_world"], dtype=float)
            bottom_center = center + surface_offset * axis_normal
            samples_z.append(float(bottom_center[2]))
            samples_y.append(float(bottom_center[1]))

        if len(samples_z) < 2:
            return np.column_stack([line_x, line_y])

        order = np.argsort(np.asarray(samples_z, dtype=float))
        z_sorted = np.asarray(samples_z, dtype=float)[order]
        y_sorted = np.asarray(samples_y, dtype=float)[order]
        guess_y = np.interp(line_x, z_sorted, y_sorted)
        guess_y[0] = boundary_start[1]
        guess_y[-1] = boundary_end[1]
        return np.column_stack([line_x, guess_y])

    def _solve_track_super_element_side(
        self,
        side: str,
        track_speed: float,
        ground_speed: float,
        state: MultibodyState | None = None,
    ) -> TrackSuperElementSideResult:
        active_state = self.state if state is None else state
        cache_key = (
            "track_super_element",
            *self._track_pose_cache_key(side, active_state, use_super_element_profile=True),
            round(track_speed, 8),
            round(ground_speed, 8),
        )
        cached = self.track_super_element_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(self.track_super_element_cache) > 32:
            self.track_super_element_cache.clear()

        node_z, rest_lengths = self._track_super_element_mesh(side, state=active_state)
        if node_z.size < 2:
            result = TrackSuperElementSideResult(
                side=side,
                node_z=np.zeros(0, dtype=float),
                node_y=np.zeros(0, dtype=float),
                rest_lengths=np.zeros(0, dtype=float),
                terrain_normal=np.zeros(0, dtype=float),
                terrain_traction=np.zeros(0, dtype=float),
                wheel_longitudinal_loads={},
                wheel_vertical_loads={},
                wheel_penetrations={},
                wheel_torques={},
                total_normal_load=0.0,
                total_traction=0.0,
                pitch_moment=0.0,
                undeformed_length=0.0,
            )
            self.track_super_element_cache[cache_key] = result
            return result

        wheel_names = self._road_wheel_names_for_side(side)
        rear_name, front_name, boundary_start, boundary_end = self._track_span_endpoints(side, state=active_state)
        wheel_centers_world = np.asarray(
            [
                self.point_for_body_to_world(
                    name,
                    np.asarray(self.body_map[name]["joint"]["origin_xyz"], dtype=float),
                    state=active_state,
                )
                for name in wheel_names
            ],
            dtype=float,
        )
        wheel_centers_zy = np.column_stack([wheel_centers_world[:, 2], wheel_centers_world[:, 1]])
        track_center_x = self.body_map[f"{side}_track"]["joint"]["origin_xyz"][0]
        lateral_world = float(self.body_to_world((track_center_x, 0.0, 0.0), state=active_state)[0])
        undeformed_length = float(np.sum(rest_lengths))
        node_pos = self._bottom_track_profile_guess(
            side,
            node_z,
            boundary_start,
            boundary_end,
            undeformed_length,
            state=active_state,
        )
        node_pos[0] = boundary_start
        node_pos[-1] = boundary_end

        tributary = np.zeros_like(node_z)
        tributary[0] = 0.5 * rest_lengths[0]
        tributary[-1] = 0.5 * rest_lengths[-1]
        if node_z.size > 2:
            tributary[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])

        slip_velocity = track_speed - ground_speed
        traction_factor = math.tanh(slip_velocity / max(self.ground.slip_velocity_regularization, 1.0e-6))
        terrain_heights = self._terrain_height_profile(
            lateral_world,
            node_pos[:, 0],
            footprint_half_length=0.5 * self.geom.shoe_length,
        )

        def evaluate_profile(
            positions: np.ndarray,
            *,
            with_wheel_outputs: bool,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
            node_count = positions.shape[0]
            force = np.zeros((node_count, 2), dtype=float)
            diag = np.full((node_count, 2), 1.0e-6, dtype=float)
            normal_nodes = np.zeros(node_count, dtype=float)
            traction_nodes = np.zeros(node_count, dtype=float)
            wheel_force_sums = None
            wheel_torque_sums = None
            wheel_penetration_sums = None
            if with_wheel_outputs:
                wheel_force_sums = {name: np.zeros(2, dtype=float) for name in wheel_names}
                wheel_torque_sums = {name: 0.0 for name in wheel_names}
                wheel_penetration_sums = {name: 0.0 for name in wheel_names}

            force[:, 1] -= self.track_weight_per_length * tributary
            diag[:, 0] += 0.22 * self.track_super_element_ea / max(self.geom.track_pitch, 1.0e-6)
            diag[:, 1] += 0.18 * self.track_super_element_terrain_penalty * tributary

            edge = positions[1:] - positions[:-1]
            current_length = np.linalg.norm(edge, axis=1)
            current_length = np.maximum(current_length, 1.0e-9)
            unit = edge / current_length[:, None]
            strain = (current_length - rest_lengths) / np.maximum(rest_lengths, 1.0e-9)
            tension = np.maximum(0.0, self.track_super_element_pretension + self.track_super_element_ea * strain)
            internal = tension[:, None] * unit
            np.add.at(force, np.arange(node_count - 1), internal)
            np.add.at(force, np.arange(1, node_count), -internal)
            k_local = np.maximum(
                self.track_super_element_pretension / np.maximum(rest_lengths, 1.0e-9),
                self.track_super_element_ea / np.maximum(rest_lengths, 1.0e-9),
            )
            np.add.at(diag[:, 0], np.arange(node_count - 1), k_local)
            np.add.at(diag[:, 0], np.arange(1, node_count), k_local)
            np.add.at(diag[:, 1], np.arange(node_count - 1), k_local)
            np.add.at(diag[:, 1], np.arange(1, node_count), k_local)

            penetration_ground = terrain_heights - positions[:, 1]
            contact_ground = penetration_ground > 0.0
            if np.any(contact_ground):
                contact_indices = np.flatnonzero(contact_ground)
                cg_z_for_velocity = self.cg_world_z(active_state)
                vertical_velocity = (
                    active_state.heave_rate
                    - (positions[contact_indices, 0] - cg_z_for_velocity) * active_state.pitch_rate
                )
                normal_force = (
                    self.track_super_element_terrain_penalty * penetration_ground[contact_indices]
                    - self.track_super_element_terrain_damping * vertical_velocity
                ) * tributary[contact_indices]
                normal_force = np.maximum(normal_force, 0.0)
                active_contact_indices = contact_indices[normal_force > 0.0]
                active_normal_force = normal_force[normal_force > 0.0]
                force[active_contact_indices, 1] += active_normal_force
                force[active_contact_indices, 0] += self.ground.friction_mu * active_normal_force * traction_factor
                diag[contact_ground, 0] += 0.3 * self.track_super_element_terrain_penalty * tributary[contact_ground]
                diag[contact_ground, 1] += self.track_super_element_terrain_penalty * tributary[contact_ground]
                normal_nodes[active_contact_indices] = active_normal_force
                traction_nodes[active_contact_indices] = self.ground.friction_mu * active_normal_force * traction_factor

            wheel_delta = positions[:, None, :] - wheel_centers_zy[None, :, :]
            wheel_distance = np.linalg.norm(wheel_delta, axis=2)
            wheel_distance = np.maximum(wheel_distance, 1.0e-9)
            wheel_penetration = self.geom.road_wheel_radius - wheel_distance
            wheel_contact = wheel_penetration > 0.0
            if np.any(wheel_contact):
                normal_force = self.track_super_element_wheel_penalty * wheel_penetration * tributary[:, None]
                normal_force = np.where(wheel_contact, normal_force, 0.0)
                direction = wheel_delta / wheel_distance[:, :, None]
                track_force = normal_force[:, :, None] * direction
                force += np.sum(track_force, axis=1)
                diag[:, 0] += np.sum(
                    wheel_contact * (self.track_super_element_wheel_penalty * tributary[:, None] * np.maximum(direction[:, :, 0] ** 2, 0.15)),
                    axis=1,
                )
                diag[:, 1] += np.sum(
                    wheel_contact * (self.track_super_element_wheel_penalty * tributary[:, None] * np.maximum(direction[:, :, 1] ** 2, 0.15)),
                    axis=1,
                )
                if with_wheel_outputs:
                    wheel_reaction = -track_force
                    for wheel_index, name in enumerate(wheel_names):
                        reaction_sum = np.sum(wheel_reaction[:, wheel_index, :], axis=0)
                        wheel_force_sums[name] = reaction_sum
                        wheel_penetration_sums[name] = float(np.max(np.where(wheel_contact[:, wheel_index], wheel_penetration[:, wheel_index], 0.0)))
                        lever = positions - wheel_centers_zy[wheel_index]
                        wheel_torque_sums[name] = float(
                            np.sum(
                                lever[:, 0] * wheel_reaction[:, wheel_index, 1]
                                - lever[:, 1] * wheel_reaction[:, wheel_index, 0]
                            )
                        )

            if with_wheel_outputs:
                wheel_longitudinal = {name: float(force_vec[0]) for name, force_vec in wheel_force_sums.items()}
                wheel_vertical = {
                    name: max(0.0, float(self.road_wheel_static_loads.get(name, 0.0) + force_vec[1]))
                    for name, force_vec in wheel_force_sums.items()
                }
                wheel_penetrations = {name: max(0.0, float(value)) for name, value in wheel_penetration_sums.items()}
            else:
                wheel_longitudinal = {}
                wheel_vertical = {}
                wheel_penetrations = {}
                wheel_torque_sums = {}
            return force, diag, normal_nodes, traction_nodes, wheel_longitudinal, wheel_vertical, wheel_penetrations, wheel_torque_sums

        relaxation = self.track_super_element_relaxation
        previous_direction: np.ndarray | None = None
        for _ in range(self.track_super_element_iterations):
            force, diag, _, _, _, _, _, _ = evaluate_profile(node_pos, with_wheel_outputs=False)
            residual = force[1:-1]
            if residual.size == 0:
                break
            if np.max(np.linalg.norm(residual, axis=1)) < 1.0e-3:
                break
            direction = residual / diag[1:-1]
            if previous_direction is not None:
                delta_direction = direction - previous_direction
                denom = float(np.sum(delta_direction * delta_direction))
                if denom > 1.0e-18:
                    relaxation = float(
                        np.clip(
                            -relaxation * float(np.sum(previous_direction * delta_direction)) / denom,
                            self.track_super_element_relaxation_min,
                            self.track_super_element_relaxation_max,
                        )
                    )
            node_pos[1:-1] += relaxation * direction
            node_pos[0] = boundary_start
            node_pos[-1] = boundary_end
            previous_direction = direction

        _, _, normal_nodes, traction_nodes, wheel_longitudinal, wheel_vertical, wheel_penetrations, wheel_torques = evaluate_profile(
            node_pos,
            with_wheel_outputs=True,
        )
        cg_world_z = self.cg_world_z(active_state)
        cg_world_y = self.cg_world_y(active_state)
        total_normal_load = float(np.sum(normal_nodes))
        total_traction = float(np.sum(traction_nodes))
        pitch_moment = 0.0
        if node_pos.size:
            arm_z = node_pos[:, 0] - cg_world_z
            arm_y = node_pos[:, 1] - cg_world_y
            pitch_moment = float(np.sum(arm_z * normal_nodes - arm_y * traction_nodes))
        self.track_super_element_seed[side] = {
            "node_s": np.asarray(node_z, dtype=float),
            "node_z": np.asarray(node_pos[:, 0], dtype=float),
            "node_y": np.asarray(node_pos[:, 1], dtype=float),
            "undeformed_length": undeformed_length,
        }
        result = TrackSuperElementSideResult(
            side=side,
            node_z=np.asarray(node_pos[:, 0], dtype=float),
            node_y=np.asarray(node_pos[:, 1], dtype=float),
            rest_lengths=np.asarray(rest_lengths, dtype=float),
            terrain_normal=np.asarray(normal_nodes, dtype=float),
            terrain_traction=np.asarray(traction_nodes, dtype=float),
            wheel_longitudinal_loads=wheel_longitudinal,
            wheel_vertical_loads=wheel_vertical,
            wheel_penetrations=wheel_penetrations,
            wheel_torques={name: float(value) for name, value in wheel_torques.items()},
            total_normal_load=total_normal_load,
            total_traction=total_traction,
            pitch_moment=float(pitch_moment),
            undeformed_length=undeformed_length,
        )
        self.track_super_element_cache[cache_key] = result
        return result

    def _drive_limited_track_result(
        self,
        result: TrackSuperElementSideResult,
        state: MultibodyState,
        drive_torque: float,
    ) -> TrackSuperElementSideResult:
        raw_traction = float(result.total_traction)
        if abs(raw_traction) <= 1.0e-12:
            return result

        sprocket_radius = max(self.geom.drive_sprocket_radius, 1.0e-9)
        traction_limit = abs(float(drive_torque)) / sprocket_radius
        if traction_limit <= 1.0e-12:
            limited_traction = 0.0
        else:
            limited_traction = math.copysign(min(abs(raw_traction), traction_limit), raw_traction)

        scale = limited_traction / raw_traction
        if abs(scale - 1.0) <= 1.0e-12:
            return result

        traction_nodes = np.asarray(result.terrain_traction, dtype=float) * scale
        normal_nodes = np.asarray(result.terrain_normal, dtype=float)
        node_z = np.asarray(result.node_z, dtype=float)
        node_y = np.asarray(result.node_y, dtype=float)
        pitch_moment = float(result.pitch_moment)
        if node_z.size == normal_nodes.size and node_y.size == traction_nodes.size:
            cg_world_z = self.cg_world_z(state)
            cg_world_y = self.cg_world_y(state)
            arm_z = node_z - cg_world_z
            arm_y = node_y - cg_world_y
            pitch_moment = float(np.sum(arm_z * normal_nodes - arm_y * traction_nodes))

        return replace(
            result,
            terrain_traction=np.asarray(traction_nodes, dtype=float),
            total_traction=float(limited_traction),
            pitch_moment=pitch_moment,
        )

    def _road_wheel_track_contact(
        self,
        station_name: str,
        state: MultibodyState | None = None,
    ) -> dict[str, Any]:
        active_state = self.state if state is None else state
        kin = self.suspension_kinematics.get(station_name)
        side = self.body_map.get(station_name, {}).get("side")
        if kin is None or side is None:
            return {
                "penetration": 0.0,
                "distance": math.inf,
                "closest_point_world": np.zeros(3, dtype=float),
                "shoe_index": -1,
            }

        side_contacts = self._road_wheel_track_contacts_for_side(side, active_state)
        cached = side_contacts.get(station_name)
        if cached is not None:
            return cached
        return {
            "penetration": 0.0,
            "distance": math.inf,
            "closest_point_world": np.zeros(3, dtype=float),
            "shoe_index": -1,
        }

    def _bottom_shoe_support_lifts(
        self,
        side: str,
        state: MultibodyState | None = None,
    ) -> dict[int, float]:
        active_state = self.state if state is None else state
        cache_key = (
            "bottom_shoe_support",
            *self._track_pose_cache_key(side, active_state, use_super_element_profile=True),
        )
        cached = self.bottom_shoe_support_cache.get(cache_key)
        if cached is not None:
            return cached

        surface_offset = _track_contact_surface_offset(self.geom)
        lifts: dict[int, float] = {}
        sample_offsets = (0.0, 0.5 * self.geom.shoe_length, -0.5 * self.geom.shoe_length)
        for pose in self.track_shoe_poses(side, state=active_state):
            if pose["segment"] != "bottom":
                continue
            center = np.asarray(pose["world_center"], dtype=float)
            axis_long = np.asarray(pose["axis_long_world"], dtype=float)
            axis_normal = np.asarray(pose["axis_normal_world"], dtype=float)
            lift = 0.0
            for long_offset in sample_offsets:
                bottom_point = center + long_offset * axis_long + surface_offset * axis_normal
                terrain_height = self._terrain_height_at_point(float(bottom_point[0]), float(bottom_point[2]))
                lift = max(lift, terrain_height - float(bottom_point[1]))
            lifts[int(pose["index"])] = max(0.0, lift)

        self.bottom_shoe_support_cache[cache_key] = lifts
        return lifts

    def _road_wheel_track_contacts_for_side(
        self,
        side: str,
        state: MultibodyState | None = None,
    ) -> dict[str, dict[str, Any]]:
        active_state = self.state if state is None else state
        cache_key = (
            "wheel_track_contact_side",
            *self._track_pose_cache_key(side, active_state, use_super_element_profile=True),
        )
        cached = self.wheel_track_contact_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(self.wheel_track_contact_cache) > 96:
            self.wheel_track_contact_cache.clear()

        station_names = self.suspension_station_names_by_side[side]
        if not station_names:
            return {}

        radius = self.geom.road_wheel_radius
        influence = radius + 0.75 * self.geom.shoe_length
        surface_offset = _track_contact_surface_offset(self.geom)
        support_lifts = self._bottom_shoe_support_lifts(side, active_state)
        shoe_segments: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        for pose in self.track_shoe_poses(side, state=active_state):
            if pose["segment"] != "bottom":
                continue

            center = np.asarray(pose["world_center"], dtype=float)
            axis_long = np.asarray(pose["axis_long_world"], dtype=float)
            axis_normal = np.asarray(pose["axis_normal_world"], dtype=float)
            support_lift = support_lifts.get(int(pose["index"]), 0.0)
            top_center = center - surface_offset * axis_normal + np.array([0.0, support_lift, 0.0], dtype=float)

            start_world = top_center - 0.5 * self.geom.shoe_length * axis_long
            end_world = top_center + 0.5 * self.geom.shoe_length * axis_long
            start_zy = np.array([float(start_world[2]), float(start_world[1])], dtype=float)
            end_zy = np.array([float(end_world[2]), float(end_world[1])], dtype=float)
            shoe_segments.append(
                (
                    int(pose["index"]),
                    start_world,
                    end_world,
                    start_zy,
                    end_zy,
                    min(float(start_world[2]), float(end_world[2])),
                    max(float(start_world[2]), float(end_world[2])),
                )
            )

        results: dict[str, dict[str, Any]] = {}
        for station_name in station_names:
            kin = self.suspension_kinematics[station_name]
            wheel_origin = np.asarray(self.body_map[kin.road_wheel_name]["joint"]["origin_xyz"], dtype=float)
            wheel_center_world = self.point_for_body_to_world(kin.road_wheel_name, wheel_origin, state=active_state)
            wheel_center_zy = np.array([float(wheel_center_world[2]), float(wheel_center_world[1])], dtype=float)
            wheel_center_z = float(wheel_center_world[2])
            best_distance = math.inf
            best_point_world = np.asarray(wheel_center_world, dtype=float)
            best_index = -1

            for shoe_index, start_world, end_world, start_zy, end_zy, z_min, z_max in shoe_segments:
                if wheel_center_z < z_min - influence or wheel_center_z > z_max + influence:
                    continue
                closest_zy, alpha = _closest_point_on_segment_2d(wheel_center_zy, start_zy, end_zy)
                distance = float(np.linalg.norm(wheel_center_zy - closest_zy))
                if distance < best_distance:
                    best_distance = distance
                    best_index = shoe_index
                    best_point_world = start_world + alpha * (end_world - start_world)

            penetration = max(0.0, radius - best_distance) if math.isfinite(best_distance) else 0.0
            results[station_name] = {
                "penetration": penetration,
                "distance": best_distance,
                "closest_point_world": best_point_world,
                "shoe_index": best_index,
            }

        self.wheel_track_contact_cache[cache_key] = results
        return results

    def _build_track_circle_chain(
        self,
        side: str,
        state: MultibodyState | None = None,
    ) -> tuple[list[TrackCircle], list[str]]:
        if hasattr(self, "return_roller_names_by_side"):
            return_rollers = self.return_roller_names_by_side[side]
        else:
            return_rollers = sorted(
                [name for name in self.role_groups["return_roller"] if name.startswith(side)],
                key=lambda item: self.body_map[item]["joint"]["origin_xyz"][2],
            )
        if hasattr(self, "road_wheel_names_by_side"):
            road_wheels = self.road_wheel_names_by_side[side]
        else:
            road_wheels = sorted(
                [name for name in self.role_groups["road_wheel"] if name.startswith(side)],
                key=lambda item: self.body_map[item]["joint"]["origin_xyz"][2],
                reverse=True,
            )
        endpoint_names = [f"{side}_drive_sprocket", f"{side}_idler"]
        endpoint_names.sort(key=lambda item: self.body_map[item]["joint"]["origin_xyz"][2])
        rear_endpoint, front_endpoint = endpoint_names
        return_rollers = sorted(
            return_rollers,
            key=lambda item: self.body_map[item]["joint"]["origin_xyz"][2],
        )
        road_wheels = sorted(
            road_wheels,
            key=lambda item: self.body_map[item]["joint"]["origin_xyz"][2],
            reverse=True,
        )
        top_chain = [rear_endpoint, *return_rollers, front_endpoint]
        circle_names = [*top_chain, *road_wheels]
        tangent_sides = ["upper"] * max(1, len(top_chain) - 1)
        tangent_sides.extend(["lower"] * max(1, len(road_wheels) + 1))

        circles: list[TrackCircle] = []
        for body_name in circle_names:
            joint_origin = np.asarray(self.body_map[body_name]["joint"]["origin_xyz"], dtype=float)
            center_model = self.point_for_body_in_model(body_name, joint_origin, state=state)
            circles.append(
                TrackCircle(
                    name=body_name,
                    center_zy=np.array([center_model[2], center_model[1]], dtype=float),
                    radius=self._body_radius(body_name),
                    arc_preference=self._track_arc_preference(body_name),
                )
            )
        return circles, tangent_sides

    def _track_path_cache_key(self, side: str, state: MultibodyState) -> tuple[Any, ...]:
        if not hasattr(self, "suspension_station_names_by_side"):
            return (side, "bootstrap")
        suspension_signature = tuple(
            state.joint_angles.get(name, 0.0) for name in self.suspension_station_names_by_side[side]
        )
        return side, suspension_signature

    def _track_arc_preference(self, body_name: str) -> str:
        role = self.body_map[body_name]["body_role"]
        if role in {"drive_sprocket", "idler"}:
            side = str(self.body_map[body_name].get("side", ""))
            other_role = "idler" if role == "drive_sprocket" else "drive_sprocket"
            other_name = f"{side}_{other_role}"
            own_z = float(self.body_map[body_name]["joint"]["origin_xyz"][2])
            other_z = float(self.body_map[other_name]["joint"]["origin_xyz"][2])
            return "front" if own_z > other_z else "rear"
        if role == "return_roller":
            return "top"
        if role == "road_wheel":
            return "bottom"
        raise ValueError(f"unsupported track circle role for {body_name}")

    def _build_track_path(
        self,
        side: str,
        state: MultibodyState | None = None,
    ) -> TrackEnvelopePath:
        active_state = self.state if state is None else state
        cache_key = None
        if hasattr(self, "track_path_cache"):
            cache_key = self._track_path_cache_key(side, active_state)
            cached = self.track_path_cache.get(cache_key)
            if cached is not None:
                return cached

        circles, tangent_sides = self._build_track_circle_chain(side, state=state)
        tangents: list[tuple[np.ndarray, np.ndarray]] = []

        for index, (circle, tangent_side) in enumerate(zip(circles, tangent_sides)):
            next_circle = circles[(index + 1) % len(circles)]
            tangents.append(_circle_external_tangent(circle, next_circle, tangent_side))

        segments: list[dict[str, Any]] = []
        for index, circle in enumerate(circles):
            previous_index = (index - 1) % len(circles)
            incoming = tangents[previous_index][1]
            outgoing = tangents[index][0]
            arc_segment = _build_track_arc_segment(circle, incoming, outgoing)
            if arc_segment is not None:
                segments.append(arc_segment)

            next_circle = circles[(index + 1) % len(circles)]
            line_tag = "top" if tangent_sides[index] == "upper" else "bottom"
            segments.extend(
                _build_track_line_segments(
                    outgoing,
                    tangents[index][1],
                    line_tag,
                    start_name=circle.name,
                    end_name=next_circle.name,
                )
            )

        path = TrackEnvelopePath(segments)
        if cache_key is not None:
            self.track_path_cache[cache_key] = path
        return path

    def _zero_metrics(self) -> MultibodyStepMetrics:
        return MultibodyStepMetrics(
            t=0.0,
            forward_pos=0.0,
            lateral_pos=0.0,
            heave=0.0,
            yaw=0.0,
            pitch=0.0,
            heave_rate=0.0,
            pitch_rate=0.0,
            forward_speed=0.0,
            yaw_rate=0.0,
            left_drive_torque=0.0,
            right_drive_torque=0.0,
            left_track_speed=0.0,
            right_track_speed=0.0,
            left_sprocket_omega=0.0,
            right_sprocket_omega=0.0,
            left_slip_ratio=0.0,
            right_slip_ratio=0.0,
            left_traction=0.0,
            right_traction=0.0,
            left_normal_load=0.5 * self.geom.mass * self.geom.gravity,
            right_normal_load=0.5 * self.geom.mass * self.geom.gravity,
            rolling_resistance=0.0,
            drag_force=0.0,
            acceleration=0.0,
            yaw_acceleration=0.0,
            left_contact_shoes=0,
            right_contact_shoes=0,
            obstacle_contact_shoes=0,
        )

    def _clear_track_related_caches(self) -> None:
        self.track_pose_cache.clear()
        self.track_path_cache.clear()
        self.wheel_track_contact_cache.clear()
        self.bottom_shoe_support_cache.clear()
        self.track_super_element_cache.clear()

    def _update_station_diagnostics_from_track_results(
        self,
        track_results: dict[str, TrackSuperElementSideResult] | None = None,
        *,
        state: MultibodyState | None = None,
    ) -> None:
        active_state = self.state if state is None else state
        normal_loads = self._station_normal_loads_from_wheel_reactions()
        if not any(value > 0.0 for value in normal_loads.values()):
            normal_loads = self._compute_station_normal_loads(state=active_state)
        track_penetrations: dict[str, float] = {}
        compressions: dict[str, float] = {}

        for station_name, kin in self.suspension_kinematics.items():
            track_penetrations[station_name] = float(
                self._road_wheel_track_contact(station_name, state=active_state)["penetration"]
            )
            compressions[station_name] = max(
                0.0,
                kin.vertical_sensitivity * active_state.joint_angles.get(station_name, 0.0),
            )

        self.last_station_normal_loads = normal_loads
        self.last_station_track_penetrations = track_penetrations
        self.last_station_compressions = compressions

    def clear_wheel_contact_reactions(self) -> None:
        self.last_wheel_contact_reactions = {}

    def apply_wheel_contact_reactions(self, reactions: dict[str, dict[str, object]] | None) -> None:
        normalized: dict[str, dict[str, Any]] = {}
        for wheel_name, record in (reactions or {}).items():
            if not isinstance(record, dict):
                continue
            force = np.asarray(record.get("force_world", np.zeros(3, dtype=float)), dtype=float)
            moment = np.asarray(record.get("moment_world", np.zeros(3, dtype=float)), dtype=float)
            if force.shape != (3,):
                force = np.zeros(3, dtype=float)
            if moment.shape != (3,):
                moment = np.zeros(3, dtype=float)
            normalized[str(wheel_name)] = {
                "force_world": force,
                "moment_world": moment,
                "sample_count": int(record.get("sample_count", 0)),
                "contact_count": int(record.get("contact_count", 0)),
            }
        self.last_wheel_contact_reactions = normalized
        normal_loads = self._station_normal_loads_from_wheel_reactions()
        if any(value > 0.0 for value in normal_loads.values()):
            self.last_station_normal_loads = normal_loads

    def _station_normal_loads_from_wheel_reactions(self) -> dict[str, float]:
        loads = {name: 0.0 for name in self.suspension_kinematics}
        for station_name, kin in self.suspension_kinematics.items():
            record = self.last_wheel_contact_reactions.get(kin.road_wheel_name)
            if not record:
                continue
            force = np.asarray(record.get("force_world", np.zeros(3, dtype=float)), dtype=float)
            if force.shape == (3,):
                loads[station_name] = max(0.0, float(force[1]))
        return loads

    def _suspension_contact_torque(
        self,
        station_name: str,
        *,
        state: MultibodyState | None = None,
        track_results: dict[str, TrackSuperElementSideResult] | None = None,
    ) -> float:
        active_state = self.state if state is None else state
        kin = self.suspension_kinematics.get(station_name)
        if kin is None:
            return 0.0

        record = self.last_wheel_contact_reactions.get(kin.road_wheel_name)
        if record:
            force = np.asarray(record.get("force_world", np.zeros(3, dtype=float)), dtype=float)
            moment = np.asarray(record.get("moment_world", np.zeros(3, dtype=float)), dtype=float)
            if force.shape == (3,) and moment.shape == (3,):
                return self._station_axis_torque_from_force_moment(station_name, force, moment, state=active_state)

        side = self.body_map.get(station_name, {}).get("side")
        active_results = self.current_track_side_results if track_results is None else track_results
        side_result = active_results.get(side) if side else None
        if side_result is None:
            return self.station_static_loads.get(station_name, 0.0) * kin.vertical_sensitivity

        vertical_load = max(0.0, float(side_result.wheel_vertical_loads.get(kin.road_wheel_name, 0.0)))
        longitudinal_load = float(side_result.wheel_longitudinal_loads.get(kin.road_wheel_name, 0.0))
        if vertical_load <= 0.0 and abs(longitudinal_load) <= 1.0e-12:
            return self.station_static_loads.get(station_name, 0.0) * kin.vertical_sensitivity

        wheel_origin = np.asarray(self.body_map[kin.road_wheel_name]["joint"]["origin_xyz"], dtype=float)
        wheel_world = self.point_for_body_to_world(kin.road_wheel_name, wheel_origin, state=active_state)
        force_world = np.array([0.0, vertical_load, longitudinal_load], dtype=float)
        moment_world = np.cross(wheel_world, force_world)
        return self._station_axis_torque_from_force_moment(station_name, force_world, moment_world, state=active_state)

    def _station_axis_torque_from_force_moment(
        self,
        station_name: str,
        force_world: np.ndarray,
        moment_world: np.ndarray,
        *,
        state: MultibodyState,
    ) -> float:
        station_body = self.body_map.get(station_name)
        if station_body is None:
            return 0.0
        joint = station_body["joint"]
        pivot = self.point_for_body_to_world(station_name, np.asarray(joint["origin_xyz"], dtype=float), state=state)
        axis_world = self.vector_to_world(np.asarray(joint["axis_xyz"], dtype=float), state=state)
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm <= 1.0e-12:
            return 0.0
        axis_world = axis_world / axis_norm
        moment_about_pivot = np.asarray(moment_world, dtype=float) - np.cross(pivot, np.asarray(force_world, dtype=float))
        return float(np.dot(moment_about_pivot, axis_world))

    def _advance_suspension_state(
        self,
        state: MultibodyState,
        dt: float,
        side: str,
        track_speed: float,
        track_results: dict[str, TrackSuperElementSideResult],
    ) -> float:
        names = self.suspension_station_names_by_side[side]
        if not names:
            return 0.0

        base_phase = state.left_track_phase if side == "left" else state.right_track_phase
        max_angle_change = 0.0
        for index, body_name in enumerate(names):
            angle = state.joint_angles.get(body_name, 0.0)
            rate = state.joint_rates.get(body_name, 0.0)
            kin = self.suspension_kinematics.get(body_name)
            if kin is None:
                continue

            wheel_rate = max(self.station_equivalent_stiffness.get(body_name, 1.0), 1.0e-9)
            wheel_damping = max(self.station_equivalent_damping.get(body_name, 0.0), 0.0)
            angular_stiffness = wheel_rate * kin.vertical_sensitivity * kin.vertical_sensitivity
            angular_damping = wheel_damping * kin.vertical_sensitivity * kin.vertical_sensitivity
            angular_inertia = max(self.station_equivalent_inertia.get(body_name, 1.0), 1.0e-9)

            rest_angle = 0.0
            if self.suspension_wave_amplitude != 0.0 and self.suspension_wave_frequency != 0.0:
                phase = self.suspension_wave_frequency * state.t + 0.55 * index + 0.08 * base_phase
                rest_angle = self.suspension_wave_amplitude * math.sin(phase) * math.tanh(abs(track_speed))

            contact_torque = self._suspension_contact_torque(body_name, state=state, track_results=track_results)
            static_preload = self.station_static_loads.get(body_name, 0.0) * kin.vertical_sensitivity
            spring_torque = angular_stiffness * (rest_angle - angle)
            damping_torque = -angular_damping * rate
            ang_acc = (contact_torque - static_preload + spring_torque + damping_torque) / angular_inertia
            next_rate = rate + ang_acc * dt
            next_angle = angle + next_rate * dt
            next_angle = max(-self.suspension_max_angle, min(self.suspension_max_angle, next_angle))
            if next_angle in (-self.suspension_max_angle, self.suspension_max_angle):
                next_rate = 0.0
            state.joint_rates[body_name] = next_rate
            state.joint_angles[body_name] = next_angle
            max_angle_change = max(max_angle_change, abs(next_angle - angle))

        return max_angle_change

    def _advance_rotating_state(
        self,
        state: MultibodyState,
        dt: float,
        side: str,
        track_speed: float,
    ) -> None:
        for body_name in self.rotating_bodies_by_side[side]:
            radius = self._body_radius(body_name)
            omega = -track_speed / max(radius, 1.0e-6)
            state.joint_rates[body_name] = omega
            state.joint_angles[body_name] = state.joint_angles.get(body_name, 0.0) + omega * dt

    def _predict_vehicle_state_from_track_results(
        self,
        base_state: MultibodyState,
        dt: float,
        left_drive_torque: float,
        right_drive_torque: float,
        left_result: TrackSuperElementSideResult,
        right_result: TrackSuperElementSideResult,
    ) -> MultibodyState:
        g = self.geom
        p = self.ground
        next_state = self.copy_state(base_state)

        x_left = g.left_track_center_x
        x_right = g.right_track_center_x
        sprocket_radius = g.drive_sprocket_radius
        left_track_speed = -base_state.left_sprocket_omega * sprocket_radius
        right_track_speed = -base_state.right_sprocket_omega * sprocket_radius

        total_vertical = left_result.total_normal_load + right_result.total_normal_load
        rolling = p.rolling_resistance * total_vertical * math.tanh(
            base_state.forward_speed / max(p.slip_velocity_regularization, 1.0e-6)
        )
        drag = 0.5 * p.air_density * p.drag_coefficient_area * base_state.forward_speed * abs(base_state.forward_speed)
        total_traction = left_result.total_traction + right_result.total_traction
        yaw_moment = -(x_left * left_result.total_traction + x_right * right_result.total_traction) - p.yaw_damping * base_state.yaw_rate
        acceleration = (total_traction - rolling - drag) / g.mass
        yaw_acceleration = yaw_moment / g.yaw_inertia

        heave_acceleration = (total_vertical - g.mass * g.gravity) / g.mass - self.heave_damping * base_state.heave_rate
        motor_reaction_pitch_moment = float(left_drive_torque) + float(right_drive_torque)
        dynamic_pitch_moment = (
            left_result.pitch_moment
            + right_result.pitch_moment
            + motor_reaction_pitch_moment
            - self.static_pitch_moment_bias
        )
        pitch_acceleration = dynamic_pitch_moment / max(g.pitch_inertia, 1.0e-6)
        pitch_acceleration -= self.pitch_damping * base_state.pitch_rate

        next_state.forward_speed = base_state.forward_speed + acceleration * dt
        next_state.yaw_rate = base_state.yaw_rate + yaw_acceleration * dt
        next_state.yaw = base_state.yaw + next_state.yaw_rate * dt
        next_state.heave_rate = base_state.heave_rate + heave_acceleration * dt
        next_state.heave = base_state.heave + next_state.heave_rate * dt
        next_state.pitch_rate = base_state.pitch_rate + pitch_acceleration * dt
        next_state.pitch = base_state.pitch + next_state.pitch_rate * dt

        left_drive_reaction = left_result.total_traction * sprocket_radius
        right_drive_reaction = right_result.total_traction * sprocket_radius
        left_alpha = (-left_drive_torque + left_drive_reaction - self.sprocket_damping * base_state.left_sprocket_omega) / self.sprocket_inertia
        right_alpha = (-right_drive_torque + right_drive_reaction - self.sprocket_damping * base_state.right_sprocket_omega) / self.sprocket_inertia
        next_state.left_sprocket_omega = base_state.left_sprocket_omega + left_alpha * dt
        next_state.right_sprocket_omega = base_state.right_sprocket_omega + right_alpha * dt

        next_left_track_speed = -next_state.left_sprocket_omega * sprocket_radius
        next_right_track_speed = -next_state.right_sprocket_omega * sprocket_radius
        next_state.lateral_pos = base_state.lateral_pos + next_state.forward_speed * math.sin(next_state.yaw) * dt
        next_state.forward_pos = base_state.forward_pos + next_state.forward_speed * math.cos(next_state.yaw) * dt
        next_state.left_track_phase = base_state.left_track_phase + next_left_track_speed * dt
        next_state.right_track_phase = base_state.right_track_phase + next_right_track_speed * dt
        next_state.t = base_state.t + dt

        track_results = {"left": left_result, "right": right_result}
        self._advance_suspension_state(next_state, dt, "left", next_left_track_speed, track_results)
        self._advance_suspension_state(next_state, dt, "right", next_right_track_speed, track_results)
        self._advance_rotating_state(next_state, dt, "left", next_left_track_speed)
        self._advance_rotating_state(next_state, dt, "right", next_right_track_speed)
        return next_state

    def _iterate_track_vehicle_coupling(
        self,
        dt: float,
        left_drive_torque: float,
        right_drive_torque: float,
    ) -> tuple[TrackSuperElementSideResult, TrackSuperElementSideResult]:
        base_state = self.copy_state()
        coupled_state = self.copy_state(base_state)
        sprocket_radius = self.geom.drive_sprocket_radius
        left_track_speed = -coupled_state.left_sprocket_omega * sprocket_radius
        right_track_speed = -coupled_state.right_sprocket_omega * sprocket_radius
        left_result = self._solve_track_super_element_side(
            "left",
            left_track_speed,
            coupled_state.forward_speed - coupled_state.yaw_rate * self.geom.left_track_center_x,
            state=coupled_state,
        )
        left_result = self._drive_limited_track_result(left_result, coupled_state, left_drive_torque)
        right_result = self._solve_track_super_element_side(
            "right",
            right_track_speed,
            coupled_state.forward_speed - coupled_state.yaw_rate * self.geom.right_track_center_x,
            state=coupled_state,
        )
        right_result = self._drive_limited_track_result(right_result, coupled_state, right_drive_torque)
        track_results = {"left": left_result, "right": right_result}

        for _ in range(self.track_vehicle_coupling_iterations):
            next_state = self._predict_vehicle_state_from_track_results(
                base_state,
                dt,
                left_drive_torque,
                right_drive_torque,
                left_result,
                right_result,
            )
            next_left_track_speed = -next_state.left_sprocket_omega * sprocket_radius
            next_right_track_speed = -next_state.right_sprocket_omega * sprocket_radius
            left_result = self._solve_track_super_element_side(
                "left",
                next_left_track_speed,
                next_state.forward_speed - next_state.yaw_rate * self.geom.left_track_center_x,
                state=next_state,
            )
            left_result = self._drive_limited_track_result(left_result, next_state, left_drive_torque)
            right_result = self._solve_track_super_element_side(
                "right",
                next_right_track_speed,
                next_state.forward_speed - next_state.yaw_rate * self.geom.right_track_center_x,
                state=next_state,
            )
            right_result = self._drive_limited_track_result(right_result, next_state, right_drive_torque)
            max_delta = 0.0
            scalar_fields = (
                "forward_pos",
                "lateral_pos",
                "heave",
                "heave_rate",
                "yaw",
                "pitch",
                "pitch_rate",
                "forward_speed",
                "yaw_rate",
                "left_track_phase",
                "right_track_phase",
                "left_sprocket_omega",
                "right_sprocket_omega",
            )
            for field_name in scalar_fields:
                max_delta = max(max_delta, abs(getattr(next_state, field_name) - getattr(coupled_state, field_name)))
            for station_name in self.suspension_kinematics:
                max_delta = max(
                    max_delta,
                    abs(next_state.joint_angles.get(station_name, 0.0) - coupled_state.joint_angles.get(station_name, 0.0)),
                )
            coupled_state = next_state
            track_results = {"left": left_result, "right": right_result}
            if max_delta <= self.track_vehicle_coupling_tolerance:
                break

        self.state = self.copy_state(coupled_state)
        self.current_track_side_results = track_results
        self._update_station_diagnostics_from_track_results(track_results, state=self.state)
        return left_result, right_result

    def set_scene_obstacles(self, obstacles: Iterable[SceneObstacle]) -> None:
        self.scene_obstacles = tuple(obstacles)
        self._clear_track_related_caches()
        self.track_super_element_seed.clear()
        self.static_pitch_moment_bias = self._static_pitch_moment_bias()

    def step(
        self,
        dt: float,
        left_drive_torque: float,
        right_drive_torque: float,
    ) -> MultibodyStepMetrics:
        self._clear_track_related_caches()
        g = self.geom
        p = self.ground
        previous_state = self.copy_state(self.state)

        left_result, right_result = self._iterate_track_vehicle_coupling(
            dt,
            left_drive_torque,
            right_drive_torque,
        )
        s = self.state
        x_left = g.left_track_center_x
        x_right = g.right_track_center_x
        sprocket_radius = g.drive_sprocket_radius
        obstacle_contact_shoes = self._count_scene_obstacle_contact_shoes()

        self._clear_track_related_caches()
        final_v_left_ground = s.forward_speed - s.yaw_rate * x_left
        final_v_right_ground = s.forward_speed - s.yaw_rate * x_right
        left_track_speed = -s.left_sprocket_omega * sprocket_radius
        right_track_speed = -s.right_sprocket_omega * sprocket_radius
        final_left_result = self._solve_track_super_element_side("left", left_track_speed, final_v_left_ground, state=s)
        final_left_result = self._drive_limited_track_result(final_left_result, s, left_drive_torque)
        final_right_result = self._solve_track_super_element_side("right", right_track_speed, final_v_right_ground, state=s)
        final_right_result = self._drive_limited_track_result(final_right_result, s, right_drive_torque)
        self.current_track_side_results = {"left": final_left_result, "right": final_right_result}
        self._update_station_diagnostics_from_track_results(self.current_track_side_results, state=s)
        left_normal_load = final_left_result.total_normal_load
        right_normal_load = final_right_result.total_normal_load
        left_traction = final_left_result.total_traction
        right_traction = final_right_result.total_traction
        left_slip_velocity = left_track_speed - final_v_left_ground
        right_slip_velocity = right_track_speed - final_v_right_ground
        rolling = p.rolling_resistance * (left_normal_load + right_normal_load) * math.tanh(
            s.forward_speed / max(p.slip_velocity_regularization, 1.0e-6)
        )
        drag = 0.5 * p.air_density * p.drag_coefficient_area * s.forward_speed * abs(s.forward_speed)
        inv_dt = 1.0 / max(dt, 1.0e-12)
        acceleration = (s.forward_speed - previous_state.forward_speed) * inv_dt
        yaw_acceleration = (s.yaw_rate - previous_state.yaw_rate) * inv_dt
        left_denom = max(abs(left_track_speed), abs(final_v_left_ground), 0.25)
        right_denom = max(abs(right_track_speed), abs(final_v_right_ground), 0.25)
        left_contact = self.contact_shoe_count("left")
        right_contact = self.contact_shoe_count("right")

        self.last_metrics = MultibodyStepMetrics(
            t=s.t,
            forward_pos=s.forward_pos,
            lateral_pos=s.lateral_pos,
            heave=s.heave,
            yaw=s.yaw,
            pitch=s.pitch,
            heave_rate=s.heave_rate,
            pitch_rate=s.pitch_rate,
            forward_speed=s.forward_speed,
            yaw_rate=s.yaw_rate,
            left_drive_torque=left_drive_torque,
            right_drive_torque=right_drive_torque,
            left_track_speed=left_track_speed,
            right_track_speed=right_track_speed,
            left_sprocket_omega=s.left_sprocket_omega,
            right_sprocket_omega=s.right_sprocket_omega,
            left_slip_ratio=left_slip_velocity / left_denom,
            right_slip_ratio=right_slip_velocity / right_denom,
            left_traction=left_traction,
            right_traction=right_traction,
            left_normal_load=left_normal_load,
            right_normal_load=right_normal_load,
            rolling_resistance=rolling,
            drag_force=drag,
            acceleration=acceleration,
            yaw_acceleration=yaw_acceleration,
            left_contact_shoes=left_contact,
            right_contact_shoes=right_contact,
            obstacle_contact_shoes=obstacle_contact_shoes,
        )
        return self.last_metrics

    def _update_rotating_bodies(self, dt: float, side: str, track_speed: float) -> None:
        self._advance_rotating_state(self.state, dt, side, track_speed)

    def _update_suspension_states(self, dt: float, side: str, track_speed: float) -> None:
        self._advance_suspension_state(
            self.state,
            dt,
            side,
            track_speed,
            self.current_track_side_results,
        )
        self._clear_track_related_caches()

    def _scene_contact_possible(self) -> bool:
        if not self.scene_obstacles:
            return False

        local_forward_offset = float(self.cg_reference[2])
        front_reach = self.state.forward_pos + max(
            self.body_map["left_idler"]["joint"]["origin_xyz"][2],
            self.body_map["right_idler"]["joint"]["origin_xyz"][2],
            max(self.geom.left_road_wheel_z),
            max(self.geom.right_road_wheel_z),
        ) - local_forward_offset + 1.1 * self.geom.shoe_length
        rear_reach = self.state.forward_pos + min(
            self.body_map["left_drive_sprocket"]["joint"]["origin_xyz"][2],
            self.body_map["right_drive_sprocket"]["joint"]["origin_xyz"][2],
            min(self.geom.left_road_wheel_z),
            min(self.geom.right_road_wheel_z),
        ) - local_forward_offset - 1.1 * self.geom.shoe_length
        for obstacle in self.scene_obstacles:
            front_face = obstacle.center[2] - 0.5 * obstacle.dims[2]
            back_face = obstacle.center[2] + 0.5 * obstacle.dims[2]
            if back_face >= rear_reach - 1.0e-6 and front_face <= front_reach + 1.0e-6:
                return True
        return False

    def _count_scene_obstacle_contact_shoes(self) -> int:
        if not self._scene_contact_possible():
            return 0

        _, obstacle_shoes = self._collect_contact_constraints()
        return obstacle_shoes

    def _resolve_scene_contact_pose(self, iterations: int = 3) -> int:
        if iterations <= 0 or not self._scene_contact_possible():
            return 0

        contact_count = 0
        for _ in range(iterations):
            constraints, obstacle_shoes = self._collect_contact_constraints()
            contact_count = obstacle_shoes
            if not constraints:
                break

            delta_heave, delta_pitch = self._solve_contact_pose_increment(constraints)
            if abs(delta_heave) < 1.0e-8 and abs(delta_pitch) < 1.0e-8:
                break

            self.state.heave = max(0.0, self.state.heave + delta_heave)
            self.state.pitch = max(-0.30, min(0.30, self.state.pitch + delta_pitch))
            if delta_heave > 0.0 and self.state.heave_rate < 0.0:
                self.state.heave_rate = 0.0
            if abs(delta_pitch) > 0.0 and self.state.pitch_rate * delta_pitch > 0.0:
                self.state.pitch_rate = 0.0
            self._enforce_obstacle_front_faces()

        return contact_count

    def _collect_contact_constraints(self) -> tuple[list[tuple[float, float]], int]:
        constraints: list[tuple[float, float]] = []
        obstacle_contact_shoes: set[tuple[str, int]] = set()
        surface_offset = _track_contact_surface_offset(self.geom)

        for side in ("left", "right"):
            for pose in self.track_shoe_poses(side):
                if pose["segment"] != "bottom":
                    continue

                offsets = (
                    0.0,
                    0.5 * self.geom.shoe_length,
                    -0.5 * self.geom.shoe_length,
                )
                shoe_has_obstacle_contact = False
                for long_offset in offsets:
                    world_tip, local_tip = self._shoe_tip_sample(pose, long_offset, surface_offset)
                    terrain_height = self._terrain_height_at_point(world_tip[0], world_tip[2])
                    penetration = terrain_height - world_tip[1]
                    if penetration > 1.0e-8:
                        z_eff = _rotate_x(local_tip - self.cg_reference, self.state.pitch)[2]
                        constraints.append((float(z_eff), float(penetration)))
                    if terrain_height > 0.0:
                        shoe_has_obstacle_contact = True

                if shoe_has_obstacle_contact:
                    obstacle_contact_shoes.add((side, int(pose["index"])))

        return constraints, len(obstacle_contact_shoes)

    def _solve_contact_pose_increment(self, constraints: list[tuple[float, float]]) -> tuple[float, float]:
        pitch_limit = 0.30
        heave_limit = 0.80
        candidates: list[tuple[float, float]] = [(max(p for _, p in constraints), 0.0), (0.0, 0.0)]

        for a, b in constraints:
            if abs(a) > 1.0e-8:
                candidates.append((0.0, b / a))

        for i, (a_i, b_i) in enumerate(constraints):
            for a_j, b_j in constraints[i + 1 :]:
                denom = a_i - a_j
                if abs(denom) <= 1.0e-8:
                    continue
                delta_pitch = (b_i - b_j) / denom
                delta_heave = b_i - a_i * delta_pitch
                candidates.append((delta_heave, delta_pitch))

        best: tuple[float, float] | None = None
        best_cost = math.inf
        pitch_scale = max(0.25 * self.geom.hull_length, 1.0)
        for delta_heave, delta_pitch in candidates:
            if not math.isfinite(delta_heave) or not math.isfinite(delta_pitch):
                continue
            if delta_heave < -1.0e-8 or delta_heave > heave_limit:
                continue
            if abs(delta_pitch) > pitch_limit:
                continue

            if any(delta_heave + a * delta_pitch + 1.0e-8 < b for a, b in constraints):
                continue

            cost = delta_heave * delta_heave + (pitch_scale * delta_pitch) * (pitch_scale * delta_pitch)
            if cost < best_cost:
                best = (delta_heave, delta_pitch)
                best_cost = cost

        if best is not None:
            return best

        max_penetration = max(b for _, b in constraints)
        return max_penetration, 0.0

    def _enforce_obstacle_front_faces(self) -> None:
        if not self.scene_obstacles:
            return

        surface_offset = _track_contact_surface_offset(self.geom)
        pushback = 0.0
        for side in ("left", "right"):
            for pose in self.track_shoe_poses(side):
                if pose["segment"] != "bottom":
                    continue

                world_front_tip, _ = self._shoe_tip_sample(pose, 0.5 * self.geom.shoe_length, surface_offset)
                for obstacle in self.scene_obstacles:
                    if obstacle.shape != "box":
                        continue
                    front_face = obstacle.center[2] - 0.5 * obstacle.dims[2]
                    back_face = obstacle.center[2] + 0.5 * obstacle.dims[2]
                    top_face = obstacle.center[1] + 0.5 * obstacle.dims[1]
                    half_width = 0.5 * obstacle.dims[0]
                    if abs(world_front_tip[0] - obstacle.center[0]) > half_width + 1.0e-6:
                        continue
                    if world_front_tip[1] >= top_face - 1.0e-4:
                        continue
                    tip_z = float(world_front_tip[2])
                    front_contact_band = front_face + 0.75 * self.geom.shoe_length
                    if tip_z < front_face - 1.0e-6 or tip_z > front_contact_band:
                        continue
                    pushback = max(pushback, tip_z - front_face)

        if pushback > 0.0:
            self.state.forward_pos -= pushback
            self.state.forward_speed = min(self.state.forward_speed, 0.0)

    def _shoe_tip_sample(
        self,
        pose: dict[str, Any],
        long_offset: float,
        surface_offset: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_point = (
            np.asarray(pose["local_center"], dtype=float)
            + long_offset * np.asarray(pose["axis_long_local"], dtype=float)
            + surface_offset * np.asarray(pose["axis_normal_local"], dtype=float)
        )
        world_point = (
            np.asarray(pose["world_center"], dtype=float)
            + long_offset * np.asarray(pose["axis_long_world"], dtype=float)
            + surface_offset * np.asarray(pose["axis_normal_world"], dtype=float)
        )
        return world_point, local_point

    def _terrain_height_at_point(self, x_world: float, z_world: float) -> float:
        height = 0.0
        for obstacle in self.scene_obstacles:
            half_width = 0.5 * obstacle.dims[0]
            if abs(x_world - obstacle.center[0]) > half_width + 1.0e-6:
                continue
            if obstacle.shape == "semicylinder":
                radius = float(obstacle.dims[1])
                dz = float(z_world - obstacle.center[2])
                if abs(dz) <= radius + 1.0e-6:
                    local_height = obstacle.center[1] + math.sqrt(max(radius * radius - dz * dz, 0.0))
                    height = max(height, local_height)
                continue

            half_length = 0.5 * obstacle.dims[2]
            if abs(z_world - obstacle.center[2]) <= half_length + 1.0e-6:
                height = max(height, obstacle.center[1] + 0.5 * obstacle.dims[1])
        return height

    def _terrain_height_profile(
        self,
        x_world: float,
        z_world: np.ndarray,
        *,
        footprint_half_length: float = 0.0,
    ) -> np.ndarray:
        heights = np.zeros_like(z_world, dtype=float)
        z_values = np.asarray(z_world, dtype=float)
        half_footprint = max(0.0, float(footprint_half_length))
        for obstacle in self.scene_obstacles:
            half_width = 0.5 * obstacle.dims[0]
            if abs(x_world - obstacle.center[0]) > half_width + 1.0e-6:
                continue
            if obstacle.shape == "semicylinder":
                radius = float(obstacle.dims[1])
                if half_footprint > 1.0e-12:
                    span_min = z_values - half_footprint
                    span_max = z_values + half_footprint
                    obstacle_min = obstacle.center[2] - radius
                    obstacle_max = obstacle.center[2] + radius
                    mask = (span_max >= obstacle_min - 1.0e-6) & (span_min <= obstacle_max + 1.0e-6)
                    z_peak = np.clip(obstacle.center[2], span_min, span_max)
                    z_peak = np.clip(z_peak, obstacle_min, obstacle_max)
                    dz = z_peak - obstacle.center[2]
                else:
                    dz = z_values - obstacle.center[2]
                    mask = np.abs(dz) <= radius + 1.0e-6
                if np.any(mask):
                    profile = obstacle.center[1] + np.sqrt(np.maximum(radius * radius - dz[mask] * dz[mask], 0.0))
                    heights[mask] = np.maximum(heights[mask], profile)
                continue

            half_length = 0.5 * obstacle.dims[2]
            if half_footprint > 1.0e-12:
                mask = (
                    z_values + half_footprint >= obstacle.center[2] - half_length - 1.0e-6
                ) & (
                    z_values - half_footprint <= obstacle.center[2] + half_length + 1.0e-6
                )
            else:
                mask = np.abs(z_values - obstacle.center[2]) <= half_length + 1.0e-6
            if np.any(mask):
                heights[mask] = np.maximum(heights[mask], obstacle.center[1] + 0.5 * obstacle.dims[1])
        return heights

    def _body_radius(self, body_name: str) -> float:
        role = self.body_map[body_name]["body_role"]
        if role == "road_wheel":
            return self.geom.road_wheel_radius
        if role == "return_roller":
            return self.geom.return_roller_radius
        if role == "drive_sprocket":
            return self.geom.drive_sprocket_radius
        if role == "idler":
            return self.geom.idler_radius
        return 1.0

    def _ensure_track_shoe_dynamic_state(self, state: MultibodyState) -> None:
        fields = (
            "track_shoe_normal_deflections",
            "track_shoe_normal_rates",
            "track_shoe_pitch_deflections",
            "track_shoe_pitch_rates",
        )
        for field_name in fields:
            mapping = getattr(state, field_name, None)
            if not isinstance(mapping, dict):
                mapping = {}
            for side in ("left", "right"):
                count = int(self.shoe_count.get(side, 0)) if hasattr(self, "shoe_count") else 0
                values = [float(value) for value in mapping.get(side, [])]
                if len(values) < count:
                    values.extend([0.0] * (count - len(values)))
                elif len(values) > count:
                    values = values[:count]
                mapping[side] = values
            setattr(state, field_name, mapping)

    def apply_track_terrain_feedback(
        self,
        *,
        patch_side_ids: Iterable[int],
        patch_shoe_ids: Iterable[float],
        patch_forces: np.ndarray,
        patch_moments: np.ndarray,
        dt: float,
    ) -> None:
        """Apply Chrono-style per-shoe terrain loads to the segmented track state.

        Forces and moments use the coupled MPM frame: x forward, y lateral, z vertical.
        """

        self._ensure_track_shoe_dynamic_state(self.state)
        forces = np.asarray(patch_forces, dtype=float)
        moments = np.asarray(patch_moments, dtype=float)
        side_ids = np.asarray(list(patch_side_ids), dtype=np.int32)
        shoe_ids = np.asarray(list(patch_shoe_ids), dtype=float)
        if forces.ndim != 2 or forces.shape[1] != 3:
            forces = np.zeros((0, 3), dtype=float)
        if moments.ndim != 2 or moments.shape[1] != 3:
            moments = np.zeros((forces.shape[0], 3), dtype=float)

        self.last_track_shoe_terrain_forces = {
            side: np.zeros((self.shoe_count[side], 3), dtype=float)
            for side in ("left", "right")
        }
        self.last_track_shoe_terrain_moments = {
            side: np.zeros((self.shoe_count[side], 3), dtype=float)
            for side in ("left", "right")
        }

        count = min(forces.shape[0], moments.shape[0], side_ids.size, shoe_ids.size)
        for patch_index in range(count):
            body_id = int(round(float(shoe_ids[patch_index])))
            side_index = self.track_shoe_body_id_to_index.get(body_id)
            if side_index is None:
                side = "left" if int(side_ids[patch_index]) == 0 else "right"
                local_index = body_id
                if not (0 <= local_index < self.shoe_count[side]):
                    continue
            else:
                side, local_index = side_index
            self.last_track_shoe_terrain_forces[side][local_index] += forces[patch_index]
            self.last_track_shoe_terrain_moments[side][local_index] += moments[patch_index]

        if not self.track_shoe_dynamics_enabled or dt <= 0.0:
            return

        step = float(max(dt, 1.0e-12))
        pitch = max(self.geom.track_pitch, 1.0e-6)
        for side in ("left", "right"):
            shoe_count = self.shoe_count[side]
            if shoe_count <= 0:
                continue

            normal = np.asarray(self.state.track_shoe_normal_deflections[side], dtype=float)
            normal_rate = np.asarray(self.state.track_shoe_normal_rates[side], dtype=float)
            pitch_angle = np.asarray(self.state.track_shoe_pitch_deflections[side], dtype=float)
            pitch_rate = np.asarray(self.state.track_shoe_pitch_rates[side], dtype=float)
            normal_force = self.last_track_shoe_terrain_forces[side][:, 2]
            pitch_moment = self.last_track_shoe_terrain_moments[side][:, 1]

            lap_normal = np.roll(normal, -1) - 2.0 * normal + np.roll(normal, 1)
            lap_rate = np.roll(normal_rate, -1) - 2.0 * normal_rate + np.roll(normal_rate, 1)
            shoe_mass = np.maximum(self.track_shoe_mass[side], 1.0e-9)
            normal_accel = (
                normal_force
                - self.track_shoe_normal_stiffness * normal
                - self.track_shoe_normal_damping * normal_rate
                + self.track_pin_bending_stiffness * lap_normal
                + self.track_pin_bending_damping * lap_rate
            ) / shoe_mass

            normal_rate = normal_rate + normal_accel * step
            normal = normal + normal_rate * step
            max_deflection = self.track_shoe_max_deflection
            if max_deflection > 0.0:
                clipped = np.clip(normal, -max_deflection, max_deflection)
                normal_rate = np.where(clipped != normal, 0.0, normal_rate)
                normal = clipped

            slope_target = np.arctan2(np.roll(normal, -1) - np.roll(normal, 1), 2.0 * pitch)
            shoe_inertia = np.maximum(self.track_shoe_pitch_inertia[side], 1.0e-9)
            pitch_accel = (
                pitch_moment
                - self.track_pin_pitch_stiffness * (pitch_angle - slope_target)
                - self.track_pin_pitch_damping * pitch_rate
            ) / shoe_inertia
            pitch_rate = pitch_rate + pitch_accel * step
            pitch_angle = pitch_angle + pitch_rate * step
            max_pitch = self.track_shoe_max_pitch
            if max_pitch > 0.0:
                clipped_pitch = np.clip(pitch_angle, -max_pitch, max_pitch)
                pitch_rate = np.where(clipped_pitch != pitch_angle, 0.0, pitch_rate)
                pitch_angle = clipped_pitch

            self.state.track_shoe_normal_deflections[side] = normal.tolist()
            self.state.track_shoe_normal_rates[side] = normal_rate.tolist()
            self.state.track_shoe_pitch_deflections[side] = pitch_angle.tolist()
            self.state.track_shoe_pitch_rates[side] = pitch_rate.tolist()

        self._clear_track_related_caches()

    def track_shoe_contact_scalars(self) -> dict[int, dict[str, float]]:
        """Return per-track-shoe VTK scalars keyed by synthetic track shoe body id."""

        self._ensure_track_shoe_dynamic_state(self.state)
        contact_area = max(self.geom.shoe_length * self.geom.track_width, 1.0e-12)
        scalars: dict[int, dict[str, float]] = {}
        for side in ("left", "right"):
            shoe_count = self.shoe_count.get(side, 0)
            normal_deflections = self.state.track_shoe_normal_deflections.get(side, [])
            pitch_deflections = self.state.track_shoe_pitch_deflections.get(side, [])
            for index in range(shoe_count):
                body_id = int(self.track_shoe_body_index[(side, index)])
                scalars[body_id] = {
                    "contact_pressure": 0.0,
                    "contact_normal_force": 0.0,
                    "contact_flag": 0.0,
                    "track_shoe_deflection": float(normal_deflections[index]) if index < len(normal_deflections) else 0.0,
                    "track_shoe_pitch": float(pitch_deflections[index]) if index < len(pitch_deflections) else 0.0,
                }

        feedback_has_load = False
        for side in ("left", "right"):
            forces = np.asarray(
                self.last_track_shoe_terrain_forces.get(side, np.zeros((0, 3), dtype=float)),
                dtype=float,
            )
            shoe_count = min(self.shoe_count.get(side, 0), forces.shape[0])
            for index in range(shoe_count):
                normal_force = max(0.0, float(forces[index, 2]))
                if normal_force <= 1.0e-12:
                    continue
                body_id = int(self.track_shoe_body_index[(side, index)])
                record = scalars[body_id]
                record["contact_normal_force"] += normal_force
                record["contact_pressure"] = record["contact_normal_force"] / contact_area
                record["contact_flag"] = 1.0
                feedback_has_load = True
        if feedback_has_load:
            return scalars

        current_results = getattr(self, "current_track_side_results", {})
        if not current_results:
            left_result, right_result = self._static_track_results_for_state(self.state)
            current_results = {"left": left_result, "right": right_result}
            self.current_track_side_results = current_results
        for side in ("left", "right"):
            result = current_results.get(side)
            if result is None:
                continue
            node_z = np.asarray(result.node_z, dtype=float)
            normal_force_nodes = np.asarray(result.terrain_normal, dtype=float)
            if node_z.size == 0 or normal_force_nodes.size != node_z.size:
                continue

            bottom_shoes: list[tuple[int, int, float]] = []
            for pose in self.track_shoe_poses(side, use_super_element_profile=True):
                if pose["segment"] != "bottom":
                    continue
                bottom_shoes.append(
                    (
                        int(pose["index"]),
                        int(pose["body_id"]),
                        float(np.asarray(pose["world_center"], dtype=float)[2]),
                    )
                )
            if not bottom_shoes:
                continue

            shoe_centers_z = np.asarray([item[2] for item in bottom_shoes], dtype=float)
            for node_z_value, node_force in zip(node_z, normal_force_nodes):
                normal_force = max(0.0, float(node_force))
                if normal_force <= 1.0e-12:
                    continue
                nearest = int(np.argmin(np.abs(shoe_centers_z - float(node_z_value))))
                _, body_id, _ = bottom_shoes[nearest]
                scalars[body_id]["contact_normal_force"] += normal_force

            for _, body_id, _ in bottom_shoes:
                record = scalars[body_id]
                normal_force = max(0.0, float(record["contact_normal_force"]))
                if normal_force > 1.0e-12:
                    record["contact_pressure"] = normal_force / contact_area
                    record["contact_flag"] = 1.0

        return scalars

    def copy_state(self, state: MultibodyState | None = None) -> MultibodyState:
        source = self.state if state is None else state
        self._ensure_track_shoe_dynamic_state(source)
        return MultibodyState(
            t=float(source.t),
            forward_pos=float(source.forward_pos),
            lateral_pos=float(source.lateral_pos),
            heave=float(source.heave),
            heave_rate=float(source.heave_rate),
            yaw=float(source.yaw),
            pitch=float(source.pitch),
            pitch_rate=float(source.pitch_rate),
            forward_speed=float(source.forward_speed),
            yaw_rate=float(source.yaw_rate),
            left_track_phase=float(source.left_track_phase),
            right_track_phase=float(source.right_track_phase),
            left_sprocket_omega=float(source.left_sprocket_omega),
            right_sprocket_omega=float(source.right_sprocket_omega),
            joint_angles=dict(source.joint_angles),
            joint_rates=dict(source.joint_rates),
            track_shoe_normal_deflections={
                side: list(values) for side, values in source.track_shoe_normal_deflections.items()
            },
            track_shoe_normal_rates={
                side: list(values) for side, values in source.track_shoe_normal_rates.items()
            },
            track_shoe_pitch_deflections={
                side: list(values) for side, values in source.track_shoe_pitch_deflections.items()
            },
            track_shoe_pitch_rates={
                side: list(values) for side, values in source.track_shoe_pitch_rates.items()
            },
        )

    def contact_shoe_count(self, side: str, state: MultibodyState | None = None) -> int:
        return len(self.ground_contact_shoe_indices(side, state=state))

    def _ground_contact_height_tolerance(self) -> float:
        ref_length = max(self.geom.track_pitch, self.geom.shoe_length, 1.0e-6)
        return max(0.015, 0.20 * ref_length)

    def ground_contact_shoe_indices(
        self,
        side: str,
        state: MultibodyState | None = None,
        *,
        surface_offset_extra: float = 0.0,
        height_tolerance: float | None = None,
    ) -> set[int]:
        poses = self.track_shoe_poses(side, state=state)
        if not poses:
            return set()

        surface_offset = _track_contact_surface_offset(self.geom) + float(surface_offset_extra)
        tol = self._ground_contact_height_tolerance() if height_tolerance is None else max(0.0, float(height_tolerance))
        heights: list[float] = []
        for pose in poses:
            center = np.asarray(pose["world_center"], dtype=float)
            axis_normal = np.asarray(pose["axis_normal_world"], dtype=float)
            surface_point = center + surface_offset * axis_normal
            heights.append(float(surface_point[1]))

        min_height = min(heights)
        return {
            int(pose["index"])
            for pose, height in zip(poses, heights)
            if height <= min_height + tol
        }

    def ground_contact_shoe_poses(
        self,
        side: str,
        state: MultibodyState | None = None,
        *,
        surface_offset_extra: float = 0.0,
        height_tolerance: float | None = None,
    ) -> list[dict[str, Any]]:
        indices = self.ground_contact_shoe_indices(
            side,
            state=state,
            surface_offset_extra=surface_offset_extra,
            height_tolerance=height_tolerance,
        )
        return [
            pose
            for pose in self.track_shoe_poses(side, state=state)
            if int(pose["index"]) in indices
        ]

    def _track_pose_cache_key(
        self,
        side: str,
        state: MultibodyState,
        *,
        use_super_element_profile: bool,
        apply_dynamic_state: bool | None = None,
    ) -> tuple[Any, ...]:
        self._ensure_track_shoe_dynamic_state(state)
        dynamic_enabled = use_super_element_profile if apply_dynamic_state is None else bool(apply_dynamic_state)
        phase = state.left_track_phase if side == "left" else state.right_track_phase
        suspension_signature = tuple(
            state.joint_angles.get(name, 0.0) for name in self.suspension_station_names_by_side[side]
        )
        dynamic_signature: tuple[Any, ...] = ()
        if dynamic_enabled and self.track_shoe_dynamics_enabled:
            dynamic_signature = (
                tuple(round(value, 8) for value in state.track_shoe_normal_deflections.get(side, ())),
                tuple(round(value, 8) for value in state.track_shoe_pitch_deflections.get(side, ())),
            )
        return (
            side,
            use_super_element_profile,
            dynamic_enabled,
            state.forward_pos,
            state.lateral_pos,
            state.heave,
            state.yaw,
            state.pitch,
            phase,
            suspension_signature,
            dynamic_signature,
        )

    def track_shoe_poses(
        self,
        side: str,
        state: MultibodyState | None = None,
        *,
        use_super_element_profile: bool = True,
        apply_dynamic_state: bool | None = None,
    ) -> list[dict[str, Any]]:
        active_state = self.state if state is None else state
        self._ensure_track_shoe_dynamic_state(active_state)
        dynamic_enabled = use_super_element_profile if apply_dynamic_state is None else bool(apply_dynamic_state)
        cache_key = self._track_pose_cache_key(
            side,
            active_state,
            use_super_element_profile=use_super_element_profile,
            apply_dynamic_state=dynamic_enabled,
        )
        cached = self.track_pose_cache.get(cache_key)
        if cached is not None:
            return cached

        track_body = self.body_map[f"{side}_track"]
        track_center_x = track_body["joint"]["origin_xyz"][0]
        phase = active_state.left_track_phase if side == "left" else active_state.right_track_phase
        if not use_super_element_profile:
            phase += self.visual_track_phase_offset.get(side, 0.0)
        path = self._build_track_path(side, state=active_state)
        count = self.shoe_count[side]
        actual_pitch = path.length / max(count, 1)

        poses: list[dict[str, Any]] = []
        current_results = getattr(self, "current_track_side_results", {})
        side_result = current_results.get(side) if use_super_element_profile and active_state is self.state else None
        result_nodes_z = None if side_result is None else np.asarray(side_result.node_z, dtype=float)
        result_nodes_y = None if side_result is None else np.asarray(side_result.node_y, dtype=float)
        if result_nodes_z is not None and result_nodes_y is not None and result_nodes_z.size == result_nodes_y.size:
            lateral_world = float(self.body_to_world((track_center_x, 0.0, 0.0), state=active_state)[0])
            profile_local = np.asarray(
                [
                    self.world_to_body((lateral_world, float(node_y), float(node_z)), state=active_state)
                    for node_z, node_y in zip(result_nodes_z, result_nodes_y)
                ],
                dtype=float,
            )
            result_nodes_z = profile_local[:, 2]
            result_nodes_y = profile_local[:, 1] + self.track_vertical_offsets.get(side, 0.0)
        profile_z_sorted = None
        profile_y_sorted = None
        if result_nodes_z is not None and result_nodes_y is not None and result_nodes_z.size >= 2:
            order = np.argsort(result_nodes_z)
            z_ordered = np.asarray(result_nodes_z[order], dtype=float)
            y_ordered = np.asarray(result_nodes_y[order], dtype=float)
            unique_z: list[float] = []
            unique_y: list[float] = []
            start = 0
            while start < z_ordered.size:
                stop = start + 1
                while stop < z_ordered.size and abs(float(z_ordered[stop] - z_ordered[start])) <= 1.0e-8:
                    stop += 1
                unique_z.append(float(np.mean(z_ordered[start:stop])))
                unique_y.append(float(np.mean(y_ordered[start:stop])))
                start = stop
            if len(unique_z) >= 2:
                profile_z_sorted = np.asarray(unique_z, dtype=float)
                profile_y_sorted = np.asarray(unique_y, dtype=float)
        for index in range(count):
            path_s = index * actual_pitch - phase
            pos_zy, tangent_zy, segment, segment_type = path.sample(path_s)
            original_tangent_zy = np.asarray(tangent_zy, dtype=float)
            if (
                segment == "bottom"
                and segment_type == "line"
                and result_nodes_z is not None
                and result_nodes_y is not None
                and profile_z_sorted is not None
                and profile_y_sorted is not None
                and float(profile_z_sorted[0]) <= float(pos_zy[0]) <= float(profile_z_sorted[-1])
            ):
                z_value = float(pos_zy[0])
                tangent_index = int(np.clip(np.searchsorted(profile_z_sorted, z_value) - 1, 0, profile_z_sorted.size - 2))
                dz = float(profile_z_sorted[tangent_index + 1] - profile_z_sorted[tangent_index])
                dy = float(profile_y_sorted[tangent_index + 1] - profile_y_sorted[tangent_index])
                slope = dy / max(abs(dz), 1.0e-9)
                slope = float(np.clip(slope, -1.5, 1.5))
                profile_tangent_zy = _unit2(np.array([1.0, slope], dtype=float))
                tangent_alignment = float(np.dot(original_tangent_zy, profile_tangent_zy))
                if abs(tangent_alignment) >= 0.82:
                    profile_y = float(np.interp(z_value, profile_z_sorted, profile_y_sorted))
                    max_visual_offset = max(0.025, 0.25 * actual_pitch)
                    profile_y = float(
                        pos_zy[1]
                        + np.clip(profile_y - float(pos_zy[1]), -max_visual_offset, max_visual_offset)
                    )
                    pos_zy = np.array([z_value, profile_y], dtype=float)
                    tangent_zy = profile_tangent_zy if tangent_alignment >= 0.0 else -profile_tangent_zy
                else:
                    tangent_zy = original_tangent_zy
            local_long = np.array([0.0, tangent_zy[1], tangent_zy[0]], dtype=float)
            local_width = np.array([1.0, 0.0, 0.0], dtype=float)
            local_normal = np.array([0.0, -local_long[2], local_long[1]], dtype=float)
            local_center = np.array(
                [track_center_x, pos_zy[1] - self.track_vertical_offsets.get(side, 0.0), pos_zy[0]],
                dtype=float,
            )
            dynamic_normal = 0.0
            dynamic_pitch = 0.0
            if dynamic_enabled and self.track_shoe_dynamics_enabled:
                dynamic_normal = float(active_state.track_shoe_normal_deflections[side][index])
                dynamic_pitch = float(active_state.track_shoe_pitch_deflections[side][index])
                if dynamic_pitch != 0.0:
                    local_long = _rotate_x(local_long, dynamic_pitch)
                    local_normal = _rotate_x(local_normal, dynamic_pitch)
                    tangent_zy = _unit2(np.array([local_long[2], local_long[1]], dtype=float))

            local_center = local_center - self._track_centerline_outward_offset() * local_normal

            if dynamic_enabled and self.track_shoe_dynamics_enabled:
                if dynamic_normal != 0.0:
                    local_center[1] += dynamic_normal

            poses.append(
                {
                    "index": index,
                    "side": side,
                    "shoe_name": self.track_shoe_names[side][index],
                    "body_id": self.track_shoe_body_index[(side, index)],
                    "local_center": local_center,
                    "axis_long_local": local_long,
                    "axis_width_local": local_width,
                    "axis_normal_local": local_normal,
                    "world_center": self.body_to_world(local_center, state=active_state),
                    "axis_long_world": self.vector_to_world(local_long, state=active_state),
                    "axis_width_world": self.vector_to_world(local_width, state=active_state),
                    "axis_normal_world": self.vector_to_world(local_normal, state=active_state),
                    "tangent_zy": tangent_zy,
                    "segment": segment,
                    "segment_type": segment_type,
                    "path_s": path_s % path.length,
                    "dynamic_normal_deflection": dynamic_normal,
                    "dynamic_pitch_deflection": dynamic_pitch,
                }
            )
        self.track_pose_cache[cache_key] = poses
        return poses

    def _track_centerline_outward_offset(self) -> float:
        return 0.5 * self.geom.shoe_thickness + float(self.track_wheel_visual_clearance)

    def body_to_world(self, local_point: Iterable[float], state: MultibodyState | None = None) -> np.ndarray:
        active_state = self.state if state is None else state
        point = np.asarray(local_point, dtype=float)
        centered = point - self.cg_reference
        rotated = _rotate_y(_rotate_x(centered, active_state.pitch), active_state.yaw)
        return np.array(
            [
                active_state.lateral_pos + rotated[0],
                active_state.heave + rotated[1],
                active_state.forward_pos + rotated[2],
            ],
            dtype=float,
        )

    def world_to_body(self, world_point: Iterable[float], state: MultibodyState | None = None) -> np.ndarray:
        active_state = self.state if state is None else state
        point = np.asarray(world_point, dtype=float)
        translated = np.array(
            [
                point[0] - active_state.lateral_pos,
                point[1] - active_state.heave,
                point[2] - active_state.forward_pos,
            ],
            dtype=float,
        )
        centered = _rotate_x(_rotate_y(translated, -active_state.yaw), -active_state.pitch)
        return self.cg_reference + centered

    def vector_to_world(self, local_vector: Iterable[float], state: MultibodyState | None = None) -> np.ndarray:
        active_state = self.state if state is None else state
        return _rotate_y(_rotate_x(np.asarray(local_vector, dtype=float), active_state.pitch), active_state.yaw)

    def body_vector_to_world(self, local_vector: Iterable[float], state: MultibodyState | None = None) -> np.ndarray:
        return self.vector_to_world(local_vector, state=state)

    def point_for_body_to_world(
        self,
        body_name: str,
        point_model: Iterable[float],
        state: MultibodyState | None = None,
        skip_joint_bodies: set[str] | None = None,
    ) -> np.ndarray:
        point_model_arr = np.asarray(point_model, dtype=float)
        transformed = self.point_for_body_in_model(
            body_name,
            point_model_arr,
            state=state,
            skip_joint_bodies=skip_joint_bodies,
        )
        return self.body_to_world(transformed, state=state)

    def points_for_body_to_world(
        self,
        body_name: str,
        points_model: np.ndarray,
        state: MultibodyState | None = None,
        skip_joint_bodies: set[str] | None = None,
    ) -> np.ndarray:
        transformed = self.points_for_body_in_model(
            body_name,
            points_model,
            state=state,
            skip_joint_bodies=skip_joint_bodies,
        )
        return self.points_body_to_world(transformed, state=state)

    def points_body_to_world(
        self,
        local_points: np.ndarray,
        state: MultibodyState | None = None,
    ) -> np.ndarray:
        active_state = self.state if state is None else state
        points = np.asarray(local_points, dtype=float)
        centered = points - self.cg_reference
        rotated = _rotate_y_batch(_rotate_x_batch(centered, active_state.pitch), active_state.yaw)
        return rotated + np.array(
            [
                active_state.lateral_pos,
                active_state.heave,
                active_state.forward_pos,
            ],
            dtype=float,
        )

    def point_for_body_in_model(
        self,
        body_name: str,
        point_model: np.ndarray,
        state: MultibodyState | None = None,
        skip_joint_bodies: set[str] | None = None,
    ) -> np.ndarray:
        active_state = self.state if state is None else state
        point = np.asarray(point_model, dtype=float)
        for joint_body in self._body_chain(body_name):
            if skip_joint_bodies is not None and joint_body in skip_joint_bodies:
                continue
            body = self.body_map[joint_body]
            joint = body["joint"]
            if joint["type"] != "revolute":
                continue
            angle = active_state.joint_angles.get(joint_body, 0.0)
            if abs(angle) < 1.0e-12:
                continue
            point = _rotate_about_axis(point, np.asarray(joint["origin_xyz"], dtype=float), np.asarray(joint["axis_xyz"], dtype=float), angle)
        return point

    def points_for_body_in_model(
        self,
        body_name: str,
        points_model: np.ndarray,
        state: MultibodyState | None = None,
        skip_joint_bodies: set[str] | None = None,
    ) -> np.ndarray:
        active_state = self.state if state is None else state
        points = np.asarray(points_model, dtype=float)
        for joint_body in self._body_chain(body_name):
            if skip_joint_bodies is not None and joint_body in skip_joint_bodies:
                continue
            body = self.body_map[joint_body]
            joint = body["joint"]
            if joint["type"] != "revolute":
                continue
            angle = active_state.joint_angles.get(joint_body, 0.0)
            if abs(angle) < 1.0e-12:
                continue
            points = _rotate_points_about_axis(
                points,
                np.asarray(joint["origin_xyz"], dtype=float),
                np.asarray(joint["axis_xyz"], dtype=float),
                angle,
            )
        return points

    def _body_chain(self, body_name: str) -> list[str]:
        if body_name in self.body_chain_cache:
            return self.body_chain_cache[body_name]

        chain: list[str] = []
        current = body_name
        while current and current in self.body_map and current != "hull":
            chain.append(current)
            current = self.body_map[current]["parent_body"]
        self.body_chain_cache[body_name] = chain
        return chain

    def joint_history_row(self) -> dict[str, float]:
        row: dict[str, float] = {
            "t": self.state.t,
            "forward_pos": self.state.forward_pos,
            "lateral_pos": self.state.lateral_pos,
            "heave": self.state.heave,
            "heave_rate": self.state.heave_rate,
            "yaw": self.state.yaw,
            "pitch": self.state.pitch,
            "pitch_rate": self.state.pitch_rate,
            "forward_speed": self.state.forward_speed,
            "yaw_rate": self.state.yaw_rate,
            "left_track_phase": self.state.left_track_phase,
            "right_track_phase": self.state.right_track_phase,
            "left_sprocket_omega": self.state.left_sprocket_omega,
            "right_sprocket_omega": self.state.right_sprocket_omega,
        }
        for station_name in sorted(self.suspension_kinematics):
            row[f"{station_name}_normal_load"] = self.last_station_normal_loads.get(station_name, 0.0)
            row[f"{station_name}_compression"] = self.last_station_compressions.get(station_name, 0.0)
            row[f"{station_name}_track_penetration"] = self.last_station_track_penetrations.get(station_name, 0.0)
        for body_name in sorted(self.state.joint_angles):
            row[f"{body_name}_angle"] = self.state.joint_angles[body_name]
            row[f"{body_name}_rate"] = self.state.joint_rates.get(body_name, 0.0)
        return row


def build_multibody_tank_mesh(
    tank: MultibodyRigidGroundTank,
    *,
    include_outer_grousers: bool = False,
    include_ground_plane: bool = True,
    visual_detail: str = VISUAL_DETAIL_SIMPLE,
    track_contact_scalars: dict[int, dict[str, float]] | None = None,
    use_deformed_track: bool = True,
    include_tracks: bool = True,
    include_running_gear: bool = True,
) -> MeshBuilder:
    return build_multibody_tank_mesh_with_obstacles(
        tank,
        obstacles=tank.scene_obstacles,
        include_outer_grousers=include_outer_grousers,
        include_ground_plane=include_ground_plane,
        visual_detail=visual_detail,
        track_contact_scalars=track_contact_scalars,
        use_deformed_track=use_deformed_track,
        include_tracks=include_tracks,
        include_running_gear=include_running_gear,
    )


def build_multibody_tank_mesh_with_obstacles(
    tank: MultibodyRigidGroundTank,
    obstacles: Iterable[SceneObstacle],
    *,
    include_outer_grousers: bool = False,
    include_ground_plane: bool = True,
    visual_detail: str = VISUAL_DETAIL_SIMPLE,
    track_contact_scalars: dict[int, dict[str, float]] | None = None,
    use_deformed_track: bool = True,
    include_tracks: bool = True,
    include_running_gear: bool = True,
) -> MeshBuilder:
    visual_detail = _normalize_visual_detail(visual_detail)
    mesh = MeshBuilder()
    if track_contact_scalars is None:
        track_contact_scalars = tank.track_shoe_contact_scalars()

    if include_ground_plane:
        center_x = tank.state.lateral_pos
        center_z = tank.state.forward_pos
        ground_extent_x = 8.0 + 0.5 * tank.geom.hull_width
        ground_extent_z = 14.0 + 0.5 * tank.geom.hull_length
        mesh.add_face(
            [
                (center_x - ground_extent_x, 0.0, center_z - ground_extent_z),
                (center_x + ground_extent_x, 0.0, center_z - ground_extent_z),
                (center_x + ground_extent_x, 0.0, center_z + ground_extent_z),
                (center_x - ground_extent_x, 0.0, center_z + ground_extent_z),
            ],
            part_kind=0,
            body_id=0,
            contact_flag=0,
        )

    if visual_detail == VISUAL_DETAIL_FULL:
        _add_obj_visual_mesh(
            mesh,
            tank,
            include_outer_grousers=include_outer_grousers,
            track_contact_scalars=track_contact_scalars,
            use_deformed_track=use_deformed_track,
            add_discrete_tracks=include_tracks,
            excluded_body_roles=None if include_running_gear else RUNNING_GEAR_BODY_ROLES,
        )
    else:
        _add_simplified_multibody_visual_mesh(
            mesh,
            tank,
            include_outer_grousers=include_outer_grousers,
            track_contact_scalars=track_contact_scalars,
            use_deformed_track=use_deformed_track,
            include_tracks=include_tracks,
            include_running_gear=include_running_gear,
        )

    for obstacle in obstacles:
        _add_scene_obstacle(mesh, obstacle)
    return mesh


def _normalize_visual_detail(visual_detail: str) -> str:
    detail = str(visual_detail).strip().lower()
    aliases = {
        "low": VISUAL_DETAIL_SIMPLE,
        "compact": VISUAL_DETAIL_SIMPLE,
        "simplified": VISUAL_DETAIL_SIMPLE,
        "obj": VISUAL_DETAIL_FULL,
        "original": VISUAL_DETAIL_FULL,
        "detailed": VISUAL_DETAIL_FULL,
    }
    detail = aliases.get(detail, detail)
    if detail not in VISUAL_DETAIL_CHOICES:
        choices = ", ".join(VISUAL_DETAIL_CHOICES)
        raise ValueError(f"visual_detail must be one of: {choices}")
    return detail


def _add_obj_visual_mesh(
    mesh: MeshBuilder,
    tank: MultibodyRigidGroundTank,
    *,
    include_outer_grousers: bool = False,
    body_roles: set[str] | None = None,
    excluded_body_roles: set[str] | None = None,
    add_discrete_tracks: bool = True,
    track_contact_scalars: dict[int, dict[str, float]] | None = None,
    use_deformed_track: bool = True,
) -> None:
    visual_model = tank.get_visual_model()
    parts_by_body: dict[str, list[OBJVisualPart]] = {}
    for part in visual_model.parts:
        if part.body_name in {"left_track", "right_track"}:
            continue
        role = tank.body_map.get(part.body_name, {}).get("body_role")
        if excluded_body_roles is not None and role in excluded_body_roles:
            continue
        if body_roles is not None and role not in body_roles:
            continue
        parts_by_body.setdefault(part.body_name, []).append(part)

    for body_name, body_parts in parts_by_body.items():
        body_id = tank.body_index.get(body_name, 0)
        unique_indices = sorted({index for part in body_parts for index in part.unique_vertex_indices})
        local_vertices = np.asarray([visual_model.vertices[index] for index in unique_indices], dtype=float)
        skip_joint_bodies = None
        if tank.body_map.get(body_name, {}).get("body_role") == "road_wheel":
            skip_joint_bodies = {body_name}
        world_vertices = tank.points_for_body_to_world(
            body_name,
            local_vertices,
            skip_joint_bodies=skip_joint_bodies,
        )
        index_lookup = {vertex_index: local_index for local_index, vertex_index in enumerate(unique_indices)}

        for part in body_parts:
            for face in part.faces:
                mesh.add_face(
                    [world_vertices[index_lookup[vertex_index]] for vertex_index in face],
                    part_kind=part.part_kind,
                    body_id=body_id,
                    contact_flag=0,
                )

    if add_discrete_tracks:
        _add_discrete_track_shoes(
            mesh,
            tank,
            include_outer_grousers=include_outer_grousers,
            track_contact_scalars=track_contact_scalars,
            use_deformed_track=use_deformed_track,
        )


def _add_simplified_multibody_visual_mesh(
    mesh: MeshBuilder,
    tank: MultibodyRigidGroundTank,
    *,
    include_outer_grousers: bool = False,
    track_contact_scalars: dict[int, dict[str, float]] | None = None,
    use_deformed_track: bool = True,
    include_tracks: bool = True,
    include_running_gear: bool = True,
) -> None:
    _add_simplified_hull(mesh, tank)
    if include_running_gear:
        _add_obj_visual_mesh(
            mesh,
            tank,
            body_roles=RUNNING_GEAR_BODY_ROLES,
            add_discrete_tracks=False,
        )
    if include_tracks:
        _add_discrete_track_shoes(
            mesh,
            tank,
            include_outer_grousers=include_outer_grousers,
            track_contact_scalars=track_contact_scalars,
            use_deformed_track=use_deformed_track,
        )


def _body_bbox_arrays(tank: MultibodyRigidGroundTank, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body = tank.body_map.get(body_name, {})
    if "bbox_min_xyz" in body and "bbox_max_xyz" in body:
        return np.asarray(body["bbox_min_xyz"], dtype=float), np.asarray(body["bbox_max_xyz"], dtype=float)

    if "center_xyz" in body and "size_xyz" in body:
        center = np.asarray(body["center_xyz"], dtype=float)
        half_size = 0.5 * np.asarray(body["size_xyz"], dtype=float)
        return center - half_size, center + half_size

    center = np.asarray(tank.base_reference, dtype=float)
    half_size = np.array(
        [
            0.5 * max(tank.geom.hull_width, tank.geom.track_width),
            0.5 * max(tank.geom.hull_height, 0.25),
            0.5 * max(tank.geom.hull_length, tank.geom.track_pitch),
        ],
        dtype=float,
    )
    return center - half_size, center + half_size


def _add_simplified_hull(mesh: MeshBuilder, tank: MultibodyRigidGroundTank) -> None:
    body_id = tank.body_index.get("hull", 0)
    bbox_min, bbox_max = _body_bbox_arrays(tank, "hull")
    center = 0.5 * (bbox_min + bbox_max)
    size = np.maximum(bbox_max - bbox_min, np.array([0.1, 0.1, 0.1], dtype=float))

    lower_y = float(bbox_min[1] + 0.18 * size[1])
    upper_y = float(bbox_max[1] - 0.04 * size[1])
    rear_z = float(bbox_min[2] + 0.04 * size[2])
    front_z = float(bbox_max[2] - 0.04 * size[2])
    top_rear_z = float(bbox_min[2] + 0.22 * size[2])
    top_front_z = float(bbox_max[2] - 0.12 * size[2])
    if top_front_z <= top_rear_z:
        top_rear_z = rear_z
        top_front_z = front_z

    inner_track_half_gap = max(
        0.35,
        min(abs(tank.geom.left_track_center_x), abs(tank.geom.right_track_center_x)) - 0.5 * tank.geom.track_width,
    )
    hull_track_clearance = max(0.06, min(0.12, 0.16 * tank.geom.track_width))
    max_hull_half_width = max(0.35, inner_track_half_gap - hull_track_clearance)
    running_gear_inner_edges: list[float] = []
    running_gear_roles = {"drive_sprocket", "idler", "return_roller", "road_wheel", "suspension_station"}
    for body in tank.data.get("bodies", []):
        if body.get("body_role") not in running_gear_roles:
            continue
        if "bbox_min_xyz" not in body or "bbox_max_xyz" not in body:
            continue
        xmin = float(body["bbox_min_xyz"][0])
        xmax = float(body["bbox_max_xyz"][0])
        if xmin <= 0.0 <= xmax:
            continue
        running_gear_inner_edges.append(min(abs(xmin), abs(xmax)))
    if running_gear_inner_edges:
        max_hull_half_width = min(max_hull_half_width, max(0.35, min(running_gear_inner_edges) - hull_track_clearance))
    bottom_half_width = min(0.44 * float(size[0]), max_hull_half_width)
    top_half_width = min(0.34 * float(size[0]), 0.88 * max_hull_half_width)
    cx = float(center[0])
    local = np.asarray(
        [
            (cx - bottom_half_width, lower_y, rear_z),
            (cx + bottom_half_width, lower_y, rear_z),
            (cx + bottom_half_width, lower_y, front_z),
            (cx - bottom_half_width, lower_y, front_z),
            (cx - top_half_width, upper_y, top_rear_z),
            (cx + top_half_width, upper_y, top_rear_z),
            (cx + top_half_width, upper_y, top_front_z),
            (cx - top_half_width, upper_y, top_front_z),
        ],
        dtype=float,
    )
    world = tank.points_body_to_world(local)
    for face in ([0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]):
        mesh.add_face([world[index] for index in face], part_kind=1, body_id=body_id)

def build_rectangular_obstacle_in_front(
    tank: MultibodyRigidGroundTank,
    *,
    height: float,
    length: float,
    clearance: float,
    width: float | None = None,
) -> SceneObstacle:
    effective_width = width
    if effective_width is None or effective_width <= 0.0:
        outer_track_span = abs(tank.geom.left_track_center_x - tank.geom.right_track_center_x) + tank.geom.track_width
        effective_width = max(tank.geom.hull_width + 0.40, outer_track_span + 0.40)

    front_z = _frontmost_track_world_z(tank)
    center = (
        float(tank.state.lateral_pos),
        0.5 * float(height),
        float(front_z + clearance + 0.5 * length),
    )
    return SceneObstacle(
        center=center,
        dims=(float(effective_width), float(height), float(length)),
        shape="box",
    )


def build_semicircular_obstacle_in_front(
    tank: MultibodyRigidGroundTank,
    *,
    height: float,
    clearance: float,
    width: float | None = None,
) -> SceneObstacle:
    effective_width = width
    if effective_width is None or effective_width <= 0.0:
        outer_track_span = abs(tank.geom.left_track_center_x - tank.geom.right_track_center_x) + tank.geom.track_width
        effective_width = max(tank.geom.hull_width + 0.40, outer_track_span + 0.40)

    radius = float(height)
    front_z = _frontmost_track_world_z(tank)
    center = (
        float(tank.state.lateral_pos),
        0.0,
        float(front_z + clearance + radius),
    )
    return SceneObstacle(
        center=center,
        dims=(float(effective_width), radius, 2.0 * radius),
        shape="semicylinder",
    )


def build_mpm_track_surface_patches(
    tank: MultibodyRigidGroundTank,
    previous_state: MultibodyState,
    dt: float,
    z_max: float,
    z_margin: float,
    surface_offset_extra: float = 0.0,
    allowed_segments: tuple[str, ...] | None = None,
    ground_contact_only: bool = False,
) -> dict[str, np.ndarray]:
    """Build MPM-ready contact patches from discrete track shoes.

    Output uses the existing MPM coordinate convention:
    - x: forward
    - y: lateral
    - z: vertical
    The multibody model uses OBJ coordinates internally:
    - x: lateral
    - y: vertical
    - z: forward
    """

    centers = []
    axes_long = []
    axes_width = []
    normals = []
    velocities = []
    half_extents = []
    patch_masses = []
    shoe_ids = []
    side_ids = []

    geom = tank.geom
    max_patch_z = z_max + z_margin
    allowed_segment_set = None if allowed_segments is None else set(allowed_segments)
    eligible_ground_contact: dict[str, set[int]] = {}
    if ground_contact_only:
        for side in ("left", "right"):
            eligible_ground_contact[side] = tank.ground_contact_shoe_indices(
                side,
                state=tank.state,
                surface_offset_extra=surface_offset_extra,
            )

    for side in ("left", "right"):
        current_poses = tank.track_shoe_poses(side, state=tank.state)
        previous_poses = tank.track_shoe_poses(side, state=previous_state)
        for pose, previous_pose in zip(current_poses, previous_poses):
            if allowed_segment_set is not None and pose["segment"] not in allowed_segment_set:
                continue
            if ground_contact_only and int(pose["index"]) not in eligible_ground_contact.get(side, set()):
                continue
            surface_offset = abs(_track_contact_surface_offset(geom)) + float(surface_offset_extra)
            contact_normal_world = _soil_facing_track_normal_world(pose)
            previous_contact_normal_world = _soil_facing_track_normal_world(previous_pose)
            center = pose["world_center"] + contact_normal_world * surface_offset
            previous_center = previous_pose["world_center"] + previous_contact_normal_world * surface_offset

            center_mpm = _multibody_world_to_mpm(center)
            previous_center_mpm = _multibody_world_to_mpm(previous_center)
            axis_long_mpm = _unit(_multibody_world_vector_to_mpm(pose["axis_long_world"]))
            axis_width_mpm = _unit(_multibody_world_vector_to_mpm(pose["axis_width_world"]))
            normal_mpm = _unit(_multibody_world_vector_to_mpm(contact_normal_world))
            half_extent = np.array([0.5 * geom.shoe_length, 0.5 * geom.track_width], dtype=float)
            contact_face_corners = np.asarray(
                [
                    center_mpm + s_long * half_extent[0] * axis_long_mpm + s_width * half_extent[1] * axis_width_mpm
                    for s_long in (-1.0, 1.0)
                    for s_width in (-1.0, 1.0)
                ],
                dtype=float,
            )
            min_contact_face_z = float(np.min(contact_face_corners[:, 2]))
            # Shared-node contact should only see shoes that are on or near the
            # local terrain surface. Deeply buried shoes remain eligible; shoes
            # well above the local surface are excluded before grid projection.
            # Use the lowest contact-face corner instead of only the face center
            # so sloped shoes near the sprocket/idler can engage a deep rut.
            if min_contact_face_z > max_patch_z:
                continue

            centers.append(center_mpm)
            axes_long.append(axis_long_mpm)
            axes_width.append(axis_width_mpm)
            normals.append(normal_mpm)
            velocities.append((center_mpm - previous_center_mpm) / max(dt, 1.0e-12))
            half_extents.append(half_extent.tolist())
            patch_masses.append(float(tank.track_shoe_mass[side][int(pose["index"])]))
            shoe_ids.append(float(pose["body_id"]))
            side_ids.append(0 if side == "left" else 1)

    if not centers:
        return {
            "center": np.zeros((0, 3), dtype=np.float32),
            "axis_long": np.zeros((0, 3), dtype=np.float32),
            "axis_width": np.zeros((0, 3), dtype=np.float32),
            "normal": np.zeros((0, 3), dtype=np.float32),
            "velocity": np.zeros((0, 3), dtype=np.float32),
            "half_extent": np.zeros((0, 2), dtype=np.float32),
            "mass": np.zeros((0,), dtype=np.float32),
            "shoe_id": np.zeros((0,), dtype=np.float32),
            "side_id": np.zeros((0,), dtype=np.int32),
        }

    return {
        "center": np.asarray(centers, dtype=np.float32),
        "axis_long": np.asarray(axes_long, dtype=np.float32),
        "axis_width": np.asarray(axes_width, dtype=np.float32),
        "normal": np.asarray(normals, dtype=np.float32),
        "velocity": np.asarray(velocities, dtype=np.float32),
        "half_extent": np.asarray(half_extents, dtype=np.float32),
        "mass": np.asarray(patch_masses, dtype=np.float32),
        "shoe_id": np.asarray(shoe_ids, dtype=np.float32),
        "side_id": np.asarray(side_ids, dtype=np.int32),
    }


def _soil_facing_track_normal_world(pose: dict[str, Any]) -> np.ndarray:
    normal = np.asarray(pose["axis_normal_world"], dtype=float)
    if normal[1] > 0.0:
        normal = -normal
    return _unit(normal)


def write_metrics_csv(path: Path, rows: list[MultibodyStepMetrics]) -> None:
    fields = list(MultibodyStepMetrics.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def write_joint_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _add_discrete_track_shoes(
    mesh: MeshBuilder,
    tank: MultibodyRigidGroundTank,
    *,
    include_outer_grousers: bool = False,
    track_contact_scalars: dict[int, dict[str, float]] | None = None,
    use_deformed_track: bool = True,
) -> None:
    plate_width = 0.96 * tank.geom.track_width
    plate_thickness = 0.82 * tank.geom.shoe_thickness
    hinge_radius = max(0.014, min(0.026, 0.22 * tank.geom.shoe_thickness))
    hinge_length = max(0.055, 0.30 * tank.geom.track_width)
    hinge_span_offset = 0.25 * (tank.geom.track_width + tank.geom.guide_horn_width)
    hinge_normal_bias = 0.12 * tank.geom.shoe_thickness
    guide_horn_length = max(0.12 * tank.geom.shoe_length, min(0.18 * tank.geom.shoe_length, 0.16 * tank.geom.track_pitch))
    drive_lug_width = max(0.038, 0.72 * tank.geom.guide_horn_width)
    drive_lug_length = max(0.040, min(0.095, 0.22 * tank.geom.track_pitch))
    drive_lug_height = max(0.040, 0.58 * tank.geom.guide_horn_height)
    contact_scalars = {} if track_contact_scalars is None else track_contact_scalars

    for side in ("left", "right"):
        side_pitch = float(tank.actual_pitch.get(side, tank.geom.track_pitch))
        plate_length = max(0.92 * tank.geom.shoe_length, min(1.02 * tank.geom.shoe_length, 0.985 * side_pitch))
        hinge_offset = 0.5 * max(plate_length, side_pitch) - 0.30 * hinge_radius
        drive_lug_long_offset = 0.5 * max(plate_length, side_pitch) - 0.65 * drive_lug_length
        for pose in tank.track_shoe_poses(
            side,
            use_super_element_profile=use_deformed_track,
            apply_dynamic_state=True,
        ):
            body_id = pose["body_id"]
            scalar_record = contact_scalars.get(int(body_id), {})
            if not isinstance(scalar_record, dict):
                scalar_record = {"contact_pressure": float(scalar_record)}
            contact_pressure = max(0.0, float(scalar_record.get("contact_pressure", 0.0)))
            contact_normal_force = max(0.0, float(scalar_record.get("contact_normal_force", 0.0)))
            track_shoe_deflection = float(
                scalar_record.get("track_shoe_deflection", pose.get("dynamic_normal_deflection", 0.0))
            )
            track_shoe_pitch = float(
                scalar_record.get("track_shoe_pitch", pose.get("dynamic_pitch_deflection", 0.0))
            )
            fallback_contact = 1 if pose["segment"] == "bottom" and not contact_scalars else 0
            is_contact = int(scalar_record.get("contact_flag", fallback_contact))
            axis_long = pose["axis_long_world"]
            axis_width = pose["axis_width_world"]
            axis_normal = pose["axis_normal_world"]
            shoe_center = np.asarray(pose["world_center"], dtype=float)

            mesh.add_box(
                center=shoe_center,
                axis_x=axis_long,
                axis_y=axis_width,
                axis_z=axis_normal,
                dims=(plate_length, plate_width, plate_thickness),
                part_kind=5,
                body_id=body_id,
                contact_flag=is_contact,
                contact_pressure=contact_pressure,
                contact_normal_force=contact_normal_force,
                track_shoe_deflection=track_shoe_deflection,
                track_shoe_pitch=track_shoe_pitch,
                contact_face_axis_z=-1,
            )

            if include_outer_grousers and tank.geom.grouser_height > 1.0e-8:
                grouser_center = shoe_center + axis_normal * (
                    0.5 * tank.geom.shoe_thickness + 0.5 * tank.geom.grouser_height
                )
                mesh.add_box(
                    center=grouser_center,
                    axis_x=axis_long,
                    axis_y=axis_width,
                    axis_z=axis_normal,
                    dims=(tank.geom.shoe_length * 0.32, tank.geom.track_width, tank.geom.grouser_height),
                    part_kind=5,
                    body_id=body_id,
                    contact_flag=is_contact,
                    contact_pressure=contact_pressure,
                    contact_normal_force=contact_normal_force,
                    track_shoe_deflection=track_shoe_deflection,
                    track_shoe_pitch=track_shoe_pitch,
                    contact_face_axis_z=-1,
                )

            guide_horn_center = shoe_center + axis_normal * (
                0.5 * tank.geom.shoe_thickness + 0.5 * tank.geom.guide_horn_height
            )
            mesh.add_box(
                center=guide_horn_center,
                axis_x=axis_long,
                axis_y=axis_width,
                axis_z=axis_normal,
                dims=(guide_horn_length, tank.geom.guide_horn_width, tank.geom.guide_horn_height),
                part_kind=5,
                body_id=body_id,
                contact_flag=0,
            )

            for long_sign in (-1.0, 1.0):
                lead_scale = 1.15 if long_sign < 0.0 else 0.92
                hinge_center_long = shoe_center + axis_long * (long_sign * hinge_offset) + axis_normal * hinge_normal_bias
                for width_sign in (-1.0, 1.0):
                    hinge_center = hinge_center_long + axis_width * (width_sign * hinge_span_offset)
                    mesh.add_cylinder(
                        center=hinge_center,
                        axis_length=axis_width,
                        radius_axis_a=axis_long,
                        radius_axis_b=axis_normal,
                        radius=hinge_radius * lead_scale,
                        width=hinge_length,
                        part_kind=5,
                        body_id=body_id,
                        segments=10,
                    )

                lug_height = drive_lug_height * lead_scale
                drive_lug_center = shoe_center + axis_long * (long_sign * drive_lug_long_offset) + axis_normal * (
                    0.5 * tank.geom.shoe_thickness + 0.5 * lug_height
                )
                mesh.add_box(
                    center=drive_lug_center,
                    axis_x=axis_long,
                    axis_y=axis_width,
                    axis_z=axis_normal,
                    dims=(drive_lug_length, drive_lug_width, lug_height),
                    part_kind=5,
                    body_id=body_id,
                    contact_flag=0,
                )


def _track_contact_surface_offset(geom: MultibodyGeometry) -> float:
    # axis_normal points toward the running-gear side of the shoe.
    return -0.5 * geom.shoe_thickness


def _track_contact_normal_extent(geom: MultibodyGeometry) -> float:
    return 0.5 * geom.shoe_thickness


def _add_scene_obstacle(mesh: MeshBuilder, obstacle: SceneObstacle) -> None:
    if obstacle.shape == "semicylinder":
        mesh.add_half_cylinder(
            center=np.asarray(obstacle.center, dtype=float),
            axis_width=np.array([1.0, 0.0, 0.0], dtype=float),
            axis_height=np.array([0.0, 1.0, 0.0], dtype=float),
            axis_long=np.array([0.0, 0.0, 1.0], dtype=float),
            radius=float(obstacle.dims[1]),
            width=float(obstacle.dims[0]),
            part_kind=obstacle.part_kind,
            body_id=obstacle.body_id,
            contact_flag=obstacle.contact_flag,
        )
        return

    mesh.add_box(
        center=np.asarray(obstacle.center, dtype=float),
        axis_x=np.array([1.0, 0.0, 0.0], dtype=float),
        axis_y=np.array([0.0, 1.0, 0.0], dtype=float),
        axis_z=np.array([0.0, 0.0, 1.0], dtype=float),
        dims=obstacle.dims,
        part_kind=obstacle.part_kind,
        body_id=obstacle.body_id,
        contact_flag=obstacle.contact_flag,
    )


def _frontmost_track_world_z(tank: MultibodyRigidGroundTank) -> float:
    front_z = -math.inf
    for side in ("left", "right"):
        for pose in tank.track_shoe_poses(side):
            axis_long = np.asarray(pose["axis_long_world"], dtype=float)
            axis_width = np.asarray(pose["axis_width_world"], dtype=float)
            axis_normal = np.asarray(pose["axis_normal_world"], dtype=float)
            extent_z = (
                0.5 * tank.geom.shoe_length * abs(axis_long[2])
                + 0.5 * tank.geom.track_width * abs(axis_width[2])
                + _track_contact_normal_extent(tank.geom) * abs(axis_normal[2])
            )
            front_z = max(front_z, float(pose["world_center"][2]) + extent_z)
    return front_z


def _build_track_line_segment(start: np.ndarray, end: np.ndarray, tag: str) -> dict[str, Any] | None:
    length = float(np.linalg.norm(end - start))
    if length < 1.0e-9:
        return None
    return {
        "type": "line",
        "start": np.asarray(start, dtype=float),
        "end": np.asarray(end, dtype=float),
        "length": length,
        "tag": tag,
    }


def _build_track_line_segments(
    start: np.ndarray,
    end: np.ndarray,
    tag: str,
    *,
    start_name: str | None = None,
    end_name: str | None = None,
) -> list[dict[str, Any]]:
    if tag != "top":
        segment = _build_track_line_segment(start, end, tag)
        return [] if segment is None else [segment]

    sag = _top_track_sag_amount(start, end, start_name=start_name, end_name=end_name)
    if sag <= 1.0e-9:
        segment = _build_track_line_segment(start, end, tag)
        return [] if segment is None else [segment]

    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    span = end_arr - start_arr
    q1 = start_arr + 0.25 * span
    mid = start_arr + 0.50 * span
    q3 = start_arr + 0.75 * span
    q1[1] -= 0.60 * sag
    mid[1] -= sag
    q3[1] -= 0.60 * sag

    segments: list[dict[str, Any]] = []
    polyline = (start_arr, q1, mid, q3, end_arr)
    for a, b in zip(polyline[:-1], polyline[1:]):
        segment = _build_track_line_segment(a, b, tag)
        if segment is not None:
            segments.append(segment)
    return segments


def _top_track_sag_amount(
    start: np.ndarray,
    end: np.ndarray,
    *,
    start_name: str | None = None,
    end_name: str | None = None,
) -> float:
    span = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(span))
    if length <= 0.20:
        return 0.0

    sag = min(0.032, max(0.006, 0.035 * length))
    if (start_name and "idler" in start_name) or (end_name and "idler" in end_name):
        sag *= 1.10
    elif (start_name and "drive_sprocket" in start_name) or (end_name and "drive_sprocket" in end_name):
        sag *= 0.55
    return sag


def _build_track_arc_segment(circle: TrackCircle, start: np.ndarray, end: np.ndarray) -> dict[str, Any] | None:
    delta = _choose_track_arc_delta(circle.center_zy, start, end, circle.arc_preference)
    if abs(delta) < 1.0e-9:
        return None

    theta0 = math.atan2(start[1] - circle.center_zy[1], start[0] - circle.center_zy[0])
    arc_length = abs(delta) * circle.radius
    if arc_length < 0.015:
        return None

    return {
        "type": "arc",
        "center": np.asarray(circle.center_zy, dtype=float),
        "radius": float(circle.radius),
        "theta0": theta0,
        "direction": 1.0 if delta >= 0.0 else -1.0,
        "length": arc_length,
        "tag": _track_arc_tag(circle.arc_preference),
        "start": np.asarray(start, dtype=float),
        "end": np.asarray(end, dtype=float),
    }


def _track_arc_tag(arc_preference: str) -> str:
    if arc_preference == "top":
        return "top"
    if arc_preference == "bottom":
        return "bottom"
    if arc_preference == "front":
        return "front_idler"
    if arc_preference == "rear":
        return "rear_sprocket"
    raise ValueError(f"unsupported track arc preference: {arc_preference}")


def _circle_external_tangent(circle_a: TrackCircle, circle_b: TrackCircle, side: str) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(circle_b.center_zy - circle_a.center_zy, dtype=float)
    distance = float(np.linalg.norm(delta))
    radius_delta = circle_a.radius - circle_b.radius
    if distance <= abs(radius_delta) + 1.0e-9:
        raise ValueError(f"cannot build {side} external tangent between {circle_a.name} and {circle_b.name}")

    theta = math.atan2(delta[1], delta[0])
    alpha = math.acos(max(-1.0, min(1.0, radius_delta / distance)))
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for angle in (theta + alpha, theta - alpha):
        normal = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        candidates.append(
            (
                circle_a.center_zy + circle_a.radius * normal,
                circle_b.center_zy + circle_b.radius * normal,
            )
        )

    if side == "upper":
        return max(candidates, key=lambda pair: 0.5 * (pair[0][1] + pair[1][1]))
    if side == "lower":
        return min(candidates, key=lambda pair: 0.5 * (pair[0][1] + pair[1][1]))
    raise ValueError(f"unsupported tangent side: {side}")


def _choose_track_arc_delta(
    center_zy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    preference: str,
) -> float:
    if float(np.linalg.norm(end - start)) < 1.0e-9:
        return 0.0

    theta_start = math.atan2(start[1] - center_zy[1], start[0] - center_zy[0])
    theta_end = math.atan2(end[1] - center_zy[1], end[0] - center_zy[0])
    ccw_delta = (theta_end - theta_start) % (2.0 * math.pi)
    candidates = [ccw_delta, ccw_delta - 2.0 * math.pi]

    if preference in {"top", "bottom"}:
        return min(candidates, key=lambda delta: abs(delta))

    if preference == "front":
        anchor_angle = 0.0
    elif preference == "rear":
        anchor_angle = math.pi
    else:
        raise ValueError(f"unsupported track arc preference: {preference}")

    best_delta = candidates[0]
    best_contains_anchor = False
    best_length = math.inf
    for delta in candidates:
        contains_anchor = _angle_is_on_arc(theta_start, delta, anchor_angle)
        if contains_anchor and not best_contains_anchor:
            best_delta = delta
            best_contains_anchor = True
            best_length = abs(delta)
            continue

        if contains_anchor == best_contains_anchor and abs(delta) < best_length:
            best_delta = delta
            best_length = abs(delta)

    return best_delta


def _angle_is_on_arc(theta_start: float, delta: float, theta_target: float) -> bool:
    two_pi = 2.0 * math.pi
    if delta >= 0.0:
        travel = (theta_target - theta_start) % two_pi
        return travel <= delta + 1.0e-12

    travel = (theta_start - theta_target) % two_pi
    return travel <= -delta + 1.0e-12


def _part_kind_from_visual_spec(spec: dict[str, Any]) -> int:
    role = spec.get("body_role")
    category = spec.get("category")
    if role == "hull":
        return 1
    if role == "turret":
        return 2
    if role == "gun":
        return 3
    if role == "road_wheel":
        return 4
    if role == "track_loop":
        return 5
    if role in {"drive_sprocket", "idler"}:
        return 6
    if role == "return_roller":
        return 7
    if role == "suspension_station":
        return 8
    if role in {"aa_mg_traverse", "aa_mg_elevation"}:
        return 9
    if category == "antenna":
        return 10
    return 8


def _parse_obj_index(token: str, vertex_count: int) -> int:
    head = token.split("/", 1)[0]
    raw = int(head)
    return vertex_count + raw if raw < 0 else raw - 1


def _multibody_world_to_mpm(point: Iterable[float]) -> np.ndarray:
    px, py, pz = np.asarray(point, dtype=float)
    return np.array([pz, px, py], dtype=float)


def _multibody_world_vector_to_mpm(vector: Iterable[float]) -> np.ndarray:
    vx, vy, vz = np.asarray(vector, dtype=float)
    return np.array([vz, vx, vy], dtype=float)


def _closest_point_on_segment_2d(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    delta = np.asarray(end - start, dtype=float)
    denom = float(np.dot(delta, delta))
    if denom <= 1.0e-12:
        return np.asarray(start, dtype=float), 0.0
    alpha = float(np.dot(point - start, delta) / denom)
    alpha = max(0.0, min(1.0, alpha))
    return np.asarray(start + alpha * delta, dtype=float), alpha


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        raise ValueError("zero-length axis")
    return v / n


def _unit2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        raise ValueError("zero-length 2D axis")
    return v / n


def _rotate_y(vector: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            c * vector[0] + s * vector[2],
            vector[1],
            -s * vector[0] + c * vector[2],
        ],
        dtype=float,
    )


def _rotate_x(vector: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            vector[0],
            c * vector[1] + s * vector[2],
            -s * vector[1] + c * vector[2],
        ],
        dtype=float,
    )


def _rotate_y_batch(vectors: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    result = np.empty_like(vectors, dtype=float)
    result[:, 0] = c * vectors[:, 0] + s * vectors[:, 2]
    result[:, 1] = vectors[:, 1]
    result[:, 2] = -s * vectors[:, 0] + c * vectors[:, 2]
    return result


def _rotate_x_batch(vectors: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    result = np.empty_like(vectors, dtype=float)
    result[:, 0] = vectors[:, 0]
    result[:, 1] = c * vectors[:, 1] + s * vectors[:, 2]
    result[:, 2] = -s * vectors[:, 1] + c * vectors[:, 2]
    return result


def _rotate_about_axis(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis_u = _unit(axis)
    rel = point - origin
    c = math.cos(angle)
    s = math.sin(angle)
    rotated = rel * c + np.cross(axis_u, rel) * s + axis_u * np.dot(axis_u, rel) * (1.0 - c)
    return origin + rotated


def _rotate_points_about_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis_u = _unit(axis)
    rel = np.asarray(points, dtype=float) - origin
    c = math.cos(angle)
    s = math.sin(angle)
    axis_row = np.broadcast_to(axis_u, rel.shape)
    cross = np.cross(axis_row, rel)
    dot = rel @ axis_u
    rotated = rel * c + cross * s + axis_row * dot[:, None] * (1.0 - c)
    return origin + rotated
