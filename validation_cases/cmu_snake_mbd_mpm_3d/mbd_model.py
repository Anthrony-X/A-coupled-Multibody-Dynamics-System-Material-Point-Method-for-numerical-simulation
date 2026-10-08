from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tank_mpm.chrono_loader import require_chrono_core
from validation_cases.cmu_snake_geometry_3d.geometry import (
    JOINT_PITCH,
    JOINT_YAW,
    SnakePose,
    build_contact_envelope,
    build_contact_patches,
    build_module_surfaces,
    build_topology_lines,
    continuous_backbone_joint_state,
)

from .config import CaseConfig
from .runtime_patches import VectorizedSnakePatchKinematics


def _np_vec(value) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float64)


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-14:
        raise ValueError("Cannot normalize a near-zero vector")
    return value / norm


@dataclass(frozen=True)
class ContactFeedback:
    patch_forces: np.ndarray | None = None
    patch_moments: np.ndarray | None = None
    module_forces: np.ndarray | None = None
    module_moments_about_origin: np.ndarray | None = None
    axial_segment_normal_loads: np.ndarray | None = None

    @classmethod
    def zeros(cls, count: int) -> "ContactFeedback":
        return cls(
            patch_forces=np.zeros((count, 3), dtype=np.float32),
            patch_moments=np.zeros((count, 3), dtype=np.float32),
        )


@dataclass(frozen=True)
class ControllerSnapshot:
    target_angle_rad: np.ndarray
    target_rate_rad_s: np.ndarray
    actual_angle_rad: np.ndarray
    actual_rate_rad_s: np.ndarray
    torque_command_Nm: np.ndarray
    saturation: np.ndarray

    @property
    def tracking_error_rad(self) -> np.ndarray:
        return self.target_angle_rad - self.actual_angle_rad


def gait_envelope(time_s: float, settle_time_s: float, ramp_time_s: float) -> tuple[float, float]:
    if time_s <= settle_time_s:
        return 0.0, 0.0
    normalized = (time_s - settle_time_s) / ramp_time_s
    if normalized >= 1.0:
        return 1.0, 0.0
    normalized = max(0.0, normalized)
    value = normalized * normalized * (3.0 - 2.0 * normalized)
    derivative = 6.0 * normalized * (1.0 - normalized) / ramp_time_s
    return value, derivative


def gait_targets(config: CaseConfig, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    model = config.model
    geometry = config.geometry_config()
    envelope, envelope_rate = gait_envelope(
        time_s, model.settle_time_s, model.ramp_time_s
    )
    gait_time = max(0.0, time_s - model.settle_time_s)
    omega = 2.0 * math.pi * model.gait_frequency_hz
    phase = -omega * gait_time
    fitted, fitted_phase_rate = continuous_backbone_joint_state(geometry, phase)
    target = envelope * fitted
    target_rate = envelope_rate * fitted - envelope * omega * fitted_phase_rate
    return target, target_rate


class CmuSnakeMBD:
    """Seventeen rigid modules connected by sixteen torque-driven revolute joints."""

    def __init__(self, config: CaseConfig) -> None:
        self.config = config
        self.geometry = config.geometry_config()
        self.chrono = require_chrono_core()
        self.system = self.chrono.ChSystemSMC()
        self.system.SetGravitationalAcceleration(
            self._chrono_vec((0.0, 0.0, -config.model.gravity))
        )
        self.system.SetSolverType(self.chrono.ChSolver.Type_SPARSE_QR)
        solver = self.system.GetSolver()
        if hasattr(solver, "SetMaxIterations"):
            solver.SetMaxIterations(200)

        self.module_mass = config.model.module_mass_kg
        self.module_pitch = config.model.module_pitch_m
        radius = config.model.radius_m
        self.module_inertia = np.array(
            [
                0.5 * self.module_mass * radius * radius,
                self.module_mass * (3.0 * radius * radius + self.module_pitch**2) / 12.0,
                self.module_mass * (3.0 * radius * radius + self.module_pitch**2) / 12.0,
            ],
            dtype=np.float64,
        )
        self.bodies = []
        self.motors = []
        self.torque_functions = []
        self.joint_types = np.empty(config.model.joint_count, dtype=np.int32)
        self.joint_types[0::2] = JOINT_YAW
        self.joint_types[1::2] = JOINT_PITCH

        initial_z = (
            config.soil.surface_z
            + config.model.radius_m
            + config.model.initial_clearance_m
        )
        head_start_x = -0.5 * config.model.total_length_m
        for module_id in range(config.model.module_count):
            center = np.array(
                [
                    head_start_x + (module_id + 0.5) * self.module_pitch,
                    0.0,
                    initial_z,
                ],
                dtype=np.float64,
            )
            body = self.chrono.ChBody()
            label = "head" if module_id == 0 else "tail" if module_id == 16 else "middle"
            body.SetName(f"snake_{label}_{module_id:02d}")
            body.SetMass(float(self.module_mass))
            body.SetInertiaXX(self._chrono_vec(self.module_inertia))
            body.SetPos(self._chrono_vec(center))
            body.SetRot(self.chrono.QuatFromAngleX(0.0))
            if hasattr(body, "SetUseSleeping"):
                body.SetUseSleeping(False)
            self.system.AddBody(body)
            self.bodies.append(body)

        for joint_id, joint_type in enumerate(self.joint_types):
            joint_position = np.array(
                [head_start_x + (joint_id + 1.0) * self.module_pitch, 0.0, initial_z],
                dtype=np.float64,
            )
            frame_rotation = (
                self.chrono.QuatFromAngleX(0.0)
                if joint_type == JOINT_YAW
                else self.chrono.QuatFromAngleX(-0.5 * math.pi)
            )
            motor = self.chrono.ChLinkMotorRotationTorque()
            joint_label = "yaw" if joint_type == JOINT_YAW else "pitch"
            motor.SetName(f"joint_{joint_id:02d}_{joint_label}")
            motor.Initialize(
                self.bodies[joint_id + 1],
                self.bodies[joint_id],
                self.chrono.ChFramed(self._chrono_vec(joint_position), frame_rotation),
            )
            torque_function = self.chrono.ChFunctionConst(0.0)
            motor.SetTorqueFunction(torque_function)
            self.system.AddLink(motor)
            self.motors.append(motor)
            self.torque_functions.append(torque_function)

        self.last_target_angle = np.zeros(config.model.joint_count, dtype=np.float64)
        self.last_target_rate = np.zeros(config.model.joint_count, dtype=np.float64)
        self.last_torque_command = np.zeros(config.model.joint_count, dtype=np.float64)
        self.last_saturation = np.zeros(config.model.joint_count, dtype=np.int32)
        self._contact_patch_cache: dict[str, np.ndarray] | None = None
        self._patch_module_ids: np.ndarray | None = None
        self._patch_axial_segment_ids: np.ndarray | None = None
        self._runtime_patch_kinematics = VectorizedSnakePatchKinematics(self.geometry)

        initial_patches = self.contact_patches()
        if initial_patches["center"].shape[0] != config.contact_patch_count:
            raise RuntimeError(
                f"Patch topology mismatch: {initial_patches['center'].shape[0]} "
                f"!= {config.contact_patch_count}"
            )

    def _chrono_vec(self, value):
        x, y, z = [float(component) for component in value]
        return self.chrono.ChVector3d(x, y, z)

    def _body_rotation(self, body) -> np.ndarray:
        columns = []
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            world = body.TransformDirectionLocalToParent(self._chrono_vec(axis))
            columns.append(_unit(_np_vec(world)))
        return np.column_stack(columns)

    def current_pose(self) -> SnakePose:
        count = self.config.model.module_count
        centers = np.asarray([_np_vec(body.GetPos()) for body in self.bodies])
        rotations = np.asarray([self._body_rotation(body) for body in self.bodies])
        boundaries = np.empty((count + 1, 3), dtype=np.float64)
        boundaries[0] = centers[0] - 0.5 * self.module_pitch * rotations[0, :, 0]
        joint_positions = np.empty((count - 1, 3), dtype=np.float64)
        joint_axes = np.empty((count - 1, 3), dtype=np.float64)
        for joint_id, joint_type in enumerate(self.joint_types):
            parent_end = centers[joint_id] + 0.5 * self.module_pitch * rotations[joint_id, :, 0]
            child_start = centers[joint_id + 1] - 0.5 * self.module_pitch * rotations[joint_id + 1, :, 0]
            joint_positions[joint_id] = 0.5 * (parent_end + child_start)
            boundaries[joint_id + 1] = joint_positions[joint_id]
            local_axis_index = 2 if joint_type == JOINT_YAW else 1
            joint_axes[joint_id] = rotations[joint_id, :, local_axis_index]
        boundaries[-1] = centers[-1] + 0.5 * self.module_pitch * rotations[-1, :, 0]
        return SnakePose(
            body_centers=centers,
            body_rotations=rotations,
            boundaries=boundaries,
            joint_positions=joint_positions,
            joint_axes=joint_axes,
            joint_types=self.joint_types.copy(),
            joint_angles_rad=self.joint_angles(),
            phase_rad=self.gait_phase_rad(),
        )

    def gait_phase_rad(self, time_s: float | None = None) -> float:
        if time_s is None:
            time_s = float(self.system.GetChTime())
        gait_time = max(0.0, float(time_s) - self.config.model.settle_time_s)
        return -2.0 * math.pi * self.config.model.gait_frequency_hz * gait_time

    def joint_angles(self) -> np.ndarray:
        return np.asarray([float(motor.GetMotorAngle()) for motor in self.motors])

    def joint_rates(self) -> np.ndarray:
        return np.asarray([float(motor.GetMotorAngleDt()) for motor in self.motors])

    def contact_patches(self) -> dict[str, np.ndarray]:
        if self._contact_patch_cache is None:
            pose = self.current_pose()
            runtime = self._runtime_patch_kinematics.evaluate(pose)
            module_ids = np.asarray(runtime["module_id"], dtype=np.int32)
            axial_segment_ids = np.asarray(runtime["axial_segment_id"], dtype=np.int32)
            centers = np.asarray(runtime["center"], dtype=np.float64)
            body_centers = np.asarray([_np_vec(body.GetPos()) for body in self.bodies])
            body_velocities = np.asarray([_np_vec(body.GetLinVel()) for body in self.bodies])
            body_omegas = np.asarray(
                [_np_vec(body.GetAngVelParent()) for body in self.bodies]
            )
            velocities = body_velocities[module_ids] + np.cross(
                body_omegas[module_ids], centers - body_centers[module_ids]
            )
            self._contact_patch_cache = {
                "center": centers.astype(np.float32),
                "axis_long": np.asarray(runtime["axis_long"], dtype=np.float32),
                "axis_width": np.asarray(runtime["axis_width"], dtype=np.float32),
                "normal": np.asarray(runtime["normal"], dtype=np.float32),
                "velocity": velocities.astype(np.float32),
                "half_extent": np.asarray(runtime["half_extent"], dtype=np.float32),
                "mass": np.asarray(runtime["mass"], dtype=np.float32),
                "side_id": np.asarray(runtime["side_id"], dtype=np.int32),
                "shoe_id": np.asarray(runtime["shoe_id"], dtype=np.float32),
                "surface_id": np.asarray(runtime["surface_id"], dtype=np.int32),
                "module_id": module_ids,
                "axial_segment_id": axial_segment_ids,
            }
            if self._patch_module_ids is None:
                self._patch_module_ids = module_ids.copy()
                self._patch_axial_segment_ids = axial_segment_ids.copy()
            elif not np.array_equal(self._patch_module_ids, module_ids):
                raise RuntimeError("Contact patch ownership changed during the simulation")
        return self._contact_patch_cache

    def controller_snapshot(self, time_s: float | None = None) -> ControllerSnapshot:
        if time_s is None:
            time_s = float(self.system.GetChTime())
        target, target_rate = gait_targets(self.config, float(time_s))
        return ControllerSnapshot(
            target_angle_rad=target,
            target_rate_rad_s=target_rate,
            actual_angle_rad=self.joint_angles(),
            actual_rate_rad_s=self.joint_rates(),
            torque_command_Nm=self.last_torque_command.copy(),
            saturation=self.last_saturation.copy(),
        )

    def update_controller(self, time_s: float) -> None:
        target, target_rate = gait_targets(self.config, time_s)
        angle = self.joint_angles()
        rate = self.joint_rates()
        raw_torque = (
            self.config.model.motor_kp_Nm_rad * (target - angle)
            + self.config.model.motor_kd_Nm_s_rad * (target_rate - rate)
        )
        limit = self.config.model.motor_torque_limit_Nm
        torque = np.clip(raw_torque, -limit, limit)
        for function, value in zip(self.torque_functions, torque):
            function.SetConstant(float(value))
        self.last_target_angle = target
        self.last_target_rate = target_rate
        self.last_torque_command = torque
        self.last_saturation = (np.abs(raw_torque) >= limit).astype(np.int32)

    def apply_contact_feedback(self, feedback: ContactFeedback) -> None:
        total_forces, total_moments = self.patch_wrenches_by_body(feedback)
        for body in self.bodies:
            body.EmptyAccumulators()
        for module_id, body in enumerate(self.bodies):
            body.AccumulateForce(
                self._chrono_vec(total_forces[module_id]), body.GetPos(), False
            )
            body.AccumulateTorque(self._chrono_vec(total_moments[module_id]), False)

    def patch_wrenches_by_body(
        self, feedback: ContactFeedback
    ) -> tuple[np.ndarray, np.ndarray]:
        if feedback.module_forces is not None:
            body_forces = np.asarray(feedback.module_forces, dtype=np.float64)
            moments_origin = np.asarray(
                feedback.module_moments_about_origin, dtype=np.float64
            )
            expected_shape = (self.config.model.module_count, 3)
            if body_forces.shape != expected_shape or moments_origin.shape != expected_shape:
                raise ValueError(
                    "Reduced contact feedback must contain one force and origin moment "
                    f"per module; got {body_forces.shape}/{moments_origin.shape}"
                )
            centers = np.asarray([_np_vec(body.GetPos()) for body in self.bodies])
            body_moments = moments_origin - np.cross(centers, body_forces)
            return body_forces.copy(), body_moments

        if feedback.patch_forces is None or feedback.patch_moments is None:
            raise ValueError("Contact feedback contains neither reduced nor patch data")
        forces = np.asarray(feedback.patch_forces, dtype=np.float64)
        moments = np.asarray(feedback.patch_moments, dtype=np.float64)
        patches = self.contact_patches()
        centers = np.asarray(patches["center"], dtype=np.float64)
        if forces.shape != centers.shape or moments.shape != centers.shape:
            raise ValueError(
                f"Contact feedback shape {forces.shape}/{moments.shape} does not match "
                f"patch centers {centers.shape}"
            )
        module_ids = np.asarray(patches["module_id"], dtype=np.int32)
        body_forces = np.zeros((self.config.model.module_count, 3), dtype=np.float64)
        body_moments = np.zeros((self.config.model.module_count, 3), dtype=np.float64)
        for module_id, body in enumerate(self.bodies):
            mask = module_ids == module_id
            if not np.any(mask):
                continue
            com = _np_vec(body.GetPos())
            body_forces[module_id] = np.sum(forces[mask], axis=0)
            body_moments[module_id] = np.sum(
                moments[mask] + np.cross(centers[mask] - com, forces[mask]), axis=0
            )
        return body_forces, body_moments

    def step(self, dt: float, feedback: ContactFeedback) -> None:
        self.apply_contact_feedback(feedback)
        self.update_controller(float(self.system.GetChTime()) + float(dt))
        self.system.DoStepDynamics(float(dt))
        self._contact_patch_cache = None

    def constraint_errors(self) -> dict[str, float | np.ndarray]:
        centers = np.asarray([_np_vec(body.GetPos()) for body in self.bodies])
        rotations = np.asarray([self._body_rotation(body) for body in self.bodies])
        position_errors = np.zeros(self.config.model.joint_count, dtype=np.float64)
        axis_errors = np.zeros(self.config.model.joint_count, dtype=np.float64)
        for joint_id, joint_type in enumerate(self.joint_types):
            parent_end = centers[joint_id] + 0.5 * self.module_pitch * rotations[joint_id, :, 0]
            child_start = centers[joint_id + 1] - 0.5 * self.module_pitch * rotations[joint_id + 1, :, 0]
            position_errors[joint_id] = np.linalg.norm(parent_end - child_start)
            axis_index = 2 if joint_type == JOINT_YAW else 1
            dot = float(
                np.clip(
                    np.dot(
                        rotations[joint_id, :, axis_index],
                        rotations[joint_id + 1, :, axis_index],
                    ),
                    -1.0,
                    1.0,
                )
            )
            axis_errors[joint_id] = math.acos(dot)
        return {
            "position_errors_m": position_errors,
            "axis_errors_rad": axis_errors,
            "max_position_error_m": float(np.max(position_errors)),
            "max_axis_error_rad": float(np.max(axis_errors)),
        }

    def center_of_mass_state(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray([_np_vec(body.GetPos()) for body in self.bodies])
        velocities = np.asarray([_np_vec(body.GetLinVel()) for body in self.bodies])
        return np.mean(positions, axis=0), np.mean(velocities, axis=0)

    def mechanical_energy(self) -> float:
        energy = 0.0
        for body in self.bodies:
            velocity = _np_vec(body.GetLinVel())
            omega_local = _np_vec(body.GetAngVelLocal())
            energy += 0.5 * self.module_mass * float(np.dot(velocity, velocity))
            energy += 0.5 * float(np.dot(self.module_inertia * omega_local, omega_local))
            energy += (
                self.module_mass
                * self.config.model.gravity
                * float(body.GetPos().z)
            )
        return energy

    def actuator_power(self) -> float:
        return float(np.dot(self.last_torque_command, self.joint_rates()))

    def patch_contact_power(self, feedback: ContactFeedback) -> float:
        if feedback.module_forces is not None:
            body_forces, body_moments = self.patch_wrenches_by_body(feedback)
            linear_velocity = np.asarray(
                [_np_vec(body.GetLinVel()) for body in self.bodies]
            )
            angular_velocity = np.asarray(
                [_np_vec(body.GetAngVelParent()) for body in self.bodies]
            )
            return float(
                np.sum(body_forces * linear_velocity)
                + np.sum(body_moments * angular_velocity)
            )
        if feedback.patch_forces is None or feedback.patch_moments is None:
            return 0.0
        forces = np.asarray(feedback.patch_forces, dtype=np.float64)
        moments = np.asarray(feedback.patch_moments, dtype=np.float64)
        patches = self.contact_patches()
        centers = np.asarray(patches["center"], dtype=np.float64)
        module_ids = np.asarray(patches["module_id"], dtype=np.int32)
        power = 0.0
        for module_id, body in enumerate(self.bodies):
            mask = module_ids == module_id
            if not np.any(mask):
                continue
            com = _np_vec(body.GetPos())
            linear_velocity = _np_vec(body.GetLinVel())
            angular_velocity = _np_vec(body.GetAngVelParent())
            point_velocity = linear_velocity + np.cross(
                angular_velocity, centers[mask] - com
            )
            power += float(np.sum(forces[mask] * point_velocity))
            power += float(np.sum(moments[mask] * angular_velocity))
        return power

    def contact_length_ratio(
        self,
        feedback: ContactFeedback,
        threshold_N: float,
    ) -> tuple[float, int, np.ndarray]:
        segment_count = self.config.model.module_count * self.geometry.axial_patch_divisions
        if feedback.axial_segment_normal_loads is not None:
            segment_load = np.asarray(
                feedback.axial_segment_normal_loads, dtype=np.float64
            )
            if segment_load.shape != (segment_count,):
                raise ValueError(
                    f"Expected {segment_count} axial loads, got {segment_load.shape}"
                )
        else:
            if feedback.patch_forces is None:
                segment_load = np.zeros(segment_count, dtype=np.float64)
            else:
                patches = self.contact_patches()
                forces = np.asarray(feedback.patch_forces, dtype=np.float64)
                normals = np.asarray(patches["normal"], dtype=np.float64)
                segment_ids = np.asarray(
                    patches["axial_segment_id"], dtype=np.int32
                )
                normal_force = np.maximum(0.0, -np.sum(forces * normals, axis=1))
                segment_load = np.zeros(segment_count, dtype=np.float64)
                valid = segment_ids >= 0
                np.add.at(segment_load, segment_ids[valid], normal_force[valid])
        active = segment_load >= float(threshold_N)
        ratio = float(np.count_nonzero(active) / segment_count)
        return ratio, int(np.count_nonzero(active)), segment_load

    def export_vtk(
        self,
        vtk_dir: Path,
        step: int,
        feedback: ContactFeedback,
        contact_force_threshold_N: float,
    ) -> None:
        vtk_dir.mkdir(parents=True, exist_ok=True)
        pose = self.current_pose()
        module_mesh = build_module_surfaces(self.geometry, pose)
        body_speed = np.asarray(
            [np.linalg.norm(_np_vec(body.GetLinVel())) for body in self.bodies],
            dtype=np.float64,
        )
        module_ids = np.asarray(module_mesh.cell_scalars["module_id"], dtype=np.int32)
        module_mesh.cell_scalars["body_speed_m_s"] = body_speed[module_ids]
        module_mesh.write_legacy_vtk(
            vtk_dir / f"snake_modules_{step:06d}.vtk",
            "CMU snake MBD rigid modules",
        )

        topology = build_topology_lines(self.geometry, pose)
        snapshot = self.controller_snapshot()
        joint_padding = np.zeros(self.config.model.module_count, dtype=np.float64)
        topology.cell_scalars["target_angle_deg"] = np.concatenate(
            (joint_padding, np.degrees(snapshot.target_angle_rad))
        )
        topology.cell_scalars["actual_angle_deg"] = np.concatenate(
            (joint_padding, np.degrees(snapshot.actual_angle_rad))
        )
        topology.cell_scalars["motor_torque_Nm"] = np.concatenate(
            (joint_padding, snapshot.torque_command_Nm)
        )
        topology.cell_scalars["motor_saturated"] = np.concatenate(
            (
                np.zeros(self.config.model.module_count, dtype=np.int32),
                snapshot.saturation,
            )
        )
        topology.write_legacy_vtk(
            vtk_dir / f"snake_topology_{step:06d}.vtk",
            "CMU snake MBD joints and control state",
        )

        contact_mesh = build_contact_patches(self.geometry, pose)
        if feedback.patch_forces is None:
            raise ValueError("VTK contact export requires per-patch feedback")
        forces = np.asarray(feedback.patch_forces, dtype=np.float64)
        normals = np.asarray(contact_mesh.cell_vectors["normal"], dtype=np.float64)
        areas = np.asarray(contact_mesh.cell_scalars["patch_area_m2"], dtype=np.float64)
        normal_force = np.maximum(0.0, -np.sum(forces * normals, axis=1))
        contact_mesh.cell_scalars["contact_force_N"] = np.linalg.norm(forces, axis=1)
        contact_mesh.cell_scalars["normal_force_N"] = normal_force
        contact_mesh.cell_scalars["contact_pressure_Pa"] = normal_force / np.maximum(
            areas, 1.0e-15
        )
        contact_mesh.cell_scalars["contact_active"] = (
            normal_force >= contact_force_threshold_N
        ).astype(np.int32)
        contact_mesh.cell_vectors["contact_force"] = forces
        contact_mesh.write_legacy_vtk(
            vtk_dir / f"snake_contact_{step:06d}.vtk",
            "CMU snake MPM contact patches",
        )

        envelope = build_contact_envelope(self.geometry, pose)
        envelope.write_legacy_vtk(
            vtk_dir / f"snake_envelope_{step:06d}.vtk",
            "CMU snake continuous contact envelope",
        )
