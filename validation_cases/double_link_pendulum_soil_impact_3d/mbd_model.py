from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from tank_mpm.chrono_loader import require_chrono_core
from tank_mpm.vtk_io import write_polydata_vtk

from .config import ModelConfig


Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _np_vec(value) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float64)


def _unit(value: Iterable[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / max(float(np.linalg.norm(array)), 1.0e-15)


def _box_inertia(mass: float, dims: tuple[float, float, float]) -> np.ndarray:
    x, y, z = dims
    return mass * np.array([y * y + z * z, x * x + z * z, x * x + y * y]) / 12.0


def hammer_end_face_ring_counts(patch_grid: tuple[int, int]) -> tuple[int, ...]:
    """Return angular patch counts for concentric rings on one hammer end face."""

    count_theta, count_axial = (int(value) for value in patch_grid)
    count_radial = max(1, int(math.ceil(0.5 * count_axial)))
    return tuple(
        max(1, int(round(count_theta * (ring + 0.5) / count_radial)))
        for ring in range(count_radial)
    )


def hammer_contact_patch_count(patch_grid: tuple[int, int]) -> int:
    """Total mantle plus two circular-end-face contact patches."""

    count_theta, count_axial = (int(value) for value in patch_grid)
    return count_theta * count_axial + 2 * sum(hammer_end_face_ring_counts(patch_grid))


@dataclass(frozen=True)
class BodyMassProperties:
    mass: float
    inertia_xx: np.ndarray
    com_from_joint: float


@dataclass(frozen=True)
class ContactFeedback:
    patch_forces: np.ndarray
    patch_moments: np.ndarray

    @classmethod
    def zeros(cls, count: int) -> "ContactFeedback":
        return cls(
            patch_forces=np.zeros((count, 3), dtype=np.float32),
            patch_moments=np.zeros((count, 3), dtype=np.float32),
        )


class _SurfaceMesh:
    def __init__(self) -> None:
        self.points: list[np.ndarray] = []
        self.faces: list[list[int]] = []
        self.part_id: list[int] = []
        self.body_id: list[int] = []
        self.contact_surface: list[int] = []
        self.contact_patch_id: list[int] = []
        self.contact_force: list[float] = []
        self.contact_pressure: list[float] = []

    def add_face(
        self,
        vertices,
        *,
        part_id: int,
        body_id: int,
        contact_surface: int = 0,
        contact_patch_id: int = -1,
        contact_force: float = 0.0,
        contact_pressure: float = 0.0,
    ) -> None:
        ids = []
        for vertex in vertices:
            ids.append(len(self.points))
            self.points.append(np.asarray(vertex, dtype=np.float64))
        self.faces.append(ids)
        self.part_id.append(int(part_id))
        self.body_id.append(int(body_id))
        self.contact_surface.append(int(contact_surface))
        self.contact_patch_id.append(int(contact_patch_id))
        self.contact_force.append(float(contact_force))
        self.contact_pressure.append(float(contact_pressure))

    def add_box(
        self,
        center: np.ndarray,
        axes: tuple[np.ndarray, np.ndarray, np.ndarray],
        dims: tuple[float, float, float],
        *,
        part_id: int,
        body_id: int,
        contact_surface: int = 0,
        contact_patch_id: int = -1,
        contact_force: float = 0.0,
        contact_pressure: float = 0.0,
    ) -> None:
        a, b, c = [_unit(axis) * (0.5 * dim) for axis, dim in zip(axes, dims)]
        corners = [
            center - a - b - c,
            center + a - b - c,
            center + a + b - c,
            center - a + b - c,
            center - a - b + c,
            center + a - b + c,
            center + a + b + c,
            center - a + b + c,
        ]
        for face in ([0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]):
            self.add_face(
                [corners[index] for index in face],
                part_id=part_id,
                body_id=body_id,
                contact_surface=contact_surface,
                contact_patch_id=contact_patch_id,
                contact_force=contact_force,
                contact_pressure=contact_pressure,
            )

    def add_cylinder(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        width: float,
        *,
        part_id: int,
        body_id: int,
        segments: int,
    ) -> None:
        axis = _unit(axis)
        seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 0.0, 1.0])
        radial_a = _unit(np.cross(axis, seed))
        radial_b = _unit(np.cross(axis, radial_a))
        cap_centers = (center - 0.5 * width * axis, center + 0.5 * width * axis)
        rings: list[list[np.ndarray]] = []
        for cap_center in cap_centers:
            rings.append(
                [
                    cap_center
                    + radius * math.cos(2.0 * math.pi * index / segments) * radial_a
                    + radius * math.sin(2.0 * math.pi * index / segments) * radial_b
                    for index in range(segments)
                ]
            )
        for index in range(segments):
            nxt = (index + 1) % segments
            self.add_face(
                [rings[0][index], rings[0][nxt], rings[1][nxt], rings[1][index]],
                part_id=part_id,
                body_id=body_id,
            )
        self.add_face(reversed(rings[0]), part_id=part_id, body_id=body_id)
        self.add_face(rings[1], part_id=part_id, body_id=body_id)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_polydata_vtk(
            path,
            "3D double-link pendulum hammer",
            np.asarray(self.points, dtype=np.float64),
            self.faces,
            {
                "part_id": ("int", self.part_id),
                "body_id": ("int", self.body_id),
                "contact_surface": ("int", self.contact_surface),
                "contact_patch_id": ("int", self.contact_patch_id),
                "contact_force_N": ("float", self.contact_force),
                "contact_pressure_Pa": ("float", self.contact_pressure),
            },
        )


class DoubleLinkPendulumMBD:
    """Two 3D rigid bodies connected by ideal revolute joints in Chrono."""

    def __init__(self, config: ModelConfig, patch_grid: tuple[int, int], visual_segments: int) -> None:
        self.config = config
        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.visual_segments = int(visual_segments)
        self.chrono = require_chrono_core()
        self.system = self.chrono.ChSystemSMC()
        self.system.SetGravitationalAcceleration(self._chrono_vec((0.0, 0.0, -config.gravity)))
        self.system.SetSolverType(self.chrono.ChSolver.Type_BARZILAIBORWEIN)

        self.link1_properties = BodyMassProperties(
            mass=config.link1_mass,
            inertia_xx=np.asarray(config.link1_inertia, dtype=np.float64),
            com_from_joint=0.5 * config.link1_length,
        )
        self.link2_properties = self._link2_mass_properties()
        self._contact_patch_template = self._build_contact_patch_template()
        self._contact_patch_cache: dict[str, np.ndarray] | None = None

        self.ground = self.chrono.ChBody()
        self.ground.SetName("pendulum_support")
        self.ground.SetFixed(True)
        self.system.AddBody(self.ground)

        angle1 = math.radians(config.initial_angle1_deg)
        angle2 = math.radians(config.initial_angle2_deg)
        rate1 = math.radians(config.initial_rate1_deg_s)
        rate2 = math.radians(config.initial_rate2_deg_s)
        pivot = np.asarray(config.pivot, dtype=np.float64)
        rotation1 = self._rotation_from_alpha(angle1)
        rotation2 = self._rotation_from_alpha(angle2)
        joint2 = pivot + rotation1 @ np.array([config.link1_length, 0.0, 0.0])

        self.body1 = self._make_body(
            "link_1",
            self.link1_properties,
            pivot + rotation1 @ np.array([self.link1_properties.com_from_joint, 0.0, 0.0]),
            angle1,
        )
        self.body2 = self._make_body(
            "link_2_hammer",
            self.link2_properties,
            joint2 + rotation2 @ np.array([self.link2_properties.com_from_joint, 0.0, 0.0]),
            angle2,
        )
        omega1 = -rate1 * Y_AXIS
        omega2 = -rate2 * Y_AXIS
        self.body1.SetAngVelParent(self._chrono_vec(omega1))
        self.body2.SetAngVelParent(self._chrono_vec(omega2))
        v1 = np.cross(omega1, _np_vec(self.body1.GetPos()) - pivot)
        v_joint2 = np.cross(omega1, joint2 - pivot)
        v2 = v_joint2 + np.cross(omega2, _np_vec(self.body2.GetPos()) - joint2)
        self.body1.SetLinVel(self._chrono_vec(v1))
        self.body2.SetLinVel(self._chrono_vec(v2))

        self.system.AddBody(self.body1)
        self.system.AddBody(self.body2)
        joint_rotation = self.chrono.QuatFromAngleX(-0.5 * math.pi)
        self.joint_ground = self.chrono.ChLinkLockRevolute()
        self.joint_ground.SetName("revolute_ground_link1")
        self.joint_ground.Initialize(
            self.body1,
            self.ground,
            self.chrono.ChFramed(self._chrono_vec(pivot), joint_rotation),
        )
        self.system.AddLink(self.joint_ground)

        self.joint_links = self.chrono.ChLinkLockRevolute()
        self.joint_links.SetName("revolute_link1_link2")
        self.joint_links.Initialize(
            self.body2,
            self.body1,
            self.chrono.ChFramed(self._chrono_vec(joint2), joint_rotation),
        )
        self.system.AddLink(self.joint_links)

    def _chrono_vec(self, value: Iterable[float]):
        x, y, z = [float(component) for component in value]
        return self.chrono.ChVector3d(x, y, z)

    @staticmethod
    def _rotation_from_alpha(angle: float) -> np.ndarray:
        """Guide convention: positive alpha rotates +x toward +z."""

        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)

    def _link2_mass_properties(self) -> BodyMassProperties:
        cfg = self.config
        rod_center = 0.5 * cfg.link2_rod_length
        head_center = cfg.hammer_center_from_joint
        total_mass = cfg.link2_rod_mass + cfg.hammer_mass
        com = (cfg.link2_rod_mass * rod_center + cfg.hammer_mass * head_center) / total_mass
        return BodyMassProperties(
            mass=total_mass,
            inertia_xx=np.asarray(cfg.link2_inertia, dtype=np.float64),
            com_from_joint=com,
        )

    def _make_body(self, name: str, properties: BodyMassProperties, position: np.ndarray, angle: float):
        body = self.chrono.ChBody()
        body.SetName(name)
        body.SetMass(float(properties.mass))
        body.SetInertiaXX(self._chrono_vec(properties.inertia_xx))
        body.SetPos(self._chrono_vec(position))
        body.SetRot(self.chrono.QuatFromAngleY(float(-angle)))
        if hasattr(body, "SetUseSleeping"):
            body.SetUseSleeping(False)
        return body

    def _body_axes(self, body) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            _unit(_np_vec(body.TransformDirectionLocalToParent(self._chrono_vec(axis))))
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )

    def _world_point(self, body, local_point: Iterable[float]) -> np.ndarray:
        return _np_vec(body.TransformPointLocalToParent(self._chrono_vec(local_point)))

    def joint2_position(self) -> np.ndarray:
        return self._world_point(self.body1, (0.5 * self.config.link1_length, 0.0, 0.0))

    def hammer_center_position(self) -> np.ndarray:
        return self._world_point(
            self.body2,
            (self.config.hammer_center_from_joint - self.link2_properties.com_from_joint, 0.0, 0.0),
        )

    def generalized_state(self) -> tuple[np.ndarray, np.ndarray]:
        axes1 = self._body_axes(self.body1)
        axes2 = self._body_axes(self.body2)
        angle1 = math.atan2(axes1[0][2], axes1[0][0])
        angle2 = math.atan2(axes2[0][2], axes2[0][0])
        rate1 = -float(self.body1.GetAngVelParent().y)
        rate2 = -float(self.body2.GetAngVelParent().y)
        return np.array([angle1, angle2]), np.array([rate1, rate2])

    @property
    def contact_patch_count(self) -> int:
        return hammer_contact_patch_count(self.patch_grid)

    @property
    def mantle_contact_patch_count(self) -> int:
        return int(self.patch_grid[0] * self.patch_grid[1])

    @property
    def end_face_contact_patch_count(self) -> int:
        return int(sum(hammer_end_face_ring_counts(self.patch_grid)))

    def _build_contact_patch_template(self) -> dict[str, np.ndarray]:
        """Build mantle and two circular end-face patches in body-local coordinates."""

        cfg = self.config
        count_theta, count_y = self.patch_grid
        theta = 2.0 * math.pi * (np.arange(count_theta, dtype=np.float32) + 0.5) / count_theta
        y = (
            -0.5 * cfg.hammer_width
            + (np.arange(count_y, dtype=np.float32) + 0.5) * cfg.hammer_width / count_y
        )
        theta_grid, y_grid = np.meshgrid(theta, y, indexing="ij")
        cosine = np.cos(theta_grid).astype(np.float32)
        sine = np.sin(theta_grid).astype(np.float32)
        center_x = np.float32(
            cfg.hammer_center_from_joint - self.link2_properties.com_from_joint
        )
        centers_local = np.stack(
            (
                center_x + np.float32(cfg.hammer_radius) * cosine,
                y_grid.astype(np.float32),
                np.float32(cfg.hammer_radius) * sine,
            ),
            axis=-1,
        ).reshape((-1, 3))
        normals_local = np.stack(
            (cosine, np.zeros_like(cosine), sine), axis=-1
        ).reshape((-1, 3))
        tangents_local = np.stack(
            (-sine, np.zeros_like(sine), cosine), axis=-1
        ).reshape((-1, 3))
        widths_local = np.repeat(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            self.mantle_contact_patch_count,
            axis=0,
        )
        half_extent = np.repeat(
            np.array(
                [[math.pi * cfg.hammer_radius / count_theta, 0.5 * cfg.hammer_width / count_y]],
                dtype=np.float32,
            ),
            self.mantle_contact_patch_count,
            axis=0,
        )
        center_parts = [centers_local]
        normal_parts = [normals_local]
        axis_long_parts = [tangents_local]
        axis_width_parts = [widths_local]
        half_extent_parts = [half_extent]
        surface_id_parts = [
            np.zeros((self.mantle_contact_patch_count,), dtype=np.int32)
        ]

        ring_counts = hammer_end_face_ring_counts(self.patch_grid)
        radial_step = cfg.hammer_radius / len(ring_counts)
        for end_sign, surface_id in ((-1.0, 1), (1.0, 2)):
            for ring, ring_count_theta in enumerate(ring_counts):
                radius_mid = (ring + 0.5) * radial_step
                ring_theta = (
                    2.0
                    * math.pi
                    * (np.arange(ring_count_theta, dtype=np.float32) + 0.5)
                    / ring_count_theta
                )
                ring_cosine = np.cos(ring_theta).astype(np.float32)
                ring_sine = np.sin(ring_theta).astype(np.float32)
                center_parts.append(
                    np.stack(
                        (
                            center_x + np.float32(radius_mid) * ring_cosine,
                            np.full(
                                (ring_count_theta,),
                                np.float32(0.5 * end_sign * cfg.hammer_width),
                                dtype=np.float32,
                            ),
                            np.float32(radius_mid) * ring_sine,
                        ),
                        axis=-1,
                    )
                )
                normal_parts.append(
                    np.repeat(
                        np.array([[0.0, end_sign, 0.0]], dtype=np.float32),
                        ring_count_theta,
                        axis=0,
                    )
                )
                axis_long_parts.append(
                    np.stack(
                        (ring_cosine, np.zeros_like(ring_cosine), ring_sine),
                        axis=-1,
                    )
                )
                axis_width_parts.append(
                    np.stack(
                        (-ring_sine, np.zeros_like(ring_sine), ring_cosine),
                        axis=-1,
                    )
                )
                half_extent_parts.append(
                    np.repeat(
                        np.array(
                            [[
                                0.5 * radial_step,
                                math.pi * radius_mid / ring_count_theta,
                            ]],
                            dtype=np.float32,
                        ),
                        ring_count_theta,
                        axis=0,
                    )
                )
                surface_id_parts.append(
                    np.full((ring_count_theta,), surface_id, dtype=np.int32)
                )

        centers_local = np.concatenate(center_parts, axis=0)
        normals_local = np.concatenate(normal_parts, axis=0)
        tangents_local = np.concatenate(axis_long_parts, axis=0)
        widths_local = np.concatenate(axis_width_parts, axis=0)
        half_extent = np.concatenate(half_extent_parts, axis=0)
        surface_id_array = np.concatenate(surface_id_parts, axis=0)
        return {
            "center_local": centers_local,
            "normal_local": normals_local,
            "axis_long_local": tangents_local,
            "axis_width_local": widths_local,
            "half_extent": half_extent,
            "surface_id": surface_id_array,
            "mass": np.full(
                (self.contact_patch_count,),
                self.link2_properties.mass / self.contact_patch_count,
                dtype=np.float32,
            ),
            "side_id": np.zeros((self.contact_patch_count,), dtype=np.int32),
            "shoe_id": np.full((self.contact_patch_count,), 2.0, dtype=np.float32),
        }

    def _body_rotation_matrix_f32(self, body) -> np.ndarray:
        return np.column_stack(self._body_axes(body)).astype(np.float32)

    def contact_patches(self) -> dict[str, np.ndarray]:
        """Return cached world-space cylinder patches using one matrix transform."""

        if self._contact_patch_cache is None:
            template = self._contact_patch_template
            rotation = self._body_rotation_matrix_f32(self.body2)
            position = _np_vec(self.body2.GetPos()).astype(np.float32)
            self._contact_patch_cache = {
                "center": template["center_local"] @ rotation.T + position,
                "axis_long": template["axis_long_local"] @ rotation.T,
                "axis_width": template["axis_width_local"] @ rotation.T,
                "normal": template["normal_local"] @ rotation.T,
                "velocity": np.zeros((self.contact_patch_count, 3), dtype=np.float32),
                "half_extent": template["half_extent"],
                "mass": template["mass"],
                "side_id": template["side_id"],
                "shoe_id": template["shoe_id"],
                "surface_id": template["surface_id"],
            }
        return self._contact_patch_cache

    def patch_wrench_about_com(self, feedback: ContactFeedback) -> tuple[np.ndarray, np.ndarray]:
        forces = np.asarray(feedback.patch_forces, dtype=np.float32)
        moments = np.asarray(feedback.patch_moments, dtype=np.float32)
        if not np.any(forces) and not np.any(moments):
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero.copy()
        patches = self.contact_patches()
        centers = np.asarray(patches["center"], dtype=np.float32)
        if forces.shape != centers.shape or moments.shape != centers.shape:
            raise ValueError("Contact feedback does not match hammer patch topology")
        com = _np_vec(self.body2.GetPos()).astype(np.float32)
        total_force = np.sum(forces, axis=0, dtype=np.float32)
        total_moment = np.sum(
            moments + np.cross(centers - com, forces), axis=0, dtype=np.float32
        )
        return total_force, total_moment

    def apply_contact_feedback(self, feedback: ContactFeedback) -> None:
        self.body1.EmptyAccumulators()
        self.body2.EmptyAccumulators()
        total_force, total_moment = self.patch_wrench_about_com(feedback)
        self.body2.AccumulateForce(self._chrono_vec(total_force), self.body2.GetPos(), False)
        self.body2.AccumulateTorque(self._chrono_vec(total_moment), False)
        self._apply_joint_damping()

    def _apply_joint_damping(self) -> None:
        omega1_y = float(self.body1.GetAngVelParent().y)
        omega2_y = float(self.body2.GetAngVelParent().y)
        relative_omega_y = omega2_y - omega1_y
        torque1_y = (
            -self.config.joint_ground_damping * omega1_y
            + self.config.joint_links_damping * relative_omega_y
        )
        torque2_y = -self.config.joint_links_damping * relative_omega_y
        self.body1.AccumulateTorque(self._chrono_vec((0.0, torque1_y, 0.0)), False)
        self.body2.AccumulateTorque(self._chrono_vec((0.0, torque2_y, 0.0)), False)

    def joint_damping_power(self) -> float:
        omega1_y = float(self.body1.GetAngVelParent().y)
        relative_omega_y = float(self.body2.GetAngVelParent().y) - omega1_y
        return -(
            self.config.joint_ground_damping * omega1_y * omega1_y
            + self.config.joint_links_damping * relative_omega_y * relative_omega_y
        )

    def step(self, dt: float, feedback: ContactFeedback) -> None:
        self.apply_contact_feedback(feedback)
        self.system.DoStepDynamics(float(dt))
        self._contact_patch_cache = None

    def generalized_contact_force(self, feedback: ContactFeedback) -> np.ndarray:
        forces = np.asarray(feedback.patch_forces, dtype=np.float32)
        moments = np.asarray(feedback.patch_moments, dtype=np.float32)
        if not np.any(forces) and not np.any(moments):
            return np.zeros(2, dtype=np.float32)
        centers = np.asarray(self.contact_patches()["center"], dtype=np.float32)
        pivot = np.asarray(self.config.pivot, dtype=np.float32)
        joint2 = self.joint2_position().astype(np.float32)
        total_force = np.sum(forces, axis=0, dtype=np.float32)
        q_force = np.zeros(2, dtype=np.float32)
        q_force[0] = -np.cross(joint2 - pivot, total_force)[1]
        q_force[1] = -np.sum(
            np.cross(centers - joint2, forces)[:, 1] + moments[:, 1],
            dtype=np.float32,
        )
        return q_force

    def patch_contact_power(self, feedback: ContactFeedback) -> float:
        forces = np.asarray(feedback.patch_forces, dtype=np.float32)
        moments = np.asarray(feedback.patch_moments, dtype=np.float32)
        if not np.any(forces) and not np.any(moments):
            return 0.0
        centers = np.asarray(self.contact_patches()["center"], dtype=np.float32)
        com = _np_vec(self.body2.GetPos()).astype(np.float32)
        linear_velocity = _np_vec(self.body2.GetLinVel()).astype(np.float32)
        angular_velocity = _np_vec(self.body2.GetAngVelParent()).astype(np.float32)
        point_velocity = linear_velocity + np.cross(angular_velocity, centers - com)
        return float(
            np.sum(forces * point_velocity, dtype=np.float32)
            + np.sum(moments * angular_velocity, dtype=np.float32)
        )

    def generalized_contact_power(self, feedback: ContactFeedback) -> float:
        _, rates = self.generalized_state()
        return float(np.dot(self.generalized_contact_force(feedback), rates))

    def mechanical_energy(self) -> float:
        energy = 0.0
        for body, properties in (
            (self.body1, self.link1_properties),
            (self.body2, self.link2_properties),
        ):
            velocity = _np_vec(body.GetLinVel())
            omega_local = _np_vec(body.GetAngVelLocal())
            z = float(body.GetPos().z)
            energy += 0.5 * properties.mass * float(np.dot(velocity, velocity))
            energy += 0.5 * float(np.dot(properties.inertia_xx * omega_local, omega_local))
            energy += properties.mass * self.config.gravity * z
        return energy

    def constraint_errors(self) -> dict[str, float]:
        pivot = np.asarray(self.config.pivot, dtype=np.float64)
        body1_proximal = self._world_point(self.body1, (-0.5 * self.config.link1_length, 0.0, 0.0))
        body1_distal = self._world_point(self.body1, (0.5 * self.config.link1_length, 0.0, 0.0))
        body2_proximal = self._world_point(
            self.body2,
            (-self.link2_properties.com_from_joint, 0.0, 0.0),
        )
        body1_axis = self._body_axes(self.body1)[1]
        body2_axis = self._body_axes(self.body2)[1]
        ground_axis_error = math.acos(float(np.clip(np.dot(body1_axis, Y_AXIS), -1.0, 1.0)))
        inter_axis_error = math.acos(float(np.clip(np.dot(body1_axis, body2_axis), -1.0, 1.0)))
        return {
            "joint_ground_position_error": float(np.linalg.norm(body1_proximal - pivot)),
            "joint_links_position_error": float(np.linalg.norm(body1_distal - body2_proximal)),
            "joint_ground_axis_error": ground_axis_error,
            "joint_links_axis_error": inter_axis_error,
            "chrono_constraint_violation_norm": max(
                self._constraint_violation_norm(self.joint_ground),
                self._constraint_violation_norm(self.joint_links),
            ),
        }

    @staticmethod
    def _constraint_violation_norm(link) -> float:
        values = link.GetConstraintViolation()
        try:
            array = np.asarray(list(values), dtype=np.float64)
        except TypeError:
            size = int(values.size()) if hasattr(values, "size") else int(values.Size())
            if hasattr(values, "GetItem"):
                array = np.asarray([values.GetItem(index) for index in range(size)], dtype=np.float64)
            else:
                array = np.asarray([values[index] for index in range(size)], dtype=np.float64)
        return float(np.linalg.norm(array)) if array.size else 0.0

    def joint_reactions(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name, link in (("ground", self.joint_ground), ("links", self.joint_links)):
            reaction = link.GetReaction1()
            force_local = _np_vec(reaction.force)
            torque_local = _np_vec(reaction.torque)
            frame = link.GetFrame1Abs()
            try:
                force_world = _np_vec(frame.TransformDirectionLocalToParent(self._chrono_vec(force_local)))
                torque_world = _np_vec(frame.TransformDirectionLocalToParent(self._chrono_vec(torque_local)))
            except AttributeError:
                force_world = force_local.copy()
                torque_world = torque_local.copy()
            result[f"{name}_force_world"] = force_world
            result[f"{name}_torque_world"] = torque_world
        return result

    def export_vtk(self, path: Path, feedback: ContactFeedback | None = None) -> None:
        mesh = _SurfaceMesh()
        axes1 = self._body_axes(self.body1)
        axes2 = self._body_axes(self.body2)
        mesh.add_box(
            _np_vec(self.body1.GetPos()),
            axes1,
            (self.config.link1_length, self.config.link1_width, self.config.link1_thickness),
            part_id=1,
            body_id=1,
        )
        rod_center_local = (
            0.5 * self.config.link2_rod_length - self.link2_properties.com_from_joint,
            0.0,
            0.0,
        )
        head_center_local = (
            self.config.hammer_center_from_joint - self.link2_properties.com_from_joint,
            0.0,
            0.0,
        )
        mesh.add_box(
            self._world_point(self.body2, rod_center_local),
            axes2,
            (
                self.config.link2_rod_length,
                self.config.link2_rod_width,
                self.config.link2_rod_thickness,
            ),
            part_id=2,
            body_id=2,
        )
        mesh.add_cylinder(
            self._world_point(self.body2, head_center_local),
            axes2[1],
            self.config.hammer_radius,
            self.config.hammer_width,
            part_id=3,
            body_id=2,
            segments=self.visual_segments,
        )
        mesh.add_cylinder(
            np.asarray(self.config.pivot, dtype=np.float64),
            Y_AXIS,
            0.065,
            1.25 * self.config.link1_width,
            part_id=4,
            body_id=0,
            segments=self.visual_segments,
        )
        mesh.add_cylinder(
            self.joint2_position(),
            Y_AXIS,
            0.055,
            1.15 * max(self.config.link1_width, self.config.link2_rod_width),
            part_id=4,
            body_id=1,
            segments=self.visual_segments,
        )
        patches = self.contact_patches()
        if feedback is None:
            feedback = ContactFeedback.zeros(self.contact_patch_count)
        patch_forces = np.asarray(feedback.patch_forces, dtype=np.float64)
        if patch_forces.shape != (self.contact_patch_count, 3):
            raise ValueError("Contact feedback does not match hammer patch topology")
        patch_area = 4.0 * np.asarray(patches["half_extent"][:, 0], dtype=np.float64) * np.asarray(
            patches["half_extent"][:, 1], dtype=np.float64
        )
        patch_surface_id = np.asarray(patches["surface_id"], dtype=np.int32)
        for patch_id, (center, normal, axis_width, axis_long, half_extent, force) in enumerate(
            zip(
                patches["center"],
                patches["normal"],
                patches["axis_width"],
                patches["axis_long"],
                patches["half_extent"],
                patch_forces,
            )
        ):
            force_magnitude = float(np.linalg.norm(force))
            normal_reaction = max(0.0, float(np.dot(force, -np.asarray(normal, dtype=np.float64))))
            pressure = normal_reaction / max(float(patch_area[patch_id]), 1.0e-15)
            mesh.add_box(
                np.asarray(center, dtype=np.float64),
                (
                    np.asarray(normal, dtype=np.float64),
                    np.asarray(axis_width, dtype=np.float64),
                    np.asarray(axis_long, dtype=np.float64),
                ),
                (0.002, 2.0 * float(half_extent[1]), 2.0 * float(half_extent[0])),
                part_id=6 + int(patch_surface_id[patch_id]),
                body_id=2,
                contact_surface=1,
                contact_patch_id=patch_id,
                contact_force=force_magnitude,
                contact_pressure=pressure,
            )
        mesh.write(path)
