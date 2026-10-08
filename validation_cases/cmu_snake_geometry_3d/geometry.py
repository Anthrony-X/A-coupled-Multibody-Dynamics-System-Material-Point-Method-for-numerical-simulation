from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


HEAD = 0
MIDDLE = 1
TAIL = 2

JOINT_NONE = 0
JOINT_YAW = 1
JOINT_PITCH = 2

SURFACE_MANTLE = 0
SURFACE_HEAD_CAP = 1
SURFACE_TAIL_CAP = 2


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length > 1.0e-12:
        return vector / length
    if fallback is None:
        raise ValueError("Cannot normalize a near-zero vector")
    return _unit(np.asarray(fallback, dtype=np.float64))


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = _unit(axis)
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    identity = np.eye(3, dtype=np.float64)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _unit(source)
    target = _unit(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1.0e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(seed, source))) > 0.8:
            seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return _rotation_about_axis(np.cross(source, seed), math.pi)
    axis = cross / sine
    return _rotation_about_axis(axis, math.atan2(sine, cosine))


def module_type_array(module_count: int) -> np.ndarray:
    result = np.full(module_count, MIDDLE, dtype=np.int32)
    result[0] = HEAD
    result[-1] = TAIL
    return result


@dataclass(frozen=True)
class SnakeGeometryConfig:
    """Contact-oriented geometry inferred from the published overall dimensions.

    The papers provide the complete robot length, mass and approximate diameter, but
    not a manufacturing CAD model. The 17 equal kinematic pitches and uniform masses
    are therefore explicit preview assumptions that can later be replaced by measured
    module properties without changing the topology or VTK field layout.
    """

    total_length_m: float = 0.94
    total_mass_kg: float = 3.150
    module_count: int = 17
    radius_m: float = 0.025
    visual_radius_m: float = 0.0235
    visual_length_ratio: float = 0.86
    circumferential_divisions: int = 16
    axial_patch_divisions: int = 3
    cap_radial_divisions: int = 2
    ground_tolerance_m: float = 0.004
    horizontal_wave_amplitude_m: float = 0.035
    vertical_wave_amplitude_m: float = 0.016
    wavelength_m: float = 0.47
    body_wave_phase_offset_rad: float = math.pi / 2.0

    def __post_init__(self) -> None:
        if self.module_count != 17:
            raise ValueError("This case requires exactly 17 rigid modules")
        if self.total_length_m <= 0.0 or self.total_mass_kg <= 0.0:
            raise ValueError("Robot length and mass must be positive")
        if not (0.0 < self.visual_radius_m <= self.radius_m):
            raise ValueError("visual_radius_m must be in (0, radius_m]")
        if not (0.0 < self.visual_length_ratio < 1.0):
            raise ValueError("visual_length_ratio must be in (0, 1)")
        if self.circumferential_divisions < 8:
            raise ValueError("At least eight circumferential divisions are required")
        if self.axial_patch_divisions < 1 or self.cap_radial_divisions < 1:
            raise ValueError("Patch division counts must be positive")
        if self.horizontal_wave_amplitude_m < 0.0 or self.vertical_wave_amplitude_m < 0.0:
            raise ValueError("Backbone wave amplitudes must be non-negative")
        if self.wavelength_m <= 0.0:
            raise ValueError("Backbone wavelength must be positive")

    @property
    def module_pitch_m(self) -> float:
        return self.total_length_m / self.module_count

    @property
    def module_mass_kg(self) -> float:
        return self.total_mass_kg / self.module_count

    @property
    def joint_count(self) -> int:
        return self.module_count - 1


@dataclass(frozen=True)
class SnakePose:
    body_centers: np.ndarray
    body_rotations: np.ndarray
    boundaries: np.ndarray
    joint_positions: np.ndarray
    joint_axes: np.ndarray
    joint_types: np.ndarray
    joint_angles_rad: np.ndarray
    phase_rad: float

    def translated(self, shift: np.ndarray) -> "SnakePose":
        shift = np.asarray(shift, dtype=np.float64)
        return replace(
            self,
            body_centers=self.body_centers + shift,
            boundaries=self.boundaries + shift,
            joint_positions=self.joint_positions + shift,
        )

    def rotated(self, rotation: np.ndarray) -> "SnakePose":
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        return replace(
            self,
            body_centers=self.body_centers @ rotation.T,
            body_rotations=np.einsum("ij,njk->nik", rotation, self.body_rotations),
            boundaries=self.boundaries @ rotation.T,
            joint_positions=self.joint_positions @ rotation.T,
            joint_axes=self.joint_axes @ rotation.T,
        )


@dataclass
class PolyData:
    points: np.ndarray
    polygons: list[list[int]]
    lines: list[list[int]]
    cell_scalars: dict[str, np.ndarray]
    cell_vectors: dict[str, np.ndarray]

    @property
    def cell_count(self) -> int:
        return len(self.lines) + len(self.polygons)

    def validate(self) -> None:
        points = np.asarray(self.points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {points.shape}")
        for name, values in self.cell_scalars.items():
            if np.asarray(values).shape != (self.cell_count,):
                raise ValueError(
                    f"Scalar {name!r} must have shape ({self.cell_count},), "
                    f"got {np.asarray(values).shape}"
                )
        for name, values in self.cell_vectors.items():
            if np.asarray(values).shape != (self.cell_count, 3):
                raise ValueError(
                    f"Vector {name!r} must have shape ({self.cell_count}, 3), "
                    f"got {np.asarray(values).shape}"
                )

    def write_legacy_vtk(self, path: Path | str, title: str) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# vtk DataFile Version 3.0\n")
            handle.write(f"{title}\n")
            handle.write("ASCII\n")
            handle.write("DATASET POLYDATA\n")
            handle.write(f"POINTS {len(self.points)} float\n")
            for point in np.asarray(self.points, dtype=np.float64):
                handle.write(f"{point[0]:.9e} {point[1]:.9e} {point[2]:.9e}\n")

            if self.lines:
                size = sum(len(line) + 1 for line in self.lines)
                handle.write(f"LINES {len(self.lines)} {size}\n")
                for line in self.lines:
                    handle.write(f"{len(line)} {' '.join(str(int(i)) for i in line)}\n")
            if self.polygons:
                size = sum(len(face) + 1 for face in self.polygons)
                handle.write(f"POLYGONS {len(self.polygons)} {size}\n")
                for face in self.polygons:
                    handle.write(f"{len(face)} {' '.join(str(int(i)) for i in face)}\n")

            if self.cell_count:
                handle.write(f"CELL_DATA {self.cell_count}\n")
                for name, raw_values in self.cell_scalars.items():
                    values = np.asarray(raw_values)
                    vtk_type = "int" if np.issubdtype(values.dtype, np.integer) else "float"
                    handle.write(f"SCALARS {name} {vtk_type} 1\n")
                    handle.write("LOOKUP_TABLE default\n")
                    if vtk_type == "int":
                        for value in values:
                            handle.write(f"{int(value)}\n")
                    else:
                        for value in values:
                            handle.write(f"{float(value):.9e}\n")
                for name, raw_values in self.cell_vectors.items():
                    handle.write(f"VECTORS {name} float\n")
                    for value in np.asarray(raw_values, dtype=np.float64):
                        handle.write(f"{value[0]:.9e} {value[1]:.9e} {value[2]:.9e}\n")


def joint_types(config: SnakeGeometryConfig) -> np.ndarray:
    result = np.empty(config.joint_count, dtype=np.int32)
    result[0::2] = JOINT_YAW
    result[1::2] = JOINT_PITCH
    return result


def _continuous_backbone_module_frames(
    config: SnakeGeometryConfig,
    phase_rad: float,
) -> np.ndarray:
    """Build twist-free target frames along the continuous elliptical backbone."""

    center_s = (
        np.arange(config.module_count, dtype=np.float64) + 0.5
    ) * config.module_pitch_m
    wave_number = 2.0 * math.pi / config.wavelength_m
    argument = wave_number * center_s + float(phase_rad)
    tangents = np.column_stack(
        (
            np.ones(config.module_count, dtype=np.float64),
            config.horizontal_wave_amplitude_m
            * wave_number
            * np.cos(argument),
            config.vertical_wave_amplitude_m
            * wave_number
            * np.cos(argument + config.body_wave_phase_offset_rad),
        )
    )
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    local_y = np.empty_like(tangents)
    local_z = np.empty_like(tangents)
    seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, tangents[0]))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    local_y[0] = _unit(seed - np.dot(seed, tangents[0]) * tangents[0])
    local_z[0] = _unit(np.cross(tangents[0], local_y[0]))
    local_y[0] = _unit(np.cross(local_z[0], tangents[0]))

    for module_id in range(1, config.module_count):
        transported = (
            _rotation_between(tangents[module_id - 1], tangents[module_id])
            @ local_y[module_id - 1]
        )
        local_y[module_id] = _unit(
            transported
            - np.dot(transported, tangents[module_id]) * tangents[module_id],
            fallback=local_y[module_id - 1],
        )
        local_z[module_id] = _unit(
            np.cross(tangents[module_id], local_y[module_id])
        )
        local_y[module_id] = _unit(
            np.cross(local_z[module_id], tangents[module_id])
        )

    return np.stack(
        [
            np.column_stack((tangents[i], local_y[i], local_z[i]))
            for i in range(config.module_count)
        ]
    )


def _fit_continuous_backbone_joint_angles(
    config: SnakeGeometryConfig,
    phase_rad: float,
) -> np.ndarray:
    """Fit the continuous Bishop frames to the alternating single-axis chain."""

    target_frames = _continuous_backbone_module_frames(config, phase_rad)
    target_frames = np.einsum(
        "ij,njk->nik", target_frames[0].T, target_frames
    )
    types = joint_types(config)
    fitted_rotation = np.eye(3, dtype=np.float64)
    angles = np.empty(config.joint_count, dtype=np.float64)

    for joint_id, joint_type in enumerate(types):
        desired_relative = fitted_rotation.T @ target_frames[joint_id + 1]
        if joint_type == JOINT_YAW:
            angle = math.atan2(
                desired_relative[1, 0] - desired_relative[0, 1],
                desired_relative[0, 0] + desired_relative[1, 1],
            )
            local_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            angle = math.atan2(
                desired_relative[0, 2] - desired_relative[2, 0],
                desired_relative[0, 0] + desired_relative[2, 2],
            )
            local_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        angles[joint_id] = angle
        fitted_rotation = fitted_rotation @ _rotation_about_axis(local_axis, angle)
    return angles


@lru_cache(maxsize=16)
def _continuous_backbone_fit_table(
    config: SnakeGeometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = 512
    phase_step = 2.0 * math.pi / sample_count
    phases = phase_step * np.arange(sample_count, dtype=np.float64)
    angles = np.asarray(
        [_fit_continuous_backbone_joint_angles(config, phase) for phase in phases],
        dtype=np.float64,
    )
    angular_delta = np.arctan2(
        np.sin(np.roll(angles, -1, axis=0) - np.roll(angles, 1, axis=0)),
        np.cos(np.roll(angles, -1, axis=0) - np.roll(angles, 1, axis=0)),
    )
    phase_derivatives = angular_delta / (2.0 * phase_step)
    angles.flags.writeable = False
    phase_derivatives.flags.writeable = False
    return angles, phase_derivatives


def continuous_backbone_joint_state(
    config: SnakeGeometryConfig,
    phase_rad: float,
    *,
    straight: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted joint angles and their derivative with respect to wave phase."""

    if straight:
        zeros = np.zeros(config.joint_count, dtype=np.float64)
        return zeros, zeros.copy()

    angles, derivatives = _continuous_backbone_fit_table(config)
    sample_count = angles.shape[0]
    phase_step = 2.0 * math.pi / sample_count
    coordinate = (float(phase_rad) % (2.0 * math.pi)) / phase_step
    index0 = int(math.floor(coordinate)) % sample_count
    alpha = coordinate - math.floor(coordinate)
    index1 = (index0 + 1) % sample_count

    q0 = angles[index0]
    q1 = q0 + np.arctan2(
        np.sin(angles[index1] - q0), np.cos(angles[index1] - q0)
    )
    m0 = derivatives[index0]
    m1 = derivatives[index1]
    a2 = alpha * alpha
    a3 = a2 * alpha
    h00 = 2.0 * a3 - 3.0 * a2 + 1.0
    h10 = a3 - 2.0 * a2 + alpha
    h01 = -2.0 * a3 + 3.0 * a2
    h11 = a3 - a2
    fitted = h00 * q0 + h10 * phase_step * m0 + h01 * q1 + h11 * phase_step * m1

    dh00 = (6.0 * a2 - 6.0 * alpha) / phase_step
    dh10 = 3.0 * a2 - 4.0 * alpha + 1.0
    dh01 = (-6.0 * a2 + 6.0 * alpha) / phase_step
    dh11 = 3.0 * a2 - 2.0 * alpha
    phase_derivative = dh00 * q0 + dh10 * m0 + dh01 * q1 + dh11 * m1
    return fitted, phase_derivative


def continuous_backbone_joint_angles(
    config: SnakeGeometryConfig,
    phase_rad: float,
    *,
    straight: bool = False,
) -> np.ndarray:
    return continuous_backbone_joint_state(
        config, phase_rad, straight=straight
    )[0]


def build_pose(
    config: SnakeGeometryConfig,
    phase_rad: float = 0.0,
    *,
    straight: bool = False,
    align_on_ground: bool = True,
) -> SnakePose:
    angles = continuous_backbone_joint_angles(config, phase_rad, straight=straight)
    types = joint_types(config)
    count = config.module_count
    pitch = config.module_pitch_m

    centers = np.zeros((count, 3), dtype=np.float64)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], count, axis=0)
    positions = np.zeros((config.joint_count, 3), dtype=np.float64)
    axes = np.zeros((config.joint_count, 3), dtype=np.float64)

    for joint_id in range(config.joint_count):
        parent_rotation = rotations[joint_id]
        positions[joint_id] = centers[joint_id] + 0.5 * pitch * parent_rotation[:, 0]
        local_axis = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if types[joint_id] == JOINT_YAW
            else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        axes[joint_id] = parent_rotation @ local_axis
        child_rotation = parent_rotation @ _rotation_about_axis(local_axis, angles[joint_id])
        rotations[joint_id + 1] = child_rotation
        centers[joint_id + 1] = positions[joint_id] + 0.5 * pitch * child_rotation[:, 0]

    boundaries = np.empty((count + 1, 3), dtype=np.float64)
    boundaries[0] = centers[0] - 0.5 * pitch * rotations[0, :, 0]
    boundaries[1:-1] = positions
    boundaries[-1] = centers[-1] + 0.5 * pitch * rotations[-1, :, 0]
    pose = SnakePose(
        body_centers=centers,
        body_rotations=rotations,
        boundaries=boundaries,
        joint_positions=positions,
        joint_axes=axes,
        joint_types=types,
        joint_angles_rad=angles,
        phase_rad=float(phase_rad),
    )
    if align_on_ground:
        # A joint-curvature wave fixes only relative body rotations.  Remove the
        # arbitrary accumulated root pitch by leveling the end-to-end chord before
        # placing the contact envelope on z=0.  This is preview normalization only;
        # an MBD solve will determine the global pose dynamically.
        chord = pose.boundaries[-1] - pose.boundaries[0]
        horizontal_chord = chord.copy()
        horizontal_chord[2] = 0.0
        if float(np.linalg.norm(horizontal_chord)) > 1.0e-12:
            pose = pose.rotated(_rotation_between(chord, horizontal_chord))
        ring_centers, _, ring_normals, ring_binormals, _ = _sample_centerline(config, pose)
        sampled = (
            ring_centers[:, None, :]
            + config.radius_m * ring_normals[:, None, :]
            * np.cos(
                2.0
                * math.pi
                * np.arange(config.circumferential_divisions, dtype=np.float64)[None, :, None]
                / config.circumferential_divisions
            )
            + config.radius_m * ring_binormals[:, None, :]
            * np.sin(
                2.0
                * math.pi
                * np.arange(config.circumferential_divisions, dtype=np.float64)[None, :, None]
                / config.circumferential_divisions
            )
        )
        flat = sampled.reshape((-1, 3))
        horizontal_mid = 0.5 * (np.min(flat[:, :2], axis=0) + np.max(flat[:, :2], axis=0))
        shift = np.array(
            [-horizontal_mid[0], -horizontal_mid[1], -float(np.min(flat[:, 2]))],
            dtype=np.float64,
        )
        pose = pose.translated(shift)
    return pose


def _sample_centerline(
    config: SnakeGeometryConfig,
    pose: SnakePose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subdivisions = config.axial_patch_divisions
    ring_centers = [pose.boundaries[0]]
    segment_owner: list[int] = []
    for module_id in range(config.module_count):
        start = pose.boundaries[module_id]
        end = pose.boundaries[module_id + 1]
        for local_index in range(1, subdivisions + 1):
            alpha = local_index / subdivisions
            ring_centers.append((1.0 - alpha) * start + alpha * end)
            segment_owner.append(module_id)
    centers = np.asarray(ring_centers, dtype=np.float64)
    segment_owner_array = np.asarray(segment_owner, dtype=np.int32)

    segment_tangents = np.asarray(
        [_unit(centers[i + 1] - centers[i]) for i in range(len(centers) - 1)],
        dtype=np.float64,
    )
    tangents = np.empty_like(centers)
    tangents[0] = segment_tangents[0]
    tangents[-1] = segment_tangents[-1]
    for index in range(1, len(centers) - 1):
        tangents[index] = _unit(
            segment_tangents[index - 1] + segment_tangents[index],
            fallback=segment_tangents[index],
        )

    normals = np.empty_like(centers)
    binormals = np.empty_like(centers)
    seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(seed, tangents[0]))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normals[0] = _unit(seed - np.dot(seed, tangents[0]) * tangents[0])
    binormals[0] = _unit(np.cross(tangents[0], normals[0]))
    for index in range(1, len(centers)):
        transported = _rotation_between(tangents[index - 1], tangents[index]) @ normals[index - 1]
        normals[index] = _unit(
            transported - np.dot(transported, tangents[index]) * tangents[index],
            fallback=normals[index - 1],
        )
        binormals[index] = _unit(np.cross(tangents[index], normals[index]))
    return centers, tangents, normals, binormals, segment_owner_array


def _polygon_normal(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    normal = np.zeros(3, dtype=np.float64)
    for index in range(len(vertices)):
        current = vertices[index]
        nxt = vertices[(index + 1) % len(vertices)]
        normal += np.cross(current, nxt)
    return _unit(normal)


def build_module_surfaces(config: SnakeGeometryConfig, pose: SnakePose) -> PolyData:
    points: list[np.ndarray] = []
    polygons: list[list[int]] = []
    ids: list[int] = []
    types: list[int] = []
    masses: list[float] = []
    part_ids: list[int] = []
    module_types = module_type_array(config.module_count)
    count_theta = config.circumferential_divisions
    visual_length = config.visual_length_ratio * config.module_pitch_m

    for module_id in range(config.module_count):
        center = pose.body_centers[module_id]
        rotation = pose.body_rotations[module_id]
        axis = rotation[:, 0]
        radial_a = rotation[:, 1]
        radial_b = rotation[:, 2]
        module_type = int(module_types[module_id])
        front_radius = config.visual_radius_m
        rear_radius = config.visual_radius_m
        if module_type == HEAD:
            front_radius *= 0.72
        elif module_type == TAIL:
            rear_radius *= 0.58

        rings: list[list[int]] = []
        for sign, radius in ((-1.0, front_radius), (1.0, rear_radius)):
            ring: list[int] = []
            ring_center = center + sign * 0.5 * visual_length * axis
            for index in range(count_theta):
                theta = 2.0 * math.pi * index / count_theta
                ring.append(len(points))
                points.append(
                    ring_center
                    + radius * math.cos(theta) * radial_a
                    + radius * math.sin(theta) * radial_b
                )
            rings.append(ring)

        for index in range(count_theta):
            nxt = (index + 1) % count_theta
            polygons.append([rings[0][index], rings[0][nxt], rings[1][nxt], rings[1][index]])
            ids.append(module_id)
            types.append(module_type)
            masses.append(config.module_mass_kg)
            part_ids.append(0)
        polygons.append(list(reversed(rings[0])))
        ids.append(module_id)
        types.append(module_type)
        masses.append(config.module_mass_kg)
        part_ids.append(1)
        polygons.append(rings[1])
        ids.append(module_id)
        types.append(module_type)
        masses.append(config.module_mass_kg)
        part_ids.append(1)

    cell_count = len(polygons)
    return PolyData(
        points=np.asarray(points, dtype=np.float64),
        polygons=polygons,
        lines=[],
        cell_scalars={
            "module_id": np.asarray(ids, dtype=np.int32),
            "module_type": np.asarray(types, dtype=np.int32),
            "part_id": np.asarray(part_ids, dtype=np.int32),
            "body_mass_kg": np.asarray(masses, dtype=np.float64),
            "contact_active": np.zeros(cell_count, dtype=np.int32),
        },
        cell_vectors={},
    )


def build_contact_envelope(config: SnakeGeometryConfig, pose: SnakePose) -> PolyData:
    centers, tangents, normals, binormals, segment_owner = _sample_centerline(config, pose)
    count_theta = config.circumferential_divisions
    theta = 2.0 * math.pi * np.arange(count_theta, dtype=np.float64) / count_theta
    rings = (
        centers[:, None, :]
        + config.radius_m * np.cos(theta)[None, :, None] * normals[:, None, :]
        + config.radius_m * np.sin(theta)[None, :, None] * binormals[:, None, :]
    )
    points = list(rings.reshape((-1, 3)))
    polygons: list[list[int]] = []
    module_ids: list[int] = []
    axial_segment_ids: list[int] = []
    surface_ids: list[int] = []
    outward_normals: list[np.ndarray] = []

    for segment_id, module_id in enumerate(segment_owner):
        for index in range(count_theta):
            nxt = (index + 1) % count_theta
            face = [
                segment_id * count_theta + index,
                segment_id * count_theta + nxt,
                (segment_id + 1) * count_theta + nxt,
                (segment_id + 1) * count_theta + index,
            ]
            polygons.append(face)
            module_ids.append(int(module_id))
            axial_segment_ids.append(segment_id)
            surface_ids.append(SURFACE_MANTLE)
            outward_normals.append(_polygon_normal(np.asarray([points[i] for i in face])))

    head_center_id = len(points)
    points.append(centers[0])
    tail_center_id = len(points)
    points.append(centers[-1])
    tail_ring_start = (len(centers) - 1) * count_theta
    for index in range(count_theta):
        nxt = (index + 1) % count_theta
        head_face = [head_center_id, nxt, index]
        polygons.append(head_face)
        module_ids.append(0)
        axial_segment_ids.append(-1)
        surface_ids.append(SURFACE_HEAD_CAP)
        outward_normals.append(-tangents[0])
        tail_face = [tail_center_id, tail_ring_start + index, tail_ring_start + nxt]
        polygons.append(tail_face)
        module_ids.append(config.module_count - 1)
        axial_segment_ids.append(-1)
        surface_ids.append(SURFACE_TAIL_CAP)
        outward_normals.append(tangents[-1])

    point_array = np.asarray(points, dtype=np.float64)
    centers_of_cells = np.asarray(
        [np.mean(point_array[np.asarray(face, dtype=np.int64)], axis=0) for face in polygons]
    )
    near_ground = (centers_of_cells[:, 2] <= config.ground_tolerance_m).astype(np.int32)
    module_id_array = np.asarray(module_ids, dtype=np.int32)
    module_types = module_type_array(config.module_count)[module_id_array]
    return PolyData(
        points=point_array,
        polygons=polygons,
        lines=[],
        cell_scalars={
            "module_id": module_id_array,
            "module_type": module_types,
            "axial_segment_id": np.asarray(axial_segment_ids, dtype=np.int32),
            "surface_id": np.asarray(surface_ids, dtype=np.int32),
            "contact_candidate": np.ones(len(polygons), dtype=np.int32),
            "initial_near_ground": near_ground,
        },
        cell_vectors={"outward_normal": np.asarray(outward_normals, dtype=np.float64)},
    )


def _append_rectangle(
    points: list[np.ndarray],
    polygons: list[list[int]],
    center: np.ndarray,
    axis_long: np.ndarray,
    axis_width: np.ndarray,
    half_length: float,
    half_width: float,
) -> None:
    axis_long = _unit(axis_long)
    axis_width = _unit(axis_width)
    long_vector = half_length * axis_long
    width_vector = half_width * axis_width
    start = len(points)
    points.extend(
        [
            center - long_vector - width_vector,
            center + long_vector - width_vector,
            center + long_vector + width_vector,
            center - long_vector + width_vector,
        ]
    )
    polygons.append([start, start + 1, start + 2, start + 3])


def build_contact_patches(config: SnakeGeometryConfig, pose: SnakePose) -> PolyData:
    centers, tangents, normals, binormals, segment_owner = _sample_centerline(config, pose)
    count_theta = config.circumferential_divisions
    theta_step = 2.0 * math.pi / count_theta
    module_types = module_type_array(config.module_count)

    points: list[np.ndarray] = []
    polygons: list[list[int]] = []
    module_ids: list[int] = []
    axial_segment_ids: list[int] = []
    surface_ids: list[int] = []
    patch_centers: list[np.ndarray] = []
    patch_normals: list[np.ndarray] = []
    axes_long: list[np.ndarray] = []
    axes_width: list[np.ndarray] = []
    half_lengths: list[float] = []
    half_widths: list[float] = []

    for segment_id, module_id in enumerate(segment_owner):
        center_mid = 0.5 * (centers[segment_id] + centers[segment_id + 1])
        tangent = _unit(centers[segment_id + 1] - centers[segment_id])
        normal_mid = _unit(normals[segment_id] + normals[segment_id + 1], fallback=normals[segment_id])
        normal_mid = _unit(normal_mid - np.dot(normal_mid, tangent) * tangent, fallback=normals[segment_id])
        binormal_mid = _unit(np.cross(tangent, normal_mid))
        half_length = 0.5 * float(np.linalg.norm(centers[segment_id + 1] - centers[segment_id]))
        half_width = 0.5 * config.radius_m * theta_step
        for index in range(count_theta):
            theta_mid = (index + 0.5) * theta_step
            radial = _unit(math.cos(theta_mid) * normal_mid + math.sin(theta_mid) * binormal_mid)
            circumferential = _unit(
                -math.sin(theta_mid) * normal_mid + math.cos(theta_mid) * binormal_mid
            )
            axis_width = -circumferential
            normal = _unit(np.cross(tangent, axis_width))
            if float(np.dot(normal, radial)) < 0.0:
                axis_width = -axis_width
                normal = -normal
            patch_center = center_mid + config.radius_m * radial
            _append_rectangle(
                points,
                polygons,
                patch_center,
                tangent,
                axis_width,
                half_length,
                half_width,
            )
            module_ids.append(int(module_id))
            axial_segment_ids.append(segment_id)
            surface_ids.append(SURFACE_MANTLE)
            patch_centers.append(patch_center)
            patch_normals.append(normal)
            axes_long.append(tangent)
            axes_width.append(axis_width)
            half_lengths.append(half_length)
            half_widths.append(half_width)

    radial_step = config.radius_m / config.cap_radial_divisions
    for end_index, (end_sign, module_id, surface_id) in enumerate(
        (
            (-1.0, 0, SURFACE_HEAD_CAP),
            (1.0, config.module_count - 1, SURFACE_TAIL_CAP),
        )
    ):
        centerline_index = 0 if end_index == 0 else -1
        cap_center = centers[centerline_index]
        cap_normal = end_sign * tangents[centerline_index]
        radial_a = normals[centerline_index]
        radial_b = binormals[centerline_index]
        for ring_index in range(config.cap_radial_divisions):
            radius_mid = (ring_index + 0.5) * radial_step
            ring_count = count_theta if ring_index else max(8, count_theta // 2)
            ring_theta_step = 2.0 * math.pi / ring_count
            for index in range(ring_count):
                theta_mid = (index + 0.5) * ring_theta_step
                radial = _unit(math.cos(theta_mid) * radial_a + math.sin(theta_mid) * radial_b)
                circumferential = _unit(
                    -math.sin(theta_mid) * radial_a + math.cos(theta_mid) * radial_b
                )
                axis_long = radial
                axis_width = circumferential
                if float(np.dot(np.cross(axis_long, axis_width), cap_normal)) < 0.0:
                    axis_width = -axis_width
                patch_center = cap_center + radius_mid * radial
                half_length = 0.5 * radial_step
                half_width = 0.5 * radius_mid * ring_theta_step
                _append_rectangle(
                    points,
                    polygons,
                    patch_center,
                    axis_long,
                    axis_width,
                    half_length,
                    half_width,
                )
                module_ids.append(module_id)
                axial_segment_ids.append(-1)
                surface_ids.append(surface_id)
                patch_centers.append(patch_center)
                patch_normals.append(cap_normal)
                axes_long.append(axis_long)
                axes_width.append(axis_width)
                half_lengths.append(half_length)
                half_widths.append(half_width)

    module_id_array = np.asarray(module_ids, dtype=np.int32)
    patch_center_array = np.asarray(patch_centers, dtype=np.float64)
    half_length_array = np.asarray(half_lengths, dtype=np.float64)
    half_width_array = np.asarray(half_widths, dtype=np.float64)
    patch_area = 4.0 * half_length_array * half_width_array
    near_ground = (patch_center_array[:, 2] <= config.ground_tolerance_m).astype(np.int32)
    per_module_count = np.bincount(module_id_array, minlength=config.module_count)
    patch_mass = np.asarray(
        [config.module_mass_kg / per_module_count[module_id] for module_id in module_id_array],
        dtype=np.float64,
    )
    return PolyData(
        points=np.asarray(points, dtype=np.float64),
        polygons=polygons,
        lines=[],
        cell_scalars={
            "patch_id": np.arange(len(polygons), dtype=np.int32),
            "module_id": module_id_array,
            "module_type": module_types[module_id_array],
            "axial_segment_id": np.asarray(axial_segment_ids, dtype=np.int32),
            "surface_id": np.asarray(surface_ids, dtype=np.int32),
            "contact_candidate": np.ones(len(polygons), dtype=np.int32),
            "initial_near_ground": near_ground,
            "half_length_m": half_length_array,
            "half_width_m": half_width_array,
            "patch_area_m2": patch_area,
            "patch_mass_kg": patch_mass,
        },
        cell_vectors={
            "normal": np.asarray(patch_normals, dtype=np.float64),
            "axis_long": np.asarray(axes_long, dtype=np.float64),
            "axis_width": np.asarray(axes_width, dtype=np.float64),
            "patch_center": patch_center_array,
        },
    )


def build_topology_lines(config: SnakeGeometryConfig, pose: SnakePose) -> PolyData:
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    entity_types: list[int] = []
    module_ids: list[int] = []
    joint_ids: list[int] = []
    joint_type_values: list[int] = []
    angle_values: list[float] = []
    direction_vectors: list[np.ndarray] = []

    for module_id in range(config.module_count):
        start = len(points)
        points.extend([pose.boundaries[module_id], pose.boundaries[module_id + 1]])
        lines.append([start, start + 1])
        entity_types.append(0)
        module_ids.append(module_id)
        joint_ids.append(-1)
        joint_type_values.append(JOINT_NONE)
        angle_values.append(0.0)
        direction_vectors.append(pose.body_rotations[module_id, :, 0])

    marker_half_length = 1.35 * config.radius_m
    for joint_id in range(config.joint_count):
        start = len(points)
        center = pose.joint_positions[joint_id]
        axis = pose.joint_axes[joint_id]
        points.extend([center - marker_half_length * axis, center + marker_half_length * axis])
        lines.append([start, start + 1])
        entity_types.append(1)
        module_ids.append(-1)
        joint_ids.append(joint_id)
        joint_type_values.append(int(pose.joint_types[joint_id]))
        angle_values.append(float(math.degrees(pose.joint_angles_rad[joint_id])))
        direction_vectors.append(axis)

    return PolyData(
        points=np.asarray(points, dtype=np.float64),
        polygons=[],
        lines=lines,
        cell_scalars={
            "entity_type": np.asarray(entity_types, dtype=np.int32),
            "module_id": np.asarray(module_ids, dtype=np.int32),
            "joint_id": np.asarray(joint_ids, dtype=np.int32),
            "joint_type": np.asarray(joint_type_values, dtype=np.int32),
            "joint_angle_deg": np.asarray(angle_values, dtype=np.float64),
        },
        cell_vectors={"axis_or_tangent": np.asarray(direction_vectors, dtype=np.float64)},
    )


def build_ground_plane(config: SnakeGeometryConfig, pose: SnakePose, margin_m: float = 0.10) -> PolyData:
    envelope = build_contact_envelope(config, pose)
    lo = np.min(envelope.points[:, :2], axis=0) - margin_m
    hi = np.max(envelope.points[:, :2], axis=0) + margin_m
    points = np.array(
        [[lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0], [hi[0], hi[1], 0.0], [lo[0], hi[1], 0.0]],
        dtype=np.float64,
    )
    return PolyData(
        points=points,
        polygons=[[0, 1, 2, 3]],
        lines=[],
        cell_scalars={"reference_surface": np.array([1], dtype=np.int32)},
        cell_vectors={"normal": np.array([[0.0, 0.0, 1.0]], dtype=np.float64)},
    )


def mesh_bounds(meshes: Sequence[PolyData]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([mesh.points for mesh in meshes if len(mesh.points)], axis=0)
    return np.min(points, axis=0), np.max(points, axis=0)


def contact_patch_summary(config: SnakeGeometryConfig, patches: PolyData) -> Mapping[str, float | int]:
    areas = np.asarray(patches.cell_scalars["patch_area_m2"], dtype=np.float64)
    half_lengths = np.asarray(patches.cell_scalars["half_length_m"], dtype=np.float64)
    half_widths = np.asarray(patches.cell_scalars["half_width_m"], dtype=np.float64)
    near_ground = np.asarray(patches.cell_scalars["initial_near_ground"], dtype=np.int32)
    return {
        "patch_count": int(patches.cell_count),
        "near_ground_patch_count": int(np.sum(near_ground)),
        "minimum_full_patch_dimension_m": float(
            min(np.min(2.0 * half_lengths), np.min(2.0 * half_widths))
        ),
        "maximum_full_patch_dimension_m": float(
            max(np.max(2.0 * half_lengths), np.max(2.0 * half_widths))
        ),
        "total_patch_area_m2": float(np.sum(areas)),
        "mass_sum_kg": float(np.sum(patches.cell_scalars["patch_mass_kg"])),
        "module_pitch_m": config.module_pitch_m,
    }


def watertight_edge_counts(mesh: PolyData) -> tuple[int, int]:
    edge_counts: dict[tuple[int, int], int] = {}
    for face in mesh.polygons:
        for index in range(len(face)):
            a = int(face[index])
            b = int(face[(index + 1) % len(face)])
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = sum(count == 1 for count in edge_counts.values())
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    return boundary_edges, nonmanifold_edges
