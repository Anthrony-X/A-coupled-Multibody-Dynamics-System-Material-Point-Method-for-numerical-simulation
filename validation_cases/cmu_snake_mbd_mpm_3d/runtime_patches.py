from __future__ import annotations

import math

import numpy as np

from validation_cases.cmu_snake_geometry_3d.geometry import (
    SURFACE_HEAD_CAP,
    SURFACE_MANTLE,
    SURFACE_TAIL_CAP,
    SnakeGeometryConfig,
    SnakePose,
    _sample_centerline,
)


def _normalize_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize an (n, 3) array while retaining the geometry builder fallback."""

    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms[:, 0] > 1.0e-12
    result = np.empty_like(values)
    result[valid] = values[valid] / norms[valid]
    if np.any(~valid):
        if fallback is None:
            raise ValueError("Cannot normalize a near-zero vector")
        fallback = np.asarray(fallback, dtype=np.float64)
        fallback_norm = np.linalg.norm(fallback[~valid], axis=1, keepdims=True)
        result[~valid] = fallback[~valid] / np.maximum(fallback_norm, 1.0e-30)
    return result


class VectorizedSnakePatchKinematics:
    """Array-only contact-patch kinematics with immutable topology.

    The ParaView geometry builder intentionally creates polygon objects and rich
    cell metadata.  That is useful at output times, but unnecessarily expensive
    at every MBD macro step.  This class preserves its patch ordering and local
    frames while generating only the arrays consumed by the MPM solver.
    """

    def __init__(self, config: SnakeGeometryConfig) -> None:
        self.config = config
        self.segment_count = config.module_count * config.axial_patch_divisions
        self.count_theta = config.circumferential_divisions
        self.theta_step = 2.0 * math.pi / self.count_theta

        self._side_theta = (
            np.arange(self.count_theta, dtype=np.float64) + 0.5
        ) * self.theta_step
        self._side_module_id = np.repeat(
            np.repeat(
                np.arange(config.module_count, dtype=np.int32),
                config.axial_patch_divisions,
            ),
            self.count_theta,
        )
        self._side_segment_id = np.repeat(
            np.arange(self.segment_count, dtype=np.int32), self.count_theta
        )

        cap_theta: list[np.ndarray] = []
        cap_radius: list[np.ndarray] = []
        cap_half_width: list[np.ndarray] = []
        radial_step = config.radius_m / config.cap_radial_divisions
        for ring_index in range(config.cap_radial_divisions):
            ring_count = self.count_theta if ring_index else max(8, self.count_theta // 2)
            ring_theta_step = 2.0 * math.pi / ring_count
            theta = (np.arange(ring_count, dtype=np.float64) + 0.5) * ring_theta_step
            radius_mid = (ring_index + 0.5) * radial_step
            cap_theta.append(theta)
            cap_radius.append(np.full(ring_count, radius_mid, dtype=np.float64))
            cap_half_width.append(
                np.full(ring_count, 0.5 * radius_mid * ring_theta_step, dtype=np.float64)
            )
        self._cap_theta = np.concatenate(cap_theta)
        self._cap_radius = np.concatenate(cap_radius)
        self._cap_half_width = np.concatenate(cap_half_width)
        self.cap_patch_count_per_end = int(self._cap_theta.size)

        side_count = self.segment_count * self.count_theta
        cap_count = 2 * self.cap_patch_count_per_end
        self.patch_count = side_count + cap_count
        self.module_id = np.concatenate(
            (
                self._side_module_id,
                np.zeros(self.cap_patch_count_per_end, dtype=np.int32),
                np.full(
                    self.cap_patch_count_per_end,
                    config.module_count - 1,
                    dtype=np.int32,
                ),
            )
        )
        self.axial_segment_id = np.concatenate(
            (
                self._side_segment_id,
                np.full(cap_count, -1, dtype=np.int32),
            )
        )
        self.surface_id = np.concatenate(
            (
                np.full(side_count, SURFACE_MANTLE, dtype=np.int32),
                np.full(self.cap_patch_count_per_end, SURFACE_HEAD_CAP, dtype=np.int32),
                np.full(self.cap_patch_count_per_end, SURFACE_TAIL_CAP, dtype=np.int32),
            )
        )
        per_module_count = np.bincount(self.module_id, minlength=config.module_count)
        self.mass = (
            config.module_mass_kg / per_module_count[self.module_id]
        ).astype(np.float64)
        self.side_id = np.zeros(self.patch_count, dtype=np.int32)
        self.shoe_id = np.arange(self.patch_count, dtype=np.float32)

        side_half_extent = np.tile(
            np.array(
                [
                    0.5 * config.module_pitch_m / config.axial_patch_divisions,
                    0.5 * config.radius_m * self.theta_step,
                ],
                dtype=np.float64,
            ),
            (side_count, 1),
        )
        cap_half_extent = np.column_stack(
            (
                np.full(
                    self.cap_patch_count_per_end,
                    0.5 * radial_step,
                    dtype=np.float64,
                ),
                self._cap_half_width,
            )
        )
        self.half_extent = np.vstack(
            (side_half_extent, cap_half_extent, cap_half_extent)
        )

    def evaluate(self, pose: SnakePose) -> dict[str, np.ndarray]:
        centers, tangents, normals, binormals, _ = _sample_centerline(
            self.config, pose
        )
        segment_delta = centers[1:] - centers[:-1]
        segment_tangent = _normalize_rows(segment_delta)
        center_mid = 0.5 * (centers[1:] + centers[:-1])

        normal_mid = _normalize_rows(normals[:-1] + normals[1:], normals[:-1])
        normal_mid = _normalize_rows(
            normal_mid
            - np.sum(normal_mid * segment_tangent, axis=1, keepdims=True)
            * segment_tangent,
            normals[:-1],
        )
        binormal_mid = _normalize_rows(np.cross(segment_tangent, normal_mid))

        cosine = np.cos(self._side_theta)[None, :, None]
        sine = np.sin(self._side_theta)[None, :, None]
        radial = _normalize_rows(
            (cosine * normal_mid[:, None, :] + sine * binormal_mid[:, None, :]).reshape(
                (-1, 3)
            )
        )
        circumferential = _normalize_rows(
            (
                -sine * normal_mid[:, None, :]
                + cosine * binormal_mid[:, None, :]
            ).reshape((-1, 3))
        )
        side_axis_long = np.repeat(segment_tangent, self.count_theta, axis=0)
        side_axis_width = -circumferential
        side_normal = _normalize_rows(np.cross(side_axis_long, side_axis_width))
        flip = np.sum(side_normal * radial, axis=1) < 0.0
        side_axis_width[flip] *= -1.0
        side_normal[flip] *= -1.0
        side_center = (
            np.repeat(center_mid, self.count_theta, axis=0)
            + self.config.radius_m * radial
        )

        cap_centers: list[np.ndarray] = []
        cap_normals: list[np.ndarray] = []
        cap_axes_long: list[np.ndarray] = []
        cap_axes_width: list[np.ndarray] = []
        cap_cosine = np.cos(self._cap_theta)[:, None]
        cap_sine = np.sin(self._cap_theta)[:, None]
        for centerline_index, end_sign in ((0, -1.0), (-1, 1.0)):
            cap_normal = end_sign * tangents[centerline_index]
            radial = _normalize_rows(
                cap_cosine * normals[centerline_index]
                + cap_sine * binormals[centerline_index]
            )
            circumferential = _normalize_rows(
                -cap_sine * normals[centerline_index]
                + cap_cosine * binormals[centerline_index]
            )
            axis_width = circumferential.copy()
            flip = np.sum(
                np.cross(radial, axis_width) * cap_normal[None, :], axis=1
            ) < 0.0
            axis_width[flip] *= -1.0
            cap_centers.append(
                centers[centerline_index]
                + self._cap_radius[:, None] * radial
            )
            cap_normals.append(
                np.repeat(cap_normal[None, :], self.cap_patch_count_per_end, axis=0)
            )
            cap_axes_long.append(radial)
            cap_axes_width.append(axis_width)

        return {
            "center": np.vstack((side_center, cap_centers[0], cap_centers[1])),
            "axis_long": np.vstack(
                (side_axis_long, cap_axes_long[0], cap_axes_long[1])
            ),
            "axis_width": np.vstack(
                (side_axis_width, cap_axes_width[0], cap_axes_width[1])
            ),
            "normal": np.vstack((side_normal, cap_normals[0], cap_normals[1])),
            "half_extent": self.half_extent,
            "mass": self.mass,
            "side_id": self.side_id,
            "shoe_id": self.shoe_id,
            "surface_id": self.surface_id,
            "module_id": self.module_id,
            "axial_segment_id": self.axial_segment_id,
        }
