from __future__ import annotations

import math
import unittest

import numpy as np

from validation_cases.cmu_snake_geometry_3d.geometry import (
    HEAD,
    JOINT_PITCH,
    JOINT_YAW,
    MIDDLE,
    TAIL,
    SnakeGeometryConfig,
    build_contact_envelope,
    build_contact_patches,
    build_module_surfaces,
    build_pose,
    build_topology_lines,
    module_type_array,
    watertight_edge_counts,
)


class SnakeGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SnakeGeometryConfig()

    def test_exact_17_body_topology(self) -> None:
        pose = build_pose(self.config, phase_rad=0.37)
        self.assertEqual(pose.body_centers.shape, (17, 3))
        self.assertEqual(pose.joint_positions.shape, (16, 3))
        np.testing.assert_array_equal(
            pose.joint_types,
            np.array([JOINT_YAW, JOINT_PITCH] * 8, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            module_type_array(17),
            np.array([HEAD] + [MIDDLE] * 15 + [TAIL], dtype=np.int32),
        )

    def test_arc_length_and_joint_closure(self) -> None:
        pose = build_pose(self.config, phase_rad=1.1, align_on_ground=False)
        segment_lengths = np.linalg.norm(np.diff(pose.boundaries, axis=0), axis=1)
        np.testing.assert_allclose(segment_lengths, self.config.module_pitch_m, atol=1.0e-12)
        self.assertAlmostEqual(float(np.sum(segment_lengths)), self.config.total_length_m, places=12)
        for joint_id in range(self.config.joint_count):
            parent_end = (
                pose.body_centers[joint_id]
                + 0.5 * self.config.module_pitch_m * pose.body_rotations[joint_id, :, 0]
            )
            child_start = (
                pose.body_centers[joint_id + 1]
                - 0.5 * self.config.module_pitch_m * pose.body_rotations[joint_id + 1, :, 0]
            )
            np.testing.assert_allclose(parent_end, child_start, atol=1.0e-12)
            np.testing.assert_allclose(parent_end, pose.joint_positions[joint_id], atol=1.0e-12)

    def test_mass_partition_sums_to_published_total(self) -> None:
        pose = build_pose(self.config, straight=True)
        patches = build_contact_patches(self.config, pose)
        self.assertAlmostEqual(
            float(np.sum(patches.cell_scalars["patch_mass_kg"])),
            self.config.total_mass_kg,
            places=12,
        )

    def test_contact_envelope_is_watertight_manifold(self) -> None:
        pose = build_pose(self.config, phase_rad=0.0)
        envelope = build_contact_envelope(self.config, pose)
        boundary_edges, nonmanifold_edges = watertight_edge_counts(envelope)
        self.assertEqual(boundary_edges, 0)
        self.assertEqual(nonmanifold_edges, 0)

    def test_contact_patch_frames_are_orthonormal_and_outward(self) -> None:
        pose = build_pose(self.config, phase_rad=0.2)
        patches = build_contact_patches(self.config, pose)
        normals = patches.cell_vectors["normal"]
        axes_long = patches.cell_vectors["axis_long"]
        axes_width = patches.cell_vectors["axis_width"]
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(np.linalg.norm(axes_long, axis=1), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(np.linalg.norm(axes_width, axis=1), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(np.sum(normals * axes_long, axis=1), 0.0, atol=1.0e-12)
        np.testing.assert_allclose(np.sum(normals * axes_width, axis=1), 0.0, atol=1.0e-12)
        np.testing.assert_allclose(np.cross(axes_long, axes_width), normals, atol=1.0e-12)
        self.assertTrue(np.all(np.asarray(patches.cell_scalars["patch_area_m2"]) > 0.0))

    def test_ground_alignment_and_output_mesh_sizes(self) -> None:
        pose = build_pose(self.config, phase_rad=0.0)
        envelope = build_contact_envelope(self.config, pose)
        self.assertAlmostEqual(float(np.min(envelope.points[:, 2])), 0.0, places=12)
        self.assertAlmostEqual(
            float(pose.boundaries[-1, 2] - pose.boundaries[0, 2]),
            0.0,
            places=12,
        )
        modules = build_module_surfaces(self.config, pose)
        topology = build_topology_lines(self.config, pose)
        self.assertEqual(len(np.unique(modules.cell_scalars["module_id"])), 17)
        self.assertEqual(topology.cell_count, 17 + 16)
        expected_side_patches = (
            self.config.module_count
            * self.config.axial_patch_divisions
            * self.config.circumferential_divisions
        )
        expected_cap_patches_per_end = max(8, self.config.circumferential_divisions // 2) + (
            self.config.cap_radial_divisions - 1
        ) * self.config.circumferential_divisions
        patches = build_contact_patches(self.config, pose)
        self.assertEqual(
            patches.cell_count,
            expected_side_patches + 2 * expected_cap_patches_per_end,
        )

    def test_fitted_joint_angles_stay_inside_physical_limits(self) -> None:
        pose = build_pose(self.config, phase_rad=math.pi / 3.0)
        yaw = np.abs(np.degrees(pose.joint_angles_rad[pose.joint_types == JOINT_YAW]))
        pitch = np.abs(np.degrees(pose.joint_angles_rad[pose.joint_types == JOINT_PITCH]))
        self.assertLess(float(np.max(yaw)), 90.0)
        self.assertLess(float(np.max(pitch)), 90.0)


if __name__ == "__main__":
    unittest.main()
