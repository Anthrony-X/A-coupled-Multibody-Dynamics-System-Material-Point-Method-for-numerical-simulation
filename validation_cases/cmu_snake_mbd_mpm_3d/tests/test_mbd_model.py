from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from validation_cases.cmu_snake_mbd_mpm_3d.config import load_case_config
from validation_cases.cmu_snake_mbd_mpm_3d.mbd_model import (
    CmuSnakeMBD,
    ContactFeedback,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "case_config.json"


class MbdModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_case_config(CONFIG_PATH, "smoke")

    def test_initial_topology_constraints_and_patch_mass(self) -> None:
        mbd = CmuSnakeMBD(self.config)
        self.assertEqual(len(mbd.bodies), 17)
        self.assertEqual(len(mbd.motors), 16)
        np.testing.assert_allclose(mbd.joint_angles(), 0.0, atol=1.0e-14)
        constraints = mbd.constraint_errors()
        self.assertLess(float(constraints["max_position_error_m"]), 1.0e-12)
        self.assertLess(float(constraints["max_axis_error_rad"]), 1.0e-12)
        patches = mbd.contact_patches()
        self.assertEqual(patches["center"].shape, (self.config.contact_patch_count, 3))
        self.assertAlmostEqual(float(np.sum(patches["mass"])), 3.15, places=6)
        self.assertEqual(len(np.unique(patches["module_id"])), 17)

    def test_patch_feedback_maps_to_owning_body_and_preserves_total_force(self) -> None:
        mbd = CmuSnakeMBD(self.config)
        patches = mbd.contact_patches()
        forces = np.zeros_like(patches["center"], dtype=np.float32)
        moments = np.zeros_like(forces)
        patch_id = int(np.flatnonzero(patches["module_id"] == 7)[0])
        forces[patch_id] = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        body_forces, _ = mbd.patch_wrenches_by_body(ContactFeedback(forces, moments))
        np.testing.assert_allclose(np.sum(body_forces, axis=0), forces[patch_id])
        np.testing.assert_allclose(body_forces[7], forces[patch_id])
        self.assertEqual(np.count_nonzero(np.linalg.norm(body_forces, axis=1)), 1)

    def test_contact_length_ratio_uses_axial_arc_segments(self) -> None:
        mbd = CmuSnakeMBD(self.config)
        patches = mbd.contact_patches()
        forces = np.zeros_like(patches["center"], dtype=np.float32)
        segment_ids = patches["axial_segment_id"]
        normal = patches["normal"]
        mask = segment_ids == 3
        forces[mask] = -0.1 * normal[mask]
        ratio, active_count, loads = mbd.contact_length_ratio(
            ContactFeedback(forces, np.zeros_like(forces)), 0.02
        )
        self.assertEqual(active_count, 1)
        self.assertAlmostEqual(ratio, 1.0 / 17.0)
        self.assertGreater(loads[3], 0.02)

    def test_reduced_feedback_matches_patch_feedback(self) -> None:
        mbd = CmuSnakeMBD(self.config)
        patches = mbd.contact_patches()
        rng = np.random.default_rng(73)
        forces = rng.normal(0.0, 0.2, patches["center"].shape)
        moments = rng.normal(0.0, 0.01, patches["center"].shape)
        patch_feedback = ContactFeedback(forces, moments)
        expected_forces, expected_moments = mbd.patch_wrenches_by_body(
            patch_feedback
        )

        module_ids = np.asarray(patches["module_id"], dtype=np.int32)
        module_forces = np.zeros_like(expected_forces)
        moments_origin = np.zeros_like(expected_moments)
        for module_id in range(self.config.model.module_count):
            mask = module_ids == module_id
            module_forces[module_id] = np.sum(forces[mask], axis=0)
            moments_origin[module_id] = np.sum(
                moments[mask]
                + np.cross(np.asarray(patches["center"])[mask], forces[mask]),
                axis=0,
            )
        segment_count = (
            self.config.model.module_count
            * self.config.resolution.axial_patch_divisions
        )
        segment_loads = np.zeros(segment_count, dtype=np.float64)
        segment_ids = np.asarray(patches["axial_segment_id"], dtype=np.int32)
        patch_loads = np.maximum(
            0.0,
            -np.sum(forces * np.asarray(patches["normal"]), axis=1),
        )
        valid = segment_ids >= 0
        np.add.at(segment_loads, segment_ids[valid], patch_loads[valid])

        reduced_feedback = ContactFeedback(
            module_forces=module_forces,
            module_moments_about_origin=moments_origin,
            axial_segment_normal_loads=segment_loads,
        )
        actual_forces, actual_moments = mbd.patch_wrenches_by_body(
            reduced_feedback
        )
        np.testing.assert_allclose(actual_forces, expected_forces, atol=1.0e-12)
        np.testing.assert_allclose(actual_moments, expected_moments, atol=2.0e-8)
        expected_ratio = mbd.contact_length_ratio(patch_feedback, 0.15)
        actual_ratio = mbd.contact_length_ratio(reduced_feedback, 0.15)
        self.assertEqual(actual_ratio[0], expected_ratio[0])
        self.assertEqual(actual_ratio[1], expected_ratio[1])
        np.testing.assert_allclose(actual_ratio[2], expected_ratio[2])

    def test_torque_controller_tracks_with_correct_sign(self) -> None:
        config = replace(
            self.config,
            model=replace(
                self.config.model,
                gravity=0.0,
                settle_time_s=0.0,
                ramp_time_s=0.01,
                gait_frequency_hz=0.2,
            ),
        )
        mbd = CmuSnakeMBD(config)
        feedback = ContactFeedback.zeros(config.contact_patch_count)
        for _ in range(200):
            mbd.step(config.resolution.mbd_dt, feedback)
        snapshot = mbd.controller_snapshot()
        correlation = float(
            np.corrcoef(snapshot.target_angle_rad, snapshot.actual_angle_rad)[0, 1]
        )
        tracking_rms_deg = float(
            np.sqrt(np.mean(np.degrees(snapshot.tracking_error_rad) ** 2))
        )
        self.assertGreater(correlation, 0.95)
        self.assertLess(tracking_rms_deg, 5.0)
        self.assertLess(mbd.constraint_errors()["max_position_error_m"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
