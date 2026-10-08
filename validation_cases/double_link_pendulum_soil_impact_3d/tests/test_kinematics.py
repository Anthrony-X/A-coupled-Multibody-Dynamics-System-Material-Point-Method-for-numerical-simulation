from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_cases.double_link_pendulum_soil_impact_3d.config import load_case_config
from validation_cases.double_link_pendulum_soil_impact_3d.mbd_model import (
    ContactFeedback,
    DoubleLinkPendulumMBD,
)


class DoubleLinkKinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_case_config(CASE_DIR / "case_config.json", "smoke")

    def test_initial_revolute_constraints_are_closed(self) -> None:
        model = DoubleLinkPendulumMBD(
            self.config.model,
            self.config.resolution.head_patch_grid,
            self.config.resolution.visual_segments,
        )
        errors = model.constraint_errors()
        self.assertLess(errors["joint_ground_position_error"], 1.0e-12)
        self.assertLess(errors["joint_links_position_error"], 1.0e-12)
        self.assertLess(errors["joint_ground_axis_error"], 1.0e-12)
        self.assertLess(errors["joint_links_axis_error"], 1.0e-12)

    def test_initial_geometry_matches_case_config(self) -> None:
        model = DoubleLinkPendulumMBD(
            self.config.model,
            self.config.resolution.head_patch_grid,
            self.config.resolution.visual_segments,
        )
        config = self.config.model
        rotation1 = model._rotation_from_alpha(math.radians(config.initial_angle1_deg))
        rotation2 = model._rotation_from_alpha(math.radians(config.initial_angle2_deg))
        expected_joint2 = np.asarray(config.pivot) + rotation1 @ np.array(
            [config.link1_length, 0.0, 0.0]
        )
        expected_hammer_center = expected_joint2 + rotation2 @ np.array(
            [config.hammer_center_from_joint, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            model.joint2_position(),
            expected_joint2,
            rtol=0.0,
            atol=7.0e-7,
        )
        np.testing.assert_allclose(
            model.hammer_center_position(),
            expected_hammer_center,
            rtol=0.0,
            atol=7.0e-7,
        )
        coordinates, rates = model.generalized_state()
        np.testing.assert_allclose(
            coordinates,
            np.radians([config.initial_angle1_deg, config.initial_angle2_deg]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            rates,
            np.radians([config.initial_rate1_deg_s, config.initial_rate2_deg_s]),
            atol=1.0e-12,
        )
        expected_mass = config.link2_rod_mass + config.hammer_mass
        expected_com = (
            config.link2_rod_mass * 0.5 * config.link2_rod_length
            + config.hammer_mass * config.hammer_center_from_joint
        ) / expected_mass
        self.assertAlmostEqual(model.link2_properties.mass, expected_mass, places=7)
        self.assertAlmostEqual(model.link2_properties.com_from_joint, expected_com, places=7)

    def test_patch_area_matches_complete_cylindrical_hammer_surface(self) -> None:
        model = DoubleLinkPendulumMBD(
            self.config.model,
            self.config.resolution.head_patch_grid,
            self.config.resolution.visual_segments,
        )
        patches = model.contact_patches()
        patch_area = np.sum(4.0 * patches["half_extent"][:, 0] * patches["half_extent"][:, 1])
        expected_area = (
            2.0
            * np.pi
            * self.config.model.hammer_radius
            * self.config.model.hammer_width
            + 2.0 * np.pi * self.config.model.hammer_radius**2
        )
        self.assertAlmostEqual(float(patch_area), expected_area, places=7)
        surface_id = patches["surface_id"]
        self.assertEqual(int(np.count_nonzero(surface_id == 1)), model.end_face_contact_patch_count)
        self.assertEqual(int(np.count_nonzero(surface_id == 2)), model.end_face_contact_patch_count)
        np.testing.assert_allclose(
            patches["normal"][surface_id == 1],
            np.repeat([[0.0, -1.0, 0.0]], model.end_face_contact_patch_count, axis=0),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            patches["normal"][surface_id == 2],
            np.repeat([[0.0, 1.0, 0.0]], model.end_face_contact_patch_count, axis=0),
            atol=1.0e-7,
        )

    def test_contact_patch_geometry_is_cached_per_body_state(self) -> None:
        model = DoubleLinkPendulumMBD(
            self.config.model,
            self.config.resolution.head_patch_grid,
            self.config.resolution.visual_segments,
        )
        initial = model.contact_patches()
        self.assertIs(initial, model.contact_patches())
        model.step(1.0e-5, ContactFeedback.zeros(model.contact_patch_count))
        self.assertIsNot(initial, model.contact_patches())

    def test_contact_wrench_and_generalized_power_are_equivalent(self) -> None:
        moving_model_config = replace(
            self.config.model,
            initial_rate1_deg_s=17.0,
            initial_rate2_deg_s=-11.0,
        )
        model = DoubleLinkPendulumMBD(
            moving_model_config,
            self.config.resolution.head_patch_grid,
            self.config.resolution.visual_segments,
        )
        rng = np.random.default_rng(20260721)
        feedback = ContactFeedback(
            patch_forces=(
                rng.normal(size=(model.contact_patch_count, 3)) * 50.0
            ).astype(np.float32),
            patch_moments=(
                rng.normal(size=(model.contact_patch_count, 3)) * 2.0
            ).astype(np.float32),
        )
        patch_power = model.patch_contact_power(feedback)
        generalized_power = model.generalized_contact_power(feedback)
        scale = max(abs(patch_power), abs(generalized_power), 1.0)
        # The CUDA/MPM path is intentionally single precision; retain a margin
        # well below the formal 1e-4 generalized-power validation threshold.
        self.assertLess(abs(patch_power - generalized_power) / scale, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
