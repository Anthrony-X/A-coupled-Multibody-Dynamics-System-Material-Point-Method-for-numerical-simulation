from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from validation_cases.cmu_snake_mbd_mpm_3d.config import load_case_config
from validation_cases.cmu_snake_mbd_mpm_3d.mbd_model import gait_envelope, gait_targets


CONFIG_PATH = Path(__file__).resolve().parents[1] / "case_config.json"


class ConfigAndControllerTests(unittest.TestCase):
    def test_all_presets_are_valid_and_patch_counts_are_stable(self) -> None:
        expected = {"smoke": 152, "standard": 592, "fine": 864}
        for preset, count in expected.items():
            config = load_case_config(CONFIG_PATH, preset)
            self.assertEqual(config.model.module_count, 17)
            self.assertEqual(config.model.joint_count, 16)
            self.assertEqual(config.contact_patch_count, count)
            self.assertEqual(config.resolution.mpm_substeps, 5)
            self.assertAlmostEqual(config.model.wavelength_m, 0.47)

    def test_gait_envelope_is_c1_at_both_transitions(self) -> None:
        settle = 0.25
        ramp = 0.50
        self.assertEqual(gait_envelope(settle, settle, ramp), (0.0, 0.0))
        value, derivative = gait_envelope(settle + ramp, settle, ramp)
        self.assertEqual(value, 1.0)
        self.assertEqual(derivative, 0.0)
        middle, middle_rate = gait_envelope(settle + 0.5 * ramp, settle, ramp)
        self.assertAlmostEqual(middle, 0.5)
        self.assertGreater(middle_rate, 0.0)

    def test_targets_are_zero_during_settle_and_inside_joint_limits(self) -> None:
        config = load_case_config(CONFIG_PATH, "smoke")
        angle, rate = gait_targets(config, 0.1)
        np.testing.assert_allclose(angle, 0.0)
        np.testing.assert_allclose(rate, 0.0)
        angle, _ = gait_targets(
            config, config.model.settle_time_s + config.model.ramp_time_s + 0.17
        )
        yaw = np.abs(np.degrees(angle[0::2]))
        pitch = np.abs(np.degrees(angle[1::2]))
        self.assertLess(float(np.max(yaw)), 90.0)
        self.assertLess(float(np.max(pitch)), 90.0)
        self.assertTrue(np.all(np.isfinite(angle)))

    def test_continuous_body_wave_phase_is_quadrature(self) -> None:
        config = load_case_config(CONFIG_PATH, "smoke")
        self.assertAlmostEqual(
            math.radians(config.model.body_wave_phase_deg),
            0.5 * math.pi,
        )


if __name__ == "__main__":
    unittest.main()
