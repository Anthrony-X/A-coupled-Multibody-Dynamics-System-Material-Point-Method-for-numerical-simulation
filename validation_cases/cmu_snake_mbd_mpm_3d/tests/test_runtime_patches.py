from __future__ import annotations

import unittest

import numpy as np

from validation_cases.cmu_snake_geometry_3d.geometry import (
    SnakeGeometryConfig,
    build_contact_patches,
    build_pose,
)
from validation_cases.cmu_snake_mbd_mpm_3d.runtime_patches import (
    VectorizedSnakePatchKinematics,
)


class RuntimePatchKinematicsTests(unittest.TestCase):
    def test_array_path_matches_vtk_geometry_path(self) -> None:
        config = SnakeGeometryConfig()
        runtime_builder = VectorizedSnakePatchKinematics(config)
        for pose in (
            build_pose(config, straight=True, align_on_ground=False),
            build_pose(config, phase_rad=0.37, align_on_ground=False),
        ):
            expected = build_contact_patches(config, pose)
            actual = runtime_builder.evaluate(pose)
            np.testing.assert_allclose(
                actual["center"], expected.cell_vectors["patch_center"], atol=1.0e-12
            )
            for key in ("axis_long", "axis_width", "normal"):
                np.testing.assert_allclose(
                    actual[key], expected.cell_vectors[key], atol=1.0e-12
                )
            np.testing.assert_allclose(
                actual["half_extent"],
                np.column_stack(
                    (
                        expected.cell_scalars["half_length_m"],
                        expected.cell_scalars["half_width_m"],
                    )
                ),
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                actual["mass"], expected.cell_scalars["patch_mass_kg"], atol=1.0e-12
            )
            np.testing.assert_array_equal(
                actual["module_id"], expected.cell_scalars["module_id"]
            )
            np.testing.assert_array_equal(
                actual["axial_segment_id"],
                expected.cell_scalars["axial_segment_id"],
            )


if __name__ == "__main__":
    unittest.main()
