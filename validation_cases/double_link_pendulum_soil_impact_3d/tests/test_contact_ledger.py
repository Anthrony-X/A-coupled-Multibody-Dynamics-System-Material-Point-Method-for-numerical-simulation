from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import taichi as ti


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tank_mpm.particles import sample_soil_particles
from tank_mpm.mpm_solver import TankTrackMpmSolver


class ContactLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)

    def test_independent_soil_and_rigid_force_ledgers_close(self) -> None:
        points, particle_volume = sample_soil_particles(
            4,
            4,
            2,
            (-0.10, -0.10, 0.00),
            (0.10, 0.10, 0.10),
            jitter_ratio=0.0,
            seed=1,
        )
        solver = TankTrackMpmSolver(
            soil_points=points,
            soil_p_vol=particle_volume,
            n_grid=16,
            dt=1.0e-4,
            domain_lo=(-0.20, -0.20, -0.10),
            domain_hi=(0.20, 0.20, 0.30),
            soil_bounds_lo=(-0.10, -0.10, 0.00),
            soil_bounds_hi=(0.10, 0.10, 0.10),
            max_track_patches=1,
            soil_density=1700.0,
            soil_E=2.0e5,
            soil_nu=0.30,
            soil_gravity_scale=0.0,
            soil_damping=1.0,
            contact_mu=0.0,
            contact_barrier_stiffness=1000.0,
            contact_barrier_radius=0.035,
            contact_barrier_min_distance_ratio=0.20,
            track_activation_height=1.0,
            mpm_precision="f64",
            particle_shape=(4, 4, 2),
        )
        patch = {
            "center": np.array([[0.0, 0.0, 0.095]], dtype=np.float32),
            "previous_center": np.array([[0.0, 0.0, 0.095]], dtype=np.float32),
            "axis_long": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "previous_axis_long": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "axis_width": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            "previous_axis_width": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            "normal": np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
            "velocity": np.zeros((1, 3), dtype=np.float32),
            "half_extent": np.array([[0.10, 0.10]], dtype=np.float32),
            "mass": np.array([1.0], dtype=np.float32),
            "side_id": np.array([0], dtype=np.int32),
            "shoe_id": np.array([2.0], dtype=np.float32),
        }
        solver.set_track_patch_keyframes(patch, patch)
        solver.update_track_patches_from_keyframes(0.0, 1.0, solver.dt)
        solver.reset_contact_force_average()
        solver.substep()
        solver.accumulate_contact_force_average()
        macro = solver.contact_macro_summary()
        rigid_force = np.sum(solver.contact_forces_by_patch(), axis=0)
        soil_force = np.sum(solver.soil_contact_forces_by_patch(), axis=0)
        scale = max(float(np.linalg.norm(rigid_force)), float(np.linalg.norm(soil_force)), 1.0)
        self.assertGreater(int(solver.contact_particle_count[None]), 0)
        self.assertGreater(float(rigid_force[2]), 0.0)
        self.assertLess(float(np.linalg.norm(rigid_force + soil_force)) / scale, 1.0e-12)
        self.assertEqual(int(macro["sample_count"]), 1)
        self.assertGreater(int(macro["max_contact_particles"]), 0)
        np.testing.assert_allclose(
            np.sum(macro["patch_forces"], axis=0),
            rigid_force,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
