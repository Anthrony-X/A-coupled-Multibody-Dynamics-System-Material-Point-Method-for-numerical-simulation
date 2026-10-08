from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tank_mpm.mpm_solver import TankTrackMpmSolver


def _solver(*, binned: bool) -> TankTrackMpmSolver:
    points = np.asarray(
        [[0.0, 0.0, 0.01], [-0.07, -0.07, 0.06]], dtype=np.float32
    )
    return TankTrackMpmSolver(
        soil_points=points,
        soil_p_vol=1.0e-6,
        n_grid=16,
        dt=1.0e-4,
        domain_lo=(-0.1, -0.1, -0.1),
        domain_hi=(0.1, 0.1, 0.1),
        soil_bounds_lo=(-0.08, -0.08, -0.08),
        soil_bounds_hi=(0.08, 0.08, 0.08),
        max_track_patches=8,
        contact_barrier_radius=0.02,
        contact_barrier_stiffness=100.0,
        track_activation_height=1.0,
        particle_shape=(2, 1, 1),
        contact_binning_enabled=binned,
        contact_bin_size=0.05,
        contact_bin_capacity=8,
    )


def _patches() -> dict[str, np.ndarray]:
    center = np.asarray([[0.0, 0.0, 0.0], [0.06, 0.06, 0.0]], dtype=np.float32)
    return {
        "center": center,
        "axis_long": np.tile([1.0, 0.0, 0.0], (2, 1)).astype(np.float32),
        "axis_width": np.tile([0.0, 1.0, 0.0], (2, 1)).astype(np.float32),
        "normal": np.tile([0.0, 0.0, 1.0], (2, 1)).astype(np.float32),
        "velocity": np.zeros((2, 3), dtype=np.float32),
        "half_extent": np.tile([0.025, 0.025], (2, 1)).astype(np.float32),
        "mass": np.ones(2, dtype=np.float32),
        "side_id": np.zeros(2, dtype=np.int32),
        "shoe_id": np.arange(2, dtype=np.float32),
        "module_id": np.arange(2, dtype=np.int32),
        "axial_segment_id": np.arange(2, dtype=np.int32),
    }


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)
    patches = _patches()
    reference = _solver(binned=False)
    optimized = _solver(binned=True)
    for solver in (reference, optimized):
        solver.set_track_patches(patches)
        solver.clear_grid()
        solver.apply_track_barrier_contact()
    ti.sync()

    np.testing.assert_allclose(
        optimized.track_contact_force.to_numpy(),
        reference.track_contact_force.to_numpy(),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        optimized.track_contact_force_by_patch.to_numpy(),
        reference.track_contact_force_by_patch.to_numpy(),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert int(optimized.contact_particle_count[None]) == int(
        reference.contact_particle_count[None]
    )
    assert int(optimized.contact_bin_overflow[None]) == 0

    force_sum = np.zeros((optimized.max_track_patches, 3), dtype=np.float32)
    moment_sum = np.zeros_like(force_sum)
    force_sum[:2] = [[2.0, 0.0, -4.0], [0.0, 2.0, -6.0]]
    moment_sum[:2] = [[0.2, 0.0, 0.0], [0.0, 0.4, 0.0]]
    optimized.contact_force_sum_by_patch.from_numpy(force_sum)
    optimized.contact_moment_sum_by_patch.from_numpy(moment_sum)
    optimized.contact_force_sample_count[None] = 2
    reduced = optimized.contact_macro_summary_reduced()
    expected_force = 0.5 * force_sum[:2]
    expected_origin_moment = 0.5 * moment_sum[:2] + np.cross(
        patches["center"], expected_force
    )
    np.testing.assert_allclose(reduced["module_forces"], expected_force, atol=1.0e-6)
    np.testing.assert_allclose(
        reduced["module_moments_about_origin"], expected_origin_moment, atol=1.0e-6
    )
    np.testing.assert_allclose(
        reduced["axial_segment_normal_loads"], [2.0, 3.0], atol=1.0e-6
    )
    print(
        "CONTACT_OPTIMIZATIONS_VERIFIED "
        f"force={np.asarray(reduced['module_forces']).tolist()} "
        f"max_bin_occupancy={int(optimized.contact_bin_max_occupancy[None])}"
    )


if __name__ == "__main__":
    main()
