import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import taichi as ti

from tank_mpm.constitutive_models.srsh_modified_dp import (
    SRSH_REFERENCE_STRAIN_RATE,
    make_srsh_modified_dp_parameters,
    srsh_modified_dp_update,
    srsh_yield_stress_scalar,
)
from tank_mpm.mpm_solver import TankTrackMpmSolver


_test_arch = ti.cuda if os.environ.get("SRSH_TEST_ARCH", "cpu") == "cuda" else ti.cpu
ti.init(arch=_test_arch, default_fp=ti.f64, offline_cache=False)

_stress_in = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
_deps_in = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
_stress_out = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
_state_out = ti.field(dtype=ti.f64, shape=2)


@ti.kernel
def _run_taichi_update(
    epbar: ti.f64,
    dt: ti.f64,
    shear_modulus: ti.f64,
    bulk_modulus: ti.f64,
    friction_slope: ti.f64,
    dilatancy_slope: ti.f64,
    cohesion_intercept: ti.f64,
    rate_exponent: ti.f64,
    rate_sensitivity: ti.f64,
    saturation_increment: ti.f64,
    reference_strain_rate: ti.f64,
    saturation_plastic_strain: ti.f64,
):
    identity = ti.Matrix.identity(ti.f64, 3)
    sigma, epbar_new, plastic_multiplier = srsh_modified_dp_update(
        _stress_in[None],
        _deps_in[None],
        epbar,
        dt,
        shear_modulus,
        bulk_modulus,
        friction_slope,
        dilatancy_slope,
        cohesion_intercept,
        rate_exponent,
        rate_sensitivity,
        saturation_increment,
        reference_strain_rate,
        saturation_plastic_strain,
        identity,
        1.0e-12,
    )
    _stress_out[None] = sigma
    _state_out[0] = epbar_new
    _state_out[1] = plastic_multiplier


def _numpy_reference_update(stress, deps, epbar, dt, params):
    identity = np.eye(3)
    tr_deps = float(np.trace(deps))
    dev_deps = deps - tr_deps * identity / 3.0
    trial = stress + 2.0 * params.shear_modulus * dev_deps
    trial += params.bulk_modulus * tr_deps * identity
    pressure_trial = float(np.trace(trial)) / 3.0
    dev_trial = trial - pressure_trial * identity
    q_trial = math.sqrt(1.5 * float(np.sum(dev_trial * dev_trial)))

    def evaluate(plastic_multiplier):
        q = max(0.0, q_trial - 3.0 * params.shear_modulus * plastic_multiplier)
        pressure = pressure_trial + (
            params.bulk_modulus * params.dilatancy_slope * plastic_multiplier
        )
        yield_stress = srsh_yield_stress_scalar(
            epbar + plastic_multiplier,
            plastic_multiplier / dt,
            params.cohesion_intercept,
            params.rate_exponent,
            params.rate_sensitivity,
            params.saturation_increment,
            params.reference_strain_rate,
            params.saturation_plastic_strain,
        )
        return q - params.friction_slope * pressure - yield_stress, q, pressure, yield_stress

    residual, q_new, pressure_new, yield_new = evaluate(0.0)
    plastic_multiplier = 0.0
    if residual > 0.0:
        lower = 0.0
        upper = max(q_trial / (3.0 * params.shear_modulus), 1.0e-16)
        while evaluate(upper)[0] > 0.0:
            upper *= 2.0
        for _ in range(100):
            plastic_multiplier = 0.5 * (lower + upper)
            residual, q_new, pressure_new, yield_new = evaluate(plastic_multiplier)
            if residual > 0.0:
                lower = plastic_multiplier
            else:
                upper = plastic_multiplier
        plastic_multiplier = 0.5 * (lower + upper)
        residual, q_new, pressure_new, yield_new = evaluate(plastic_multiplier)

    dev_scale = q_new / q_trial if q_trial > 0.0 else 0.0
    stress_new = pressure_new * identity + dev_scale * dev_trial
    return stress_new, epbar + plastic_multiplier, plastic_multiplier


class SrshModifiedDpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = make_srsh_modified_dp_parameters(
            soil_E=3.0e6,
            soil_nu=0.27,
            soil_phi_deg=30.0,
            soil_psi_deg=5.0,
            soil_cohesion=1.5e5,
            rate_exponent=0.3,
            rate_sensitivity=0.1,
            saturation_increment=1.1,
            saturation_plastic_strain=0.5,
        )

    def _run(self, stress, deps, epbar=0.0, dt=1.0):
        _stress_in[None] = np.asarray(stress, dtype=np.float64)
        _deps_in[None] = np.asarray(deps, dtype=np.float64)
        p = self.params
        _run_taichi_update(
            epbar,
            dt,
            p.shear_modulus,
            p.bulk_modulus,
            p.friction_slope,
            p.dilatancy_slope,
            p.cohesion_intercept,
            p.rate_exponent,
            p.rate_sensitivity,
            p.saturation_increment,
            p.reference_strain_rate,
            p.saturation_plastic_strain,
        )
        return _stress_out.to_numpy(), _state_out.to_numpy()

    def test_source_parameter_transforms_are_exact(self):
        p = self.params
        sin_phi = math.sin(math.radians(30.0))
        saturation_normalization = 1.0 + 1.1 - 1.1 / math.exp(1.0)
        self.assertAlmostEqual(p.friction_slope, 6.0 * sin_phi / (3.0 - sin_phi))
        self.assertAlmostEqual(
            p.cohesion_intercept,
            6.0 * 1.5e5 / (3.0 - sin_phi) / saturation_normalization,
        )
        self.assertAlmostEqual(p.dilatancy_slope, math.tan(math.radians(5.0)))
        self.assertEqual(p.reference_strain_rate, SRSH_REFERENCE_STRAIN_RATE)

    def test_line_222_saturation_hardening_is_active(self):
        p = self.params
        initial = srsh_yield_stress_scalar(
            0.0,
            p.reference_strain_rate,
            p.cohesion_intercept,
            p.rate_exponent,
            p.rate_sensitivity,
            p.saturation_increment,
            p.reference_strain_rate,
            p.saturation_plastic_strain,
        )
        saturated = srsh_yield_stress_scalar(
            20.0 * p.saturation_plastic_strain,
            p.reference_strain_rate,
            p.cohesion_intercept,
            p.rate_exponent,
            p.rate_sensitivity,
            p.saturation_increment,
            p.reference_strain_rate,
            p.saturation_plastic_strain,
        )
        self.assertAlmostEqual(initial, p.cohesion_intercept)
        self.assertAlmostEqual(
            saturated,
            p.cohesion_intercept * (1.0 + p.saturation_increment),
            places=7,
        )

    def test_hydrostatic_compression_remains_elastic(self):
        stress, state = self._run(np.zeros((3, 3)), 0.01 * np.eye(3))
        expected = 0.03 * self.params.bulk_modulus * np.eye(3)
        np.testing.assert_allclose(stress, expected, rtol=1.0e-12, atol=1.0e-8)
        self.assertAlmostEqual(state[0], 0.0, places=14)
        self.assertAlmostEqual(state[1], 0.0, places=14)

    def test_pressure_increases_shear_strength(self):
        deps = np.diag([0.08, -0.04, -0.04])
        _, low_pressure_state = self._run(np.zeros((3, 3)), deps)
        _, high_pressure_state = self._run(1.0e5 * np.eye(3), deps)
        self.assertGreater(low_pressure_state[1], 0.0)
        self.assertAlmostEqual(high_pressure_state[1], 0.0, places=14)

    def test_plastic_return_matches_independent_bisection(self):
        old_stress = 2.0e4 * np.eye(3) + np.diag([3.0e4, -1.5e4, -1.5e4])
        deps = np.diag([0.16, -0.08, -0.08])
        expected = _numpy_reference_update(
            old_stress,
            deps,
            epbar=0.08,
            dt=0.02,
            params=self.params,
        )
        stress, state = self._run(
            old_stress,
            deps,
            epbar=0.08,
            dt=0.02,
        )
        np.testing.assert_allclose(stress, expected[0], rtol=1.0e-9, atol=2.0e-3)
        np.testing.assert_allclose(
            state,
            np.array(expected[1:]),
            rtol=1.0e-9,
            atol=1.0e-9,
        )
        pressure = float(np.trace(stress)) / 3.0
        dev = stress - pressure * np.eye(3)
        q = math.sqrt(1.5 * float(np.sum(dev * dev)))
        yield_stress = srsh_yield_stress_scalar(
            state[0],
            state[1] / 0.02,
            self.params.cohesion_intercept,
            self.params.rate_exponent,
            self.params.rate_sensitivity,
            self.params.saturation_increment,
            self.params.reference_strain_rate,
            self.params.saturation_plastic_strain,
        )
        self.assertLess(abs(q - self.params.friction_slope * pressure - yield_stress), 0.01)

    def test_static_surface_is_rechecked_after_rate_dependent_loading(self):
        loaded_stress, loaded_state = self._run(
            np.zeros((3, 3)),
            np.diag([0.30, -0.15, -0.15]),
            dt=1.0e-3,
        )
        _, unloaded_state = self._run(
            loaded_stress,
            np.zeros((3, 3)),
            epbar=loaded_state[0],
            dt=1.0,
        )
        self.assertGreater(unloaded_state[1], 0.0)

    def test_tank_mpm_solver_state_round_trip(self):
        coordinates = np.array([0.3, 0.7], dtype=np.float32)
        points = np.array(
            [[x, y, z] for x in coordinates for y in coordinates for z in coordinates],
            dtype=np.float32,
        )

        def make_solver():
            return TankTrackMpmSolver(
                soil_points=points,
                soil_p_vol=0.008,
                n_grid=8,
                dt=1.0e-5,
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                soil_bounds_lo=(0.2, 0.2, 0.2),
                soil_bounds_hi=(0.8, 0.8, 0.8),
                max_track_patches=2,
                soil_density=1950.0,
                soil_E=3.0e6,
                soil_nu=0.27,
                soil_phi_deg=30.0,
                soil_psi_deg=5.0,
                soil_cohesion=1.5e5,
                soil_constitutive_model="srsh",
                soil_gravity_scale=0.0,
                particle_shape=(2, 2, 2),
            )

        solver = make_solver()
        solver.substep()
        self.assertTrue(np.isfinite(solver.p_stress.to_numpy()).all())
        self.assertTrue((solver.p_srsh_epbar.to_numpy() >= 0.0).all())

        tmp_root = Path("tmp")
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            state_path = Path(tmp_dir) / "srsh_state.npz"
            solver.save_state(state_path)
            restored = make_solver()
            restored.load_state(state_path)
            np.testing.assert_allclose(
                restored.p_srsh_epbar.to_numpy(),
                solver.p_srsh_epbar.to_numpy(),
            )
            with np.load(state_path, allow_pickle=False) as state:
                self.assertNotIn("p_srsh_yield_stress", state.files)

if __name__ == "__main__":
    unittest.main()
