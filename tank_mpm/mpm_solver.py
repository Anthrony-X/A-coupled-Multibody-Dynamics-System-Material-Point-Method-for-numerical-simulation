from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import taichi as ti

from .constitutive_models import (
    CONSTITUTIVE_MODEL_DRUCKER_PRAGER,
    CONSTITUTIVE_MODEL_IDS,
    CONSTITUTIVE_MODEL_MODIFIED_CAM_CLAY,
    CONSTITUTIVE_MODEL_MOHR_COULOMB,
    CONSTITUTIVE_MODEL_PURE_WATER,
    CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP,
    SRSH_MODEL_REVISION,
    corotate_stress_increment,
    drucker_prager_update,
    make_drucker_prager_parameters,
    make_modified_cam_clay_parameters,
    make_mohr_coulomb_parameters,
    make_pure_water_parameters,
    make_srsh_modified_dp_parameters,
    modified_cam_clay_update,
    mohr_coulomb_update,
    normalize_constitutive_model_name,
    pure_water_update,
    srsh_modified_dp_update,
)
from .vtk_io import write_soil_vtk


@ti.data_oriented
class TankTrackMpmSolver:
    def __init__(
        self,
        soil_points: np.ndarray,
        soil_p_vol: float,
        n_grid: int,
        dt: float,
        domain_lo: tuple[float, float, float],
        domain_hi: tuple[float, float, float],
        soil_bounds_lo: tuple[float, float, float],
        soil_bounds_hi: tuple[float, float, float],
        max_track_patches: int = 256,
        soil_density: float = 1700.0,
        soil_E: float = 2.0e6,
        soil_nu: float = 0.30,
        soil_phi_deg: float = 18.0,
        soil_psi_deg: float = 2.0,
        soil_cohesion: float = 4.0e3,
        soil_constitutive_model: str = "drucker-prager",
        srsh_rate_exponent: float = 0.3,
        srsh_rate_sensitivity: float = 0.1,
        srsh_saturation_increment: float = 1.1,
        srsh_saturation_plastic_strain: float = 0.5,
        soil_mcc_m: float | None = None,
        soil_mcc_lambda: float = 0.12,
        soil_mcc_kappa: float = 0.02,
        soil_mcc_e0: float = 0.8,
        soil_mcc_pc0: float = 1.0e5,
        soil_mcc_min_pressure: float = 1.0,
        water_bulk_modulus: float = 2.2e9,
        water_dynamic_viscosity: float = 1.002e-3,
        water_cavitation_pressure: float = 0.0,
        soil_gravity_scale: float = 0.0,
        soil_damping: float = 1.0,
        contact_mu: float = 0.65,
        contact_barrier_stiffness: float | None = None,
        contact_barrier_radius: float | None = None,
        contact_barrier_min_distance_ratio: float = 0.15,
        contact_slip_smoothing_distance: float | None = None,
        track_activation_height: float = 0.6,
        mpm_precision: str = "f32",
        particle_shape: tuple[int, int, int] | None = None,
        moving_window_enabled: bool = False,
        moving_window_template_layers: int = 8,
        moving_window_axis: str = "x",
        moving_window_direction: int = 1,
        contact_binning_enabled: bool = False,
        contact_bin_size: float | None = None,
        contact_bin_capacity: int = 256,
    ):
        self.n_grid = int(n_grid)
        self.dt = float(dt)
        self.n_particles = int(soil_points.shape[0])
        self.max_track_patches = int(max_track_patches)
        if particle_shape is None:
            particle_shape = (self.n_particles, 1, 1)
        self.particle_shape = tuple(int(value) for value in particle_shape)
        if len(self.particle_shape) != 3 or any(value <= 0 for value in self.particle_shape):
            raise ValueError(f"Invalid particle_shape={self.particle_shape!r}")
        if int(np.prod(self.particle_shape)) != self.n_particles:
            raise ValueError(
                f"particle_shape product {int(np.prod(self.particle_shape))} does not match "
                f"n_particles={self.n_particles}"
            )
        self.particle_layers_x = int(self.particle_shape[0])
        self.particles_per_x_layer = int(self.particle_shape[1] * self.particle_shape[2])
        axis_name = str(moving_window_axis).strip().lower()
        if axis_name not in {"x", "y"}:
            raise ValueError(
                f"Unsupported moving_window_axis={moving_window_axis!r}; use 'x' or 'y'"
            )
        self.moving_window_axis_name = axis_name
        self.moving_window_axis = 0 if axis_name == "x" else 1
        self.moving_window_direction = int(moving_window_direction)
        if self.moving_window_direction not in {-1, 1}:
            raise ValueError("moving_window_direction must be -1 or +1")
        self.particle_layers_moving = int(self.particle_shape[self.moving_window_axis])
        self.particles_per_moving_layer = int(
            self.n_particles // self.particle_layers_moving
        )
        self.moving_window_enabled = bool(moving_window_enabled)
        self.moving_window_template_layers = min(
            self.particle_layers_moving,
            max(1, int(moving_window_template_layers)),
        )
        self.moving_window_ring_head = 0
        self.moving_window_shift_cells = 0
        self.moving_window_anchor: float | None = None
        self.moving_window_template_ready = False
        precision = str(mpm_precision).lower()
        if precision not in {"f32", "f64"}:
            raise ValueError(f"Unsupported mpm_precision={mpm_precision!r}; use 'f32' or 'f64'")
        self.mpm_precision = precision
        self.real = ti.f64 if precision == "f64" else ti.f32
        self.numpy_real = np.float64 if precision == "f64" else np.float32

        self.domain_lo_np = np.asarray(domain_lo, dtype=np.float64)
        self.domain_hi_np = np.asarray(domain_hi, dtype=np.float64)
        domain_span = self.domain_hi_np - self.domain_lo_np
        self.domain_extent = float(np.max(domain_span))
        if self.domain_extent <= 1.0e-12:
            raise ValueError("Invalid MPM domain extent")
        self.dx = self.domain_extent / self.n_grid
        self.inv_dx = 1.0 / self.dx
        grid_counts = np.maximum(3, np.ceil(domain_span * self.inv_dx).astype(np.int32))
        self.n_grid_x = int(grid_counts[0])
        self.n_grid_y = int(grid_counts[1])
        self.n_grid_z = int(grid_counts[2])
        self.domain_hi_np = self.domain_lo_np + grid_counts.astype(np.float64) * self.dx
        self.grid_shape = (self.n_grid_x, self.n_grid_y, self.n_grid_z)
        self.soil_bounds_lo = np.asarray(soil_bounds_lo, dtype=np.float64)
        self.soil_bounds_hi = np.asarray(soil_bounds_hi, dtype=np.float64)
        self.initial_domain_lo_np = self.domain_lo_np.copy()
        self.initial_domain_hi_np = self.domain_hi_np.copy()
        self.initial_soil_bounds_lo = self.soil_bounds_lo.copy()
        self.initial_soil_bounds_hi = self.soil_bounds_hi.copy()
        self.initial_soil_axis_min_value = float(
            self.initial_soil_bounds_lo[self.moving_window_axis]
        )
        self.initial_domain_axis_min_value = float(
            self.initial_domain_lo_np[self.moving_window_axis]
        )
        self.initial_domain_axis_max_value = float(
            self.initial_domain_hi_np[self.moving_window_axis]
        )
        self.particle_layer_spacing = float(
            (
                self.initial_soil_bounds_hi[self.moving_window_axis]
                - self.initial_soil_bounds_lo[self.moving_window_axis]
            )
            / self.particle_layers_moving
        )
        if self.moving_window_enabled:
            alignment_error = abs(self.particle_layer_spacing - self.dx)
            alignment_tolerance = max(1.0e-7, 1.0e-5 * self.dx)
            if alignment_error > alignment_tolerance:
                raise ValueError(
                    "Moving MPM window requires one particle layer per background-grid cell "
                    f"along {self.moving_window_axis_name}: "
                    f"particle_spacing={self.particle_layer_spacing:.9g}, grid_dx={self.dx:.9g}, "
                    f"error={alignment_error:.3e}. Adjust the grid, particle count, or "
                    f"{self.moving_window_axis_name} bounds."
                )

        self.p_vol = float(soil_p_vol)
        self.p_rho = float(soil_density)
        self.p_mass = self.p_vol * self.p_rho
        self.soil_young_modulus = float(soil_E)
        self.soil_poisson_ratio = float(soil_nu)
        self.soil_friction_angle_deg = float(soil_phi_deg)
        self.soil_dilatancy_angle_deg = float(soil_psi_deg)
        self.soil_reference_cohesion = float(soil_cohesion)
        self.soil_gravity_scale = float(soil_gravity_scale)
        self.soil_damping = float(max(0.0, min(1.0, soil_damping)))
        self.contact_mu = float(max(0.0, contact_mu))
        self.particle_radius = float(max((3.0 * self.p_vol / (4.0 * math.pi)) ** (1.0 / 3.0), 1.0e-9))
        self.contact_barrier_radius_value = float(
            self.particle_radius
            if contact_barrier_radius is None or float(contact_barrier_radius) <= 0.0
            else contact_barrier_radius
        )
        self.contact_barrier_stiffness_value = float(
            max(1.0e-9, soil_E * self.contact_barrier_radius_value)
            if contact_barrier_stiffness is None or float(contact_barrier_stiffness) <= 0.0
            else contact_barrier_stiffness
        )
        min_ratio = float(max(1.0e-4, min(0.95, contact_barrier_min_distance_ratio)))
        self.contact_barrier_min_distance_ratio_value = min_ratio
        self.contact_barrier_min_distance_value = min_ratio * self.contact_barrier_radius_value
        self.contact_slip_smoothing_distance_value = float(
            self.contact_barrier_radius_value
            if contact_slip_smoothing_distance is None or float(contact_slip_smoothing_distance) <= 0.0
            else contact_slip_smoothing_distance
        )
        self.track_activation_height_value = float(max(0.0, track_activation_height))
        self.contact_binning_enabled = bool(contact_binning_enabled)
        self.contact_bin_size_value = float(
            max(4.0 * self.dx, 4.0 * self.contact_barrier_radius_value)
            if contact_bin_size is None or float(contact_bin_size) <= 0.0
            else contact_bin_size
        )
        self.contact_bin_capacity = max(1, int(contact_bin_capacity))
        self.contact_bin_nx = max(
            1, int(math.ceil(domain_span[0] / self.contact_bin_size_value))
        )
        self.contact_bin_ny = max(
            1, int(math.ceil(domain_span[1] / self.contact_bin_size_value))
        )

        self.soil_constitutive_model_name = normalize_constitutive_model_name(soil_constitutive_model)
        self.soil_constitutive_model_id_value = CONSTITUTIVE_MODEL_IDS[
            self.soil_constitutive_model_name
        ]
        dp_params = make_drucker_prager_parameters(
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_phi_deg=soil_phi_deg,
            soil_psi_deg=soil_psi_deg,
            soil_cohesion=soil_cohesion,
        )
        self.shear_modulus = dp_params.shear_modulus
        self.bulk_modulus = dp_params.bulk_modulus
        self.dp_alpha = dp_params.alpha
        self.dp_k = dp_params.k
        self.dp_beta = dp_params.beta
        # MBD-MPM soil convention: compression stress is positive and tension is negative.
        # Therefore f_DP = sqrt(J2) - alpha * I1 - k uses i1_sign = -1.
        self.stress_sign_convention = "compression-positive"
        self.dp_i1_sign = -1.0
        srsh_params = make_srsh_modified_dp_parameters(
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_phi_deg=soil_phi_deg,
            soil_psi_deg=soil_psi_deg,
            soil_cohesion=soil_cohesion,
            rate_exponent=srsh_rate_exponent,
            rate_sensitivity=srsh_rate_sensitivity,
            saturation_increment=srsh_saturation_increment,
            saturation_plastic_strain=srsh_saturation_plastic_strain,
        )
        self.srsh_friction_slope = srsh_params.friction_slope
        self.srsh_dilatancy_slope = srsh_params.dilatancy_slope
        self.srsh_cohesion_intercept = srsh_params.cohesion_intercept
        self.srsh_rate_exponent = srsh_params.rate_exponent
        self.srsh_rate_sensitivity = srsh_params.rate_sensitivity
        self.srsh_saturation_increment = srsh_params.saturation_increment
        self.srsh_source_rate0 = srsh_params.reference_strain_rate
        self.srsh_saturation_plastic_strain = srsh_params.saturation_plastic_strain
        mc_params = make_mohr_coulomb_parameters(
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_phi_deg=soil_phi_deg,
            soil_psi_deg=soil_psi_deg,
            soil_cohesion=soil_cohesion,
        )
        self.mc_sin_phi = mc_params.sin_phi
        self.mc_cos_phi = mc_params.cos_phi
        self.mc_sin_psi = mc_params.sin_psi
        self.mc_cohesion = mc_params.cohesion
        mcc_params = make_modified_cam_clay_parameters(
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_phi_deg=soil_phi_deg,
            critical_state_m=soil_mcc_m,
            virgin_compression_lambda=soil_mcc_lambda,
            recompression_kappa=soil_mcc_kappa,
            initial_void_ratio=soil_mcc_e0,
            initial_preconsolidation_pressure=soil_mcc_pc0,
            min_pressure=soil_mcc_min_pressure,
        )
        self.mcc_M = mcc_params.critical_state_m
        self.mcc_lambda = mcc_params.virgin_compression_lambda
        self.mcc_kappa = mcc_params.recompression_kappa
        self.mcc_e0 = mcc_params.initial_void_ratio
        self.mcc_pc0 = mcc_params.initial_preconsolidation_pressure
        self.mcc_min_pressure = mcc_params.min_pressure
        water_params = make_pure_water_parameters(
            bulk_modulus=water_bulk_modulus,
            dynamic_viscosity=water_dynamic_viscosity,
            cavitation_pressure=water_cavitation_pressure,
        )
        self.water_bulk_modulus = water_params.bulk_modulus
        self.water_dynamic_viscosity = water_params.dynamic_viscosity
        self.water_cavitation_pressure = water_params.cavitation_pressure
        self.water_acoustic_speed = math.sqrt(self.water_bulk_modulus / self.p_rho)
        self.water_recommended_dt = 0.4 * self.dx / self.water_acoustic_speed
        if (
            self.soil_constitutive_model_id_value == CONSTITUTIVE_MODEL_PURE_WATER
            and self.dt > self.water_recommended_dt
        ):
            print(
                "[Warn] Pure-water acoustic CFL may be violated: "
                f"dt={self.dt:.3e}s, recommended <= {self.water_recommended_dt:.3e}s "
                f"for K={self.water_bulk_modulus:.3e}Pa and rho={self.p_rho:.3f}kg/m^3."
            )

        self.bc_ix_min = int((self.soil_bounds_lo[0] - self.domain_lo_np[0]) * self.inv_dx)
        self.bc_iy_min = int((self.soil_bounds_lo[1] - self.domain_lo_np[1]) * self.inv_dx)
        self.bc_iz_min = int((self.soil_bounds_lo[2] - self.domain_lo_np[2]) * self.inv_dx)
        self.bc_ix_max = int((self.soil_bounds_hi[0] - self.domain_lo_np[0]) * self.inv_dx)
        self.bc_iy_max = int((self.soil_bounds_hi[1] - self.domain_lo_np[1]) * self.inv_dx)

        self.x = ti.Vector.field(3, dtype=self.real, shape=self.n_particles)
        self.x0 = ti.Vector.field(3, dtype=self.real, shape=self.n_particles)
        self.v = ti.Vector.field(3, dtype=self.real, shape=self.n_particles)
        self.C = ti.Matrix.field(3, 3, dtype=self.real, shape=self.n_particles)
        self.p_stress = ti.Matrix.field(3, 3, dtype=self.real, shape=self.n_particles)
        self.p_pc = ti.field(dtype=self.real, shape=self.n_particles)
        self.p_srsh_epbar = ti.field(dtype=self.real, shape=self.n_particles)
        self.p_contact_shear_disp = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.p_contact_patch = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.p_contact_last_step = ti.field(dtype=ti.i32, shape=self.n_particles)

        template_shape = (
            (self.moving_window_template_layers, self.particles_per_moving_layer)
            if self.moving_window_enabled
            else (1, 1)
        )
        self.window_template_x = ti.Vector.field(3, dtype=self.real, shape=template_shape)
        self.window_template_x0 = ti.Vector.field(3, dtype=self.real, shape=template_shape)
        self.window_template_stress = ti.Matrix.field(3, 3, dtype=self.real, shape=template_shape)
        self.window_template_pc = ti.field(dtype=self.real, shape=template_shape)
        self.window_template_srsh_epbar = ti.field(dtype=self.real, shape=template_shape)

        self.grid_m = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_shape)

        self.n_patches = ti.field(dtype=ti.i32, shape=())
        self.patch_center = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_previous_center = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_axis_long = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_previous_axis_long = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_axis_width = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_previous_axis_width = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_normal = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_half_extent = ti.Vector.field(2, dtype=ti.f32, shape=self.max_track_patches)
        self.patch_side = ti.field(dtype=ti.i32, shape=self.max_track_patches)
        self.patch_shoe_id = ti.field(dtype=ti.f32, shape=self.max_track_patches)
        self.patch_module_id = ti.field(dtype=ti.i32, shape=self.max_track_patches)
        self.patch_axial_segment_id = ti.field(dtype=ti.i32, shape=self.max_track_patches)
        self.active_patch_count = ti.field(dtype=ti.i32, shape=())
        self.patch_active = ti.field(dtype=ti.i32, shape=self.max_track_patches)
        self.active_patch_ids = ti.field(dtype=ti.i32, shape=self.max_track_patches)
        self.patch_activation_min_z = ti.field(dtype=ti.f32, shape=self.max_track_patches)
        self.patch_query_lo = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.patch_query_hi = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.track_activation_height = ti.field(dtype=ti.f32, shape=())
        self.track_bottom_z_by_side = ti.field(dtype=ti.f32, shape=2)
        bin_shape = (
            (self.contact_bin_nx, self.contact_bin_ny)
            if self.contact_binning_enabled
            else (1, 1)
        )
        bin_id_shape = (
            (self.contact_bin_nx, self.contact_bin_ny, self.contact_bin_capacity)
            if self.contact_binning_enabled
            else (1, 1, 1)
        )
        self.contact_bin_count = ti.field(dtype=ti.i32, shape=bin_shape)
        self.contact_bin_patch_ids = ti.field(dtype=ti.i32, shape=bin_id_shape)
        self.contact_bin_overflow = ti.field(dtype=ti.i32, shape=())
        self.contact_bin_max_occupancy = ti.field(dtype=ti.i32, shape=())

        self.keyframe_center0 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_center1 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_axis_long0 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_axis_long1 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_axis_width0 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_axis_width1 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_normal0 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.keyframe_normal1 = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.track_keyframe_alpha_previous = ti.field(dtype=ti.f32, shape=())
        self.track_keyframe_alpha_current = ti.field(dtype=ti.f32, shape=())
        self.track_keyframe_block_dt = ti.field(dtype=ti.f32, shape=())

        self.surface_query_count = ti.field(dtype=ti.i32, shape=())
        self.surface_query_center = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.surface_query_axis_long = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.surface_query_axis_width = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.surface_query_half_extent = ti.Vector.field(2, dtype=ti.f32, shape=self.max_track_patches)
        self.surface_sample_count = ti.field(dtype=ti.i32, shape=())
        self.surface_z_samples = ti.field(dtype=self.real, shape=self.n_particles)

        self.domain_lo = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.domain_hi = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.window_template_source_start = ti.field(dtype=ti.i32, shape=())
        self.window_old_ring_head = ti.field(dtype=ti.i32, shape=())
        self.window_old_shift_cells = ti.field(dtype=ti.i32, shape=())
        self.window_shift_layers = ti.field(dtype=ti.i32, shape=())
        self.soil_kinematic_scale = ti.field(dtype=ti.f32, shape=())
        self.contact_barrier_stiffness = ti.field(dtype=ti.f32, shape=())
        self.contact_barrier_radius = ti.field(dtype=ti.f32, shape=())
        self.contact_barrier_min_distance = ti.field(dtype=ti.f32, shape=())
        self.contact_slip_smoothing_distance = ti.field(dtype=ti.f32, shape=())
        self.contact_step_index = ti.field(dtype=ti.i32, shape=())
        self.track_contact_force = ti.Vector.field(3, dtype=ti.f32, shape=())
        # Keep an independent soil-side force ledger so that Newton's
        # third-law residual is measured directly by validation cases.
        self.soil_contact_force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.contact_particle_count = ti.field(dtype=ti.i32, shape=())
        self.track_contact_force_by_side = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.contact_particle_count_by_side = ti.field(dtype=ti.i32, shape=2)
        self.track_contact_force_by_patch = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.soil_contact_force_by_patch = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.track_contact_moment_by_patch = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.track_contact_power = ti.field(dtype=ti.f32, shape=())
        self.soil_contact_power = ti.field(dtype=ti.f32, shape=())
        self.track_contact_power_by_patch = ti.field(dtype=ti.f32, shape=self.max_track_patches)
        self.soil_internal_work_increment = ti.field(dtype=ti.f32, shape=())
        self.soil_kinetic_energy = ti.field(dtype=ti.f32, shape=())
        self.soil_potential_energy = ti.field(dtype=ti.f32, shape=())
        self.contact_particle_count_by_patch = ti.field(dtype=ti.i32, shape=self.max_track_patches)
        self.contact_force_sum_by_side = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.contact_force_sum_by_patch = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.soil_contact_force_sum_by_patch = ti.Vector.field(
            3, dtype=ti.f32, shape=self.max_track_patches
        )
        self.contact_moment_sum_by_patch = ti.Vector.field(3, dtype=ti.f32, shape=self.max_track_patches)
        self.contact_force_sample_count = ti.field(dtype=ti.i32, shape=())
        # Macro metrics: rigid work, soil work, internal work, max action residual,
        # and max relative action residual. Counts: contact samples and maximum.
        self.contact_macro_metrics = ti.field(dtype=ti.f32, shape=5)
        self.contact_macro_counts = ti.field(dtype=ti.i32, shape=2)
        self.reduced_module_wrench = ti.Vector.field(
            6, dtype=ti.f32, shape=self.max_track_patches
        )
        self.reduced_axial_normal_load = ti.field(
            dtype=ti.f32, shape=self.max_track_patches
        )
        self.contact_debug_max_force_norm = ti.field(dtype=ti.f32, shape=())
        self.contact_debug_max_force_z = ti.field(dtype=ti.f32, shape=())
        self.contact_debug_min_distance = ti.field(dtype=ti.f32, shape=())
        self.contact_debug_max_normal_force = ti.field(dtype=ti.f32, shape=())
        self.contact_debug_max_slip_speed = ti.field(dtype=ti.f32, shape=())

        self.domain_lo[None] = ti.Vector(self.domain_lo_np.astype(np.float32).tolist())
        self.domain_hi[None] = ti.Vector(self.domain_hi_np.astype(np.float32).tolist())
        self.gravity[None] = ti.Vector([0.0, 0.0, -9.81])
        self.window_template_source_start[None] = 0
        self.window_old_ring_head[None] = 0
        self.window_old_shift_cells[None] = 0
        self.window_shift_layers[None] = 0
        self.soil_kinematic_scale[None] = 1.0
        self.contact_barrier_stiffness[None] = self.contact_barrier_stiffness_value
        self.contact_barrier_radius[None] = self.contact_barrier_radius_value
        self.contact_barrier_min_distance[None] = self.contact_barrier_min_distance_value
        self.contact_slip_smoothing_distance[None] = self.contact_slip_smoothing_distance_value
        self.contact_step_index[None] = 0
        self.patch_query_lo[None] = ti.Vector([1.0e30, 1.0e30, 1.0e30])
        self.patch_query_hi[None] = ti.Vector([-1.0e30, -1.0e30, -1.0e30])

        soil_points_real = soil_points.astype(self.numpy_real)
        self.soil_x0_np = soil_points_real.astype(np.float64)
        self.window_template_x0_host = np.zeros(
            (self.moving_window_template_layers, self.particles_per_moving_layer, 3),
            dtype=np.float64,
        )
        self.x.from_numpy(soil_points_real)
        self.x0.from_numpy(soil_points_real)
        self.v.from_numpy(np.zeros((self.n_particles, 3), dtype=self.numpy_real))
        self.C.from_numpy(np.zeros((self.n_particles, 3, 3), dtype=self.numpy_real))
        self.p_stress.from_numpy(np.zeros((self.n_particles, 3, 3), dtype=self.numpy_real))
        self.p_pc.from_numpy(np.full((self.n_particles,), self.mcc_pc0, dtype=self.numpy_real))
        self.p_srsh_epbar.fill(0.0)
        self.n_patches[None] = 0
        self.active_patch_count[None] = 0
        self.track_activation_height[None] = self.track_activation_height_value
        for side in range(2):
            self.track_bottom_z_by_side[side] = 1.0e30
        self.surface_query_count[None] = 0
        self.surface_sample_count[None] = 0
        self.track_keyframe_alpha_previous[None] = 0.0
        self.track_keyframe_alpha_current[None] = 0.0
        self.track_keyframe_block_dt[None] = float(self.dt)
        self.contact_force_sample_count[None] = 0
        self.contact_bin_overflow[None] = 0
        self.contact_bin_max_occupancy[None] = 0
        self.contact_module_count_host = 0
        self.contact_axial_segment_count_host = 0
        self._patch_center_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._patch_axis_long_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._patch_axis_width_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._patch_normal_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._patch_velocity_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._patch_half_extent_host = np.zeros((self.max_track_patches, 2), dtype=np.float32)
        self._patch_side_host = np.zeros((self.max_track_patches,), dtype=np.int32)
        self._patch_shoe_id_host = np.zeros((self.max_track_patches,), dtype=np.float32)
        self._patch_mass_host = np.zeros((self.max_track_patches,), dtype=np.float32)
        self._patch_module_id_host = np.full(
            (self.max_track_patches,), -1, dtype=np.int32
        )
        self._patch_axial_segment_id_host = np.full(
            (self.max_track_patches,), -1, dtype=np.int32
        )
        self._patch_active_host = np.zeros((self.max_track_patches,), dtype=np.int32)
        self._keyframe_center0_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_center1_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_axis_long0_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_axis_long1_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_axis_width0_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_axis_width1_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_normal0_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._keyframe_normal1_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._surface_query_center_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._surface_query_axis_long_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._surface_query_axis_width_host = np.zeros((self.max_track_patches, 3), dtype=np.float32)
        self._surface_query_half_extent_host = np.zeros((self.max_track_patches, 2), dtype=np.float32)
        self.coupled_step = 0
        self.initialize_contact_history()

    def _set_patch_query_bounds(
        self,
        centers: np.ndarray,
        axis_long: np.ndarray,
        axis_width: np.ndarray,
        half_extent: np.ndarray,
        *,
        previous_centers: np.ndarray | None = None,
        previous_axis_long: np.ndarray | None = None,
        previous_axis_width: np.ndarray | None = None,
    ) -> None:
        """Upload a conservative swept AABB used to cull particle-patch searches."""

        centers = np.asarray(centers, dtype=np.float32)
        if centers.shape[0] == 0:
            self.patch_query_lo[None] = ti.Vector([1.0e30, 1.0e30, 1.0e30])
            self.patch_query_hi[None] = ti.Vector([-1.0e30, -1.0e30, -1.0e30])
            return

        axis_long = np.asarray(axis_long, dtype=np.float32)
        axis_width = np.asarray(axis_width, dtype=np.float32)
        half_extent = np.asarray(half_extent, dtype=np.float32)
        extent = (
            np.abs(axis_long) * half_extent[:, 0:1]
            + np.abs(axis_width) * half_extent[:, 1:2]
        )
        lower = centers - extent
        upper = centers + extent
        if previous_centers is not None:
            previous_centers = np.asarray(previous_centers, dtype=np.float32)
            previous_axis_long = np.asarray(
                axis_long if previous_axis_long is None else previous_axis_long,
                dtype=np.float32,
            )
            previous_axis_width = np.asarray(
                axis_width if previous_axis_width is None else previous_axis_width,
                dtype=np.float32,
            )
            previous_extent = (
                np.abs(previous_axis_long) * half_extent[:, 0:1]
                + np.abs(previous_axis_width) * half_extent[:, 1:2]
            )
            lower = np.vstack((lower, previous_centers - previous_extent))
            upper = np.vstack((upper, previous_centers + previous_extent))
        padding = self.contact_barrier_radius_value
        query_lo = np.min(lower, axis=0) - padding
        query_hi = np.max(upper, axis=0) + padding
        self.patch_query_lo[None] = ti.Vector(query_lo.astype(np.float32).tolist())
        self.patch_query_hi[None] = ti.Vector(query_hi.astype(np.float32).tolist())

    def set_track_patches(self, patches: dict[str, np.ndarray]) -> None:
        n = int(patches["center"].shape[0])
        if n > self.max_track_patches:
            raise ValueError(f"Too many track patches: {n} > {self.max_track_patches}")
        self.n_patches[None] = n
        self.active_patch_count[None] = n
        if n == 0:
            self.contact_module_count_host = 0
            self.contact_axial_segment_count_host = 0
            self.contact_bin_overflow[None] = 0
            self.contact_bin_max_occupancy[None] = 0
            self._patch_active_host[:] = 0
            self.patch_active.from_numpy(self._patch_active_host)
            self._set_patch_query_bounds(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
            return
        side_id = patches.get("side_id")
        if side_id is None:
            side_id = np.zeros((n,), dtype=np.int32)
        shoe_id = patches.get("shoe_id")
        if shoe_id is None:
            shoe_id = np.arange(n, dtype=np.float32)
        module_id = patches.get("module_id")
        if module_id is None:
            module_id = np.full((n,), -1, dtype=np.int32)
        axial_segment_id = patches.get("axial_segment_id")
        if axial_segment_id is None:
            axial_segment_id = np.full((n,), -1, dtype=np.int32)
        patch_mass = patches.get("mass")
        if patch_mass is None:
            raise ValueError("Track patches with n > 0 must provide per-patch mass")
        patch_mass_arr = np.asarray(patch_mass, dtype=np.float32)
        center_arr = np.asarray(patches["center"], dtype=np.float32)
        axis_long_arr = np.asarray(patches["axis_long"], dtype=np.float32)
        axis_width_arr = np.asarray(patches["axis_width"], dtype=np.float32)
        normal_arr = np.asarray(patches["normal"], dtype=np.float32)
        velocity_arr = np.asarray(patches["velocity"], dtype=np.float32)
        half_extent_arr = np.asarray(patches["half_extent"], dtype=np.float32)
        previous_center_arr = (
            np.asarray(patches["previous_center"], dtype=np.float32)
            if "previous_center" in patches
            else None
        )
        previous_axis_long_arr = (
            np.asarray(patches["previous_axis_long"], dtype=np.float32)
            if "previous_axis_long" in patches
            else None
        )
        previous_axis_width_arr = (
            np.asarray(patches["previous_axis_width"], dtype=np.float32)
            if "previous_axis_width" in patches
            else None
        )
        side_arr = np.asarray(side_id, dtype=np.int32)
        shoe_arr = np.asarray(shoe_id, dtype=np.float32)
        module_arr = np.asarray(module_id, dtype=np.int32)
        axial_segment_arr = np.asarray(axial_segment_id, dtype=np.int32)

        self._patch_center_host[:n, :] = center_arr
        self._patch_axis_long_host[:n, :] = axis_long_arr
        self._patch_axis_width_host[:n, :] = axis_width_arr
        self._patch_normal_host[:n, :] = normal_arr
        self._patch_velocity_host[:n, :] = velocity_arr
        self._patch_half_extent_host[:n, :] = half_extent_arr
        self._patch_side_host[:n] = side_arr
        self._patch_shoe_id_host[:n] = shoe_arr
        self._patch_mass_host[:n] = patch_mass_arr
        self._patch_module_id_host[:n] = module_arr
        self._patch_axial_segment_id_host[:n] = axial_segment_arr
        if n < self.max_track_patches:
            self._patch_module_id_host[n:] = -1
            self._patch_axial_segment_id_host[n:] = -1
        self.contact_module_count_host = (
            int(np.max(module_arr)) + 1 if np.any(module_arr >= 0) else 0
        )
        self.contact_axial_segment_count_host = (
            int(np.max(axial_segment_arr)) + 1
            if np.any(axial_segment_arr >= 0)
            else 0
        )
        self._patch_active_host[:n] = 1
        if n < self.max_track_patches:
            self._patch_active_host[n:] = 0
        self.patch_center.from_numpy(self._patch_center_host)
        self.patch_previous_center.from_numpy(
            self._patch_center_host if previous_center_arr is None else np.vstack(
                [previous_center_arr, self._patch_center_host[n:]]
            ).astype(np.float32)
        )
        self.patch_axis_long.from_numpy(self._patch_axis_long_host)
        self.patch_previous_axis_long.from_numpy(
            self._patch_axis_long_host if previous_axis_long_arr is None else np.vstack(
                [previous_axis_long_arr, self._patch_axis_long_host[n:]]
            ).astype(np.float32)
        )
        self.patch_axis_width.from_numpy(self._patch_axis_width_host)
        self.patch_previous_axis_width.from_numpy(
            self._patch_axis_width_host if previous_axis_width_arr is None else np.vstack(
                [previous_axis_width_arr, self._patch_axis_width_host[n:]]
            ).astype(np.float32)
        )
        self.patch_normal.from_numpy(self._patch_normal_host)
        self.patch_velocity.from_numpy(self._patch_velocity_host)
        self.patch_half_extent.from_numpy(self._patch_half_extent_host)
        self.patch_side.from_numpy(self._patch_side_host)
        self.patch_shoe_id.from_numpy(self._patch_shoe_id_host)
        self.patch_module_id.from_numpy(self._patch_module_id_host)
        self.patch_axial_segment_id.from_numpy(self._patch_axial_segment_id_host)
        self.patch_active.from_numpy(self._patch_active_host)

        # Initialize keyframes once as well.  Subsequent macro steps upload only
        # centers and frames; half extents and ownership remain in patch fields.
        previous_center_full = self._patch_center_host.copy()
        previous_axis_long_full = self._patch_axis_long_host.copy()
        previous_axis_width_full = self._patch_axis_width_host.copy()
        if previous_center_arr is not None:
            previous_center_full[:n] = previous_center_arr
        if previous_axis_long_arr is not None:
            previous_axis_long_full[:n] = previous_axis_long_arr
        if previous_axis_width_arr is not None:
            previous_axis_width_full[:n] = previous_axis_width_arr
        self.keyframe_center0.from_numpy(previous_center_full)
        self.keyframe_center1.from_numpy(self._patch_center_host)
        self.keyframe_axis_long0.from_numpy(previous_axis_long_full)
        self.keyframe_axis_long1.from_numpy(self._patch_axis_long_host)
        self.keyframe_axis_width0.from_numpy(previous_axis_width_full)
        self.keyframe_axis_width1.from_numpy(self._patch_axis_width_host)
        self.keyframe_normal0.from_numpy(self._patch_normal_host)
        self.keyframe_normal1.from_numpy(self._patch_normal_host)
        self._set_patch_query_bounds(
            center_arr,
            axis_long_arr,
            axis_width_arr,
            half_extent_arr,
            previous_centers=previous_center_arr,
            previous_axis_long=previous_axis_long_arr,
            previous_axis_width=previous_axis_width_arr,
        )
        self.update_active_track_patches_by_height()
        if self.contact_binning_enabled:
            self.rebuild_contact_patch_bins()

    def set_track_patch_keyframes(
        self,
        start: dict[str, np.ndarray],
        end: dict[str, np.ndarray],
    ) -> None:
        """Upload two track patch keyframes for on-device SDF contact updates."""

        n = int(end["center"].shape[0])
        if n > self.max_track_patches:
            raise ValueError(f"Too many track patches: {n} > {self.max_track_patches}")
        if start["center"].shape[0] != n:
            raise ValueError("Track patch keyframes must have matching patch counts")
        for key in ("axis_long", "axis_width", "normal", "half_extent", "mass", "side_id", "shoe_id"):
            if start[key].shape[0] != n or end[key].shape[0] != n:
                raise ValueError(f"Track patch keyframes have mismatched {key!r} arrays")

        # Preserve the historical convenience API: a first keyframe pair may
        # establish topology when set_track_patches was not called explicitly.
        if int(self.n_patches[None]) == 0 and n > 0:
            self.set_track_patches(end)

        side_start = np.asarray(start["side_id"], dtype=np.int32)
        side_end = np.asarray(end["side_id"], dtype=np.int32)
        shoe_start = np.asarray(start["shoe_id"], dtype=np.float32)
        shoe_end = np.asarray(end["shoe_id"], dtype=np.float32)
        if not np.array_equal(side_start, side_end) or not np.array_equal(shoe_start, shoe_end):
            raise ValueError("Track patch keyframes must have identical side_id/shoe_id topology")

        if n != int(self.n_patches[None]):
            self.set_track_patches(end)
        if n == 0:
            self._set_patch_query_bounds(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
            return

        center0 = np.asarray(start["center"], dtype=np.float32)
        center1 = np.asarray(end["center"], dtype=np.float32)
        axis_long0 = np.asarray(start["axis_long"], dtype=np.float32)
        axis_long1 = np.asarray(end["axis_long"], dtype=np.float32)
        axis_width0 = np.asarray(start["axis_width"], dtype=np.float32)
        axis_width1 = np.asarray(end["axis_width"], dtype=np.float32)
        normal0 = np.asarray(start["normal"], dtype=np.float32)
        normal1 = np.asarray(end["normal"], dtype=np.float32)
        half_extent = np.asarray(end["half_extent"], dtype=np.float32)
        patch_mass = np.asarray(end["mass"], dtype=np.float32)
        if not np.array_equal(
            half_extent, self._patch_half_extent_host[:n]
        ) or not np.array_equal(patch_mass, self._patch_mass_host[:n]):
            self.set_track_patches(end)
        if not np.array_equal(side_end, self._patch_side_host[:n]) or not np.array_equal(
            shoe_end, self._patch_shoe_id_host[:n]
        ):
            self.set_track_patches(end)

        self._keyframe_center0_host[:n, :] = center0
        self._keyframe_center1_host[:n, :] = center1
        self._keyframe_axis_long0_host[:n, :] = axis_long0
        self._keyframe_axis_long1_host[:n, :] = axis_long1
        self._keyframe_axis_width0_host[:n, :] = axis_width0
        self._keyframe_axis_width1_host[:n, :] = axis_width1
        self._keyframe_normal0_host[:n, :] = normal0
        self._keyframe_normal1_host[:n, :] = normal1
        # Host copies support diagnostics and VTK metadata without another GPU read.
        self._patch_center_host[:n, :] = center1
        self._patch_axis_long_host[:n, :] = axis_long1
        self._patch_axis_width_host[:n, :] = axis_width1
        self._patch_normal_host[:n, :] = normal1
        self._patch_velocity_host[:n, :] = (center1 - center0) / max(self.dt, 1.0e-12)

        self.keyframe_center0.from_numpy(self._keyframe_center0_host)
        self.keyframe_center1.from_numpy(self._keyframe_center1_host)
        self.keyframe_axis_long0.from_numpy(self._keyframe_axis_long0_host)
        self.keyframe_axis_long1.from_numpy(self._keyframe_axis_long1_host)
        self.keyframe_axis_width0.from_numpy(self._keyframe_axis_width0_host)
        self.keyframe_axis_width1.from_numpy(self._keyframe_axis_width1_host)
        self.keyframe_normal0.from_numpy(self._keyframe_normal0_host)
        self.keyframe_normal1.from_numpy(self._keyframe_normal1_host)
        self._set_patch_query_bounds(
            center1,
            axis_long1,
            axis_width1,
            half_extent,
            previous_centers=center0,
            previous_axis_long=axis_long0,
            previous_axis_width=axis_width0,
        )
        if self.contact_binning_enabled:
            self.rebuild_contact_patch_bins()

    @ti.func
    def _normalized_f32(self, value):
        result = value
        norm = result.norm()
        if norm > 1.0e-12:
            result = result / norm
        return result

    def update_track_patches_from_keyframes(self, alpha_previous: float, alpha_current: float, block_dt: float) -> None:
        self.track_keyframe_alpha_previous[None] = float(alpha_previous)
        self.track_keyframe_alpha_current[None] = float(alpha_current)
        self.track_keyframe_block_dt[None] = float(block_dt)
        self._update_track_patches_from_keyframes_kernel()
        self.update_active_track_patches_by_height()

    def rebuild_contact_patch_bins(self) -> None:
        """Build a swept 2-D broad phase once per MBD macro step."""

        if self.contact_binning_enabled:
            self._rebuild_contact_patch_bins_kernel()

    @ti.kernel
    def _rebuild_contact_patch_bins_kernel(self):
        self.contact_bin_overflow[None] = 0
        self.contact_bin_max_occupancy[None] = 0
        for I in ti.grouped(self.contact_bin_count):
            self.contact_bin_count[I] = 0

        for p in range(self.n_patches[None]):
            # A circular in-plane bound remains conservative for all interpolated
            # patch orientations between the two MBD keyframes.
            half_extent = self.patch_half_extent[p]
            extent = ti.sqrt(
                half_extent[0] * half_extent[0]
                + half_extent[1] * half_extent[1]
            ) + self.contact_barrier_radius[None]
            c0 = self.keyframe_center0[p]
            c1 = self.keyframe_center1[p]
            lo_x = ti.min(c0[0], c1[0]) - extent
            hi_x = ti.max(c0[0], c1[0]) + extent
            lo_y = ti.min(c0[1], c1[1]) - extent
            hi_y = ti.max(c0[1], c1[1]) + extent
            ix0 = ti.max(
                0,
                ti.min(
                    self.contact_bin_nx - 1,
                    ti.cast(
                        ti.floor(
                            (lo_x - self.domain_lo[None][0])
                            / self.contact_bin_size_value
                        ),
                        ti.i32,
                    ),
                ),
            )
            ix1 = ti.max(
                0,
                ti.min(
                    self.contact_bin_nx - 1,
                    ti.cast(
                        ti.floor(
                            (hi_x - self.domain_lo[None][0])
                            / self.contact_bin_size_value
                        ),
                        ti.i32,
                    ),
                ),
            )
            iy0 = ti.max(
                0,
                ti.min(
                    self.contact_bin_ny - 1,
                    ti.cast(
                        ti.floor(
                            (lo_y - self.domain_lo[None][1])
                            / self.contact_bin_size_value
                        ),
                        ti.i32,
                    ),
                ),
            )
            iy1 = ti.max(
                0,
                ti.min(
                    self.contact_bin_ny - 1,
                    ti.cast(
                        ti.floor(
                            (hi_y - self.domain_lo[None][1])
                            / self.contact_bin_size_value
                        ),
                        ti.i32,
                    ),
                ),
            )
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    slot = ti.atomic_add(self.contact_bin_count[ix, iy], 1)
                    ti.atomic_max(self.contact_bin_max_occupancy[None], slot + 1)
                    if slot < self.contact_bin_capacity:
                        self.contact_bin_patch_ids[ix, iy, slot] = p
                    else:
                        self.contact_bin_overflow[None] = 1

    @ti.kernel
    def _update_track_patches_from_keyframes_kernel(self):
        a0 = ti.max(0.0, ti.min(1.0, self.track_keyframe_alpha_previous[None]))
        a1 = ti.max(0.0, ti.min(1.0, self.track_keyframe_alpha_current[None]))
        inv_dt = 1.0 / ti.max(self.track_keyframe_block_dt[None], 1.0e-12)
        for p in range(self.n_patches[None]):
            c0_key = self.keyframe_center0[p]
            c1_key = self.keyframe_center1[p]
            previous_center = c0_key + a0 * (c1_key - c0_key)
            current_center = c0_key + a1 * (c1_key - c0_key)
            previous_axis_long = self._normalized_f32(
                self.keyframe_axis_long0[p]
                + a0 * (self.keyframe_axis_long1[p] - self.keyframe_axis_long0[p])
            )
            axis_long = self._normalized_f32(
                self.keyframe_axis_long0[p]
                + a1 * (self.keyframe_axis_long1[p] - self.keyframe_axis_long0[p])
            )
            previous_axis_width = self._normalized_f32(
                self.keyframe_axis_width0[p]
                + a0 * (self.keyframe_axis_width1[p] - self.keyframe_axis_width0[p])
            )
            axis_width = self._normalized_f32(
                self.keyframe_axis_width0[p]
                + a1 * (self.keyframe_axis_width1[p] - self.keyframe_axis_width0[p])
            )
            normal = self._normalized_f32(
                self.keyframe_normal0[p]
                + a1 * (self.keyframe_normal1[p] - self.keyframe_normal0[p])
            )
            self.patch_center[p] = current_center
            self.patch_previous_center[p] = previous_center
            self.patch_axis_long[p] = axis_long
            self.patch_previous_axis_long[p] = previous_axis_long
            self.patch_axis_width[p] = axis_width
            self.patch_previous_axis_width[p] = previous_axis_width
            self.patch_normal[p] = normal
            self.patch_velocity[p] = (current_center - previous_center) * inv_dt

    @ti.func
    def _patch_surface_min_z(self, center, axis_long, axis_width, half_extent):
        h_long = ti.max(half_extent[0], 0.0)
        h_width = ti.max(half_extent[1], 0.0)
        min_z = 1.0e30
        for i, j in ti.static(ti.ndrange(2, 2)):
            s_long = -1.0 if i == 0 else 1.0
            s_width = -1.0 if j == 0 else 1.0
            corner = center + s_long * h_long * axis_long + s_width * h_width * axis_width
            min_z = ti.min(min_z, corner[2])
        return min_z

    @ti.func
    def _patch_swept_surface_min_z(self, p):
        current_min_z = self._patch_surface_min_z(
            self.patch_center[p],
            self.patch_axis_long[p],
            self.patch_axis_width[p],
            self.patch_half_extent[p],
        )
        previous_min_z = self._patch_surface_min_z(
            self.patch_previous_center[p],
            self.patch_previous_axis_long[p],
            self.patch_previous_axis_width[p],
            self.patch_half_extent[p],
        )
        return ti.min(current_min_z, previous_min_z)

    @ti.kernel
    def update_active_track_patches_by_height(self):
        self.active_patch_count[None] = 0
        for side in ti.static(range(2)):
            self.track_bottom_z_by_side[side] = 1.0e30
        for p in range(self.max_track_patches):
            self.patch_active[p] = 0
            self.active_patch_ids[p] = 0
            self.patch_activation_min_z[p] = 1.0e30

        for p in range(self.n_patches[None]):
            side = self.patch_side[p]
            if side < 0:
                side = 0
            if side > 1:
                side = 1
            min_z = self._patch_swept_surface_min_z(p)
            self.patch_activation_min_z[p] = min_z
            ti.atomic_min(self.track_bottom_z_by_side[side], min_z)

        for p in range(self.n_patches[None]):
            side = self.patch_side[p]
            if side < 0:
                side = 0
            if side > 1:
                side = 1
            if self.patch_activation_min_z[p] <= self.track_bottom_z_by_side[side] + self.track_activation_height[None]:
                self.patch_active[p] = 1
                slot = ti.atomic_add(self.active_patch_count[None], 1)
                self.active_patch_ids[slot] = p

    def set_surface_query_patches(self, surfaces: list[dict[str, np.ndarray]]) -> None:
        n = min(len(surfaces), self.max_track_patches)
        self.surface_query_count[None] = n
        if n == 0:
            return
        for idx, surface in enumerate(surfaces[:n]):
            self._surface_query_center_host[idx, :] = np.asarray(surface["center"], dtype=np.float32)
            self._surface_query_axis_long_host[idx, :] = np.asarray(surface["axis_long"], dtype=np.float32)
            self._surface_query_axis_width_host[idx, :] = np.asarray(surface["axis_width"], dtype=np.float32)
            self._surface_query_half_extent_host[idx, :] = np.asarray(surface["half_extent"], dtype=np.float32)
        self.surface_query_center.from_numpy(self._surface_query_center_host)
        self.surface_query_axis_long.from_numpy(self._surface_query_axis_long_host)
        self.surface_query_axis_width.from_numpy(self._surface_query_axis_width_host)
        self.surface_query_half_extent.from_numpy(self._surface_query_half_extent_host)

    @ti.kernel
    def collect_surface_z_samples(self):
        self.surface_sample_count[None] = 0
        for p in range(self.n_particles):
            pos = self.x[p]
            inside = False
            for query in range(self.surface_query_count[None]):
                q = ti.cast(pos, ti.f32) - self.surface_query_center[query]
                half_extent = self.surface_query_half_extent[query]
                a = q.dot(self.surface_query_axis_long[query])
                b = q.dot(self.surface_query_axis_width[query])
                half_l = half_extent[0] + self.dx
                half_w = half_extent[1] + self.dx
                if ti.abs(a) <= half_l and ti.abs(b) <= half_w:
                    inside = True
            if inside:
                sample_id = ti.atomic_add(self.surface_sample_count[None], 1)
                if sample_id < self.n_particles:
                    self.surface_z_samples[sample_id] = pos[2]

    def estimate_surface_z_from_patches(self, surfaces: list[dict[str, np.ndarray]], percentile: float = 99.0) -> float | None:
        if not surfaces:
            return None
        self.set_surface_query_patches(surfaces)
        self.collect_surface_z_samples()
        sample_count = int(self.surface_sample_count[None])
        if sample_count <= 0:
            return None
        samples = self.surface_z_samples.to_numpy()[:sample_count]
        q = float(max(0.0, min(100.0, percentile)))
        return float(np.percentile(samples, q))

    @ti.kernel
    def initialize_contact_history(self):
        for p in range(self.n_particles):
            self.p_contact_shear_disp[p] = 0.0
            self.p_contact_patch[p] = -1
            self.p_contact_last_step[p] = -2147483647

    @ti.func
    def _moving_window_particle_id(self, layer, layer_particle):
        particle_id = 0
        if ti.static(self.moving_window_axis == 0):
            particle_id = layer * self.particles_per_moving_layer + layer_particle
        else:
            z_count = ti.static(self.particle_shape[2])
            y_count = ti.static(self.particle_shape[1])
            x_index = layer_particle // z_count
            z_index = layer_particle - x_index * z_count
            particle_id = (x_index * y_count + layer) * z_count + z_index
        return particle_id

    @ti.kernel
    def _capture_moving_window_template_kernel(self):
        for template_layer, layer_particle in ti.ndrange(
            self.moving_window_template_layers,
            self.particles_per_moving_layer,
        ):
            source_layer = self.window_template_source_start[None] + template_layer
            p = self._moving_window_particle_id(source_layer, layer_particle)
            source_center = self.initial_soil_axis_min_value + (
                ti.cast(source_layer, self.real) + 0.5
            ) * self.particle_layer_spacing
            current_local = self.x[p]
            reference_local = self.x0[p]
            if ti.static(self.moving_window_axis == 0):
                current_local[0] -= source_center
                reference_local[0] -= source_center
            else:
                current_local[1] -= source_center
                reference_local[1] -= source_center
            self.window_template_x[template_layer, layer_particle] = current_local
            self.window_template_x0[template_layer, layer_particle] = reference_local
            self.window_template_stress[template_layer, layer_particle] = self.p_stress[p]
            self.window_template_pc[template_layer, layer_particle] = self.p_pc[p]
            self.window_template_srsh_epbar[
                template_layer, layer_particle
            ] = self.p_srsh_epbar[p]

    @ti.kernel
    def _recycle_particle_layers_kernel(self):
        for flat_index in range(self.window_shift_layers[None] * self.particles_per_moving_layer):
            layer_offset = flat_index // self.particles_per_moving_layer
            layer_particle = flat_index - layer_offset * self.particles_per_moving_layer
            slot_layer = 0
            global_layer = 0
            if ti.static(self.moving_window_direction > 0):
                slot_layer = (
                    self.window_old_ring_head[None] + layer_offset
                ) % self.particle_layers_moving
                global_layer = (
                    self.window_old_shift_cells[None]
                    + self.particle_layers_moving
                    + layer_offset
                )
            else:
                slot_layer = (
                    self.window_old_ring_head[None]
                    - 1
                    - layer_offset
                    + self.particle_layers_moving
                ) % self.particle_layers_moving
                global_layer = (
                    -self.window_old_shift_cells[None] - 1 - layer_offset
                )
            p = self._moving_window_particle_id(slot_layer, layer_particle)
            template_layer = (
                (
                    global_layer % self.moving_window_template_layers
                    + self.moving_window_template_layers
                )
                % self.moving_window_template_layers
            )
            target_center = self.initial_soil_axis_min_value + (
                ti.cast(global_layer, self.real) + 0.5
            ) * self.particle_layer_spacing

            current = self.window_template_x[template_layer, layer_particle]
            reference = self.window_template_x0[template_layer, layer_particle]
            if ti.static(self.moving_window_axis == 0):
                current[0] += target_center
                reference[0] += target_center
            else:
                current[1] += target_center
                reference[1] += target_center
            self.x[p] = current
            self.x0[p] = reference
            self.v[p] = ti.Vector.zero(self.real, 3)
            self.C[p] = ti.Matrix.zero(self.real, 3, 3)
            self.p_stress[p] = self.window_template_stress[template_layer, layer_particle]
            self.p_pc[p] = self.window_template_pc[template_layer, layer_particle]
            self.p_srsh_epbar[p] = self.window_template_srsh_epbar[
                template_layer, layer_particle
            ]
            self.p_contact_shear_disp[p] = 0.0
            self.p_contact_patch[p] = -1
            self.p_contact_last_step[p] = -2147483647

    @ti.kernel
    def _translate_window_grid_bounds_kernel(self):
        absolute_shift = self.window_old_shift_cells[None] + self.window_shift_layers[None]
        delta = (
            ti.cast(absolute_shift, ti.f32)
            * self.dx
            * self.moving_window_direction
        )
        if ti.static(self.moving_window_axis == 0):
            self.domain_lo[None][0] = self.initial_domain_axis_min_value + delta
            self.domain_hi[None][0] = self.initial_domain_axis_max_value + delta
        else:
            self.domain_lo[None][1] = self.initial_domain_axis_min_value + delta
            self.domain_hi[None][1] = self.initial_domain_axis_max_value + delta

    def _moving_window_layer_particle_ids(self, layer: int) -> np.ndarray:
        if self.moving_window_axis == 0:
            first = int(layer) * self.particles_per_moving_layer
            return np.arange(
                first,
                first + self.particles_per_moving_layer,
                dtype=np.int64,
            )
        nx, ny, nz = self.particle_shape
        x_index = np.arange(nx, dtype=np.int64)[:, None]
        z_index = np.arange(nz, dtype=np.int64)[None, :]
        return ((x_index * ny + int(layer)) * nz + z_index).reshape(-1)

    def capture_moving_window_template(self) -> None:
        if not self.moving_window_enabled:
            return
        if self.moving_window_shift_cells != 0 or self.moving_window_ring_head != 0:
            raise RuntimeError("Moving-window pristine template must be captured before the first window shift.")
        source_start = max(
            0,
            (self.particle_layers_moving - self.moving_window_template_layers) // 2,
        )
        self.window_template_source_start[None] = int(source_start)
        self._capture_moving_window_template_kernel()
        particle_grid = self.soil_x0_np.reshape((*self.particle_shape, 3))
        layered_grid = np.moveaxis(particle_grid, self.moving_window_axis, 0)
        source = layered_grid[
            source_start : source_start + self.moving_window_template_layers
        ].copy().reshape(
            self.moving_window_template_layers,
            self.particles_per_moving_layer,
            3,
        )
        source_centers = self.initial_soil_bounds_lo[self.moving_window_axis] + (
            np.arange(source_start, source_start + self.moving_window_template_layers, dtype=np.float64)
            + 0.5
        ) * self.particle_layer_spacing
        source[:, :, self.moving_window_axis] -= source_centers[:, None]
        self.window_template_x0_host[:, :, :] = source
        self.moving_window_template_ready = True

    def configure_moving_window_anchor(self, vehicle_coordinate: float) -> None:
        if not self.moving_window_enabled:
            return
        if not self.moving_window_template_ready:
            raise RuntimeError(
                "Moving-window pristine soil template is unavailable. Rerun geostatic with "
                "--moving-window enabled before starting drive."
            )
        if self.moving_window_anchor is None:
            self.moving_window_anchor = float(vehicle_coordinate)

    @property
    def moving_window_anchor_x(self) -> float | None:
        """Legacy x-window alias retained for the tracked-vehicle driver."""
        return self.moving_window_anchor

    @moving_window_anchor_x.setter
    def moving_window_anchor_x(self, value: float | None) -> None:
        self.moving_window_anchor = value

    def advance_moving_window(self, vehicle_coordinate: float) -> int:
        if not self.moving_window_enabled:
            return 0
        self.configure_moving_window_anchor(vehicle_coordinate)
        relative_travel = self.moving_window_direction * (
            float(vehicle_coordinate) - float(self.moving_window_anchor)
        )
        target_shift = max(0, int(math.floor(relative_travel / self.dx + 1.0e-9)))
        shift_layers = target_shift - int(self.moving_window_shift_cells)
        if shift_layers <= 0:
            return 0
        if shift_layers >= self.particle_layers_moving:
            raise RuntimeError(
                "Vehicle advanced farther than the complete MPM window in one update: "
                f"shift_layers={shift_layers}, window_layers={self.particle_layers_moving}"
            )

        old_ring_head = int(self.moving_window_ring_head)
        old_shift_cells = int(self.moving_window_shift_cells)
        self.window_old_ring_head[None] = old_ring_head
        self.window_old_shift_cells[None] = old_shift_cells
        self.window_shift_layers[None] = int(shift_layers)
        self._recycle_particle_layers_kernel()
        self._translate_window_grid_bounds_kernel()

        for layer_offset in range(int(shift_layers)):
            if self.moving_window_direction > 0:
                slot_layer = (
                    old_ring_head + layer_offset
                ) % self.particle_layers_moving
                global_layer = (
                    old_shift_cells + self.particle_layers_moving + layer_offset
                )
            else:
                slot_layer = (
                    old_ring_head - 1 - layer_offset
                ) % self.particle_layers_moving
                global_layer = -old_shift_cells - 1 - layer_offset
            template_layer = global_layer % self.moving_window_template_layers
            target_center = self.initial_soil_bounds_lo[self.moving_window_axis] + (
                float(global_layer) + 0.5
            ) * self.particle_layer_spacing
            layer_reference = self.window_template_x0_host[template_layer].copy()
            layer_reference[:, self.moving_window_axis] += target_center
            particle_ids = self._moving_window_layer_particle_ids(slot_layer)
            self.soil_x0_np[particle_ids, :] = layer_reference

        self.moving_window_ring_head = (
            old_ring_head + self.moving_window_direction * int(shift_layers)
        ) % self.particle_layers_moving
        self.moving_window_shift_cells = old_shift_cells + int(shift_layers)
        absolute_delta = (
            self.moving_window_direction
            * float(self.moving_window_shift_cells)
            * self.dx
        )
        axis = self.moving_window_axis
        self.domain_lo_np[axis] = self.initial_domain_lo_np[axis] + absolute_delta
        self.domain_hi_np[axis] = self.initial_domain_hi_np[axis] + absolute_delta
        self.soil_bounds_lo[axis] = self.initial_soil_bounds_lo[axis] + absolute_delta
        self.soil_bounds_hi[axis] = self.initial_soil_bounds_hi[axis] + absolute_delta
        return int(shift_layers)

    @ti.kernel
    def clear_grid(self):
        self.contact_particle_count[None] = 0
        self.track_contact_force[None] = ti.Vector.zero(ti.f32, 3)
        self.soil_contact_force[None] = ti.Vector.zero(ti.f32, 3)
        self.track_contact_power[None] = 0.0
        self.soil_contact_power[None] = 0.0
        self.soil_internal_work_increment[None] = 0.0
        self.contact_debug_max_force_norm[None] = 0.0
        self.contact_debug_max_force_z[None] = 0.0
        self.contact_debug_min_distance[None] = 1.0e30
        self.contact_debug_max_normal_force[None] = 0.0
        self.contact_debug_max_slip_speed[None] = 0.0
        for s in ti.static(range(2)):
            self.contact_particle_count_by_side[s] = 0
            self.track_contact_force_by_side[s] = ti.Vector.zero(ti.f32, 3)
        for p in range(self.max_track_patches):
            self.contact_particle_count_by_patch[p] = 0
            self.track_contact_force_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.soil_contact_force_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.track_contact_moment_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.track_contact_power_by_patch[p] = 0.0
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def clear_soil_grid(self):
        self.contact_particle_count[None] = 0
        self.track_contact_force[None] = ti.Vector.zero(ti.f32, 3)
        self.soil_contact_force[None] = ti.Vector.zero(ti.f32, 3)
        self.track_contact_power[None] = 0.0
        self.soil_contact_power[None] = 0.0
        self.soil_internal_work_increment[None] = 0.0
        self.contact_debug_max_force_norm[None] = 0.0
        self.contact_debug_max_force_z[None] = 0.0
        self.contact_debug_min_distance[None] = 1.0e30
        self.contact_debug_max_normal_force[None] = 0.0
        self.contact_debug_max_slip_speed[None] = 0.0
        for s in ti.static(range(2)):
            self.contact_particle_count_by_side[s] = 0
            self.track_contact_force_by_side[s] = ti.Vector.zero(ti.f32, 3)
        for p in range(self.max_track_patches):
            self.contact_particle_count_by_patch[p] = 0
            self.track_contact_force_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.soil_contact_force_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.track_contact_moment_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.track_contact_power_by_patch[p] = 0.0
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def p2g(self):
        for p in range(self.n_particles):
            xp = self.x[p] - ti.cast(self.domain_lo[None], self.real)
            base = (xp * self.inv_dx - 0.5).cast(int)
            fx = xp * self.inv_dx - base.cast(ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            sigma = self.p_stress[p]
            # p_stress stores compression-positive stress S=-sigma_Cauchy.  The
            # physical internal nodal force -V*sigma_Cauchy*grad(N) is therefore
            # +V*S*grad(N).
            affine = (
                self.dt * self.p_vol * 4.0 * self.inv_dx * self.inv_dx
            ) * sigma + self.p_mass * self.C[p]
            momentum = self.p_mass * self.v[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if 0 <= node[0] < self.n_grid_x and 0 <= node[1] < self.n_grid_y and 0 <= node[2] < self.n_grid_z:
                    dpos = (offset.cast(ti.f32) - fx) * self.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    w32 = ti.cast(weight, ti.f32)
                    self.grid_m[node] += w32 * ti.cast(self.p_mass, ti.f32)
                    self.grid_v[node] += w32 * ti.cast(momentum + affine @ dpos, ti.f32)

    @ti.kernel
    def normalize_grid(self):
        b = 2
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 1.0e-12:
                self.grid_v[I] = self.grid_v[I] / m + self.dt * (self.soil_gravity_scale * self.gravity[None])
                if I[2] <= self.bc_iz_min + 1:
                    self.grid_v[I] = ti.Vector.zero(ti.f32, 3)
                else:
                    if I[0] <= self.bc_ix_min + 1 or I[0] >= self.bc_ix_max - 1:
                        self.grid_v[I][0] = 0.0
                    if I[1] <= self.bc_iy_min + 1 or I[1] >= self.bc_iy_max - 1:
                        self.grid_v[I][1] = 0.0

                if I[0] < b and self.grid_v[I][0] < 0.0:
                    self.grid_v[I][0] = 0.0
                if I[0] >= self.n_grid_x - b and self.grid_v[I][0] > 0.0:
                    self.grid_v[I][0] = 0.0
                if I[1] < b and self.grid_v[I][1] < 0.0:
                    self.grid_v[I][1] = 0.0
                if I[1] >= self.n_grid_y - b and self.grid_v[I][1] > 0.0:
                    self.grid_v[I][1] = 0.0
                if I[2] < b and self.grid_v[I][2] < 0.0:
                    self.grid_v[I][2] = 0.0
                if I[2] >= self.n_grid_z - b and self.grid_v[I][2] > 0.0:
                    self.grid_v[I][2] = 0.0

    @ti.func
    def _barrier_normal_force(self, distance):
        r = ti.max(self.contact_barrier_radius[None], 1.0e-9)
        d = ti.max(distance, self.contact_barrier_min_distance[None])
        force = 0.0
        if d < r:
            bracket = 2.0 * ti.log(d / r) - r / d + 1.0
            force = self.contact_barrier_stiffness[None] * (d - r) * bracket
        return ti.max(force, 0.0)

    @ti.func
    def _smoothed_coulomb_factor(self, shear_disp):
        eps = ti.max(self.contact_slip_smoothing_distance[None], 1.0e-9)
        s = ti.abs(shear_disp) / eps
        factor = 1.0
        if s < 1.0:
            factor = -s * s + 2.0 * s
        return ti.max(0.0, ti.min(1.0, factor))

    @ti.kernel
    def apply_track_barrier_contact(self):
        r = self.contact_barrier_radius[None]
        for p in range(self.n_particles):
            pos = ti.cast(self.x[p], ti.f32)
            if (
                pos[0] < self.patch_query_lo[None][0]
                or pos[0] > self.patch_query_hi[None][0]
                or pos[1] < self.patch_query_lo[None][1]
                or pos[1] > self.patch_query_hi[None][1]
                or pos[2] < self.patch_query_lo[None][2]
                or pos[2] > self.patch_query_hi[None][2]
            ):
                continue
            best_patch = -1
            best_distance = r
            best_normal = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            best_u = 0.0
            best_v = 0.0
            candidate_count = self.active_patch_count[None]
            bin_x = 0
            bin_y = 0
            use_bin = 0
            if ti.static(self.contact_binning_enabled):
                bin_x = ti.cast(
                    ti.floor(
                        (pos[0] - self.domain_lo[None][0])
                        / self.contact_bin_size_value
                    ),
                    ti.i32,
                )
                bin_y = ti.cast(
                    ti.floor(
                        (pos[1] - self.domain_lo[None][1])
                        / self.contact_bin_size_value
                    ),
                    ti.i32,
                )
                if (
                    self.contact_bin_overflow[None] == 0
                    and 0 <= bin_x < self.contact_bin_nx
                    and 0 <= bin_y < self.contact_bin_ny
                ):
                    use_bin = 1
                    candidate_count = ti.min(
                        self.contact_bin_count[bin_x, bin_y],
                        self.contact_bin_capacity,
                    )
            for candidate_id in range(candidate_count):
                patch_id = self.active_patch_ids[candidate_id]
                if ti.static(self.contact_binning_enabled):
                    if use_bin == 1:
                        patch_id = self.contact_bin_patch_ids[
                            bin_x, bin_y, candidate_id
                        ]
                if self.patch_active[patch_id] == 0:
                    continue
                center = self.patch_center[patch_id]
                axis_long = self.patch_axis_long[patch_id]
                axis_width = self.patch_axis_width[patch_id]
                normal = self.patch_normal[patch_id]
                half_extent = self.patch_half_extent[patch_id]
                rel = pos - center
                u = rel.dot(axis_long)
                v = rel.dot(axis_width)
                w = rel.dot(normal)
                h_long = ti.max(half_extent[0], 0.0)
                h_width = ti.max(half_extent[1], 0.0)
                cu = ti.max(-h_long, ti.min(h_long, u))
                cv = ti.max(-h_width, ti.min(h_width, v))
                du = u - cu
                dv = v - cv
                distance = r + 1.0
                contact_normal = normal
                if ti.abs(u) <= h_long and ti.abs(v) <= h_width:
                    distance = w
                    contact_normal = normal
                else:
                    outward = du * axis_long + dv * axis_width + ti.max(w, 0.0) * normal
                    outward_norm = outward.norm()
                    if outward_norm > 1.0e-12:
                        distance = outward_norm
                        contact_normal = outward / outward_norm

                if distance < best_distance and w < r:
                    best_patch = patch_id
                    best_distance = distance
                    best_normal = contact_normal
                    best_u = cu
                    best_v = cv

            if best_patch >= 0:
                d_eff = ti.max(best_distance, self.contact_barrier_min_distance[None])
                normal_force = self._barrier_normal_force(d_eff)
                if normal_force > 0.0:
                    center = self.patch_center[best_patch]
                    prev_center = self.patch_previous_center[best_patch]
                    axis_long = self.patch_axis_long[best_patch]
                    prev_axis_long = self.patch_previous_axis_long[best_patch]
                    axis_width = self.patch_axis_width[best_patch]
                    prev_axis_width = self.patch_previous_axis_width[best_patch]
                    contact_point = center + best_u * axis_long + best_v * axis_width
                    previous_contact_point = prev_center + best_u * prev_axis_long + best_v * prev_axis_width
                    patch_velocity = (contact_point - previous_contact_point) / ti.max(self.track_keyframe_block_dt[None], self.dt)

                    particle_velocity = ti.cast(self.v[p], ti.f32)
                    relative_velocity = particle_velocity - patch_velocity
                    normal_component = relative_velocity.dot(best_normal)
                    tangent_velocity = relative_velocity - normal_component * best_normal
                    slip_speed = tangent_velocity.norm()
                    force_on_soil = normal_force * best_normal

                    if slip_speed > 1.0e-8:
                        if (
                            self.p_contact_patch[p] != best_patch
                            or self.p_contact_last_step[p] != self.contact_step_index[None] - 1
                        ):
                            self.p_contact_shear_disp[p] = 0.0
                        shear_disp = self.p_contact_shear_disp[p] + slip_speed * self.dt
                        self.p_contact_shear_disp[p] = shear_disp
                        self.p_contact_patch[p] = best_patch
                        self.p_contact_last_step[p] = self.contact_step_index[None]
                        tangent_dir = -tangent_velocity / slip_speed
                        tangent_force = self._smoothed_coulomb_factor(shear_disp) * self.contact_mu * normal_force
                        force_on_soil += tangent_force * tangent_dir
                    else:
                        self.p_contact_patch[p] = best_patch
                        self.p_contact_last_step[p] = self.contact_step_index[None]

                    xp = pos - self.domain_lo[None]
                    base = (xp * self.inv_dx - 0.5).cast(int)
                    fx = xp * self.inv_dx - base.cast(ti.f32)
                    weights = [
                        0.5 * (1.5 - fx) ** 2,
                        0.75 - (fx - 1.0) ** 2,
                        0.5 * (fx - 0.5) ** 2,
                    ]
                    contact_impulse = self.dt * force_on_soil
                    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                        offset = ti.Vector([i, j, k])
                        node = base + offset
                        if 0 <= node[0] < self.n_grid_x and 0 <= node[1] < self.n_grid_y and 0 <= node[2] < self.n_grid_z:
                            weight = weights[i][0] * weights[j][1] * weights[k][2]
                            self.grid_v[node] += ti.cast(weight * contact_impulse, ti.f32)

                    force_on_track = -force_on_soil
                    force_on_soil_f32 = force_on_soil
                    moment_on_track = (contact_point - center).cross(force_on_track)
                    side = self.patch_side[best_patch]
                    ti.atomic_add(self.contact_particle_count[None], 1)
                    ti.atomic_add(self.track_contact_force[None], force_on_track)
                    ti.atomic_add(self.soil_contact_force[None], force_on_soil_f32)
                    ti.atomic_add(self.track_contact_force_by_patch[best_patch], force_on_track)
                    ti.atomic_add(self.soil_contact_force_by_patch[best_patch], force_on_soil_f32)
                    ti.atomic_add(self.track_contact_moment_by_patch[best_patch], moment_on_track)
                    track_power = force_on_track.dot(patch_velocity)
                    soil_power = force_on_soil_f32.dot(particle_velocity)
                    ti.atomic_add(self.track_contact_power[None], track_power)
                    ti.atomic_add(self.soil_contact_power[None], soil_power)
                    ti.atomic_add(self.track_contact_power_by_patch[best_patch], track_power)
                    ti.atomic_add(self.contact_particle_count_by_patch[best_patch], 1)
                    if side >= 0 and side < 2:
                        ti.atomic_add(self.contact_particle_count_by_side[side], 1)
                        ti.atomic_add(self.track_contact_force_by_side[side], force_on_track)
                    ti.atomic_max(self.contact_debug_max_force_norm[None], force_on_track.norm())
                    ti.atomic_max(self.contact_debug_max_force_z[None], force_on_track[2])
                    ti.atomic_min(self.contact_debug_min_distance[None], best_distance)
                    ti.atomic_max(self.contact_debug_max_normal_force[None], normal_force)
                    ti.atomic_max(self.contact_debug_max_slip_speed[None], slip_speed)

    @ti.kernel
    def g2p(self):
        I3 = ti.Matrix.identity(self.real, 3)
        eps = 1.0e-7
        for p in range(self.n_particles):
            xp = self.x[p] - ti.cast(self.domain_lo[None], self.real)
            base = (xp * self.inv_dx - 0.5).cast(int)
            fx = xp * self.inv_dx - base.cast(ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            new_v = ti.Vector.zero(self.real, 3)
            new_C = ti.Matrix.zero(self.real, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if 0 <= node[0] < self.n_grid_x and 0 <= node[1] < self.n_grid_y and 0 <= node[2] < self.n_grid_z:
                    weight = w[i][0] * w[j][1] * w[k][2]
                    g_v = self.grid_v[node]
                    dpos = offset.cast(ti.f32) - fx
                    new_v += weight * g_v
                    new_C += 4.0 * self.inv_dx * weight * g_v.outer_product(dpos)

            deps = 0.5 * (new_C + new_C.transpose()) * self.dt
            sigma_old = self.p_stress[p]
            if ti.static(self.soil_constitutive_model_id_value == CONSTITUTIVE_MODEL_DRUCKER_PRAGER):
                # The velocity gradient gives extension-positive strain.  The
                # DP routine receives the conjugate compression-positive strain.
                self.p_stress[p] = drucker_prager_update(
                    self.p_stress[p],
                    -deps,
                    self.shear_modulus,
                    self.bulk_modulus,
                    self.dp_alpha,
                    self.dp_beta,
                    self.dp_k,
                    self.dp_i1_sign,
                    I3,
                    eps,
                )
            elif ti.static(
                self.soil_constitutive_model_id_value
                == CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP
            ):
                (
                    sigma_srsh,
                    epbar_new,
                    _plastic_increment,
                ) = srsh_modified_dp_update(
                    corotate_stress_increment(
                        self.p_stress[p], new_C, self.dt, I3
                    ),
                    -deps,
                    self.p_srsh_epbar[p],
                    self.dt,
                    self.shear_modulus,
                    self.bulk_modulus,
                    self.srsh_friction_slope,
                    self.srsh_dilatancy_slope,
                    self.srsh_cohesion_intercept,
                    self.srsh_rate_exponent,
                    self.srsh_rate_sensitivity,
                    self.srsh_saturation_increment,
                    self.srsh_source_rate0,
                    self.srsh_saturation_plastic_strain,
                    I3,
                    eps,
                )
                self.p_stress[p] = sigma_srsh
                self.p_srsh_epbar[p] = epbar_new
            elif ti.static(self.soil_constitutive_model_id_value == CONSTITUTIVE_MODEL_MOHR_COULOMB):
                # This material routine uses tension-positive Cauchy stress at
                # its interface; convert both entering and returned stress.
                sigma_tension_positive = mohr_coulomb_update(
                    -self.p_stress[p],
                    deps,
                    self.shear_modulus,
                    self.bulk_modulus,
                    self.mc_sin_phi,
                    self.mc_cos_phi,
                    self.mc_sin_psi,
                    self.mc_cohesion,
                    I3,
                    eps,
                )
                self.p_stress[p] = -sigma_tension_positive
            elif ti.static(self.soil_constitutive_model_id_value == CONSTITUTIVE_MODEL_MODIFIED_CAM_CLAY):
                sigma_tension_positive, pc_new = modified_cam_clay_update(
                    -self.p_stress[p],
                    deps,
                    self.p_pc[p],
                    self.shear_modulus,
                    self.bulk_modulus,
                    self.mcc_M,
                    self.mcc_lambda,
                    self.mcc_kappa,
                    self.mcc_e0,
                    self.mcc_min_pressure,
                    I3,
                    eps,
                )
                self.p_stress[p] = -sigma_tension_positive
                self.p_pc[p] = pc_new
            elif ti.static(self.soil_constitutive_model_id_value == CONSTITUTIVE_MODEL_PURE_WATER):
                self.p_stress[p] = pure_water_update(
                    self.p_stress[p],
                    deps,
                    self.dt,
                    self.water_bulk_modulus,
                    self.water_dynamic_viscosity,
                    self.water_cavitation_pressure,
                    I3,
                    eps,
                )
            sigma_new = self.p_stress[p]
            internal_work_density_increment = ti.cast(0.0, self.real)
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Physical stress power is sigma_Cauchy:deps = -S:deps when
                # the stored tensor S is compression-positive.
                internal_work_density_increment -= (
                    0.5 * (sigma_old[i, j] + sigma_new[i, j]) * deps[i, j]
                )
            ti.atomic_add(
                self.soil_internal_work_increment[None],
                ti.cast(self.p_vol * internal_work_density_increment, ti.f32),
            )
            self.v[p] = self.soil_damping * new_v
            self.C[p] = new_C
            self.x[p] += self.dt * self.v[p]

            for d in ti.static(range(3)):
                lo_d = ti.cast(self.domain_lo[None][d], self.real) + self.dx
                hi_d = ti.cast(self.domain_hi[None][d], self.real) - self.dx
                if self.x[p][d] < lo_d:
                    self.x[p][d] = lo_d
                    if self.v[p][d] < 0.0:
                        self.v[p][d] = 0.0
                if self.x[p][d] > hi_d:
                    self.x[p][d] = hi_d
                    if self.v[p][d] > 0.0:
                        self.v[p][d] = 0.0

    def substep(self):
        self.contact_step_index[None] = int(self.coupled_step)
        self.clear_grid()
        self.p2g()
        self.apply_track_barrier_contact()
        self.normalize_grid()
        self.g2p()
        self.coupled_step += 1

    def substep_geostatic(self):
        self.clear_soil_grid()
        self.p2g()
        self.normalize_grid()
        self.g2p()
        self.coupled_step += 1

    @ti.kernel
    def _compute_soil_mechanical_energy_kernel(self):
        self.soil_kinetic_energy[None] = 0.0
        self.soil_potential_energy[None] = 0.0
        gravity_z = -self.gravity[None][2]
        for p in range(self.n_particles):
            velocity = ti.cast(self.v[p], ti.f32)
            height = ti.cast(self.x[p][2], ti.f32)
            ti.atomic_add(self.soil_kinetic_energy[None], 0.5 * self.p_mass * velocity.dot(velocity))
            ti.atomic_add(self.soil_potential_energy[None], self.p_mass * gravity_z * height)

    def soil_mechanical_energy(self) -> tuple[float, float]:
        """Return particle kinetic and gravitational potential energies."""

        self._compute_soil_mechanical_energy_kernel()
        return float(self.soil_kinetic_energy[None]), float(self.soil_potential_energy[None])

    def initialize_geostatic_stress(self, k0: float, gravity_scale: float):
        ti.sync()
        pos = self.x.to_numpy().astype(np.float64)
        stress = np.zeros((self.n_particles, 3, 3), dtype=self.numpy_real)
        gamma = self.p_rho * abs(float(self.gravity[None][2])) * float(gravity_scale)
        depth = np.maximum(0.0, float(self.soil_bounds_hi[2]) - pos[:, 2])
        sigma_v = gamma * depth
        sigma_h = float(k0) * sigma_v
        stress[:, 0, 0] = sigma_h
        stress[:, 1, 1] = sigma_h
        stress[:, 2, 2] = sigma_v
        self.p_stress.from_numpy(stress)
        p_mean = (2.0 * sigma_h + sigma_v) / 3.0
        pc = np.maximum(self.mcc_pc0, 1.2 * np.maximum(self.mcc_min_pressure, p_mean))
        self.p_pc.from_numpy(pc.astype(self.numpy_real))
        self.v.from_numpy(np.zeros((self.n_particles, 3), dtype=self.numpy_real))
        self.C.from_numpy(np.zeros((self.n_particles, 3, 3), dtype=self.numpy_real))

    def scale_soil_kinematics(self, factor: float):
        self.soil_kinematic_scale[None] = float(max(0.0, min(1.0, factor)))
        self._scale_soil_kinematics_kernel()

    @ti.kernel
    def _scale_soil_kinematics_kernel(self):
        scale = self.soil_kinematic_scale[None]
        for p in range(self.n_particles):
            self.v[p] *= scale
            self.C[p] *= scale

    def get_soil_surface_z(self, percentile: float = 99.0) -> float:
        ti.sync()
        z = self.x.to_numpy()[:, 2]
        q = float(max(0.0, min(100.0, percentile)))
        return float(np.percentile(z, q))

    def soil_contact_force_total(self) -> np.ndarray:
        """Force exerted on MPM particles, accumulated independently."""

        return np.asarray(self.soil_contact_force[None], dtype=np.float64)

    def contact_power_pair(self) -> tuple[float, float]:
        """Return instantaneous rigid-side and soil-side interface powers."""

        return float(self.track_contact_power[None]), float(self.soil_contact_power[None])

    def contact_power_by_patch(self) -> np.ndarray:
        n = int(self.n_patches[None])
        return self.track_contact_power_by_patch.to_numpy()[:n]

    def soil_internal_work_increment_value(self) -> float:
        """Stress-work increment accumulated during the latest MPM substep."""

        return float(self.soil_internal_work_increment[None])

    def contact_particle_counts_by_side(self) -> np.ndarray:
        return self.contact_particle_count_by_side.to_numpy()

    def contact_forces_by_patch(self) -> np.ndarray:
        n = int(self.n_patches[None])
        return self.track_contact_force_by_patch.to_numpy()[:n]

    def soil_contact_forces_by_patch(self) -> np.ndarray:
        n = int(self.n_patches[None])
        return self.soil_contact_force_by_patch.to_numpy()[:n]

    def contact_moments_by_patch(self) -> np.ndarray:
        n = int(self.n_patches[None])
        return self.track_contact_moment_by_patch.to_numpy()[:n]

    def contact_particle_counts_by_patch(self) -> np.ndarray:
        n = int(self.n_patches[None])
        return self.contact_particle_count_by_patch.to_numpy()[:n]

    def track_patch_metadata(self) -> dict[str, np.ndarray]:
        n = int(self.n_patches[None])
        return {
            "center": self._patch_center_host[:n].copy(),
            "axis_long": self._patch_axis_long_host[:n].copy(),
            "axis_width": self._patch_axis_width_host[:n].copy(),
            "normal": self._patch_normal_host[:n].copy(),
            "half_extent": self._patch_half_extent_host[:n].copy(),
            "side_id": self._patch_side_host[:n].copy(),
            "shoe_id": self._patch_shoe_id_host[:n].copy(),
            "active": self.patch_active.to_numpy()[:n].copy(),
        }

    def active_patch_count_value(self) -> int:
        return int(self.active_patch_count[None])

    def contact_debug_summary(self) -> dict[str, float]:
        summary = {
            "max_contact_force_norm": float(self.contact_debug_max_force_norm[None]),
            "max_contact_force_z": float(self.contact_debug_max_force_z[None]),
            "min_contact_distance": float(self.contact_debug_min_distance[None]),
            "max_normal_force": float(self.contact_debug_max_normal_force[None]),
            "max_slip_speed": float(self.contact_debug_max_slip_speed[None]),
        }
        if self.contact_binning_enabled:
            summary.update(
                {
                    "contact_bin_overflow": float(self.contact_bin_overflow[None]),
                    "contact_bin_max_occupancy": float(
                        self.contact_bin_max_occupancy[None]
                    ),
                }
            )
        return summary

    @ti.kernel
    def reset_contact_force_average(self):
        self.contact_force_sample_count[None] = 0
        for s in ti.static(range(2)):
            self.contact_force_sum_by_side[s] = ti.Vector.zero(ti.f32, 3)
        for i in ti.static(range(5)):
            self.contact_macro_metrics[i] = 0.0
        for i in ti.static(range(2)):
            self.contact_macro_counts[i] = 0
        for p in range(self.max_track_patches):
            self.contact_force_sum_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.soil_contact_force_sum_by_patch[p] = ti.Vector.zero(ti.f32, 3)
            self.contact_moment_sum_by_patch[p] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def accumulate_contact_force_average(self):
        for s in ti.static(range(2)):
            self.contact_force_sum_by_side[s] += self.track_contact_force_by_side[s]
        for p in range(self.n_patches[None]):
            self.contact_force_sum_by_patch[p] += self.track_contact_force_by_patch[p]
            self.soil_contact_force_sum_by_patch[p] += self.soil_contact_force_by_patch[p]
            self.contact_moment_sum_by_patch[p] += self.track_contact_moment_by_patch[p]
        track_total = self.track_contact_force[None]
        soil_total = self.soil_contact_force[None]
        action_absolute = (track_total + soil_total).norm()
        action_scale = ti.max(ti.max(track_total.norm(), soil_total.norm()), 1.0e-12)
        self.contact_macro_metrics[0] += self.track_contact_power[None] * self.dt
        self.contact_macro_metrics[1] += self.soil_contact_power[None] * self.dt
        self.contact_macro_metrics[2] += self.soil_internal_work_increment[None]
        self.contact_macro_metrics[3] = ti.max(
            self.contact_macro_metrics[3], action_absolute
        )
        self.contact_macro_metrics[4] = ti.max(
            self.contact_macro_metrics[4], action_absolute / action_scale
        )
        self.contact_macro_counts[0] += self.contact_particle_count[None]
        self.contact_macro_counts[1] = ti.max(
            self.contact_macro_counts[1], self.contact_particle_count[None]
        )
        self.contact_force_sample_count[None] += 1

    def averaged_contact_forces_by_side(self) -> np.ndarray:
        count = max(1, int(self.contact_force_sample_count[None]))
        return self.contact_force_sum_by_side.to_numpy() / float(count)

    def averaged_track_contact_by_patch(self) -> dict[str, np.ndarray]:
        n = int(self.n_patches[None])
        count = max(1, int(self.contact_force_sample_count[None]))
        metadata = self.track_patch_metadata()
        metadata["force"] = self.contact_force_sum_by_patch.to_numpy()[:n] / float(count)
        metadata["soil_force"] = self.soil_contact_force_sum_by_patch.to_numpy()[:n] / float(count)
        metadata["moment"] = self.contact_moment_sum_by_patch.to_numpy()[:n] / float(count)
        metadata["contact_particles"] = self.contact_particle_counts_by_patch()
        return metadata

    def contact_macro_summary(self) -> dict[str, np.ndarray | float | int]:
        """Read one completed MBD macro-step ledger with a single synchronization."""

        ti.sync()
        n = int(self.n_patches[None])
        count = max(1, int(self.contact_force_sample_count[None]))
        metrics = self.contact_macro_metrics.to_numpy()
        counts = self.contact_macro_counts.to_numpy()
        return {
            "patch_forces": self.contact_force_sum_by_patch.to_numpy()[:n] / float(count),
            "soil_patch_forces": self.soil_contact_force_sum_by_patch.to_numpy()[:n]
            / float(count),
            "patch_moments": self.contact_moment_sum_by_patch.to_numpy()[:n] / float(count),
            "rigid_work": float(metrics[0]),
            "soil_work": float(metrics[1]),
            "internal_work": float(metrics[2]),
            "max_action_absolute": float(metrics[3]),
            "max_action_relative": float(metrics[4]),
            "contact_particle_samples": int(counts[0]),
            "max_contact_particles": int(counts[1]),
            "sample_count": int(self.contact_force_sample_count[None]),
        }

    @ti.kernel
    def _reduce_contact_feedback_kernel(self):
        for i in range(self.max_track_patches):
            self.reduced_module_wrench[i] = ti.Vector.zero(ti.f32, 6)
            self.reduced_axial_normal_load[i] = 0.0

        sample_count = ti.max(self.contact_force_sample_count[None], 1)
        inv_count = 1.0 / ti.cast(sample_count, ti.f32)
        for p in range(self.n_patches[None]):
            force = self.contact_force_sum_by_patch[p] * inv_count
            moment_at_patch = self.contact_moment_sum_by_patch[p] * inv_count
            module_id = self.patch_module_id[p]
            if 0 <= module_id < self.max_track_patches:
                moment_at_origin = moment_at_patch + self.patch_center[p].cross(force)
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        self.reduced_module_wrench[module_id][component],
                        force[component],
                    )
                    ti.atomic_add(
                        self.reduced_module_wrench[module_id][component + 3],
                        moment_at_origin[component],
                    )

            segment_id = self.patch_axial_segment_id[p]
            if 0 <= segment_id < self.max_track_patches:
                normal_load = ti.max(0.0, -force.dot(self.patch_normal[p]))
                ti.atomic_add(
                    self.reduced_axial_normal_load[segment_id], normal_load
                )

    def contact_macro_summary_reduced(
        self, *, include_patch_data: bool = False
    ) -> dict[str, np.ndarray | float | int]:
        """Return 17-body wrenches and axial loads with one device reduction.

        The default transfer is O(number of rigid bodies), not O(number of contact
        patches).  Per-patch arrays are copied only for requested VTK frames.
        """

        self._reduce_contact_feedback_kernel()
        ti.sync()
        module_count = self.contact_module_count_host
        segment_count = self.contact_axial_segment_count_host
        wrench = self.reduced_module_wrench.to_numpy()[:module_count]
        metrics = self.contact_macro_metrics.to_numpy()
        counts = self.contact_macro_counts.to_numpy()
        result: dict[str, np.ndarray | float | int] = {
            "module_forces": wrench[:, :3].copy(),
            "module_moments_about_origin": wrench[:, 3:].copy(),
            "axial_segment_normal_loads": self.reduced_axial_normal_load.to_numpy()[
                :segment_count
            ].copy(),
            "rigid_work": float(metrics[0]),
            "soil_work": float(metrics[1]),
            "internal_work": float(metrics[2]),
            "max_action_absolute": float(metrics[3]),
            "max_action_relative": float(metrics[4]),
            "contact_particle_samples": int(counts[0]),
            "max_contact_particles": int(counts[1]),
            "sample_count": int(self.contact_force_sample_count[None]),
            "contact_bin_overflow": int(self.contact_bin_overflow[None]),
            "contact_bin_max_occupancy": int(
                self.contact_bin_max_occupancy[None]
            ),
        }
        if include_patch_data:
            n = int(self.n_patches[None])
            count = max(1, int(self.contact_force_sample_count[None]))
            result["patch_forces"] = (
                self.contact_force_sum_by_patch.to_numpy()[:n] / float(count)
            )
            result["patch_moments"] = (
                self.contact_moment_sum_by_patch.to_numpy()[:n] / float(count)
            )
        return result

    def save_state(self, state_path: Path | str, soil_gravity_scale: float | None = None) -> None:
        ti.sync()
        state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        soil_reference = self.soil_x0_np.astype(np.float64)
        state_dict = dict(
            version=np.array([8], dtype=np.int32),
            n_particles=np.array([self.n_particles], dtype=np.int32),
            particle_shape=np.asarray(self.particle_shape, dtype=np.int32),
            coupled_step=np.array([int(self.coupled_step)], dtype=np.int64),
            soil_constitutive_model_id=np.array(
                [int(self.soil_constitutive_model_id_value)], dtype=np.int32
            ),
            srsh_model_revision=np.array([SRSH_MODEL_REVISION], dtype=np.int32),
            water_bulk_modulus=np.array([self.water_bulk_modulus], dtype=np.float64),
            water_dynamic_viscosity=np.array(
                [self.water_dynamic_viscosity], dtype=np.float64
            ),
            water_cavitation_pressure=np.array(
                [self.water_cavitation_pressure], dtype=np.float64
            ),
            srsh_parameters=np.array(
                [
                    self.soil_young_modulus,
                    self.soil_poisson_ratio,
                    self.soil_friction_angle_deg,
                    self.soil_dilatancy_angle_deg,
                    self.soil_reference_cohesion,
                    self.srsh_rate_exponent,
                    self.srsh_rate_sensitivity,
                    self.srsh_saturation_increment,
                    self.srsh_source_rate0,
                    self.srsh_saturation_plastic_strain,
                    self.p_rho,
                ],
                dtype=np.float64,
            ),
            mpm_precision=np.array([0 if self.mpm_precision == "f32" else 1], dtype=np.int32),
            stress_compression_positive=np.array([1], dtype=np.int32),
            grid_shape=np.array(self.grid_shape, dtype=np.int32),
            domain_lo=np.asarray(self.domain_lo_np, dtype=np.float64),
            domain_hi=np.asarray(self.domain_hi_np, dtype=np.float64),
            soil_bounds_lo=np.asarray(self.soil_bounds_lo, dtype=np.float64),
            soil_bounds_hi=np.asarray(self.soil_bounds_hi, dtype=np.float64),
            contact_barrier_stiffness=np.array([self.contact_barrier_stiffness_value], dtype=np.float64),
            contact_barrier_radius=np.array([self.contact_barrier_radius_value], dtype=np.float64),
            contact_barrier_min_distance_ratio=np.array(
                [self.contact_barrier_min_distance_ratio_value], dtype=np.float64
            ),
            contact_slip_smoothing_distance=np.array(
                [self.contact_slip_smoothing_distance_value], dtype=np.float64
            ),
            x=self.x.to_numpy().astype(np.float64),
            v=self.v.to_numpy().astype(np.float64),
            C=self.C.to_numpy().astype(np.float64),
            p_stress=self.p_stress.to_numpy().astype(np.float64),
            p_pc=self.p_pc.to_numpy().astype(np.float64),
            p_srsh_epbar=self.p_srsh_epbar.to_numpy().astype(np.float64),
            soil_x0_np=soil_reference,
            particle_contact_shear_disp=self.p_contact_shear_disp.to_numpy().astype(np.float32),
            particle_contact_patch=self.p_contact_patch.to_numpy().astype(np.int32),
            particle_contact_last_step=self.p_contact_last_step.to_numpy().astype(np.int32),
            track_contact_force=np.asarray(self.track_contact_force[None], dtype=np.float64),
            track_contact_force_by_side=self.track_contact_force_by_side.to_numpy().astype(np.float64),
            contact_particle_count=np.array([int(self.contact_particle_count[None])], dtype=np.int32),
            contact_particle_count_by_side=self.contact_particle_count_by_side.to_numpy().astype(np.int32),
            moving_window_enabled=np.array([int(self.moving_window_enabled)], dtype=np.int32),
            moving_window_axis=np.array([self.moving_window_axis], dtype=np.int32),
            moving_window_direction=np.array(
                [self.moving_window_direction], dtype=np.int32
            ),
            moving_window_shift_cells=np.array([int(self.moving_window_shift_cells)], dtype=np.int64),
            moving_window_ring_head=np.array([int(self.moving_window_ring_head)], dtype=np.int32),
            moving_window_anchor=np.array(
                [np.nan if self.moving_window_anchor is None else float(self.moving_window_anchor)],
                dtype=np.float64,
            ),
            moving_window_anchor_x=np.array(
                [np.nan if self.moving_window_anchor is None else float(self.moving_window_anchor)],
                dtype=np.float64,
            ),
            moving_window_template_ready=np.array(
                [int(self.moving_window_template_ready)], dtype=np.int32
            ),
        )
        if self.moving_window_enabled and self.moving_window_template_ready:
            state_dict.update(
                moving_window_template_layers=np.array(
                    [self.moving_window_template_layers], dtype=np.int32
                ),
                moving_window_template_x=self.window_template_x.to_numpy().astype(np.float64),
                moving_window_template_x0=self.window_template_x0_host.astype(np.float64),
                moving_window_template_stress=self.window_template_stress.to_numpy().astype(np.float64),
                moving_window_template_pc=self.window_template_pc.to_numpy().astype(np.float64),
                moving_window_template_srsh_epbar=self.window_template_srsh_epbar.to_numpy().astype(np.float64),
            )
        if soil_gravity_scale is not None:
            state_dict["soil_gravity_scale_saved"] = np.array([float(soil_gravity_scale)], dtype=np.float64)
        np.savez_compressed(state_path, **state_dict)

    def load_state(self, state_path: Path | str) -> None:
        state_path = Path(state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        data = np.load(state_path, allow_pickle=False)
        if (
            "stress_compression_positive" not in data
            or int(data["stress_compression_positive"][0]) != 1
        ):
            raise RuntimeError(
                "State stress convention is not compression-positive. Regenerate this state "
                "with the current MBD-MPM solver instead of mixing stress sign conventions."
            )
        if (
            "soil_constitutive_model_id" in data
            and int(data["soil_constitutive_model_id"][0])
            != self.soil_constitutive_model_id_value
        ):
            raise RuntimeError(
                "State constitutive model does not match the currently selected model."
            )
        if self.soil_constitutive_model_id_value == CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP:
            if (
                "srsh_model_revision" not in data
                or int(data["srsh_model_revision"][0]) != SRSH_MODEL_REVISION
            ):
                raise RuntimeError(
                    "This SRSH state uses an incompatible constitutive revision. Regenerate it."
                )
            required_srsh_keys = (
                "srsh_parameters",
                "p_srsh_epbar",
            )
            missing_srsh_keys = [key for key in required_srsh_keys if key not in data]
            if missing_srsh_keys:
                raise RuntimeError(
                    "SRSH state is missing constitutive history: "
                    + ", ".join(missing_srsh_keys)
                )
            expected_srsh_parameters = np.array(
                [
                    self.soil_young_modulus,
                    self.soil_poisson_ratio,
                    self.soil_friction_angle_deg,
                    self.soil_dilatancy_angle_deg,
                    self.soil_reference_cohesion,
                    self.srsh_rate_exponent,
                    self.srsh_rate_sensitivity,
                    self.srsh_saturation_increment,
                    self.srsh_source_rate0,
                    self.srsh_saturation_plastic_strain,
                    self.p_rho,
                ],
                dtype=np.float64,
            )
            loaded_srsh_parameters = np.asarray(data["srsh_parameters"], dtype=np.float64)
            if loaded_srsh_parameters.shape != expected_srsh_parameters.shape or not np.allclose(
                loaded_srsh_parameters,
                expected_srsh_parameters,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "SRSH state parameters do not match the current solver configuration."
                )
        if "n_particles" in data and int(data["n_particles"][0]) != self.n_particles:
            raise RuntimeError(
                f"State n_particles mismatch: file={int(data['n_particles'][0])}, solver={self.n_particles}"
            )
        if "particle_shape" in data:
            file_particle_shape = tuple(np.asarray(data["particle_shape"], dtype=np.int32).tolist())
            if file_particle_shape != self.particle_shape:
                raise RuntimeError(
                    f"State particle_shape mismatch: file={file_particle_shape}, solver={self.particle_shape}"
                )
        if "grid_shape" in data and tuple(np.asarray(data["grid_shape"], dtype=np.int32).tolist()) != self.grid_shape:
            raise RuntimeError(
                f"State grid_shape mismatch: file={tuple(np.asarray(data['grid_shape'], dtype=np.int32).tolist())}, "
                f"solver={self.grid_shape}"
            )
        file_moving_window = bool(int(data["moving_window_enabled"][0])) if "moving_window_enabled" in data else False
        if file_moving_window and not self.moving_window_enabled:
            raise RuntimeError("State uses a moving MPM window but the current solver has it disabled.")
        file_moving_window_axis = (
            int(data["moving_window_axis"][0]) if "moving_window_axis" in data else 0
        )
        file_moving_window_direction = (
            int(data["moving_window_direction"][0])
            if "moving_window_direction" in data
            else 1
        )
        if (
            file_moving_window
            and self.moving_window_enabled
            and file_moving_window_axis != self.moving_window_axis
        ):
            raise RuntimeError(
                "State moving-window axis does not match the current solver: "
                f"file={file_moving_window_axis}, solver={self.moving_window_axis}."
            )
        if (
            file_moving_window
            and self.moving_window_enabled
            and file_moving_window_direction != self.moving_window_direction
        ):
            raise RuntimeError(
                "State moving-window direction does not match the current solver: "
                f"file={file_moving_window_direction}, "
                f"solver={self.moving_window_direction}."
            )

        loaded_soil_lo = np.asarray(
            data["soil_bounds_lo"] if "soil_bounds_lo" in data else self.initial_soil_bounds_lo,
            dtype=np.float64,
        )
        loaded_soil_hi = np.asarray(
            data["soil_bounds_hi"] if "soil_bounds_hi" in data else self.initial_soil_bounds_hi,
            dtype=np.float64,
        )
        loaded_domain_lo = np.asarray(
            data["domain_lo"] if "domain_lo" in data else self.initial_domain_lo_np,
            dtype=np.float64,
        )
        loaded_domain_hi = np.asarray(
            data["domain_hi"] if "domain_hi" in data else self.initial_domain_hi_np,
            dtype=np.float64,
        )
        if self.moving_window_enabled:
            atol = max(1.0e-8, 1.0e-5 * self.dx)
            axis = self.moving_window_axis
            transverse_axes = [index for index in range(3) if index != axis]
            if not np.allclose(
                loaded_soil_hi - loaded_soil_lo,
                self.initial_soil_bounds_hi - self.initial_soil_bounds_lo,
                rtol=0.0,
                atol=atol,
            ):
                raise RuntimeError("State moving-window soil span does not match the current solver.")
            if not np.allclose(
                loaded_domain_hi - loaded_domain_lo,
                self.initial_domain_hi_np - self.initial_domain_lo_np,
                rtol=0.0,
                atol=atol,
            ):
                raise RuntimeError("State moving-window domain span does not match the current solver.")
            soil_delta = float(
                loaded_soil_lo[axis] - self.initial_soil_bounds_lo[axis]
            )
            domain_delta = float(
                loaded_domain_lo[axis] - self.initial_domain_lo_np[axis]
            )
            inferred_shift = int(
                round(soil_delta / (self.moving_window_direction * self.dx))
            )
            expected_delta = (
                self.moving_window_direction * inferred_shift * self.dx
            )
            if (
                inferred_shift < 0
                or abs(soil_delta - expected_delta) > atol
                or abs(domain_delta - soil_delta) > atol
            ):
                raise RuntimeError(
                    f"State moving-window {self.moving_window_axis_name} offset is not an "
                    "integer number of current grid cells."
                )
            if not np.allclose(
                loaded_soil_lo[transverse_axes],
                self.initial_soil_bounds_lo[transverse_axes],
                rtol=0.0,
                atol=atol,
            ):
                raise RuntimeError(
                    "State moving-window transverse soil bounds do not match the current solver."
                )
            if not np.allclose(
                loaded_domain_lo[transverse_axes],
                self.initial_domain_lo_np[transverse_axes],
                rtol=0.0,
                atol=atol,
            ):
                raise RuntimeError(
                    "State moving-window transverse domain bounds do not match the current solver."
                )
            saved_shift = int(data["moving_window_shift_cells"][0]) if "moving_window_shift_cells" in data else inferred_shift
            if saved_shift != inferred_shift:
                raise RuntimeError(
                    f"State moving-window shift mismatch: metadata={saved_shift}, bounds={inferred_shift}"
                )
            self.moving_window_shift_cells = saved_shift
            self.moving_window_ring_head = (
                int(data["moving_window_ring_head"][0])
                if "moving_window_ring_head" in data
                else (
                    self.moving_window_direction * saved_shift
                ) % self.particle_layers_moving
            )
            expected_ring_head = (
                self.moving_window_direction * saved_shift
            ) % self.particle_layers_moving
            if self.moving_window_ring_head != expected_ring_head:
                raise RuntimeError("State moving-window ring head is inconsistent with its cell shift.")
            self.domain_lo_np = loaded_domain_lo.copy()
            self.domain_hi_np = loaded_domain_hi.copy()
            self.soil_bounds_lo = loaded_soil_lo.copy()
            self.soil_bounds_hi = loaded_soil_hi.copy()
            self.domain_lo[None] = ti.Vector(self.domain_lo_np.astype(np.float32).tolist())
            self.domain_hi[None] = ti.Vector(self.domain_hi_np.astype(np.float32).tolist())
        else:
            for name, loaded, expected in (
                ("soil_bounds_lo", loaded_soil_lo, self.soil_bounds_lo),
                ("soil_bounds_hi", loaded_soil_hi, self.soil_bounds_hi),
                ("domain_lo", loaded_domain_lo, self.domain_lo_np),
                ("domain_hi", loaded_domain_hi, self.domain_hi_np),
            ):
                if not np.allclose(loaded, expected, rtol=0.0, atol=1.0e-9):
                    raise RuntimeError(
                        f"State {name} mismatch: file={loaded.tolist()}, solver={expected.tolist()}"
                    )

        self.x.from_numpy(data["x"].astype(self.numpy_real))
        self.v.from_numpy(data["v"].astype(self.numpy_real))
        self.C.from_numpy(data["C"].astype(self.numpy_real))
        self.p_stress.from_numpy(data["p_stress"].astype(self.numpy_real))
        if "p_pc" in data:
            self.p_pc.from_numpy(data["p_pc"].astype(self.numpy_real))
        else:
            self.p_pc.from_numpy(np.full((self.n_particles,), self.mcc_pc0, dtype=self.numpy_real))
        if "p_srsh_epbar" in data:
            self.p_srsh_epbar.from_numpy(data["p_srsh_epbar"].astype(self.numpy_real))
        else:
            self.p_srsh_epbar.fill(0.0)
        soil_reference = (
            data["soil_x0_np"].astype(self.numpy_real)
            if "soil_x0_np" in data
            else data["x"].astype(self.numpy_real)
        )
        self.soil_x0_np = soil_reference.astype(np.float64)
        self.x0.from_numpy(soil_reference)
        if self.moving_window_enabled:
            if "moving_window_anchor" in data:
                anchor = float(data["moving_window_anchor"][0])
            elif "moving_window_anchor_x" in data:
                anchor = float(data["moving_window_anchor_x"][0])
            else:
                anchor = math.nan
            self.moving_window_anchor = None if not math.isfinite(anchor) else anchor
            template_keys = [
                "moving_window_template_x",
                "moving_window_template_x0",
                "moving_window_template_stress",
                "moving_window_template_pc",
            ]
            if (
                self.soil_constitutive_model_id_value
                == CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP
            ):
                template_keys.extend(
                    [
                        "moving_window_template_srsh_epbar",
                    ]
                )
            template_ready = all(key in data for key in template_keys)
            if template_ready:
                file_template_layers = int(data["moving_window_template_layers"][0])
                if file_template_layers != self.moving_window_template_layers:
                    raise RuntimeError(
                        "State moving-window template layer count mismatch: "
                        f"file={file_template_layers}, solver={self.moving_window_template_layers}"
                    )
                self.window_template_x.from_numpy(data["moving_window_template_x"].astype(self.numpy_real))
                self.window_template_x0.from_numpy(data["moving_window_template_x0"].astype(self.numpy_real))
                self.window_template_x0_host = data["moving_window_template_x0"].astype(np.float64)
                self.window_template_stress.from_numpy(
                    data["moving_window_template_stress"].astype(self.numpy_real)
                )
                self.window_template_pc.from_numpy(data["moving_window_template_pc"].astype(self.numpy_real))
                if "moving_window_template_srsh_epbar" in data:
                    self.window_template_srsh_epbar.from_numpy(
                        data["moving_window_template_srsh_epbar"].astype(self.numpy_real)
                    )
            self.moving_window_template_ready = bool(template_ready)
        if "coupled_step" in data:
            self.coupled_step = int(data["coupled_step"][0])
        if (
            "particle_contact_shear_disp" in data
            and "particle_contact_patch" in data
            and "particle_contact_last_step" in data
            and tuple(data["particle_contact_shear_disp"].shape) == (self.n_particles,)
            and tuple(data["particle_contact_patch"].shape) == (self.n_particles,)
            and tuple(data["particle_contact_last_step"].shape) == (self.n_particles,)
        ):
            self.p_contact_shear_disp.from_numpy(data["particle_contact_shear_disp"].astype(np.float32))
            self.p_contact_patch.from_numpy(data["particle_contact_patch"].astype(np.int32))
            self.p_contact_last_step.from_numpy(data["particle_contact_last_step"].astype(np.int32))
        else:
            self.initialize_contact_history()
        if "track_contact_force" in data:
            self.track_contact_force[None] = data["track_contact_force"].astype(np.float64).tolist()
        if "track_contact_force_by_side" in data:
            self.track_contact_force_by_side.from_numpy(data["track_contact_force_by_side"].astype(np.float64))
        if "contact_particle_count" in data:
            self.contact_particle_count[None] = int(data["contact_particle_count"][0])
        if "contact_particle_count_by_side" in data:
            self.contact_particle_count_by_side.from_numpy(data["contact_particle_count_by_side"].astype(np.int32))

    def export_soil(self, out_dir: Path, step: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ti.sync()
        soil_pos = self.x.to_numpy()
        soil_vel = self.v.to_numpy()
        soil_stress = self.p_stress.to_numpy()
        soil_disp = soil_pos - self.soil_x0_np

        soil_pos = np.nan_to_num(soil_pos, nan=0.0, posinf=0.0, neginf=0.0)
        soil_vel = np.nan_to_num(soil_vel, nan=0.0, posinf=0.0, neginf=0.0)
        soil_stress = np.nan_to_num(soil_stress, nan=0.0, posinf=0.0, neginf=0.0)
        soil_disp = np.nan_to_num(soil_disp, nan=0.0, posinf=0.0, neginf=0.0)

        write_soil_vtk(out_dir / f"soil_{step:06d}.vtk", soil_pos, soil_vel, soil_disp, soil_stress)
