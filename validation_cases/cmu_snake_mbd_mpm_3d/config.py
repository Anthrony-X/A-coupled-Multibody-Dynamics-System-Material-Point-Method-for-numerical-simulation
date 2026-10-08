from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from validation_cases.cmu_snake_geometry_3d.geometry import SnakeGeometryConfig


@dataclass(frozen=True)
class ResolutionConfig:
    name: str
    duration: float
    mpm_dt: float
    mbd_dt: float
    save_interval: float
    soil_particles: tuple[int, int, int]
    n_grid_x: int
    circumferential_divisions: int
    axial_patch_divisions: int
    cap_radial_divisions: int
    visual_segments: int
    contact_barrier_stiffness_scale: float

    @property
    def mpm_substeps(self) -> int:
        ratio = self.mbd_dt / self.mpm_dt
        count = int(round(ratio))
        if count < 1 or abs(ratio - count) > 1.0e-10:
            raise ValueError(
                f"mbd_dt/mpm_dt must be a positive integer, got {self.mbd_dt}/{self.mpm_dt}"
            )
        return count


@dataclass(frozen=True)
class ModelConfig:
    gravity: float
    total_length_m: float
    total_mass_kg: float
    radius_m: float
    visual_radius_m: float
    initial_clearance_m: float
    horizontal_wave_amplitude_m: float
    vertical_wave_amplitude_m: float
    wavelength_ratio: float
    body_wave_phase_deg: float
    gait_frequency_hz: float
    settle_time_s: float
    ramp_time_s: float
    motor_kp_Nm_rad: float
    motor_kd_Nm_s_rad: float
    motor_torque_limit_Nm: float

    @property
    def module_count(self) -> int:
        return 17

    @property
    def joint_count(self) -> int:
        return 16

    @property
    def module_pitch_m(self) -> float:
        return self.total_length_m / self.module_count

    @property
    def module_mass_kg(self) -> float:
        return self.total_mass_kg / self.module_count

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_ratio * self.total_length_m


@dataclass(frozen=True)
class SoilConfig:
    bounds_lo: tuple[float, float, float]
    bounds_hi: tuple[float, float, float]
    domain_lo: tuple[float, float, float]
    domain_hi: tuple[float, float, float]
    moving_window_enabled: bool
    moving_window_axis: str
    moving_window_direction: int
    moving_window_template_layers: int
    density: float
    constitutive_model: str
    young_modulus: float
    poisson_ratio: float
    friction_angle_deg: float
    dilation_angle_deg: float
    cohesion: float
    contact_mu: float
    contact_barrier_stiffness: float | None
    contact_barrier_min_distance_ratio: float
    contact_activation_height: float
    particle_jitter_ratio: float
    seed: int

    @property
    def surface_z(self) -> float:
        return self.bounds_hi[2]


@dataclass(frozen=True)
class ValidationConfig:
    contact_force_threshold_N: float
    max_joint_position_error_m: float
    max_joint_axis_error_rad: float
    max_action_reaction_relative: float
    max_tracking_rms_deg: float
    steady_state_delay_s: float


@dataclass(frozen=True)
class CaseConfig:
    resolution: ResolutionConfig
    model: ModelConfig
    soil: SoilConfig
    validation: ValidationConfig

    def geometry_config(self) -> SnakeGeometryConfig:
        return SnakeGeometryConfig(
            total_length_m=self.model.total_length_m,
            total_mass_kg=self.model.total_mass_kg,
            radius_m=self.model.radius_m,
            visual_radius_m=self.model.visual_radius_m,
            circumferential_divisions=self.resolution.circumferential_divisions,
            axial_patch_divisions=self.resolution.axial_patch_divisions,
            cap_radial_divisions=self.resolution.cap_radial_divisions,
            horizontal_wave_amplitude_m=self.model.horizontal_wave_amplitude_m,
            vertical_wave_amplitude_m=self.model.vertical_wave_amplitude_m,
            wavelength_m=self.model.wavelength_m,
            body_wave_phase_offset_rad=math.radians(self.model.body_wave_phase_deg),
        )

    @property
    def contact_patch_count(self) -> int:
        c = self.resolution.circumferential_divisions
        axial = self.model.module_count * self.resolution.axial_patch_divisions * c
        cap_per_end = max(8, c // 2) + (self.resolution.cap_radial_divisions - 1) * c
        return axial + 2 * cap_per_end


def _tuple(values: Any, length: int, label: str, cast=float) -> tuple:
    result = tuple(cast(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values, got {result!r}")
    return result


def load_case_config(path: Path | str, preset: str) -> CaseConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if preset not in {"smoke", "standard", "fine"}:
        raise ValueError(f"Unknown preset {preset!r}; choose smoke, standard, or fine")

    resolution_raw = raw[preset]
    model_raw = raw["model"]
    soil_raw = raw["soil"]
    validation_raw = raw["validation"]
    resolution = ResolutionConfig(
        name=preset,
        duration=float(resolution_raw["duration"]),
        mpm_dt=float(resolution_raw["mpm_dt"]),
        mbd_dt=float(resolution_raw["mbd_dt"]),
        save_interval=float(resolution_raw["save_interval"]),
        soil_particles=_tuple(resolution_raw["soil_particles"], 3, "soil_particles", int),
        n_grid_x=int(resolution_raw["n_grid_x"]),
        circumferential_divisions=int(resolution_raw["circumferential_divisions"]),
        axial_patch_divisions=int(resolution_raw["axial_patch_divisions"]),
        cap_radial_divisions=int(resolution_raw["cap_radial_divisions"]),
        visual_segments=int(resolution_raw["visual_segments"]),
        contact_barrier_stiffness_scale=float(
            resolution_raw["contact_barrier_stiffness_scale"]
        ),
    )
    model = ModelConfig(**{key: float(value) for key, value in model_raw.items()})
    barrier = soil_raw.get("contact_barrier_stiffness")
    soil = SoilConfig(
        bounds_lo=_tuple(soil_raw["bounds_lo"], 3, "bounds_lo"),
        bounds_hi=_tuple(soil_raw["bounds_hi"], 3, "bounds_hi"),
        domain_lo=_tuple(soil_raw["domain_lo"], 3, "domain_lo"),
        domain_hi=_tuple(soil_raw["domain_hi"], 3, "domain_hi"),
        moving_window_enabled=bool(soil_raw.get("moving_window_enabled", False)),
        moving_window_axis=str(soil_raw.get("moving_window_axis", "x")).strip().lower(),
        moving_window_direction=int(soil_raw.get("moving_window_direction", 1)),
        moving_window_template_layers=int(
            soil_raw.get("moving_window_template_layers", 8)
        ),
        density=float(soil_raw["density"]),
        constitutive_model=str(soil_raw["constitutive_model"]),
        young_modulus=float(soil_raw["young_modulus"]),
        poisson_ratio=float(soil_raw["poisson_ratio"]),
        friction_angle_deg=float(soil_raw["friction_angle_deg"]),
        dilation_angle_deg=float(soil_raw["dilation_angle_deg"]),
        cohesion=float(soil_raw["cohesion"]),
        contact_mu=float(soil_raw["contact_mu"]),
        contact_barrier_stiffness=None if barrier is None else float(barrier),
        contact_barrier_min_distance_ratio=float(
            soil_raw["contact_barrier_min_distance_ratio"]
        ),
        contact_activation_height=float(soil_raw["contact_activation_height"]),
        particle_jitter_ratio=float(soil_raw["particle_jitter_ratio"]),
        seed=int(soil_raw["seed"]),
    )
    validation = ValidationConfig(
        **{key: float(value) for key, value in validation_raw.items()}
    )
    config = CaseConfig(
        resolution=resolution,
        model=model,
        soil=soil,
        validation=validation,
    )
    _validate(config)
    return config


def _validate(config: CaseConfig) -> None:
    r = config.resolution
    m = config.model
    s = config.soil
    if any(value <= 0 for value in r.soil_particles):
        raise ValueError("All soil particle counts must be positive")
    if r.n_grid_x < 16:
        raise ValueError("n_grid_x must be at least 16")
    if r.contact_barrier_stiffness_scale <= 0.0:
        raise ValueError("contact_barrier_stiffness_scale must be positive")
    _ = r.mpm_substeps
    _ = config.geometry_config()
    if m.total_length_m <= 0.0 or m.total_mass_kg <= 0.0 or m.radius_m <= 0.0:
        raise ValueError("Robot dimensions and mass must be positive")
    if m.motor_torque_limit_Nm <= 0.0 or m.motor_kp_Nm_rad <= 0.0:
        raise ValueError("Motor limit and proportional gain must be positive")
    if m.horizontal_wave_amplitude_m < 0.0 or m.vertical_wave_amplitude_m < 0.0:
        raise ValueError("Backbone wave amplitudes must be non-negative")
    if m.wavelength_ratio <= 0.0:
        raise ValueError("Backbone wavelength ratio must be positive")
    if m.settle_time_s < 0.0 or m.ramp_time_s <= 0.0 or m.gait_frequency_hz <= 0.0:
        raise ValueError("Gait timing values are invalid")
    if any(hi <= lo for lo, hi in zip(s.domain_lo, s.domain_hi)):
        raise ValueError("Invalid MPM domain bounds")
    if any(hi <= lo for lo, hi in zip(s.bounds_lo, s.bounds_hi)):
        raise ValueError("Invalid soil bounds")
    if s.moving_window_axis not in {"x", "y"}:
        raise ValueError("moving_window_axis must be 'x' or 'y'")
    if s.moving_window_direction not in {-1, 1}:
        raise ValueError("moving_window_direction must be -1 or +1")
    if s.moving_window_template_layers <= 0:
        raise ValueError("moving_window_template_layers must be positive")
    if not all(
        domain_lo <= soil_lo < soil_hi <= domain_hi
        for domain_lo, soil_lo, soil_hi, domain_hi in zip(
            s.domain_lo, s.bounds_lo, s.bounds_hi, s.domain_hi
        )
    ):
        raise ValueError("Soil bounds must lie inside the MPM domain")
