from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolutionConfig:
    name: str
    duration: float
    mpm_dt: float
    mbd_dt: float
    save_interval: float
    soil_particles: tuple[int, int, int]
    n_grid_x: int
    head_patch_grid: tuple[int, int]
    visual_segments: int

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
    pivot: tuple[float, float, float]
    link1_length: float
    link1_width: float
    link1_thickness: float
    link1_mass: float
    link1_inertia: tuple[float, float, float]
    link2_rod_length: float
    link2_rod_width: float
    link2_rod_thickness: float
    link2_rod_mass: float
    hammer_center_from_joint: float
    hammer_radius: float
    hammer_width: float
    hammer_mass: float
    link2_inertia: tuple[float, float, float]
    joint_ground_damping: float
    joint_links_damping: float
    initial_angle1_deg: float
    initial_angle2_deg: float
    initial_rate1_deg_s: float
    initial_rate2_deg_s: float


@dataclass(frozen=True)
class SoilConfig:
    bounds_lo: tuple[float, float, float]
    bounds_hi: tuple[float, float, float]
    domain_lo: tuple[float, float, float]
    domain_hi: tuple[float, float, float]
    density: float
    constitutive_model: str
    young_modulus: float
    poisson_ratio: float
    friction_angle_deg: float
    dilation_angle_deg: float
    cohesion: float
    water_density: float
    water_bulk_modulus: float
    water_dynamic_viscosity: float
    water_cavitation_pressure: float
    contact_mu: float
    contact_barrier_stiffness: float | None
    contact_barrier_min_distance_ratio: float
    contact_activation_height: float
    particle_jitter_ratio: float
    seed: int


@dataclass(frozen=True)
class ValidationLimits:
    joint_position_error_m: float
    joint_axis_error_rad: float
    action_reaction_relative: float
    generalized_power_relative: float
    mbd_energy_relative: float
    total_energy_relative: float


@dataclass(frozen=True)
class CaseConfig:
    resolution: ResolutionConfig
    model: ModelConfig
    soil: SoilConfig
    limits: ValidationLimits


def _tuple(values: Any, length: int, label: str, cast=float) -> tuple:
    result = tuple(cast(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values, got {result!r}")
    return result


def load_case_config(path: Path | str, preset: str) -> CaseConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if preset not in {"fine", "standard", "smoke"}:
        raise ValueError(f"Unknown preset {preset!r}; choose fine, standard, or smoke")
    resolution_raw = raw[preset]
    model_raw = raw["model"]
    soil_raw = raw["soil"]
    limits_raw = raw["limits"]

    resolution = ResolutionConfig(
        name=preset,
        duration=float(resolution_raw["duration"]),
        mpm_dt=float(resolution_raw["mpm_dt"]),
        mbd_dt=float(resolution_raw["mbd_dt"]),
        save_interval=float(resolution_raw["save_interval"]),
        soil_particles=_tuple(resolution_raw["soil_particles"], 3, "soil_particles", int),
        n_grid_x=int(resolution_raw["n_grid_x"]),
        head_patch_grid=_tuple(resolution_raw["head_patch_grid"], 2, "head_patch_grid", int),
        visual_segments=int(resolution_raw["visual_segments"]),
    )
    model = ModelConfig(
        gravity=float(model_raw["gravity"]),
        pivot=_tuple(model_raw["pivot"], 3, "pivot"),
        link1_length=float(model_raw["link1_length"]),
        link1_width=float(model_raw["link1_width"]),
        link1_thickness=float(model_raw["link1_thickness"]),
        link1_mass=float(model_raw["link1_mass"]),
        link1_inertia=_tuple(model_raw["link1_inertia"], 3, "link1_inertia"),
        link2_rod_length=float(model_raw["link2_rod_length"]),
        link2_rod_width=float(model_raw["link2_rod_width"]),
        link2_rod_thickness=float(model_raw["link2_rod_thickness"]),
        link2_rod_mass=float(model_raw["link2_rod_mass"]),
        hammer_center_from_joint=float(model_raw["hammer_center_from_joint"]),
        hammer_radius=float(model_raw["hammer_radius"]),
        hammer_width=float(model_raw["hammer_width"]),
        hammer_mass=float(model_raw["hammer_mass"]),
        link2_inertia=_tuple(model_raw["link2_inertia"], 3, "link2_inertia"),
        joint_ground_damping=float(model_raw["joint_ground_damping"]),
        joint_links_damping=float(model_raw["joint_links_damping"]),
        initial_angle1_deg=float(model_raw["initial_angle1_deg"]),
        initial_angle2_deg=float(model_raw["initial_angle2_deg"]),
        initial_rate1_deg_s=float(model_raw["initial_rate1_deg_s"]),
        initial_rate2_deg_s=float(model_raw["initial_rate2_deg_s"]),
    )
    contact_barrier_stiffness_raw = soil_raw["contact_barrier_stiffness"]
    soil = SoilConfig(
        bounds_lo=_tuple(soil_raw["bounds_lo"], 3, "bounds_lo"),
        bounds_hi=_tuple(soil_raw["bounds_hi"], 3, "bounds_hi"),
        domain_lo=_tuple(soil_raw["domain_lo"], 3, "domain_lo"),
        domain_hi=_tuple(soil_raw["domain_hi"], 3, "domain_hi"),
        density=float(soil_raw["density"]),
        constitutive_model=str(soil_raw.get("constitutive_model", "drucker-prager")),
        young_modulus=float(soil_raw["young_modulus"]),
        poisson_ratio=float(soil_raw["poisson_ratio"]),
        friction_angle_deg=float(soil_raw["friction_angle_deg"]),
        dilation_angle_deg=float(soil_raw["dilation_angle_deg"]),
        cohesion=float(soil_raw["cohesion"]),
        water_density=float(soil_raw.get("water_density", 1000.0)),
        water_bulk_modulus=float(soil_raw.get("water_bulk_modulus", 2.2e9)),
        water_dynamic_viscosity=float(
            soil_raw.get("water_dynamic_viscosity", 1.002e-3)
        ),
        water_cavitation_pressure=float(
            soil_raw.get("water_cavitation_pressure", 0.0)
        ),
        contact_mu=float(soil_raw["contact_mu"]),
        contact_barrier_stiffness=(
            None
            if contact_barrier_stiffness_raw is None
            else float(contact_barrier_stiffness_raw)
        ),
        contact_barrier_min_distance_ratio=float(soil_raw["contact_barrier_min_distance_ratio"]),
        contact_activation_height=float(soil_raw["contact_activation_height"]),
        particle_jitter_ratio=float(soil_raw["particle_jitter_ratio"]),
        seed=int(soil_raw["seed"]),
    )
    limits = ValidationLimits(**{key: float(value) for key, value in limits_raw.items()})

    if any(value <= 0 for value in resolution.soil_particles):
        raise ValueError("All soil particle counts must be positive")
    if any(value <= 0 for value in resolution.head_patch_grid):
        raise ValueError("Both hammer contact-patch counts must be positive")
    if model.hammer_radius <= 0 or model.hammer_width <= 0:
        raise ValueError("hammer_radius and hammer_width must be positive")
    if any(value <= 0 for value in model.link1_inertia + model.link2_inertia):
        raise ValueError("Rigid-body principal inertias must be positive")
    _ = resolution.mpm_substeps
    return CaseConfig(resolution=resolution, model=model, soil=soil, limits=limits)
