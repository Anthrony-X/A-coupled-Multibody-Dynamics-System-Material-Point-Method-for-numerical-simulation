from __future__ import annotations

import math
from dataclasses import dataclass

import taichi as ti


SRSH_MODEL_REVISION = 4
SRSH_REFERENCE_STRAIN_RATE = 1.0e-4


@dataclass(frozen=True)
class SrshModifiedDpParameters:
    shear_modulus: float
    bulk_modulus: float
    friction_slope: float
    dilatancy_slope: float
    cohesion_intercept: float
    rate_exponent: float
    rate_sensitivity: float
    saturation_increment: float
    reference_strain_rate: float
    saturation_plastic_strain: float


def make_srsh_modified_dp_parameters(
    soil_E: float,
    soil_nu: float,
    soil_phi_deg: float,
    soil_psi_deg: float,
    soil_cohesion: float,
    rate_exponent: float = 0.185,
    rate_sensitivity: float = 0.0,
    saturation_increment: float = 0.0,
    saturation_plastic_strain: float = 0.6,
) -> SrshModifiedDpParameters:
    young_modulus = float(soil_E)
    poisson_ratio = float(soil_nu)
    friction_angle_deg = float(soil_phi_deg)
    dilatancy_angle_deg = float(soil_psi_deg)
    reference_cohesion = float(soil_cohesion)
    if young_modulus <= 0.0:
        raise ValueError("SRSH Young's modulus must be positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("SRSH Poisson's ratio must be in (-1, 0.5)")
    if not 0.0 <= friction_angle_deg < 90.0:
        raise ValueError("SRSH friction angle must be in [0, 90) degrees")
    if not 0.0 <= dilatancy_angle_deg < 90.0:
        raise ValueError("SRSH dilatancy angle must be in [0, 90) degrees")
    if reference_cohesion < 0.0:
        raise ValueError("SRSH reference cohesion must be nonnegative")
    if float(rate_exponent) < 0.0:
        raise ValueError("SRSH rate exponent beta must be nonnegative")
    if float(rate_sensitivity) < 0.0:
        raise ValueError("SRSH rate sensitivity eta must be nonnegative")
    if float(saturation_increment) < 0.0:
        raise ValueError("SRSH saturation increment delta must be nonnegative")
    if float(saturation_plastic_strain) <= 0.0:
        raise ValueError("SRSH saturation plastic strain must be positive")

    friction_angle = math.radians(friction_angle_deg)
    sin_phi = math.sin(friction_angle)
    source_denominator = 3.0 - sin_phi
    saturation_normalization = (
        1.0
        + float(saturation_increment)
        - float(saturation_increment) / math.exp(1.0)
    )
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    bulk_modulus = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    return SrshModifiedDpParameters(
        shear_modulus=shear_modulus,
        bulk_modulus=bulk_modulus,
        friction_slope=6.0 * sin_phi / source_denominator,
        dilatancy_slope=math.tan(math.radians(dilatancy_angle_deg)),
        cohesion_intercept=(
            6.0
            * reference_cohesion
            / source_denominator
            / saturation_normalization
        ),
        rate_exponent=float(rate_exponent),
        rate_sensitivity=float(rate_sensitivity),
        saturation_increment=float(saturation_increment),
        reference_strain_rate=SRSH_REFERENCE_STRAIN_RATE,
        saturation_plastic_strain=float(saturation_plastic_strain),
    )


def srsh_yield_stress_scalar(
    equivalent_plastic_strain: float,
    equivalent_plastic_strain_rate: float,
    cohesion_intercept: float,
    rate_exponent: float,
    rate_sensitivity: float,
    saturation_increment: float,
    reference_strain_rate: float,
    saturation_plastic_strain: float,
) -> float:
    rate = max(
        abs(float(equivalent_plastic_strain_rate)),
        float(reference_strain_rate),
    )
    rate_ratio = rate / float(reference_strain_rate)
    rate_factor = (
        1.0 + float(rate_sensitivity) * rate_ratio ** float(rate_exponent)
    ) / (1.0 + float(rate_sensitivity))
    hardening_factor = (
        1.0
        + float(saturation_increment)
        - float(saturation_increment)
        * math.exp(
            -3.0
            * float(equivalent_plastic_strain)
            / float(saturation_plastic_strain)
        )
    )
    return float(cohesion_intercept) * rate_factor * hardening_factor


@ti.func
def _srsh_yield_stress_and_derivative(
    equivalent_plastic_strain_old,
    plastic_multiplier,
    dt,
    cohesion_intercept,
    rate_exponent,
    rate_sensitivity,
    saturation_increment,
    reference_strain_rate,
    saturation_plastic_strain,
):
    rate_ratio = ti.max(
        1.0,
        plastic_multiplier / (dt * reference_strain_rate),
    )
    rate_power = ti.pow(rate_ratio, rate_exponent)
    rate_factor = (
        1.0 + rate_sensitivity * rate_power
    ) / (1.0 + rate_sensitivity)
    rate_derivative = 0.0 * plastic_multiplier
    if rate_ratio > 1.0:
        rate_derivative = (
            rate_sensitivity
            * rate_exponent
            * ti.pow(rate_ratio, rate_exponent - 1.0)
            / (
                (1.0 + rate_sensitivity)
                * dt
                * reference_strain_rate
            )
        )

    epbar_new = equivalent_plastic_strain_old + plastic_multiplier
    hardening_exponential = ti.exp(
        -3.0 * epbar_new / saturation_plastic_strain
    )
    hardening_factor = (
        1.0
        + saturation_increment
        - saturation_increment * hardening_exponential
    )
    hardening_derivative = (
        3.0
        * saturation_increment
        * hardening_exponential
        / saturation_plastic_strain
    )
    model_yield_stress = (
        cohesion_intercept * rate_factor * hardening_factor
    )
    model_derivative = cohesion_intercept * (
        rate_derivative * hardening_factor
        + rate_factor * hardening_derivative
    )

    return model_yield_stress, model_derivative


@ti.func
def _srsh_consistency_residual(
    plastic_multiplier,
    pressure_trial,
    equivalent_stress_trial,
    equivalent_plastic_strain_old,
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
):
    deviatoric_measure = equivalent_stress_trial - (
        3.0 * shear_modulus * plastic_multiplier
    )
    equivalent_stress = ti.max(0.0, deviatoric_measure)
    pressure = pressure_trial + (
        bulk_modulus * dilatancy_slope * plastic_multiplier
    )
    yield_stress, yield_derivative = _srsh_yield_stress_and_derivative(
        equivalent_plastic_strain_old,
        plastic_multiplier,
        dt,
        cohesion_intercept,
        rate_exponent,
        rate_sensitivity,
        saturation_increment,
        reference_strain_rate,
        saturation_plastic_strain,
    )
    residual = equivalent_stress - friction_slope * pressure - yield_stress
    residual_derivative = (
        -friction_slope * bulk_modulus * dilatancy_slope
        - yield_derivative
    )
    if deviatoric_measure > 0.0:
        residual_derivative -= 3.0 * shear_modulus
    return residual, residual_derivative, equivalent_stress, pressure, yield_stress


@ti.func
def corotate_stress_increment(stress, velocity_gradient, dt, identity):
    """Rotate global stress over one step to match VUMAT's corotational frame."""

    spin_increment = 0.5 * (velocity_gradient - velocity_gradient.transpose()) * dt
    rotation_increment = (
        identity
        + spin_increment
        + 0.5 * spin_increment @ spin_increment
    )
    return rotation_increment @ stress @ rotation_increment.transpose()


@ti.func
def srsh_modified_dp_update(
    compression_positive_stress,
    compression_positive_strain_increment,
    equivalent_plastic_strain,
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
    eps,
):
    """Backward-Euler non-associated return for the revised SRSH yield surface.

    Stress and strain use the solver's compression-positive conjugate convention:
    f = q - M p - sigma_y and g = q - M_psi p.
    """

    tr_deps = compression_positive_strain_increment.trace()
    dev_deps = compression_positive_strain_increment - (
        tr_deps / 3.0
    ) * identity
    trial_stress = (
        compression_positive_stress
        + 2.0 * shear_modulus * dev_deps
        + bulk_modulus * tr_deps * identity
    )
    pressure_trial = trial_stress.trace() / 3.0
    deviatoric_trial = trial_stress - pressure_trial * identity
    q_trial = ti.sqrt(1.5 * (deviatoric_trial * deviatoric_trial).sum() + eps)

    (
        trial_residual,
        _trial_derivative,
        _trial_q,
        _trial_pressure,
        _trial_yield_stress,
    ) = _srsh_consistency_residual(
        0.0,
        pressure_trial,
        q_trial,
        equivalent_plastic_strain,
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
    )

    stress_new = trial_stress
    plastic_multiplier = 0.0 * q_trial
    if trial_residual > 0.0:
        lower = 0.0 * q_trial
        upper = q_trial / (3.0 * shear_modulus + eps)
        upper = ti.max(
            upper,
            trial_residual
            / (
                3.0 * shear_modulus
                + friction_slope * bulk_modulus * ti.abs(dilatancy_slope)
                + eps
            ),
        )
        upper_residual = trial_residual
        for _ in range(24):
            if upper_residual > 0.0:
                (
                    upper_residual,
                    _upper_derivative,
                    _upper_q,
                    _upper_pressure,
                    _upper_yield,
                ) = _srsh_consistency_residual(
                    upper,
                    pressure_trial,
                    q_trial,
                    equivalent_plastic_strain,
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
                )
                if upper_residual > 0.0:
                    upper = 2.0 * upper + eps

        plastic_multiplier = 0.5 * (lower + upper)
        young_modulus = (
            9.0
            * bulk_modulus
            * shear_modulus
            / (3.0 * bulk_modulus + shear_modulus + eps)
        )
        tolerance = young_modulus * 1.0e-9
        residual = trial_residual
        for _ in range(32):
            if ti.abs(residual) > tolerance:
                (
                    residual,
                    residual_derivative,
                    _candidate_q,
                    _candidate_pressure,
                    _candidate_yield,
                ) = _srsh_consistency_residual(
                    plastic_multiplier,
                    pressure_trial,
                    q_trial,
                    equivalent_plastic_strain,
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
                )
                if ti.abs(residual) > tolerance:
                    if residual > 0.0:
                        lower = plastic_multiplier
                    else:
                        upper = plastic_multiplier

                    newton_candidate = plastic_multiplier - residual / (
                        residual_derivative - eps
                    )
                    if newton_candidate <= lower or newton_candidate >= upper:
                        newton_candidate = 0.5 * (lower + upper)
                    plastic_multiplier = newton_candidate

        (
            _final_residual,
            _final_derivative,
            q_new,
            pressure_new,
            _final_yield_stress,
        ) = _srsh_consistency_residual(
            plastic_multiplier,
            pressure_trial,
            q_trial,
            equivalent_plastic_strain,
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
        )
        deviatoric_scale = q_new / (q_trial + eps)
        stress_new = pressure_new * identity + deviatoric_scale * deviatoric_trial

    return (
        stress_new,
        equivalent_plastic_strain + plastic_multiplier,
        plastic_multiplier,
    )
