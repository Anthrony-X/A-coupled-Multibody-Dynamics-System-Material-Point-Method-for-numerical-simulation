from __future__ import annotations

import math
from dataclasses import dataclass

import taichi as ti


@dataclass(frozen=True)
class DruckerPragerParameters:
    shear_modulus: float
    bulk_modulus: float
    alpha: float
    beta: float
    k: float


def make_drucker_prager_parameters(
    soil_E: float,
    soil_nu: float,
    soil_phi_deg: float,
    soil_psi_deg: float,
    soil_cohesion: float,
) -> DruckerPragerParameters:
    shear_modulus = float(soil_E) / (2.0 * (1.0 + float(soil_nu)))
    bulk_modulus = float(soil_E) / (3.0 * (1.0 - 2.0 * float(soil_nu)))

    phi = math.radians(float(soil_phi_deg))
    psi = math.radians(float(soil_psi_deg))
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_psi = math.sin(psi)
    denom = math.sqrt(3.0) * (3.0 - sin_phi) + 1.0e-12
    denom_psi = math.sqrt(3.0) * (3.0 - sin_psi) + 1.0e-12
    return DruckerPragerParameters(
        shear_modulus=shear_modulus,
        bulk_modulus=bulk_modulus,
        alpha=float(2.0 * sin_phi / denom),
        beta=float(2.0 * sin_psi / denom_psi),
        k=float(6.0 * float(soil_cohesion) * cos_phi / denom),
    )


@ti.func
def drucker_prager_update(
    stress,
    strain_increment,
    shear_modulus,
    bulk_modulus,
    alpha,
    beta,
    k,
    i1_sign,
    identity,
    eps,
):
    tr_deps = strain_increment.trace()
    dev_deps = strain_increment - (tr_deps / 3.0) * identity
    sigma_trial = stress + 2.0 * shear_modulus * dev_deps + bulk_modulus * tr_deps * identity

    I1_trial = sigma_trial.trace()
    s_trial = sigma_trial - (I1_trial / 3.0) * identity
    sqrt_J2_trial = ti.sqrt(0.5 * (s_trial * s_trial).sum() + eps)
    f_trial = sqrt_J2_trial + i1_sign * alpha * I1_trial - k

    sigma_new = sigma_trial
    if f_trial > 0.0:
        dg = f_trial / (shear_modulus + 9.0 * bulk_modulus * alpha * beta + eps)
        s_scale = ti.max(0.0, 1.0 - shear_modulus * dg / sqrt_J2_trial)
        s_new = s_scale * s_trial
        I1_new = I1_trial - i1_sign * 9.0 * bulk_modulus * beta * dg
        sigma_new = s_new + (I1_new / 3.0) * identity

    return sigma_new
