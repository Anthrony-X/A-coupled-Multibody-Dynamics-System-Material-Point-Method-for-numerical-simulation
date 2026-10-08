from __future__ import annotations

import math
from dataclasses import dataclass

import taichi as ti


MODEL_NAME = "modified-cam-clay"


@dataclass(frozen=True)
class ModifiedCamClayParameters:
    shear_modulus: float
    bulk_modulus: float
    critical_state_m: float
    virgin_compression_lambda: float
    recompression_kappa: float
    initial_void_ratio: float
    initial_preconsolidation_pressure: float
    min_pressure: float


def make_modified_cam_clay_parameters(
    soil_E: float,
    soil_nu: float,
    soil_phi_deg: float,
    critical_state_m: float | None = None,
    virgin_compression_lambda: float = 0.12,
    recompression_kappa: float = 0.02,
    initial_void_ratio: float = 0.8,
    initial_preconsolidation_pressure: float = 1.0e5,
    min_pressure: float = 1.0,
) -> ModifiedCamClayParameters:
    sin_phi = math.sin(math.radians(float(soil_phi_deg)))
    m_default = 6.0 * sin_phi / (3.0 - sin_phi + 1.0e-12)
    mcc_m = m_default if critical_state_m is None else float(critical_state_m)
    kappa = max(1.0e-6, float(recompression_kappa))
    lamb = max(kappa + 1.0e-6, float(virgin_compression_lambda))
    return ModifiedCamClayParameters(
        shear_modulus=float(soil_E) / (2.0 * (1.0 + float(soil_nu))),
        bulk_modulus=float(soil_E) / (3.0 * (1.0 - 2.0 * float(soil_nu))),
        critical_state_m=max(1.0e-6, mcc_m),
        virgin_compression_lambda=lamb,
        recompression_kappa=kappa,
        initial_void_ratio=max(0.0, float(initial_void_ratio)),
        initial_preconsolidation_pressure=max(float(min_pressure), float(initial_preconsolidation_pressure)),
        min_pressure=max(1.0e-9, float(min_pressure)),
    )


@ti.func
def modified_cam_clay_update(
    stress,
    strain_increment,
    preconsolidation_pressure,
    shear_modulus,
    bulk_modulus,
    critical_state_m,
    virgin_compression_lambda,
    recompression_kappa,
    initial_void_ratio,
    min_pressure,
    identity,
    eps,
):
    tr_deps = strain_increment.trace()
    dev_deps = strain_increment - (tr_deps / 3.0) * identity
    sigma_trial = stress + 2.0 * shear_modulus * dev_deps + bulk_modulus * tr_deps * identity

    p_trial = ti.max(min_pressure, -sigma_trial.trace() / 3.0)
    s_trial = sigma_trial + p_trial * identity
    q_trial = ti.sqrt(1.5 * (s_trial * s_trial).sum() + eps)
    pc_trial = ti.max(preconsolidation_pressure, min_pressure)
    m2 = critical_state_m * critical_state_m
    f_trial = q_trial * q_trial + m2 * p_trial * (p_trial - pc_trial)

    sigma_new = sigma_trial
    pc_new = pc_trial
    if f_trial > 0.0:
        df_dp = m2 * (2.0 * p_trial - pc_trial)
        df_dq = 2.0 * q_trial
        denom = bulk_modulus * df_dp * df_dp + 3.0 * shear_modulus * df_dq * df_dq + eps
        dgamma = f_trial / denom

        p_new = p_trial - dgamma * bulk_modulus * df_dp
        q_new = q_trial - dgamma * 3.0 * shear_modulus * df_dq
        q_new = ti.max(0.0, q_new)

        hardening_mod = (1.0 + initial_void_ratio) / (
            virgin_compression_lambda - recompression_kappa + eps
        )
        dep_v_p = dgamma * df_dp
        pc_new = pc_trial * ti.exp(ti.max(-20.0, ti.min(20.0, hardening_mod * dep_v_p)))
        pc_new = ti.max(pc_new, min_pressure)

        p_new = ti.max(min_pressure, ti.min(pc_new - min_pressure, p_new))
        q_cap = critical_state_m * ti.sqrt(ti.max(0.0, p_new * (pc_new - p_new)))
        q_new = ti.min(q_new, q_cap)

        scale = 0.0
        if q_trial > eps:
            scale = q_new / q_trial
        sigma_new = -p_new * identity + scale * s_trial

    return sigma_new, pc_new
