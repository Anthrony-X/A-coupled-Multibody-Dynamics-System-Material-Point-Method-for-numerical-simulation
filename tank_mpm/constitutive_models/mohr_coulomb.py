from __future__ import annotations

import math
from dataclasses import dataclass

import taichi as ti


MODEL_NAME = "mohr-coulomb"


@dataclass(frozen=True)
class MohrCoulombParameters:
    shear_modulus: float
    bulk_modulus: float
    sin_phi: float
    cos_phi: float
    sin_psi: float
    cohesion: float


def make_mohr_coulomb_parameters(
    soil_E: float,
    soil_nu: float,
    soil_phi_deg: float,
    soil_psi_deg: float,
    soil_cohesion: float,
) -> MohrCoulombParameters:
    phi = math.radians(float(soil_phi_deg))
    psi = math.radians(float(soil_psi_deg))
    return MohrCoulombParameters(
        shear_modulus=float(soil_E) / (2.0 * (1.0 + float(soil_nu))),
        bulk_modulus=float(soil_E) / (3.0 * (1.0 - 2.0 * float(soil_nu))),
        sin_phi=math.sin(phi),
        cos_phi=math.cos(phi),
        sin_psi=math.sin(psi),
        cohesion=max(0.0, float(soil_cohesion)),
    )


@ti.func
def _swap_principal(a, va, b, vb):
    out_a = a
    out_va = va
    out_b = b
    out_vb = vb
    if out_a < out_b:
        tmp = out_a
        tmp_v = out_va
        out_a = out_b
        out_va = out_vb
        out_b = tmp
        out_vb = tmp_v
    return out_a, out_va, out_b, out_vb


@ti.func
def mohr_coulomb_update(
    stress,
    strain_increment,
    shear_modulus,
    bulk_modulus,
    sin_phi,
    cos_phi,
    sin_psi,
    cohesion,
    identity,
    eps,
):
    tr_deps = strain_increment.trace()
    dev_deps = strain_increment - (tr_deps / 3.0) * identity
    sigma_trial = stress + 2.0 * shear_modulus * dev_deps + bulk_modulus * tr_deps * identity

    compression_trial = -sigma_trial
    eigvals, eigvecs = ti.sym_eig(compression_trial)
    s0 = eigvals[0]
    s1 = eigvals[1]
    s2 = eigvals[2]
    v0 = ti.Vector([eigvecs[0, 0], eigvecs[1, 0], eigvecs[2, 0]])
    v1 = ti.Vector([eigvecs[0, 1], eigvecs[1, 1], eigvecs[2, 1]])
    v2 = ti.Vector([eigvecs[0, 2], eigvecs[1, 2], eigvecs[2, 2]])

    s0, v0, s1, v1 = _swap_principal(s0, v0, s1, v1)
    s1, v1, s2, v2 = _swap_principal(s1, v1, s2, v2)
    s0, v0, s1, v1 = _swap_principal(s0, v0, s1, v1)

    a_phi = 1.0 - sin_phi
    b_phi = -(1.0 + sin_phi)
    f_trial = a_phi * s0 + b_phi * s2 - 2.0 * cohesion * cos_phi

    sc0 = s0
    sc1 = s1
    sc2 = s2
    if f_trial > 0.0:
        a_psi = 1.0 - sin_psi
        b_psi = -(1.0 + sin_psi)
        m0 = a_psi
        m1 = 0.0
        m2 = b_psi
        m_trace = m0 + m1 + m2
        m_mean = m_trace / 3.0
        cm0 = 2.0 * shear_modulus * (m0 - m_mean) + bulk_modulus * m_trace
        cm1 = 2.0 * shear_modulus * (m1 - m_mean) + bulk_modulus * m_trace
        cm2 = 2.0 * shear_modulus * (m2 - m_mean) + bulk_modulus * m_trace
        denom = a_phi * cm0 + b_phi * cm2
        dgamma = f_trial / (denom + eps)
        if denom > eps:
            sc0 = s0 - dgamma * cm0
            sc1 = s1 - dgamma * cm1
            sc2 = s2 - dgamma * cm2

    compression_new = (
        sc0 * v0.outer_product(v0)
        + sc1 * v1.outer_product(v1)
        + sc2 * v2.outer_product(v2)
    )
    return -compression_new
