from __future__ import annotations

from dataclasses import dataclass

import taichi as ti


MODEL_NAME = "pure-water"


@dataclass(frozen=True)
class PureWaterParameters:
    bulk_modulus: float
    dynamic_viscosity: float
    cavitation_pressure: float


def make_pure_water_parameters(
    bulk_modulus: float = 2.2e9,
    dynamic_viscosity: float = 1.002e-3,
    cavitation_pressure: float = 0.0,
) -> PureWaterParameters:
    """Return weakly-compressible Newtonian parameters for pure liquid water.

    Pressure is gauge pressure.  The default zero cavitation pressure prevents
    a free-surface water phase from carrying tensile hydrostatic stress.
    """

    if float(bulk_modulus) <= 0.0:
        raise ValueError("Pure-water bulk modulus must be positive")
    if float(dynamic_viscosity) < 0.0:
        raise ValueError("Pure-water dynamic viscosity cannot be negative")
    return PureWaterParameters(
        bulk_modulus=float(bulk_modulus),
        dynamic_viscosity=float(dynamic_viscosity),
        cavitation_pressure=float(cavitation_pressure),
    )


@ti.func
def pure_water_update(
    compression_positive_stress,
    extension_positive_strain_increment,
    time_step,
    bulk_modulus,
    dynamic_viscosity,
    cavitation_pressure,
    identity,
    eps,
):
    """Update compression-positive stress for a weakly-compressible fluid.

    The volumetric response is elastic and the deviatoric response is
    Newtonian viscous.  The function intentionally carries no elastic shear
    history, so a stationary liquid cannot sustain shear stress.
    """

    tr_deps = extension_positive_strain_increment.trace()
    pressure_old = compression_positive_stress.trace() / 3.0
    pressure_new = ti.max(
        cavitation_pressure,
        pressure_old - bulk_modulus * tr_deps,
    )

    dev_deps = extension_positive_strain_increment - (tr_deps / 3.0) * identity
    strain_rate_dev = dev_deps / ti.max(time_step, eps)
    compression_positive_viscous_stress = -2.0 * dynamic_viscosity * strain_rate_dev
    return pressure_new * identity + compression_positive_viscous_stress
