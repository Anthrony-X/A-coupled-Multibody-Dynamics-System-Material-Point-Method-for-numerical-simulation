# Constitutive Models

This package keeps soil stress-update laws separate from the MPM solvers.

Implemented now:

- `drucker-prager` (`dp` alias): current elastoplastic Drucker-Prager return mapping.
- `mohr-coulomb` (`mc` alias): principal-stress Mohr-Coulomb return with optional non-associated dilatancy.
- `modified-cam-clay` (`mcc` alias): p-q Modified Cam Clay return with per-particle preconsolidation pressure `pc`.
- `pure-water` (`water` alias): weakly-compressible Newtonian pure water with
  bulk pressure, dynamic viscosity, and a gauge-pressure cavitation cutoff.
- `srsh-modified-dp` (`srsh` and `improved-dp` aliases): the rate-dependent,
  pressure-sensitive, saturating-hardening model whose yield criterion is
  translated from `C:\Users\90522\Desktop\SRSH.for`. Stress integration uses a
  safeguarded backward-Euler non-associated return mapping.

The solvers select the model through `--soil-constitutive-model`. New models
should add their Taichi stress update in this folder, register a model id in
`registry.py`, and branch from the solver-side stress update wrapper.

The full SRSH equation mapping, SI units, source parameter conversion, state
variables, and usage are documented in `docs/srsh_constitutive_model.md`.

Pure-water defaults correspond to approximately room-temperature water:
`bulk_modulus = 2.2e9 Pa`, `dynamic_viscosity = 1.002e-3 Pa s`, and
`cavitation_pressure = 0 Pa` gauge. Use a fluid density of about `1000 kg/m^3`
and choose the MPM time step from the acoustic CFL limit.
