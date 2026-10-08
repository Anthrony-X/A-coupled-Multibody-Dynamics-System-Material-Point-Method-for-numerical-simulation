# Publication figures

The four editable SVG figures compare the water and sand diagnostics datasets:

1. `fig01_penetration_depth.svg` - hammer penetration depth.
2. `fig02_action_reaction.svg` - body/medium vertical contact-force pair and Newton-third-law residual.
3. `fig03_generalized_power_residual.svg` - absolute and relative generalized-power residuals.
4. `fig04_energy_balance.svg` - MBD/medium energy-work pairs and normalized energy residuals.
5. `fig05_model_boundary_schematic.svg` - three-dimensional coupled model and MPM boundary conditions.

All text remains editable in SVG-aware software such as Adobe Illustrator,
Inkscape, Microsoft PowerPoint, and recent versions of Microsoft Word. The
figures use Arial/Helvetica, colorblind-safe colors, vector paths, 175 mm journal
width, and English axis labels.

Regenerate the figures after replacing either diagnostics file:

```powershell
powershell -ExecutionPolicy Bypass -File .\make_publication_figures.ps1
```

The script reads:

- `..\outputs\diagnostics_water.csv` as the water result;
- `..\outputs\diagnostics_sand.csv` as the sand result.

`figure_metrics.csv` records the headline values used to verify the plotted
data. `plot_data_water.csv` and `plot_data_sand.csv` contain every derived curve
shown in the figures, so the plots can also be rebuilt in Origin, MATLAB, or
another graphics package. Export SVG to PDF/EPS only after the final text and
line placement have been checked in the target journal template.

The model schematic uses the reference release configuration
`alpha1 = 150 deg`, `alpha2 = 170 deg`, and `omega1 = omega2 = 0`. It therefore
does not use the nonzero angular-rate checkpoint currently stored in
`case_config.json`. Geometry and MPM/domain bounds are read from the current
configuration file. Regenerate it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\make_model_boundary_schematic.ps1
```
