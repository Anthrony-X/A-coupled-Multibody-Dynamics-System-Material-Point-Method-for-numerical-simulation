## Requirements

- Windows and Python 3.10
- Project Chrono Python modules available in the active environment
- CUDA-capable GPU for production MPM runs
- Python packages listed in `requirements.txt`

The configured environment is currently:

```powershell
C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe
```

Verify Project Chrono before a long run:

```powershell
& "C:\Users\miniconda3\envs\mpm_taichi\python.exe" tools\smoke_project_chrono_core.py
```
# Examples: Tracked Vehicle on soft soil; Snake robot walk on sand

This project couples a Project Chrono Vehicle tracked and snake robot multibody model with a
Taichi Material Point Method soil solver. 

# Simulation Results
Tracked Vehicle on soft soil
https://github.com/user-attachments/assets/4fbf447b-050c-4229-94f5-ab6c2571a57d
Snake robot walk on sand (reference: DOI: 10.1126/science.1255718  published by Science)
[fig01_cmu_snake_model_boundary_schematic.tif](https://github.com/user-attachments/files/33222991/fig01_cmu_snake_model_boundary_schematic.tif)
[算例2-渲染模型.tif](https://github.com/user-attachments/files/33222974/2-.tif)
## Run

Open the launcher:

```powershell
.\start_tank_mpm_launcher.bat
```

Run the complete Chrono-MPM workflow directly:

```powershell
& "C:\Users\miniconda3\envs\mpm_taichi\python.exe" `
  .\main_multibody_tank_mpm.py `
  --stage all `
  --arch cuda `
  --model .\ZTZ_96\multibody\ztz96_multibody_model.json `
  --out .\outputs\output_multibody_tank_mpm
```

The stages are:

1. `geostatic`: equilibrate the soil without the vehicle.
2. `settle`: place the Chrono vehicle and establish track-soil support.
3. `drive`: apply sprocket torque and advance the coupled system.

The launcher stores its current values in `tank_mpm_launcher_config.json`.
Command-line runs use parser defaults unless the corresponding options are
passed explicitly.

## Rigid-Ground Check

Use the same Project Chrono tracked-vehicle backend without MPM soil:

```powershell
& "C:\Users\miniconda3\envs\mpm_taichi\python.exe" `
  .\main_tank_rigid_ground.py `
  --model .\ZTZ_96\multibody\ztz96_multibody_model.json `
  --out .\outputs\output_tank_rigid_ground
```

## Constitutive Tests

Run all unit-cell models using the saved launcher configuration:

```powershell
& "C:\Users\miniconda3\envs\mpm_taichi\python.exe" `
  .\constitutive_unit_cell_test.py `
  --config .\tank_mpm_launcher_config.json
```

SRSH parameters are read from the JSON `fields` object. Editing fallback values
inside the test script does not override values present in that file.

## Structure

```text
|-- tank_mpm/                    core Python package
|   |-- constitutive_models/     soil and water stress updates
|   |-- chrono_vehicle.py        Project Chrono Vehicle backend
|   |-- mpm_solver.py            Taichi MPM and track contact
|   |-- vehicle.py               vehicle model and VTK geometry
|   `-- ...
|-- main_multibody_tank_mpm.py   production coupled entry point
|-- main_tank_rigid_ground.py    rigid-ground verification
|-- tank_mpm_launcher.py         GUI launcher
|-- constitutive_unit_cell_test.py
|-- ZTZ_96/ and ZBD_04A/         vehicle assets
|-- validation_cases/            independent validation cases
|-- tools/                       focused utilities
|-- docs/                        method notes and source papers
`-- outputs/                     generated results
```

See `docs/project_structure.md` for module ownership and dependency rules.

## Verification

```powershell
& "C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe" -m unittest discover
```

Generated outputs, Python bytecode, Taichi caches, temporary probes, and rendered
document intermediates are ignored by `.gitignore` and should not be stored in
the source tree.

