@echo off
setlocal
cd /d "%~dp0"
set "PYTHONW=C:\Users\90522\miniconda3\envs\mpm_taichi\pythonw.exe"
if exist "%PYTHONW%" (
    start "" "%PYTHONW%" "%~dp0tank_mpm_launcher.py"
) else (
    "C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe" "%~dp0tank_mpm_launcher.py"
)
endlocal
