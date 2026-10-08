from __future__ import annotations

import importlib.util
import importlib
import os
import pathlib
import sys
import sysconfig
from types import ModuleType


_CORE_MODULE: ModuleType | None = None
_STANDARD_CORE_MODULE: ModuleType | None = None
_VEHICLE_MODULE: ModuleType | None = None


def prepare_chrono_dll_path() -> pathlib.Path:
    """Make Chrono's conda DLL directory visible to Python extension modules."""

    prefix = pathlib.Path(sys.prefix)
    bin_dir = prefix / "Library" / "bin"
    if not bin_dir.exists():
        return bin_dir

    bin_text = str(bin_dir)
    current_path = os.environ.get("PATH", "")
    parts = [part for part in current_path.split(os.pathsep) if part]
    if not any(os.path.normcase(part) == os.path.normcase(bin_text) for part in parts):
        os.environ["PATH"] = bin_text + os.pathsep + current_path

    dlls_dir = prefix / "DLLs"
    for extra in (dlls_dir, prefix):
        if not extra.exists():
            continue
        extra_text = str(extra)
        current_path = os.environ.get("PATH", "")
        parts = [part for part in current_path.split(os.pathsep) if part]
        if not any(os.path.normcase(part) == os.path.normcase(extra_text) for part in parts):
            os.environ["PATH"] = extra_text + os.pathsep + current_path

    return bin_dir


def load_chrono_core() -> ModuleType:
    """Load PyChrono's core wrapper without importing pychrono.__init__.

    The conda PyChrono package imports optional modules such as vehicle and
    sensor from ``pychrono/__init__.py``.  On this workstation the vehicle
    module can block during import, while the base Chrono core module loads
    correctly.  Directly loading ``pychrono/core.py`` gives access to
    ``ChSystemSMC``, ``ChBody``, links, vectors, and solvers without touching
    optional modules.
    """

    global _CORE_MODULE
    if _CORE_MODULE is not None:
        return _CORE_MODULE

    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    core_path = purelib / "pychrono" / "core.py"
    if not core_path.exists():
        raise ModuleNotFoundError(f"Cannot find PyChrono core wrapper at {core_path}")

    spec = importlib.util.spec_from_file_location("chrono_core_direct", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PyChrono core wrapper from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CORE_MODULE = module
    return module


def load_standard_chrono_core() -> ModuleType:
    """Load the regular ``pychrono.core`` module after fixing DLL lookup."""

    global _STANDARD_CORE_MODULE
    if _STANDARD_CORE_MODULE is not None:
        return _STANDARD_CORE_MODULE

    prepare_chrono_dll_path()
    module = importlib.import_module("pychrono.core")
    _STANDARD_CORE_MODULE = module
    return module


def load_chrono_vehicle() -> ModuleType:
    """Load Chrono's built-in vehicle module."""

    global _VEHICLE_MODULE
    if _VEHICLE_MODULE is not None:
        return _VEHICLE_MODULE

    prepare_chrono_dll_path()
    load_standard_chrono_core()
    module = importlib.import_module("pychrono.vehicle")
    _VEHICLE_MODULE = module
    return module


def require_chrono_core() -> ModuleType:
    chrono = load_chrono_core()
    required = ("ChSystemSMC", "ChBody", "ChVector3d")
    missing = [name for name in required if not hasattr(chrono, name)]
    if missing:
        raise ImportError(f"PyChrono core wrapper is missing required symbols: {missing}")
    return chrono


def require_chrono_vehicle() -> tuple[ModuleType, ModuleType]:
    chrono = load_standard_chrono_core()
    vehicle = load_chrono_vehicle()
    missing = [name for name in ("TrackedVehicle", "RigidTerrain", "DriverInputs") if not hasattr(vehicle, name)]
    if missing:
        raise ImportError(f"PyChrono vehicle wrapper is missing required symbols: {missing}")
    return chrono, vehicle
