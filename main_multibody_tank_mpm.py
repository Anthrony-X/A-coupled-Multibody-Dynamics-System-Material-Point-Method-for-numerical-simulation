from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import taichi as ti

from tank_mpm.constitutive_models import CONSTITUTIVE_MODEL_CHOICES, normalize_constitutive_model_name
from tank_mpm.particles import sample_soil_particles
from tank_mpm.vehicle import (
    GroundParams,
    MultibodyRigidGroundTank,
    MultibodyState,
    VISUAL_DETAIL_CHOICES,
    VISUAL_DETAIL_SIMPLE,
    build_multibody_tank_mesh,
)
from tank_mpm.mpm_solver import TankTrackMpmSolver
from tank_mpm.chrono_vehicle import ProjectChronoVehicleBackend


TrackBackend = ProjectChronoVehicleBackend


@dataclass
class MultibodyMpmDiagnostics:
    stage: str
    step: int
    stage_time: float
    t: float
    tank_x: float
    tank_y: float
    tank_heave: float
    tank_yaw: float
    tank_pitch: float
    tank_forward_speed: float
    tank_heave_rate: float
    left_track_speed: float
    right_track_speed: float
    left_sprocket_omega: float
    right_sprocket_omega: float
    active_patches: int
    contact_particles: int
    max_track_shoe_deflection: float
    max_track_shoe_pitch: float
    left_contact_particles: int
    right_contact_particles: int
    left_force_x: float
    left_force_y: float
    left_force_z: float
    right_force_x: float
    right_force_y: float
    right_force_z: float
    total_force_x: float
    total_force_y: float
    total_force_z: float
    max_soil_speed: float
    mean_soil_z: float
    min_soil_z: float
    max_soil_z: float


@dataclass
class TrackTerrainFeedback:
    """Chrono-style external terrain feedback grouped by track contact patch.

    Coordinates follow this MPM driver: x forward, y lateral, z vertical.
    """

    side_forces: np.ndarray
    patch_centers: np.ndarray
    patch_forces: np.ndarray
    patch_moments: np.ndarray
    patch_side_ids: np.ndarray
    patch_shoe_ids: np.ndarray

    @classmethod
    def empty(cls) -> "TrackTerrainFeedback":
        return cls(
            side_forces=np.zeros((2, 3), dtype=np.float64),
            patch_centers=np.zeros((0, 3), dtype=np.float64),
            patch_forces=np.zeros((0, 3), dtype=np.float64),
            patch_moments=np.zeros((0, 3), dtype=np.float64),
            patch_side_ids=np.zeros((0,), dtype=np.int32),
            patch_shoe_ids=np.zeros((0,), dtype=np.float64),
        )


def track_feedback_from_solver(solver: TankTrackMpmSolver) -> TrackTerrainFeedback:
    patch_feedback = solver.averaged_track_contact_by_patch()
    side_forces = solver.averaged_contact_forces_by_side()
    return TrackTerrainFeedback(
        side_forces=np.asarray(side_forces, dtype=np.float64),
        patch_centers=np.asarray(patch_feedback["center"], dtype=np.float64),
        patch_forces=np.asarray(patch_feedback["force"], dtype=np.float64),
        patch_moments=np.asarray(patch_feedback["moment"], dtype=np.float64),
        patch_side_ids=np.asarray(patch_feedback["side_id"], dtype=np.int32),
        patch_shoe_ids=np.asarray(patch_feedback["shoe_id"], dtype=np.float64),
    )
@dataclass
class KineticDampingState:
    previous_energy: float | None = None
    event_count: int = 0


def smooth_ramp(t: float, ramp_time: float) -> float:
    if ramp_time <= 0.0:
        return 1.0
    a = max(0.0, min(1.0, t / ramp_time))
    return a * a * (3.0 - 2.0 * a)


def cubic_domain(bounds_lo: np.ndarray, bounds_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    span = bounds_hi - bounds_lo
    extent = float(np.max(span))
    center = 0.5 * (bounds_lo + bounds_hi)
    return center - 0.5 * extent, center + 0.5 * extent


def resolve_geostatic_k0(args: argparse.Namespace) -> float:
    if args.geostatic_k0 is not None:
        return float(args.geostatic_k0)
    if args.soil_constitutive_model == "pure-water":
        return 1.0
    return 1.0 - math.sin(math.radians(float(args.soil_phi_deg)))


def parse_args(
    argv: list[str] | None = None,
    namespace: argparse.Namespace | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-stage multibody tank and MPM soil coupled simulation."
    )
    p.add_argument("--stage", choices=["geostatic", "settle", "drive", "all"], default="all")
    p.add_argument("--out", type=str, default="outputs/output_multibody_tank_mpm")
    p.add_argument("--model", type=str, default="ZTZ_96/multibody/ztz96_multibody_model.json")
    p.add_argument("--arch", type=str, default="cuda", choices=["cpu", "cuda", "vulkan", "gpu"])
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Taichi CPU worker threads. Use 0 for Taichi default.",
    )
    p.add_argument(
        "--taichi-offline-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Taichi offline kernel cache. Disabled by default to avoid cache-lock stalls on this workspace.",
    )
    p.add_argument("--dt", type=float, default=1.0e-5, help="MPM time step in seconds, kept for compatibility")
    p.add_argument("--mpm-dt", type=float, default=None, help="MPM substep in seconds. Overrides --dt when set.")
    p.add_argument(
        "--mbd-dt",
        type=float,
        default=None,
        help="MBD macro time step in seconds. Default is 50 MPM substeps, snapped to an integer MPM-step multiple.",
    )
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue the selected single stage from its saved state instead of starting it from the previous stage.",
    )

    p.add_argument("--geostatic-steps", type=int, default=5000)
    p.add_argument("--settle-steps", type=int, default=10000)
    p.add_argument("--drive-steps", type=int, default=15000)
    p.add_argument("--geostatic-state", type=str, default=None)
    p.add_argument("--settled-state", type=str, default=None)
    p.add_argument("--settled-tank-state", type=str, default=None)
    p.add_argument("--drive-state", type=str, default=None)
    p.add_argument("--drive-tank-state", type=str, default=None)

    p.add_argument("--grid", type=int, default=152)
    p.add_argument("--soil-nx", type=int, default=144)
    p.add_argument("--soil-ny", type=int, default=60)
    p.add_argument("--soil-nz", type=int, default=32)
    p.add_argument("--soil-jitter", type=float, default=0.15)
    p.add_argument("--soil-xmin", type=float, default=-1.5)
    p.add_argument("--soil-xmax", type=float, default=10.5)
    p.add_argument("--soil-ymin", type=float, default=-2.5)
    p.add_argument("--soil-ymax", type=float, default=2.5)
    p.add_argument("--soil-zmin", type=float, default=-1.5)
    p.add_argument("--soil-zmax", type=float, default=0.5)
    p.add_argument("--domain-padding", type=float, default=1.0 / 3.0)
    p.add_argument(
        "--domain-shape",
        choices=["rect", "cubic"],
        default="rect",
        help="MPM background domain shape. rect keeps the requested x-resolution and trims empty y/z grid space; cubic restores the old cube.",
    )
    p.add_argument(
        "--mpm-precision",
        choices=["f32", "f64"],
        default="f32",
        help="Floating-point precision for MPM particle state and stress fields. f32 is faster; f64 restores the previous high-precision mode.",
    )
    p.add_argument(
        "--moving-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recycle x-directed particle layers on the GPU so the fixed-size MPM window follows the vehicle during drive.",
    )
    p.add_argument(
        "--moving-window-template-layers",
        type=int,
        default=8,
        help="Number of pristine geostatic x-layers retained as the repeating inlet-soil template.",
    )

    p.add_argument("--soil-density", type=float, default=1700.0)
    p.add_argument("--soil-E", type=float, default=1.5e6)
    p.add_argument("--soil-nu", type=float, default=0.30)
    p.add_argument("--soil-phi-deg", type=float, default=18.0)
    p.add_argument("--soil-psi-deg", type=float, default=2.0)
    p.add_argument("--soil-cohesion", type=float, default=4.0e3)
    p.add_argument(
        "--soil-constitutive-model",
        type=normalize_constitutive_model_name,
        default="drucker-prager",
        metavar="MODEL",
        help=f"Soil constitutive model. Available: {', '.join(CONSTITUTIVE_MODEL_CHOICES)}",
    )
    p.add_argument("--srsh-rate-exponent", type=float, default=0.3, help="SRSH rate exponent beta")
    p.add_argument("--srsh-rate-sensitivity", type=float, default=0.1, help="SRSH rate sensitivity eta")
    p.add_argument("--srsh-saturation-increment", type=float, default=1.1, help="SRSH saturation hardening increment delta")
    p.add_argument("--srsh-saturation-plastic-strain", type=float, default=0.5, help="SRSH plastic strain at about 95 percent saturation")
    p.add_argument("--soil-mcc-m", type=float, default=None, help="Modified Cam Clay critical-state slope M. Omit to estimate from phi.")
    p.add_argument("--soil-mcc-lambda", type=float, default=0.12, help="Modified Cam Clay virgin compression index lambda")
    p.add_argument("--soil-mcc-kappa", type=float, default=0.02, help="Modified Cam Clay recompression index kappa")
    p.add_argument("--soil-mcc-e0", type=float, default=0.8, help="Modified Cam Clay initial void ratio")
    p.add_argument("--soil-mcc-pc0", type=float, default=1.0e5, help="Modified Cam Clay initial preconsolidation pressure Pa")
    p.add_argument("--soil-mcc-min-pressure", type=float, default=1.0, help="Modified Cam Clay minimum compressive pressure Pa")
    p.add_argument("--water-bulk-modulus", type=float, default=2.2e9, help="Pure-water bulk modulus Pa")
    p.add_argument("--water-dynamic-viscosity", type=float, default=1.002e-3, help="Pure-water dynamic viscosity Pa s")
    p.add_argument("--water-cavitation-pressure", type=float, default=0.0, help="Pure-water minimum gauge pressure Pa")
    p.add_argument("--soil-gravity-scale", type=float, default=1.0)
    p.add_argument("--soil-damping", type=float, default=1.0)
    p.add_argument("--geostatic-soil-gravity-scale", type=float, default=1.0)
    p.add_argument("--geostatic-soil-damping", type=float, default=1.0)
    p.add_argument(
        "--geostatic-k0",
        type=float,
        default=None,
        help="Earth-pressure-at-rest coefficient for geostatic stress. Omit to use Jaky K0 = 1 - sin(phi).",
    )
    p.add_argument("--geostatic-initial-stress", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--contact-mu", type=float, default=0.50)
    p.add_argument(
        "--contact-barrier-stiffness",
        type=float,
        default=None,
        help="Barrier normal contact stiffness kappa for particle-to-track SDF contact, N/m. Omit to use soil_E * particle_radius.",
    )
    p.add_argument(
        "--contact-barrier-radius",
        type=float,
        default=None,
        help="Barrier activation distance around each track shoe contact surface, m. Omit to use the equivalent spherical particle radius.",
    )
    p.add_argument(
        "--contact-barrier-min-distance-ratio",
        type=float,
        default=0.15,
        help="Lower clamp d_min/r for the logarithmic barrier distance to keep explicit MPM stable under small initial overlap.",
    )
    p.add_argument(
        "--contact-slip-smoothing-distance",
        type=float,
        default=None,
        help="Slip distance for smoothed Coulomb transition, m. Omit to use the barrier radius.",
    )
    p.add_argument(
        "--patch-update-every",
        type=int,
        default=1,
        help=(
            "Upload/rebuild moving track contact patches every N MPM substeps inside one MBD macro step. "
            "Use 1 for accuracy-preserving runs; values above 1 are approximate speedups."
        ),
    )
    p.add_argument(
        "--surface-estimator",
        choices=["gpu", "cpu"],
        default="gpu",
        help=(
            "Soil-surface percentile estimator used once for initial settle placement. "
            "gpu filters candidate particles in Taichi before copying only z samples."
        ),
    )
    p.add_argument(
        "--track-update-mode",
        choices=["gpu", "cpu"],
        default="gpu",
        help=(
            "Track contact update path. gpu uploads patch keyframes and updates near-surface "
            "SDF contact patch state on the device; cpu updates interpolated patches from numpy."
        ),
    )
    p.add_argument(
        "--track-dynamics-model",
        choices=["chrono-vehicle", "project-chrono"],
        default="chrono-vehicle",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--chrono-max-substep-dt", type=float, default=5.0e-4)
    p.add_argument("--chrono-contact-young-modulus", type=float, default=2.0e7)
    p.add_argument("--chrono-contact-kn", type=float, default=3.0e6)
    p.add_argument("--chrono-contact-gn", type=float, default=4.0e4)
    p.add_argument("--chrono-contact-kt", type=float, default=1.0e6)
    p.add_argument("--chrono-contact-gt", type=float, default=1.0e4)
    p.add_argument("--chrono-wheel-mass", type=float, default=120.0)
    p.add_argument("--chrono-vehicle-asset-dir", type=str, default="", help="Directory for generated Chrono vehicle JSON. Empty uses an ASCII temp directory.")
    p.add_argument("--chrono-chassis-mass", type=float, default=0.0, help="Chrono vehicle chassis mass, kg. Use <=0 to auto-subtract tracks/running gear from --mass.")
    p.add_argument("--chrono-roller-mass", type=float, default=0.0, help="Return roller mass, kg. Use <=0 for an estimate from --chrono-wheel-mass.")
    p.add_argument("--chrono-sprocket-mass", type=float, default=0.0, help="Drive sprocket gear mass, kg. Use <=0 for an estimate from --chrono-wheel-mass.")
    p.add_argument("--chrono-idler-mass", type=float, default=0.0, help="Idler wheel mass, kg. Use <=0 to use --chrono-wheel-mass.")
    p.add_argument("--chrono-suspension-arm-mass", type=float, default=75.0)
    p.add_argument("--chrono-suspension-spring-constant", type=float, default=8.0e4)
    p.add_argument("--chrono-suspension-damping-coefficient", type=float, default=2.0e3)
    p.add_argument("--chrono-suspension-preload", type=float, default=-1.0e4)
    p.add_argument("--chrono-aux-damper-coefficient", type=float, default=1.0e2)
    p.add_argument("--chrono-tensioner-preload", type=float, default=2.0e4)
    p.add_argument("--chrono-tensioner-free-length", type=float, default=0.75)
    p.add_argument("--chrono-tensioner-stiffness", type=float, default=1.0e6)
    p.add_argument("--chrono-tensioner-damping", type=float, default=1.4e4)
    p.add_argument("--chrono-track-shoe-count", type=int, default=0, help="Track shoe count per side for Chrono vehicle. Use 0 for model count plus padding.")
    p.add_argument("--chrono-track-shoe-count-padding", type=int, default=2, help="Extra shoes added in Chrono vehicle assembly when --chrono-track-shoe-count is 0.")
    p.add_argument("--chrono-track-shoe-pitch", type=float, default=None, help="Optional Chrono assembly pitch override, m.")
    p.add_argument("--chrono-idler-assembly-retraction", type=float, default=None, help="Signed initial idler shift toward the sprocket, m.")
    p.add_argument("--chrono-drive-torque-sign", type=float, default=-1.0, help="Sign applied to direct sprocket axle torques in the Chrono vehicle frame.")
    p.add_argument("--chrono-max-penetration-recovery-speed", type=float, default=0.25)
    p.add_argument(
        "--chrono-use-rigid-terrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For MPM coupling this should remain false; true adds Chrono's rigid terrain in addition to MPM soil.",
    )
    p.add_argument("--sprocket-tooth-count", type=int, default=0, help="Drive sprocket tooth count. Use 0 to infer from sprocket radius and track pitch.")
    p.add_argument(
        "--include-grousers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the outer grousers in track-soil contact and tank VTK export.",
    )
    p.add_argument(
        "--tank-vtk-detail",
        choices=VISUAL_DETAIL_CHOICES,
        default=VISUAL_DETAIL_SIMPLE,
        help="Tank VTK visual detail. simple writes a low-face primitive model; full restores the OBJ visual mesh.",
    )

    p.add_argument(
        "--mass",
        type=float,
        default=None,
        help="Vehicle mass, kg. Omit to use mass_properties.mass_kg from the selected model JSON.",
    )
    p.add_argument(
        "--track-pitch",
        type=float,
        default=None,
        help="Track shoe pitch, m. Default prefers calibrated model geometry and otherwise matches shoe length.",
    )
    p.add_argument(
        "--track-width",
        type=float,
        default=None,
        help="Track width, m. Default prefers calibrated model geometry and otherwise uses the identified loop span.",
    )
    p.add_argument(
        "--shoe-length",
        type=float,
        default=None,
        help="Track shoe length, m. Default prefers calibrated model geometry and otherwise falls back to the legacy placeholder.",
    )
    p.add_argument(
        "--track-shoe-mass",
        type=float,
        default=None,
        help=(
            "Mass of one track shoe body, kg. Default uses the model JSON calibrated inertial property, "
            "Chrono-style, instead of scaling with --mass."
        ),
    )
    p.add_argument(
        "--track-shoe-pitch-inertia",
        type=float,
        default=None,
        help=(
            "Pitch inertia of one track shoe body, kg*m^2. Default uses the model JSON value or a box estimate "
            "from the resolved shoe mass and dimensions."
        ),
    )
    p.add_argument("--initial-forward", type=float, default=4.0)
    p.add_argument("--initial-lateral", type=float, default=0.0)
    p.add_argument("--initial-sinkage", type=float, default=0.0)
    p.add_argument("--suspension-stiffness", type=float, default=800.0)
    p.add_argument("--suspension-damping", type=float, default=220.0)
    p.add_argument("--suspension-max-angle", type=float, default=0.35)
    p.add_argument("--sprocket-inertia", type=float, default=220.0)
    p.add_argument("--sprocket-damping", type=float, default=450.0)
    p.add_argument("--max-sprocket-omega", type=float, default=90.0)
    p.add_argument(
        "--track-shoe-dynamics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Chrono-style segmented track shoe compliance driven by per-shoe terrain loads.",
    )
    p.add_argument("--track-shoe-normal-stiffness", type=float, default=2.0e6)
    p.add_argument("--track-shoe-normal-damping", type=float, default=2.5e4)
    p.add_argument("--track-pin-bending-stiffness", type=float, default=4.0e5)
    p.add_argument("--track-pin-bending-damping", type=float, default=8.0e3)
    p.add_argument("--track-pin-pitch-stiffness", type=float, default=2.0e4)
    p.add_argument("--track-pin-pitch-damping", type=float, default=1.2e3)
    p.add_argument("--track-shoe-max-deflection", type=float, default=0.080)
    p.add_argument("--track-shoe-max-pitch", type=float, default=0.35)

    p.add_argument(
        "--drive-torque",
        type=float,
        default=-12000.0,
        help=(
            "Default drive torque per sprocket, N*m. With the current track phase convention, "
            "negative torque drives the bottom track surface rearward and moves the vehicle forward."
        ),
    )
    p.add_argument("--left-drive-torque", type=float, default=None)
    p.add_argument("--right-drive-torque", type=float, default=None)
    p.add_argument("--ramp-time", type=float, default=1.0)
    p.add_argument("--yaw-damping", type=float, default=0.0)
    p.add_argument("--body-drag-coefficient-area", type=float, default=0.0)
    p.add_argument("--air-density", type=float, default=1.225)
    p.add_argument("--body-rolling-resistance", type=float, default=0.0)
    p.add_argument("--slip-regularization", type=float, default=0.35)
    p.add_argument("--vertical-damping", type=float, default=0.0)
    p.add_argument("--pitch-rate-damping", type=float, default=0.0)
    p.add_argument("--max-feedback-normal-factor", type=float, default=4.0)
    p.add_argument("--max-feedback-traction-factor", type=float, default=2.0)
    p.add_argument(
        "--settle-lock-planar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain chassis forward/lateral displacement and yaw during settle while leaving vertical settling and track motion free.",
    )
    p.add_argument(
        "--settle-initial-clearance",
        type=float,
        default=0.025,
        help="Initial clearance between the lowest track contact surface and the estimated local soil surface before settle, m.",
    )
    p.add_argument("--settle-surface-percentile", type=float, default=95.0)
    p.add_argument(
        "--settle-weight-ramp-steps",
        type=int,
        default=0,
        help="Deprecated compatibility option. Settle now always applies full tank self-weight immediately.",
    )
    p.add_argument(
        "--settle-soil-damping",
        type=float,
        default=None,
        help="Optional soil velocity damping only for settle. Omit to keep --soil-damping.",
    )
    p.add_argument(
        "--settle-vertical-damping",
        type=float,
        default=None,
        help="Optional heave-rate damping only for settle. Omit to keep --vertical-damping.",
    )
    p.add_argument(
        "--settle-pitch-rate-damping",
        type=float,
        default=None,
        help="Optional pitch-rate damping only for settle. Omit to keep --pitch-rate-damping.",
    )
    p.add_argument(
        "--settle-suspension-damping",
        type=float,
        default=None,
        help="Optional suspension damping override only for settle. Omit to keep --suspension-damping.",
    )
    p.add_argument("--settle-kinetic-damping", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--settle-kinetic-damping-factor",
        type=float,
        default=0.2,
        help="Velocity multiplier applied to MBD rates after a settle kinetic-energy peak.",
    )
    p.add_argument(
        "--settle-soil-kinetic-damping-factor",
        type=float,
        default=0.5,
        help="Velocity multiplier applied to MPM soil kinematics after a settle kinetic-energy peak.",
    )
    p.add_argument("--settle-kinetic-damping-min-energy", type=float, default=10.0)
    return p.parse_args(argv, namespace=namespace)


def resolve_state_paths(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if args.geostatic_state is None:
        args.geostatic_state = str(out_dir / "geostatic_state.npz")
    if args.settled_state is None:
        args.settled_state = str(out_dir / "tank_settled_state.npz")
    if args.settled_tank_state is None:
        args.settled_tank_state = str(out_dir / "tank_settled_state.tank.json")
    if args.drive_state is None:
        args.drive_state = str(out_dir / "tank_drive_final_state.npz")
    if args.drive_tank_state is None:
        args.drive_tank_state = str(out_dir / "tank_drive_final_state.tank.json")


def resolve_time_steps(args: argparse.Namespace) -> None:
    mpm_dt = float(args.dt if args.mpm_dt is None else args.mpm_dt)
    if mpm_dt <= 0.0:
        raise ValueError("MPM time step must be positive")

    requested_mbd_dt = 50.0 * mpm_dt if args.mbd_dt is None else float(args.mbd_dt)
    if requested_mbd_dt <= 0.0:
        raise ValueError("MBD time step must be positive")

    mbd_substeps = max(1, int(round(requested_mbd_dt / mpm_dt)))
    args.mpm_dt = mpm_dt
    args.requested_mbd_dt = requested_mbd_dt
    args.mbd_substeps = mbd_substeps
    args.mbd_dt = mbd_substeps * mpm_dt
    # Keep existing helpers and saved progress files compatible with older --dt semantics.
    args.dt = mpm_dt


def init_taichi(arch_name: str, *, cpu_threads: int = 0, offline_cache: bool = False) -> None:
    arch_map = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "gpu": ti.gpu,
    }
    init_kwargs = {
        "arch": arch_map[arch_name],
        "default_fp": ti.f32,
        "offline_cache": bool(offline_cache),
    }
    if int(cpu_threads) > 0:
        init_kwargs["cpu_max_num_threads"] = int(cpu_threads)
    try:
        ti.init(**init_kwargs)
    except Exception as exc:
        print(f"[Warn] Failed to init Taichi arch={arch_name}: {exc}")
        print("[Warn] Falling back to CPU.")
        init_kwargs["arch"] = ti.cpu
        ti.init(**init_kwargs)


def build_tank(args: argparse.Namespace) -> MultibodyRigidGroundTank:
    ground = GroundParams(
        friction_mu=args.contact_mu,
        rolling_resistance=args.body_rolling_resistance,
        slip_velocity_regularization=args.slip_regularization,
        yaw_damping=args.yaw_damping,
        drag_coefficient_area=args.body_drag_coefficient_area,
        air_density=args.air_density,
    )
    tank = MultibodyRigidGroundTank(
        model_path=Path(args.model),
        ground=ground,
        mass=args.mass,
        track_pitch=args.track_pitch,
        track_width=args.track_width,
        shoe_length=args.shoe_length,
        track_shoe_mass=args.track_shoe_mass,
        track_shoe_pitch_inertia=args.track_shoe_pitch_inertia,
        suspension_stiffness=args.suspension_stiffness,
        suspension_damping=args.suspension_damping,
        suspension_max_angle=args.suspension_max_angle,
        sprocket_inertia=args.sprocket_inertia,
        sprocket_damping=args.sprocket_damping,
        track_shoe_dynamics=args.track_shoe_dynamics,
        track_shoe_normal_stiffness=args.track_shoe_normal_stiffness,
        track_shoe_normal_damping=args.track_shoe_normal_damping,
        track_pin_bending_stiffness=args.track_pin_bending_stiffness,
        track_pin_bending_damping=args.track_pin_bending_damping,
        track_pin_pitch_stiffness=args.track_pin_pitch_stiffness,
        track_pin_pitch_damping=args.track_pin_pitch_damping,
        track_shoe_max_deflection=args.track_shoe_max_deflection,
        track_shoe_max_pitch=args.track_shoe_max_pitch,
    )
    tank.state.forward_pos = float(args.initial_forward)
    tank.state.lateral_pos = float(args.initial_lateral)
    place_tank_on_soil(tank, soil_zmax=float(args.soil_zmax), initial_sinkage=float(args.initial_sinkage), args=args)
    return tank


def build_track_backend(args: argparse.Namespace, tank: MultibodyRigidGroundTank) -> TrackBackend:
    backend = ProjectChronoVehicleBackend(args=args, tank=tank, obstacles=[])
    backend.initialize()
    return backend


def apply_track_feedback_to_backend(
    track_backend: TrackBackend,
    tank: MultibodyRigidGroundTank,
    feedback: TrackTerrainFeedback,
) -> None:
    if hasattr(track_backend, "clear_external_loads"):
        track_backend.clear_external_loads()
    centers = np.asarray(feedback.patch_centers, dtype=np.float64)
    forces = np.asarray(feedback.patch_forces, dtype=np.float64)
    moments = np.asarray(feedback.patch_moments, dtype=np.float64)
    shoe_ids = np.asarray(feedback.patch_shoe_ids, dtype=np.float64)
    if centers.size == 0 or forces.size == 0 or shoe_ids.size == 0:
        return
    count = min(centers.shape[0], forces.shape[0], shoe_ids.shape[0])
    for patch_index in range(count):
        body_id = int(round(float(shoe_ids[patch_index])))
        force_world = mpm_vector_to_multibody(forces[patch_index])
        point_world = mpm_point_to_multibody(centers[patch_index])
        track_backend.add_shoe_force_by_body_id(
            body_id,
            force_world,
            application_point_world=point_world,
        )
        if moments.size and patch_index < moments.shape[0]:
            track_backend.add_shoe_torque_by_body_id(body_id, mpm_vector_to_multibody(moments[patch_index]))


def build_solver(args: argparse.Namespace, tank: MultibodyRigidGroundTank) -> TankTrackMpmSolver:
    if bool(args.moving_window) and args.domain_shape != "rect":
        raise ValueError("--moving-window requires --domain-shape rect.")
    soil_lo = np.array([args.soil_xmin, args.soil_ymin, args.soil_zmin], dtype=np.float64)
    soil_hi = np.array([args.soil_xmax, args.soil_ymax, args.soil_zmax], dtype=np.float64)
    padded_lo = soil_lo - args.domain_padding
    padded_hi = soil_hi + args.domain_padding
    if args.domain_shape == "cubic":
        domain_lo, domain_hi = cubic_domain(padded_lo, padded_hi)
    else:
        domain_lo, domain_hi = padded_lo, padded_hi

    soil_points, soil_p_vol = sample_soil_particles(
        args.soil_nx,
        args.soil_ny,
        args.soil_nz,
        lo=soil_lo,
        hi=soil_hi,
        jitter_ratio=args.soil_jitter,
        seed=args.seed,
    )
    chrono_shoes = int(getattr(args, "chrono_track_shoe_count", 0))
    if chrono_shoes <= 0:
        chrono_shoes = max(tank.shoe_count["left"], tank.shoe_count["right"]) + max(
            0,
            int(getattr(args, "chrono_track_shoe_count_padding", 2)),
        )
    max_track_patches = 2 * chrono_shoes
    return TankTrackMpmSolver(
        soil_points=soil_points,
        soil_p_vol=soil_p_vol,
        n_grid=args.grid,
        dt=args.dt,
        domain_lo=tuple(domain_lo.tolist()),
        domain_hi=tuple(domain_hi.tolist()),
        soil_bounds_lo=tuple(soil_lo.tolist()),
        soil_bounds_hi=tuple(soil_hi.tolist()),
        max_track_patches=max_track_patches,
        soil_density=args.soil_density,
        soil_E=args.soil_E,
        soil_nu=args.soil_nu,
        soil_phi_deg=args.soil_phi_deg,
        soil_psi_deg=args.soil_psi_deg,
        soil_cohesion=args.soil_cohesion,
        soil_constitutive_model=args.soil_constitutive_model,
        srsh_rate_exponent=args.srsh_rate_exponent,
        srsh_rate_sensitivity=args.srsh_rate_sensitivity,
        srsh_saturation_increment=args.srsh_saturation_increment,
        srsh_saturation_plastic_strain=args.srsh_saturation_plastic_strain,
        soil_mcc_m=args.soil_mcc_m,
        soil_mcc_lambda=args.soil_mcc_lambda,
        soil_mcc_kappa=args.soil_mcc_kappa,
        soil_mcc_e0=args.soil_mcc_e0,
        soil_mcc_pc0=args.soil_mcc_pc0,
        soil_mcc_min_pressure=args.soil_mcc_min_pressure,
        water_bulk_modulus=args.water_bulk_modulus,
        water_dynamic_viscosity=args.water_dynamic_viscosity,
        water_cavitation_pressure=args.water_cavitation_pressure,
        soil_gravity_scale=args.soil_gravity_scale,
        soil_damping=args.soil_damping,
        contact_mu=args.contact_mu,
        contact_barrier_stiffness=args.contact_barrier_stiffness,
        contact_barrier_radius=args.contact_barrier_radius,
        contact_barrier_min_distance_ratio=args.contact_barrier_min_distance_ratio,
        contact_slip_smoothing_distance=args.contact_slip_smoothing_distance,
        mpm_precision=args.mpm_precision,
        particle_shape=(args.soil_nx, args.soil_ny, args.soil_nz),
        moving_window_enabled=args.moving_window,
        moving_window_template_layers=args.moving_window_template_layers,
    )


def contact_surface_extra(tank: MultibodyRigidGroundTank, args: argparse.Namespace) -> float:
    return tank.geom.grouser_height if args.include_grousers else 0.0


def soil_facing_track_normal_world(pose: dict[str, np.ndarray]) -> np.ndarray:
    normal = np.asarray(pose["axis_normal_world"], dtype=float)
    if normal[1] > 0.0:
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1.0e-12)


def place_tank_on_soil(
    tank: MultibodyRigidGroundTank,
    *,
    soil_zmax: float,
    initial_sinkage: float,
    args: argparse.Namespace,
    clearance: float = 0.0,
) -> None:
    tank.state.heave = 0.0
    tank.state.heave_rate = 0.0
    tank.state.pitch = 0.0
    tank.state.pitch_rate = 0.0
    tank.track_pose_cache.clear()
    surface_offset = 0.5 * tank.geom.shoe_thickness + contact_surface_extra(tank, args)
    contact_z_values = []
    for side in ("left", "right"):
        for pose in tank.ground_contact_shoe_poses(side, surface_offset_extra=contact_surface_extra(tank, args)):
            surface_point = pose["world_center"] + soil_facing_track_normal_world(pose) * surface_offset
            contact_z_values.append(float(surface_point[1]))
    if not contact_z_values:
        raise RuntimeError("No near-ground track shoes were found for initial tank placement.")

    target_z = float(soil_zmax) + max(0.0, float(clearance)) - float(initial_sinkage)
    tank.state.heave += target_z - min(contact_z_values)
    tank.track_pose_cache.clear()


def align_chrono_track_backend_on_soil(
    track_backend: TrackBackend | None,
    tank: MultibodyRigidGroundTank,
    args: argparse.Namespace,
    *,
    soil_surface_z: float,
    label: str,
) -> None:
    del tank
    if track_backend is None or not hasattr(track_backend, "align_track_contact_to_model_y"):
        return
    clearance = max(0.0, float(getattr(args, "settle_initial_clearance", 0.0)))
    target_y = float(soil_surface_z) + clearance - float(args.initial_sinkage)
    before = float(
        track_backend.track_contact_min_model_y(
            surface_offset_extra=contact_surface_extra(track_backend.tank, args),
            bottom_run_only=True,
        )
    )
    delta = float(
        track_backend.align_track_contact_to_model_y(
            target_y,
            surface_offset_extra=contact_surface_extra(track_backend.tank, args),
        )
    )
    after = float(
        track_backend.track_contact_min_model_y(
            surface_offset_extra=contact_surface_extra(track_backend.tank, args),
            bottom_run_only=True,
        )
    )
    print(
        f"[Info] Chrono track placement ({label}): target_contact_z={target_y:.6f}, "
        f"clearance={clearance:.6f}, before={before:.6f}, after={after:.6f}, vehicle_shift_dy={delta:.6f}"
    )


def bottom_track_contact_surfaces_mpm(tank: MultibodyRigidGroundTank, args: argparse.Namespace) -> list[dict[str, np.ndarray]]:
    surfaces: list[dict[str, np.ndarray]] = []
    surface_offset = 0.5 * tank.geom.shoe_thickness + contact_surface_extra(tank, args)
    for side in ("left", "right"):
        for pose in tank.ground_contact_shoe_poses(side, surface_offset_extra=contact_surface_extra(tank, args)):
            center_world = pose["world_center"] + soil_facing_track_normal_world(pose) * surface_offset
            axis_long = multibody_vector_to_mpm(pose["axis_long_world"])
            axis_width = multibody_vector_to_mpm(pose["axis_width_world"])
            axis_long = axis_long / (np.linalg.norm(axis_long) + 1.0e-12)
            axis_width = axis_width / (np.linalg.norm(axis_width) + 1.0e-12)
            surfaces.append(
                {
                    "center": multibody_point_to_mpm(center_world),
                    "axis_long": axis_long,
                    "axis_width": axis_width,
                    "half_extent": np.array([0.5 * tank.geom.shoe_length, 0.5 * tank.geom.track_width], dtype=float),
                }
            )
    return surfaces


def estimate_soil_surface_under_tracks(
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank,
    args: argparse.Namespace,
) -> float:
    surfaces = bottom_track_contact_surfaces_mpm(tank, args)
    percentile = float(args.settle_surface_percentile)
    if getattr(args, "surface_estimator", "gpu") == "gpu":
        gpu_estimate = solver.estimate_surface_z_from_patches(surfaces, percentile)
        if gpu_estimate is not None:
            return gpu_estimate

    ti.sync()
    soil_pos = solver.x.to_numpy().astype(np.float64)
    z_samples: list[np.ndarray] = []
    for surface in surfaces:
        center = surface["center"]
        q = soil_pos - center[None, :]
        a = q @ surface["axis_long"]
        b = q @ surface["axis_width"]
        half_l = float(surface["half_extent"][0]) + solver.dx
        half_w = float(surface["half_extent"][1]) + solver.dx
        mask = (np.abs(a) <= half_l) & (np.abs(b) <= half_w)
        if np.any(mask):
            z_samples.append(soil_pos[mask, 2])

    if not z_samples:
        return solver.get_soil_surface_z(percentile)

    z = np.concatenate(z_samples)
    q = float(max(0.0, min(100.0, percentile)))
    return float(np.percentile(z, q))


def multibody_point_to_mpm(point: Iterable[float]) -> np.ndarray:
    px, py, pz = np.asarray(point, dtype=float)
    return np.array([pz, px, py], dtype=float)


def multibody_vector_to_mpm(vector: Iterable[float]) -> np.ndarray:
    vx, vy, vz = np.asarray(vector, dtype=float)
    return np.array([vz, vx, vy], dtype=float)


def mpm_point_to_multibody(point: Iterable[float]) -> np.ndarray:
    px, py, pz = np.asarray(point, dtype=float)
    return np.array([py, pz, px], dtype=float)


def mpm_vector_to_multibody(vector: Iterable[float]) -> np.ndarray:
    vx, vy, vz = np.asarray(vector, dtype=float)
    return np.array([vy, vz, vx], dtype=float)


def append_mesh(target, source) -> None:
    point_offset = len(target.points)
    target.points.extend(np.asarray(point, dtype=float) for point in source.points)
    target.faces.extend([[point_offset + int(index) for index in face] for face in source.faces])
    target.part_kind.extend(source.part_kind)
    target.body_id.extend(source.body_id)
    target.contact_flag.extend(source.contact_flag)
    target.contact_pressure.extend(source.contact_pressure)
    target.contact_normal_force.extend(source.contact_normal_force)
    target.track_shoe_deflection.extend(source.track_shoe_deflection)
    target.track_shoe_pitch.extend(source.track_shoe_pitch)


def write_tank_mpm_vtk(
    tank: MultibodyRigidGroundTank,
    path: Path,
    *,
    include_outer_grousers: bool = False,
    visual_detail: str = VISUAL_DETAIL_SIMPLE,
    track_contact_scalars: dict[int, dict[str, float]] | None = None,
    use_deformed_track: bool = False,
    track_backend: TrackBackend | None = None,
) -> None:
    mesh = build_multibody_tank_mesh(
        tank,
        include_outer_grousers=include_outer_grousers,
        include_ground_plane=False,
        visual_detail=visual_detail,
        track_contact_scalars=track_contact_scalars,
        use_deformed_track=use_deformed_track,
        include_tracks=track_backend is None,
        include_running_gear=track_backend is None,
    )
    if track_backend is not None:
        append_mesh(
            mesh,
            track_backend.build_mesh(
                include_outer_grousers=include_outer_grousers,
                track_contact_scalars=track_contact_scalars,
            ),
        )
    mesh.points = [multibody_point_to_mpm(point) for point in mesh.points]
    mesh.write_vtk(path)


def track_contact_scalars_from_solver(
    tank: MultibodyRigidGroundTank,
    solver: TankTrackMpmSolver,
    track_backend: TrackBackend | None = None,
) -> dict[int, dict[str, float]]:
    tank._ensure_track_shoe_dynamic_state(tank.state)
    scalars: dict[int, dict[str, float]] = {}
    if track_backend is not None and hasattr(track_backend, "shoe_records"):
        for record in track_backend.shoe_records():
            body_id = int(record["body_id"])
            scalars[body_id] = {
                "contact_pressure": 0.0,
                "contact_normal_force": 0.0,
                "contact_flag": 0.0,
                "track_shoe_deflection": 0.0,
                "track_shoe_pitch": 0.0,
                "_contact_area": 0.0,
            }
    else:
        for side in ("left", "right"):
            normal_deflections = tank.state.track_shoe_normal_deflections.get(side, [])
            pitch_deflections = tank.state.track_shoe_pitch_deflections.get(side, [])
            for index in range(tank.shoe_count.get(side, 0)):
                body_id = int(tank.track_shoe_body_index[(side, index)])
                scalars[body_id] = {
                    "contact_pressure": 0.0,
                    "contact_normal_force": 0.0,
                    "contact_flag": 0.0,
                    "track_shoe_deflection": float(normal_deflections[index]) if index < len(normal_deflections) else 0.0,
                    "track_shoe_pitch": float(pitch_deflections[index]) if index < len(pitch_deflections) else 0.0,
                    "_contact_area": 0.0,
                }

    metadata = solver.averaged_track_contact_by_patch()
    forces = np.asarray(metadata.get("force", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    shoe_ids = np.asarray(metadata.get("shoe_id", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    normals = np.asarray(metadata.get("normal", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    half_extents = np.asarray(metadata.get("half_extent", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    count = min(forces.shape[0], shoe_ids.size, normals.shape[0], half_extents.shape[0])
    for patch_index in range(count):
        body_id = int(round(float(shoe_ids[patch_index])))
        record = scalars.setdefault(
            body_id,
            {
                "contact_pressure": 0.0,
                "contact_normal_force": 0.0,
                "contact_flag": 0.0,
                "track_shoe_deflection": 0.0,
                "track_shoe_pitch": 0.0,
                "_contact_area": 0.0,
            },
        )
        area = max(4.0 * float(half_extents[patch_index, 0]) * float(half_extents[patch_index, 1]), 1.0e-12)
        normal = normals[patch_index]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1.0e-12:
            normal = normal / normal_norm
            normal_force = max(0.0, float(-np.dot(forces[patch_index], normal)))
        else:
            normal_force = max(0.0, float(forces[patch_index, 2]))
        if normal_force <= 1.0e-12:
            continue
        record["contact_normal_force"] += normal_force
        record["_contact_area"] += area
        record["contact_pressure"] = record["contact_normal_force"] / max(record["_contact_area"], 1.0e-12)
        record["contact_flag"] = 1.0

    for record in scalars.values():
        record.pop("_contact_area", None)
    return scalars


def save_tank_state(path: Path | str, tank: MultibodyRigidGroundTank, args: argparse.Namespace, stage: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "model": str(Path(args.model)),
        "pose_reference": "cg",
        "state": asdict(tank.state),
        "mass": tank.geom.mass,
        "track_pitch": tank.geom.track_pitch,
        "track_width": tank.geom.track_width,
        "shoe_length": tank.geom.shoe_length,
        "track_shoe_mass": float(tank.track_shoe_mass_value),
        "track_shoe_pitch_inertia": float(tank.track_shoe_pitch_inertia_value),
        "total_track_shoe_mass": float(tank.total_track_shoe_mass),
        "track_shoe_dynamics": bool(tank.track_shoe_dynamics_enabled),
        "track_shoe_normal_stiffness": float(tank.track_shoe_normal_stiffness),
        "track_shoe_normal_damping": float(tank.track_shoe_normal_damping),
        "track_pin_bending_stiffness": float(tank.track_pin_bending_stiffness),
        "track_pin_bending_damping": float(tank.track_pin_bending_damping),
        "track_pin_pitch_stiffness": float(tank.track_pin_pitch_stiffness),
        "track_pin_pitch_damping": float(tank.track_pin_pitch_damping),
        "track_shoe_max_deflection": float(tank.track_shoe_max_deflection),
        "track_shoe_max_pitch": float(tank.track_shoe_max_pitch),
        "mpm_dt": float(args.mpm_dt),
        "mbd_dt": float(args.mbd_dt),
        "mbd_substeps": int(args.mbd_substeps),
        "moving_window": bool(args.moving_window),
        "moving_window_template_layers": int(args.moving_window_template_layers),
        "soil_constitutive_model": args.soil_constitutive_model,
        "srsh_rate_exponent": float(args.srsh_rate_exponent),
        "srsh_rate_sensitivity": float(args.srsh_rate_sensitivity),
        "srsh_saturation_increment": float(args.srsh_saturation_increment),
        "srsh_saturation_plastic_strain": float(
            args.srsh_saturation_plastic_strain
        ),
        "soil_mcc_m": args.soil_mcc_m,
        "soil_mcc_lambda": float(args.soil_mcc_lambda),
        "soil_mcc_kappa": float(args.soil_mcc_kappa),
        "soil_mcc_e0": float(args.soil_mcc_e0),
        "soil_mcc_pc0": float(args.soil_mcc_pc0),
        "soil_mcc_min_pressure": float(args.soil_mcc_min_pressure),
        "water_bulk_modulus": float(args.water_bulk_modulus),
        "water_dynamic_viscosity": float(args.water_dynamic_viscosity),
        "water_cavitation_pressure": float(args.water_cavitation_pressure),
        "contact_barrier_stiffness": (
            None if args.contact_barrier_stiffness is None else float(args.contact_barrier_stiffness)
        ),
        "contact_barrier_radius": None if args.contact_barrier_radius is None else float(args.contact_barrier_radius),
        "contact_barrier_min_distance_ratio": float(args.contact_barrier_min_distance_ratio),
        "contact_slip_smoothing_distance": (
            None if args.contact_slip_smoothing_distance is None else float(args.contact_slip_smoothing_distance)
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tank_state(path: Path | str, tank: MultibodyRigidGroundTank) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tank state file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["state"]
    pose_reference = str(payload.get("pose_reference", "base_reference"))
    joint_angles = {name: float(value) for name, value in raw.get("joint_angles", {}).items()}
    joint_rates = {name: float(value) for name, value in raw.get("joint_rates", {}).items()}
    for name in tank.state.joint_angles:
        joint_angles.setdefault(name, 0.0)
        joint_rates.setdefault(name, 0.0)

    def track_state_dict(name: str) -> dict[str, list[float]]:
        raw_mapping = raw.get(name, {})
        if not isinstance(raw_mapping, dict):
            raw_mapping = {}
        return {
            side: [float(value) for value in raw_mapping.get(side, [])]
            for side in ("left", "right")
        }

    loaded_state = MultibodyState(
        t=float(raw.get("t", 0.0)),
        forward_pos=float(raw.get("forward_pos", 0.0)),
        lateral_pos=float(raw.get("lateral_pos", 0.0)),
        heave=float(raw.get("heave", 0.0)),
        heave_rate=float(raw.get("heave_rate", 0.0)),
        yaw=float(raw.get("yaw", 0.0)),
        pitch=float(raw.get("pitch", 0.0)),
        pitch_rate=float(raw.get("pitch_rate", 0.0)),
        forward_speed=float(raw.get("forward_speed", 0.0)),
        yaw_rate=float(raw.get("yaw_rate", 0.0)),
        left_track_phase=float(raw.get("left_track_phase", 0.0)),
        right_track_phase=float(raw.get("right_track_phase", 0.0)),
        left_sprocket_omega=float(raw.get("left_sprocket_omega", 0.0)),
        right_sprocket_omega=float(raw.get("right_sprocket_omega", 0.0)),
        joint_angles=joint_angles,
        joint_rates=joint_rates,
        track_shoe_normal_deflections=track_state_dict("track_shoe_normal_deflections"),
        track_shoe_normal_rates=track_state_dict("track_shoe_normal_rates"),
        track_shoe_pitch_deflections=track_state_dict("track_shoe_pitch_deflections"),
        track_shoe_pitch_rates=track_state_dict("track_shoe_pitch_rates"),
    )
    if pose_reference != "cg":
        cg_offset = tank.vector_to_world(tank.cg_reference - tank.base_reference, state=loaded_state)
        loaded_state.lateral_pos += float(cg_offset[0])
        loaded_state.heave += float(cg_offset[1])
        loaded_state.forward_pos += float(cg_offset[2])
    tank.state = loaded_state
    tank._ensure_track_shoe_dynamic_state(tank.state)
    tank.track_pose_cache.clear()


def scale_tank_rates(tank: MultibodyRigidGroundTank, factor: float) -> None:
    scale = max(0.0, min(1.0, float(factor)))
    tank.state.forward_speed *= scale
    tank.state.heave_rate *= scale
    tank.state.yaw_rate *= scale
    tank.state.pitch_rate *= scale
    tank.state.left_sprocket_omega *= scale
    tank.state.right_sprocket_omega *= scale
    for key in list(tank.state.joint_rates):
        tank.state.joint_rates[key] *= scale
    for side, values in tank.state.track_shoe_normal_rates.items():
        tank.state.track_shoe_normal_rates[side] = [float(value) * scale for value in values]
    for side, values in tank.state.track_shoe_pitch_rates.items():
        tank.state.track_shoe_pitch_rates[side] = [float(value) * scale for value in values]
    tank.track_pose_cache.clear()


def settle_weight_scale(args: argparse.Namespace, step: int) -> float:
    return 1.0


def tank_kinetic_energy_estimate(tank: MultibodyRigidGroundTank) -> float:
    g = tank.geom
    s = tank.state
    translational = 0.5 * g.mass * (s.forward_speed * s.forward_speed + s.heave_rate * s.heave_rate)
    yaw = 0.5 * g.yaw_inertia * s.yaw_rate * s.yaw_rate
    pitch = 0.5 * g.pitch_inertia * s.pitch_rate * s.pitch_rate
    sprockets = 0.5 * tank.sprocket_inertia * (
        s.left_sprocket_omega * s.left_sprocket_omega
        + s.right_sprocket_omega * s.right_sprocket_omega
    )
    station_mass = g.mass / max(1, len(tank.state.joint_rates))
    joint = 0.5 * station_mass * sum(rate * rate for rate in tank.state.joint_rates.values())
    track_shoes = 0.0
    for side in ("left", "right"):
        normal_rates = np.asarray(tank.state.track_shoe_normal_rates.get(side, []), dtype=float)
        pitch_rates = np.asarray(tank.state.track_shoe_pitch_rates.get(side, []), dtype=float)
        if normal_rates.size:
            mass = tank.track_shoe_mass.get(side, np.zeros(normal_rates.size, dtype=float))
            track_shoes += float(0.5 * np.sum(mass[: normal_rates.size] * normal_rates * normal_rates))
        if pitch_rates.size:
            inertia = tank.track_shoe_pitch_inertia.get(side, np.zeros(pitch_rates.size, dtype=float))
            track_shoes += float(0.5 * np.sum(inertia[: pitch_rates.size] * pitch_rates * pitch_rates))
    return float(translational + yaw + pitch + sprockets + joint + track_shoes)


def apply_settle_kinetic_damping(
    args: argparse.Namespace,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank,
    state: KineticDampingState,
) -> None:
    energy = tank_kinetic_energy_estimate(tank)
    previous = state.previous_energy
    min_energy = max(0.0, float(args.settle_kinetic_damping_min_energy))
    if previous is not None and previous > min_energy and energy < previous:
        scale_tank_rates(tank, float(args.settle_kinetic_damping_factor))
        solver.scale_soil_kinematics(float(args.settle_soil_kinetic_damping_factor))
        state.event_count += 1
        energy = tank_kinetic_energy_estimate(tank)
    state.previous_energy = energy


def feedback_force_limit_scales(
    tank: MultibodyRigidGroundTank,
    forces: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float]:
    sampled = np.asarray(forces, dtype=np.float64)
    if sampled.shape != (2, 3):
        return 1.0, 1.0
    weight = tank.geom.mass * tank.geom.gravity
    normal_limit = max(0.1, float(args.max_feedback_normal_factor)) * weight
    traction_limit = max(0.1, float(args.max_feedback_traction_factor)) * weight

    normal_scale = 1.0
    total_fz = float(np.sum(sampled[:, 2]))
    if total_fz > normal_limit:
        normal_scale = normal_limit / max(total_fz, 1.0e-12)
    elif total_fz < -normal_limit:
        normal_scale = -normal_limit / min(total_fz, -1.0e-12)

    horizontal_scale = 1.0
    horizontal = np.sum(sampled[:, :2], axis=0)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm > traction_limit:
        horizontal_scale = traction_limit / max(horizontal_norm, 1.0e-12)
    return float(normal_scale), float(horizontal_scale)


def scale_track_feedback(
    feedback: TrackTerrainFeedback,
    *,
    normal_scale: float,
    horizontal_scale: float,
) -> TrackTerrainFeedback:
    side_forces = np.asarray(feedback.side_forces, dtype=np.float64).copy()
    side_forces[:, :2] *= horizontal_scale
    side_forces[:, 2] *= normal_scale

    patch_forces = np.asarray(feedback.patch_forces, dtype=np.float64).copy()
    if patch_forces.size:
        patch_forces[:, :2] *= horizontal_scale
        patch_forces[:, 2] *= normal_scale

    moment_scale = min(abs(normal_scale), abs(horizontal_scale))
    return TrackTerrainFeedback(
        side_forces=side_forces,
        patch_centers=np.asarray(feedback.patch_centers, dtype=np.float64),
        patch_forces=patch_forces,
        patch_moments=np.asarray(feedback.patch_moments, dtype=np.float64) * moment_scale,
        patch_side_ids=np.asarray(feedback.patch_side_ids, dtype=np.int32),
        patch_shoe_ids=np.asarray(feedback.patch_shoe_ids, dtype=np.float64),
    )


def make_track_patches_from_chrono_records(
    records: list[dict[str, np.ndarray]],
    tank: MultibodyRigidGroundTank,
    args: argparse.Namespace,
    *,
    previous_records: list[dict[str, np.ndarray]],
    contact_dt: float,
) -> dict[str, np.ndarray]:
    centers = []
    axes_long = []
    axes_width = []
    normals = []
    velocities = []
    half_extents = []
    patch_masses = []
    shoe_ids = []
    side_ids = []
    previous_by_key = {
        (str(record["side"]), int(record["index"])): record
        for record in previous_records
    }
    surface_offset = 0.5 * tank.geom.shoe_thickness + contact_surface_extra(tank, args)
    dt = max(float(contact_dt), 1.0e-12)

    for record in records:
        side = str(record["side"])
        index = int(record["index"])
        previous = previous_by_key.get((side, index), record)
        center_world = np.asarray(record["center"], dtype=np.float64)
        previous_center_world = np.asarray(previous["center"], dtype=np.float64)
        axis_long_world = np.asarray(record["axis_long"], dtype=np.float64)
        axis_width_world = np.asarray(record["axis_width"], dtype=np.float64)
        normal_world = np.asarray(record["axis_normal"], dtype=np.float64)
        previous_normal_world = np.asarray(previous["axis_normal"], dtype=np.float64)
        if normal_world[1] > 0.0:
            normal_world = -normal_world
        if previous_normal_world[1] > 0.0:
            previous_normal_world = -previous_normal_world

        contact_center_world = center_world + normal_world * surface_offset
        previous_contact_center_world = previous_center_world + previous_normal_world * surface_offset
        center_mpm = multibody_point_to_mpm(contact_center_world)
        previous_center_mpm = multibody_point_to_mpm(previous_contact_center_world)
        axis_long_mpm = multibody_vector_to_mpm(axis_long_world)
        axis_width_mpm = multibody_vector_to_mpm(axis_width_world)
        normal_mpm = multibody_vector_to_mpm(normal_world)
        axis_long_mpm = axis_long_mpm / (np.linalg.norm(axis_long_mpm) + 1.0e-12)
        axis_width_mpm = axis_width_mpm / (np.linalg.norm(axis_width_mpm) + 1.0e-12)
        normal_mpm = normal_mpm / (np.linalg.norm(normal_mpm) + 1.0e-12)
        half_extent = np.array([0.5 * tank.geom.shoe_length, 0.5 * tank.geom.track_width], dtype=np.float64)

        centers.append(center_mpm)
        axes_long.append(axis_long_mpm)
        axes_width.append(axis_width_mpm)
        normals.append(normal_mpm)
        velocities.append((center_mpm - previous_center_mpm) / dt)
        half_extents.append(half_extent)
        side_masses = tank.track_shoe_mass.get(side, np.zeros((0,), dtype=float))
        if 0 <= index < len(side_masses):
            patch_masses.append(float(side_masses[index]))
        else:
            patch_masses.append(float(tank.track_shoe_mass_value))
        shoe_ids.append(float(record["body_id"]))
        side_ids.append(0 if side == "left" else 1)

    if not centers:
        return {
            "center": np.zeros((0, 3), dtype=np.float32),
            "axis_long": np.zeros((0, 3), dtype=np.float32),
            "axis_width": np.zeros((0, 3), dtype=np.float32),
            "normal": np.zeros((0, 3), dtype=np.float32),
            "velocity": np.zeros((0, 3), dtype=np.float32),
            "half_extent": np.zeros((0, 2), dtype=np.float32),
            "mass": np.zeros((0,), dtype=np.float32),
            "shoe_id": np.zeros((0,), dtype=np.float32),
            "side_id": np.zeros((0,), dtype=np.int32),
        }
    return {
        "center": np.asarray(centers, dtype=np.float32),
        "axis_long": np.asarray(axes_long, dtype=np.float32),
        "axis_width": np.asarray(axes_width, dtype=np.float32),
        "normal": np.asarray(normals, dtype=np.float32),
        "velocity": np.asarray(velocities, dtype=np.float32),
        "half_extent": np.asarray(half_extents, dtype=np.float32),
        "mass": np.asarray(patch_masses, dtype=np.float32),
        "shoe_id": np.asarray(shoe_ids, dtype=np.float32),
        "side_id": np.asarray(side_ids, dtype=np.int32),
    }


def track_patch_topology_matches(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> bool:
    if first["center"].shape != second["center"].shape:
        return False
    if first["side_id"].shape != second["side_id"].shape:
        return False
    if first["shoe_id"].shape != second["shoe_id"].shape:
        return False
    return bool(
        np.array_equal(first["side_id"], second["side_id"])
        and np.array_equal(first["shoe_id"], second["shoe_id"])
    )


def unit_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=True)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1.0e-12)).astype(np.float32, copy=False)


def interpolate_track_patches(
    start: dict[str, np.ndarray],
    end: dict[str, np.ndarray],
    alpha_previous: float,
    alpha_current: float,
    dt: float,
) -> dict[str, np.ndarray]:
    a0 = max(0.0, min(1.0, float(alpha_previous)))
    a1 = max(0.0, min(1.0, float(alpha_current)))
    start_center = np.asarray(start["center"], dtype=np.float32)
    end_center = np.asarray(end["center"], dtype=np.float32)
    delta_center = end_center - start_center
    previous_center = start_center + a0 * delta_center
    current_center = start_center + a1 * delta_center
    start_axis_long = np.asarray(start["axis_long"], dtype=np.float32)
    end_axis_long = np.asarray(end["axis_long"], dtype=np.float32)
    start_axis_width = np.asarray(start["axis_width"], dtype=np.float32)
    end_axis_width = np.asarray(end["axis_width"], dtype=np.float32)
    start_normal = np.asarray(start["normal"], dtype=np.float32)
    end_normal = np.asarray(end["normal"], dtype=np.float32)
    previous_axis_long = unit_rows(start_axis_long + a0 * (end_axis_long - start_axis_long))
    current_axis_long = unit_rows(start_axis_long + a1 * (end_axis_long - start_axis_long))
    previous_axis_width = unit_rows(start_axis_width + a0 * (end_axis_width - start_axis_width))
    current_axis_width = unit_rows(start_axis_width + a1 * (end_axis_width - start_axis_width))

    patches = {
        "center": current_center.astype(np.float32, copy=False),
        "axis_long": current_axis_long,
        "axis_width": current_axis_width,
        "normal": unit_rows(start_normal + a1 * (end_normal - start_normal)),
        "velocity": ((current_center - previous_center) / max(float(dt), 1.0e-12)).astype(np.float32, copy=False),
        "previous_center": previous_center.astype(np.float32, copy=False),
        "previous_axis_long": previous_axis_long,
        "previous_axis_width": previous_axis_width,
        "dt": np.float32(max(float(dt), 1.0e-12)),
        "half_extent": np.asarray(end["half_extent"], dtype=np.float32),
        "mass": np.asarray(end["mass"], dtype=np.float32),
        "shoe_id": np.asarray(end["shoe_id"], dtype=np.float32),
        "side_id": np.asarray(end["side_id"], dtype=np.int32),
    }
    return patches


def macro_substeps_until_sync(args: argparse.Namespace, step: int, end_step: int) -> int:
    count = min(int(args.mbd_substeps), int(end_step - step))
    if args.save_every > 0:
        next_save = ((int(step) // int(args.save_every)) + 1) * int(args.save_every)
        if int(step) < next_save <= int(end_step):
            count = min(count, next_save - int(step))
    return max(1, int(count))


def advance_mpm_with_chrono_track_motion(
    args: argparse.Namespace,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank,
    *,
    substeps: int,
    macro_dt: float,
    chrono_start_records: list[dict[str, Any]],
    chrono_end_records: list[dict[str, Any]],
) -> None:
    use_gpu_chrono_track = getattr(args, "track_update_mode", "gpu") == "gpu"
    start_patches = make_track_patches_from_chrono_records(
        chrono_start_records,
        tank,
        args,
        previous_records=chrono_start_records,
        contact_dt=macro_dt,
    )
    end_patches = make_track_patches_from_chrono_records(
        chrono_end_records,
        tank,
        args,
        previous_records=chrono_start_records,
        contact_dt=macro_dt,
    )
    solver.reset_contact_force_average()
    topology_matches = track_patch_topology_matches(start_patches, end_patches)
    patch_update_every = max(1, int(getattr(args, "patch_update_every", 1)))
    if use_gpu_chrono_track and topology_matches:
        solver.set_track_patch_keyframes(
            start_patches,
            end_patches,
        )
        for block_start in range(0, substeps, patch_update_every):
            block_count = min(patch_update_every, substeps - block_start)
            solver.update_track_patches_from_keyframes(
                block_start / float(substeps),
                (block_start + block_count) / float(substeps),
                block_count * args.mpm_dt,
            )
            for _ in range(block_count):
                solver.substep()
                solver.accumulate_contact_force_average()
        return

    for block_start in range(0, substeps, patch_update_every):
        block_count = min(patch_update_every, substeps - block_start)
        if topology_matches:
            solver.set_track_patches(
                interpolate_track_patches(
                    start_patches,
                    end_patches,
                    block_start / float(substeps),
                    (block_start + block_count) / float(substeps),
                    block_count * args.mpm_dt,
                )
            )
        else:
            solver.set_track_patches(end_patches)
        for _ in range(block_count):
            solver.substep()
            solver.accumulate_contact_force_average()


def advance_multirate_coupled_macro(
    args: argparse.Namespace,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank,
    *,
    substeps: int,
    left_drive_torque: float,
    right_drive_torque: float,
    feedback_forces: TrackTerrainFeedback,
    track_backend: TrackBackend | None = None,
    body_gravity_scale: float = 1.0,
    settle_planar_constraint: bool = False,
    update_moving_window: bool = False,
) -> TrackTerrainFeedback:
    if track_backend is None:
        raise RuntimeError("Chrono vehicle track backend is required for coupled MPM calculation.")
    if not hasattr(track_backend, "capture_state") or not hasattr(track_backend, "restore_state"):
        raise RuntimeError("Predictor-corrector coupling requires a Chrono backend with state capture/restore.")
    macro_dt = float(substeps) * float(args.mpm_dt)
    chrono_snapshot = track_backend.capture_state()
    chrono_start_records = list(track_backend.shoe_records())

    predictor_feedback = feedback_forces
    predictor_normal_scale, predictor_horizontal_scale = feedback_force_limit_scales(
        tank,
        predictor_feedback.side_forces,
        args,
    )
    predictor_limited_feedback = scale_track_feedback(
        predictor_feedback,
        normal_scale=predictor_normal_scale,
        horizontal_scale=predictor_horizontal_scale,
    )
    apply_track_feedback_to_backend(track_backend, tank, predictor_limited_feedback)
    track_backend.step(
        left_drive_torque=left_drive_torque,
        right_drive_torque=right_drive_torque,
        dt=macro_dt,
        settle_planar_constraint=settle_planar_constraint,
        gravity_scale=body_gravity_scale,
    )
    chrono_predicted_end_records = list(track_backend.shoe_records())

    advance_mpm_with_chrono_track_motion(
        args,
        solver,
        tank,
        substeps=substeps,
        macro_dt=macro_dt,
        chrono_start_records=chrono_start_records,
        chrono_end_records=chrono_predicted_end_records,
    )
    current_feedback = track_feedback_from_solver(solver)

    track_backend.restore_state(chrono_snapshot)
    current_normal_scale, current_horizontal_scale = feedback_force_limit_scales(
        tank,
        current_feedback.side_forces,
        args,
    )
    current_limited_feedback = scale_track_feedback(
        current_feedback,
        normal_scale=current_normal_scale,
        horizontal_scale=current_horizontal_scale,
    )
    apply_track_feedback_to_backend(track_backend, tank, current_limited_feedback)
    track_backend.step(
        left_drive_torque=left_drive_torque,
        right_drive_torque=right_drive_torque,
        dt=macro_dt,
        settle_planar_constraint=settle_planar_constraint,
        gravity_scale=body_gravity_scale,
    )
    tank.track_pose_cache.clear()
    if update_moving_window:
        solver.advance_moving_window(float(tank.state.forward_pos))
    return current_feedback


def export_snapshot(
    out_dir: Path,
    stage: str,
    step: int,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank | None = None,
    *,
    include_outer_grousers: bool = False,
    visual_detail: str = VISUAL_DETAIL_SIMPLE,
    track_backend: TrackBackend | None = None,
) -> None:
    stage_dir = out_dir / stage
    solver.export_soil(stage_dir, step)
    if tank is not None:
        write_tank_mpm_vtk(
            tank,
            stage_dir / f"tank_{step:06d}.vtk",
            include_outer_grousers=include_outer_grousers,
            visual_detail=visual_detail,
            track_contact_scalars=track_contact_scalars_from_solver(tank, solver, track_backend=track_backend),
            track_backend=track_backend,
        )


def diagnostics_row(
    stage: str,
    step: int,
    stage_time: float,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank | None,
) -> MultibodyMpmDiagnostics:
    ti.sync()
    soil_pos = solver.x.to_numpy()
    soil_vel = solver.v.to_numpy()
    speed = np.linalg.norm(soil_vel, axis=1)
    side_forces = solver.averaged_contact_forces_by_side()
    side_particles = solver.contact_particle_counts_by_side()
    total_force = side_forces.sum(axis=0)

    if tank is None:
        tank_x = tank_y = heave = yaw = pitch = forward_speed = heave_rate = 0.0
        left_track_speed = right_track_speed = left_omega = right_omega = 0.0
        t = stage_time
    else:
        sprocket_radius = tank.geom.drive_sprocket_radius
        tank_x = tank.state.forward_pos
        tank_y = tank.state.lateral_pos
        heave = tank.state.heave
        yaw = tank.state.yaw
        pitch = tank.state.pitch
        forward_speed = tank.state.forward_speed
        heave_rate = tank.state.heave_rate
        left_omega = tank.state.left_sprocket_omega
        right_omega = tank.state.right_sprocket_omega
        left_track_speed = -left_omega * sprocket_radius
        right_track_speed = -right_omega * sprocket_radius
        t = tank.state.t
        max_track_shoe_deflection = max(
            (
                max((abs(float(value)) for value in tank.state.track_shoe_normal_deflections.get(side, [])), default=0.0)
                for side in ("left", "right")
            ),
            default=0.0,
        )
        max_track_shoe_pitch = max(
            (
                max((abs(float(value)) for value in tank.state.track_shoe_pitch_deflections.get(side, [])), default=0.0)
                for side in ("left", "right")
            ),
            default=0.0,
        )
    if tank is None:
        max_track_shoe_deflection = 0.0
        max_track_shoe_pitch = 0.0

    return MultibodyMpmDiagnostics(
        stage=stage,
        step=step,
        stage_time=stage_time,
        t=t,
        tank_x=tank_x,
        tank_y=tank_y,
        tank_heave=heave,
        tank_yaw=yaw,
        tank_pitch=pitch,
        tank_forward_speed=forward_speed,
        tank_heave_rate=heave_rate,
        left_track_speed=left_track_speed,
        right_track_speed=right_track_speed,
        left_sprocket_omega=left_omega,
        right_sprocket_omega=right_omega,
        active_patches=solver.active_patch_count_value(),
        contact_particles=int(solver.contact_particle_count[None]),
        max_track_shoe_deflection=float(max_track_shoe_deflection),
        max_track_shoe_pitch=float(max_track_shoe_pitch),
        left_contact_particles=int(side_particles[0]),
        right_contact_particles=int(side_particles[1]),
        left_force_x=float(side_forces[0, 0]),
        left_force_y=float(side_forces[0, 1]),
        left_force_z=float(side_forces[0, 2]),
        right_force_x=float(side_forces[1, 0]),
        right_force_y=float(side_forces[1, 1]),
        right_force_z=float(side_forces[1, 2]),
        total_force_x=float(total_force[0]),
        total_force_y=float(total_force[1]),
        total_force_z=float(total_force[2]),
        max_soil_speed=float(np.max(speed)) if speed.size else 0.0,
        mean_soil_z=float(np.mean(soil_pos[:, 2])) if soil_pos.size else 0.0,
        min_soil_z=float(np.min(soil_pos[:, 2])) if soil_pos.size else 0.0,
        max_soil_z=float(np.max(soil_pos[:, 2])) if soil_pos.size else 0.0,
    )


def write_diagnostics_csv(path: Path, rows: list[MultibodyMpmDiagnostics], append: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MultibodyMpmDiagnostics.__dataclass_fields__.keys())
    write_header = (not append) or (not path.exists()) or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def stage_progress_path(out_dir: Path, stage: str) -> Path:
    return out_dir / f"{stage}_progress.json"


def load_next_stage_step(out_dir: Path, stage: str) -> int:
    progress_path = stage_progress_path(out_dir, stage)
    if progress_path.exists():
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            return int(payload.get("last_step", -1)) + 1
        except Exception:
            pass

    diagnostics_path = out_dir / f"{stage}_diagnostics.csv"
    if not diagnostics_path.exists():
        return 0
    max_step = -1
    try:
        with diagnostics_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                max_step = max(max_step, int(float(row.get("step", -1))))
    except Exception:
        return 0
    return max_step + 1


def save_stage_progress(
    out_dir: Path,
    stage: str,
    *,
    last_step: int,
    dt: float,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank | None = None,
) -> None:
    payload = {
        "stage": stage,
        "last_step": int(last_step),
        "stage_time": float(last_step) * float(dt),
        "solver_coupled_step": int(solver.coupled_step),
    }
    if tank is not None:
        payload["tank_t"] = float(tank.state.t)
    stage_progress_path(out_dir, stage).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_geostatic(
    args: argparse.Namespace,
    solver: TankTrackMpmSolver,
    out_dir: Path,
    *,
    start_step: int = 0,
    append_diagnostics: bool = False,
    initialize_stress: bool = True,
) -> None:
    rows: list[MultibodyMpmDiagnostics] = []
    solver.set_track_patches(
        {
            "center": np.zeros((0, 3), dtype=np.float32),
            "axis_long": np.zeros((0, 3), dtype=np.float32),
            "axis_width": np.zeros((0, 3), dtype=np.float32),
            "normal": np.zeros((0, 3), dtype=np.float32),
            "velocity": np.zeros((0, 3), dtype=np.float32),
            "half_extent": np.zeros((0, 2), dtype=np.float32),
            "side_id": np.zeros((0,), dtype=np.int32),
        }
    )
    saved_gravity = solver.soil_gravity_scale
    saved_damping = solver.soil_damping
    solver.soil_gravity_scale = float(args.geostatic_soil_gravity_scale)
    solver.soil_damping = float(max(0.0, min(1.0, args.geostatic_soil_damping)))
    if args.geostatic_initial_stress and initialize_stress:
        k0 = resolve_geostatic_k0(args)
        solver.initialize_geostatic_stress(k0, float(args.geostatic_soil_gravity_scale))
        print(f"[Info] Geostatic initial stress enabled, K0={k0:.6f}")
    elif not initialize_stress:
        print("[Info] Geostatic resume: keeping loaded stress/particle state.")

    print(
        f"[Info] Stage 1 geostatic: start_step={start_step}, "
        f"steps={args.geostatic_steps}, mpm_dt={args.mpm_dt:.3e}"
    )
    for local_step in range(args.geostatic_steps + 1):
        step = start_step + local_step
        stage_time = step * args.dt
        if args.save_every > 0 and step % args.save_every == 0:
            export_snapshot(out_dir, "geostatic", step, solver, tank=None)
            row = diagnostics_row("geostatic", step, stage_time, solver, None)
            rows.append(row)
            print(
                f"[Output] geostatic step={step:06d} "
                f"surface_z={row.max_soil_z:.4f} max_v={row.max_soil_speed:.3e}"
            )
        if local_step == args.geostatic_steps:
            break
        solver.substep_geostatic()

    solver.soil_gravity_scale = saved_gravity
    solver.soil_damping = saved_damping
    solver.capture_moving_window_template()
    solver.save_state(args.geostatic_state, soil_gravity_scale=args.geostatic_soil_gravity_scale)
    write_diagnostics_csv(out_dir / "geostatic_diagnostics.csv", rows, append=append_diagnostics)
    save_stage_progress(
        out_dir,
        "geostatic",
        last_step=start_step + args.geostatic_steps,
        dt=args.dt,
        solver=solver,
    )
    print(f"[Info] Geostatic state saved: {Path(args.geostatic_state).resolve()}")


def run_settle(
    args: argparse.Namespace,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank,
    out_dir: Path,
    *,
    start_step: int = 0,
    append_diagnostics: bool = False,
    track_backend: TrackBackend | None = None,
) -> None:
    rows: list[MultibodyMpmDiagnostics] = []
    last_forces = track_feedback_from_solver(solver)
    kinetic_state = KineticDampingState()
    saved_soil_damping = solver.soil_damping
    saved_vertical_damping = args.vertical_damping
    saved_pitch_rate_damping = args.pitch_rate_damping
    saved_suspension_damping = tank.suspension_damping
    if args.settle_soil_damping is not None:
        solver.soil_damping = float(max(0.0, min(1.0, args.settle_soil_damping)))
    if args.settle_vertical_damping is not None:
        args.vertical_damping = float(max(0.0, args.settle_vertical_damping))
    if args.settle_pitch_rate_damping is not None:
        args.pitch_rate_damping = float(max(0.0, args.settle_pitch_rate_damping))
    if args.settle_suspension_damping is not None:
        tank.suspension_damping = float(max(0.0, args.settle_suspension_damping))
    print(
        f"[Info] Stage 2 tank self-weight settling: start_step={start_step}, "
        f"steps={args.settle_steps}, mpm_dt={args.mpm_dt:.3e}, mbd_dt={args.mbd_dt:.3e}, "
        f"mbd_substeps={args.mbd_substeps}, "
        f"patch_update_every={max(1, int(args.patch_update_every))}, "
        f"weight_ramp=disabled, "
        f"initial_weight_scale={settle_weight_scale(args, start_step):.3f}, "
        f"lock_planar={args.settle_lock_planar}"
    )
    step = int(start_step)
    end_step = int(start_step + args.settle_steps)
    settle_overrides = [
        name
        for name, value in (
            ("soil", args.settle_soil_damping),
            ("vertical", args.settle_vertical_damping),
            ("pitch_rate", args.settle_pitch_rate_damping),
            ("suspension", args.settle_suspension_damping),
        )
        if value is not None
    ]
    print(
        f"[Info] Settle damping: soil_damping={solver.soil_damping:.3f}, "
        f"vertical_damping={args.vertical_damping:.3e}, "
        f"pitch_rate_damping={args.pitch_rate_damping:.3e}, "
        f"suspension_damping={tank.suspension_damping:.3e}, "
        f"kinetic_damping={args.settle_kinetic_damping}, "
        f"overrides={','.join(settle_overrides) if settle_overrides else 'none'}"
    )
    try:
        while step <= end_step:
            stage_time = step * args.mpm_dt
            if args.save_every > 0 and step % args.save_every == 0:
                export_snapshot(
                    out_dir,
                    "settle",
                    step,
                    solver,
                    tank,
                    include_outer_grousers=args.include_grousers,
                    visual_detail=args.tank_vtk_detail,
                    track_backend=track_backend,
                )
                row = diagnostics_row("settle", step, stage_time, solver, tank)
                rows.append(row)
                print(
                    f"[Output] settle step={step:06d} heave={row.tank_heave:.4f} "
                    f"vz={row.tank_heave_rate:.3e} Fz={row.total_force_z:.3e} "
                    f"wscale={settle_weight_scale(args, step):.3f}"
                )
            if step == end_step:
                break

            substeps = macro_substeps_until_sync(args, step, end_step)
            next_step = step + substeps
            last_forces = advance_multirate_coupled_macro(
                args,
                solver,
                tank,
                substeps=substeps,
                left_drive_torque=0.0,
                right_drive_torque=0.0,
                feedback_forces=last_forces,
                track_backend=track_backend,
                body_gravity_scale=settle_weight_scale(args, next_step),
                settle_planar_constraint=args.settle_lock_planar,
            )
            if args.settle_kinetic_damping:
                apply_settle_kinetic_damping(args, solver, tank, kinetic_state)
            step = next_step
    finally:
        solver.soil_damping = saved_soil_damping
        args.vertical_damping = saved_vertical_damping
        args.pitch_rate_damping = saved_pitch_rate_damping
        tank.suspension_damping = saved_suspension_damping
        tank.track_pose_cache.clear()
    if kinetic_state.event_count:
        print(f"[Info] Settle kinetic damping events: {kinetic_state.event_count}")

    final_step = start_step + args.settle_steps
    if args.save_every > 0 and args.settle_steps % args.save_every == 0:
        export_snapshot(
            out_dir,
            "settle",
            final_step,
            solver,
            tank,
            include_outer_grousers=args.include_grousers,
            visual_detail=args.tank_vtk_detail,
            track_backend=track_backend,
        )
    if rows and rows[-1].step == final_step:
        rows[-1] = diagnostics_row("settle", final_step, final_step * args.dt, solver, tank)
    solver.save_state(args.settled_state, soil_gravity_scale=args.soil_gravity_scale)
    save_tank_state(args.settled_tank_state, tank, args, "settle")
    write_diagnostics_csv(out_dir / "settle_diagnostics.csv", rows, append=append_diagnostics)
    save_stage_progress(
        out_dir,
        "settle",
        last_step=final_step,
        dt=args.dt,
        solver=solver,
        tank=tank,
    )
    print(f"[Info] Settled MPM state saved: {Path(args.settled_state).resolve()}")
    print(f"[Info] Settled tank state saved: {Path(args.settled_tank_state).resolve()}")


def run_drive(
    args: argparse.Namespace,
    solver: TankTrackMpmSolver,
    tank: MultibodyRigidGroundTank,
    out_dir: Path,
    *,
    start_step: int = 0,
    append_diagnostics: bool = False,
    track_backend: TrackBackend | None = None,
) -> None:
    rows: list[MultibodyMpmDiagnostics] = []
    solver.configure_moving_window_anchor(float(tank.state.forward_pos))
    last_forces = track_feedback_from_solver(solver)
    left_torque_max = args.drive_torque if args.left_drive_torque is None else args.left_drive_torque
    right_torque_max = args.drive_torque if args.right_drive_torque is None else args.right_drive_torque
    print(
        f"[Info] Stage 3 drive: start_step={start_step}, steps={args.drive_steps}, "
        f"mpm_dt={args.mpm_dt:.3e}, mbd_dt={args.mbd_dt:.3e}, mbd_substeps={args.mbd_substeps}, "
        f"left_torque={left_torque_max:.1f}, right_torque={right_torque_max:.1f}, "
        f"patch_update_every={max(1, int(args.patch_update_every))}"
    )
    if solver.moving_window_enabled:
        print(
            f"[Info] Moving MPM window: anchor_x={float(solver.moving_window_anchor_x):.6f}, "
            f"shift_cells={solver.moving_window_shift_cells}, "
            f"soil_x=({solver.soil_bounds_lo[0]:.6f},{solver.soil_bounds_hi[0]:.6f})"
        )
    step = int(start_step)
    end_step = int(start_step + args.drive_steps)
    while step <= end_step:
        stage_time = step * args.mpm_dt
        if args.save_every > 0 and step % args.save_every == 0:
            export_snapshot(
                out_dir,
                "drive",
                step,
                solver,
                tank,
                include_outer_grousers=args.include_grousers,
                visual_detail=args.tank_vtk_detail,
                track_backend=track_backend,
            )
            row = diagnostics_row("drive", step, stage_time, solver, tank)
            rows.append(row)
            print(
                f"[Output] drive step={step:06d} x={row.tank_x:.4f} "
                f"v={row.tank_forward_speed:.3e} F=({row.total_force_x:.2e},{row.total_force_z:.2e})"
            )
        if step == end_step:
            break

        ramp = smooth_ramp(stage_time, args.ramp_time)
        substeps = macro_substeps_until_sync(args, step, end_step)
        last_forces = advance_multirate_coupled_macro(
            args,
            solver,
            tank,
            substeps=substeps,
            left_drive_torque=left_torque_max * ramp,
            right_drive_torque=right_torque_max * ramp,
            feedback_forces=last_forces,
            track_backend=track_backend,
            update_moving_window=True,
        )
        step += substeps

    write_diagnostics_csv(out_dir / "drive_diagnostics.csv", rows, append=append_diagnostics)
    solver.save_state(args.drive_state, soil_gravity_scale=args.soil_gravity_scale)
    save_tank_state(args.drive_tank_state, tank, args, "drive")
    save_stage_progress(
        out_dir,
        "drive",
        last_step=start_step + args.drive_steps,
        dt=args.dt,
        solver=solver,
        tank=tank,
    )
    print("[Done] Drive stage completed.")


def main() -> None:
    args = parse_args()
    resolve_time_steps(args)
    resolve_state_paths(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_taichi(
        args.arch,
        cpu_threads=args.cpu_threads,
        offline_cache=args.taichi_offline_cache,
    )

    tank = build_tank(args)
    solver = build_solver(args, tank)
    print("[Info] Multibody tank + MPM soil")
    print(
        f"[Info] soil_box=({args.soil_xmin:.2f},{args.soil_ymin:.2f},{args.soil_zmin:.2f})"
        f" -> ({args.soil_xmax:.2f},{args.soil_ymax:.2f},{args.soil_zmax:.2f}), "
        f"particles={solver.n_particles}, grid={solver.n_grid_x}x{solver.n_grid_y}x{solver.n_grid_z}, dx={solver.dx:.4f}"
    )
    print(f"[Info] MPM precision={solver.mpm_precision}")
    print(
        f"[Info] moving_window={solver.moving_window_enabled}, "
        f"particle_layers_x={solver.particle_layers_x}, "
        f"particles_per_layer={solver.particles_per_x_layer}, "
        f"template_layers={solver.moving_window_template_layers}"
    )
    print(
        f"[Info] time stepping: mpm_dt={args.mpm_dt:.3e}, "
        f"mbd_dt={args.mbd_dt:.3e} ({args.mbd_substeps} MPM substeps per MBD macro step)"
    )
    print("[Info] Coupling scheme=Chrono predictor-corrector: replay Chrono with current MPM feedback each macro step.")
    if abs(float(args.mbd_dt) - float(args.requested_mbd_dt)) > 1.0e-15:
        print(
            f"[Info] requested mbd_dt={args.requested_mbd_dt:.3e} was snapped to "
            "an integer multiple of mpm_dt."
        )
    print(
        f"[Info] tank mass={tank.geom.mass:.1f} kg, shoes=({tank.shoe_count['left']},{tank.shoe_count['right']}), "
        f"initial x={tank.state.forward_pos:.3f}, heave={tank.state.heave:.3f}"
    )
    print(
        f"[Info] track shoe inertia: mass={tank.track_shoe_mass_value:.6g} kg "
        f"({tank.track_shoe_mass_source}), pitch_inertia={tank.track_shoe_pitch_inertia_value:.6g} kg*m^2 "
        f"({tank.track_shoe_pitch_inertia_source}), total_track_shoe_mass={tank.total_track_shoe_mass:.3f} kg"
    )
    print(f"[Info] tank VTK detail={args.tank_vtk_detail}")
    print(
        f"[Info] resolved track_width={tank.geom.track_width:.5f} m, "
        f"track_pitch={tank.geom.track_pitch:.5f} m, "
        f"shoe_length={tank.geom.shoe_length:.5f} m, "
        f"actual_pitch=({tank.actual_pitch['left']:.5f},{tank.actual_pitch['right']:.5f}) m"
    )
    print(f"[Info] Soil constitutive model={solver.soil_constitutive_model_name}")
    if solver.soil_constitutive_model_name == "modified-cam-clay":
        print(
            f"[Info] MCC parameters: M={solver.mcc_M:.4f}, lambda={solver.mcc_lambda:.4f}, "
            f"kappa={solver.mcc_kappa:.4f}, e0={solver.mcc_e0:.3f}, pc0={solver.mcc_pc0:.3e} Pa"
        )
    elif solver.soil_constitutive_model_name == "pure-water":
        print(
            f"[Info] Pure water K={solver.water_bulk_modulus:.3e}Pa, "
            f"mu={solver.water_dynamic_viscosity:.6e}Pa s, "
            f"acoustic CFL dt<={solver.water_recommended_dt:.3e}s"
        )
    elif solver.soil_constitutive_model_name == "srsh-modified-dp":
        print(
            f"[Info] SRSH parameters: beta={solver.srsh_rate_exponent:.4g}, "
            f"eta={solver.srsh_rate_sensitivity:.4g}, "
            f"delta={solver.srsh_saturation_increment:.4g}, "
            f"M={solver.srsh_friction_slope:.6g}, "
            f"D={solver.srsh_dilatancy_slope:.6g}, "
            f"C={solver.srsh_cohesion_intercept:.6e}Pa, "
            f"rate0={solver.srsh_source_rate0:.6e}1/s, "
            f"ep95={solver.srsh_saturation_plastic_strain:.6g}"
        )
    print(
        "[Info] Track-soil contact=particle-to-track level-set barrier contact, "
        f"kappa={solver.contact_barrier_stiffness_value:.3e} N/m, "
        f"radius={solver.contact_barrier_radius_value:.5f} m, "
        f"d_min/r={solver.contact_barrier_min_distance_ratio_value:.3f}"
    )
    print(
        f"[Info] Interface shear law=smoothed Coulomb, mu={solver.contact_mu:.4f}, "
        f"slip_smoothing={solver.contact_slip_smoothing_distance_value:.5f} m"
    )
    print(f"[Info] output={out_dir.resolve()}")
    track_backend: TrackBackend | None = None
    print(
        "[Info] Track dynamics model=chrono-vehicle(pychrono.vehicle TrackedVehicle, MPM terrain coupling) "
        f"(max_substep_dt={float(args.chrono_max_substep_dt):.3e}, "
        f"rigid_terrain={bool(args.chrono_use_rigid_terrain)}, "
        f"shoe_count={int(args.chrono_track_shoe_count) if int(args.chrono_track_shoe_count) > 0 else 'auto+' + str(int(args.chrono_track_shoe_count_padding))})"
    )

    if args.resume and args.stage == "all":
        print("[Warn] --resume is ignored with --stage all; use --stage geostatic/settle/drive for staged continuation.")

    if args.stage in {"geostatic", "all"}:
        geostatic_resumed = False
        geostatic_start = 0
        if args.stage == "geostatic" and args.resume:
            geostatic_path = Path(args.geostatic_state)
            if geostatic_path.exists():
                print(f"[Info] Resuming geostatic state: {geostatic_path.resolve()}")
                solver.load_state(geostatic_path)
                geostatic_start = load_next_stage_step(out_dir, "geostatic")
                geostatic_resumed = True
            else:
                print(f"[Warn] --resume requested but geostatic state does not exist: {geostatic_path.resolve()}")
        run_geostatic(
            args,
            solver,
            out_dir,
            start_step=geostatic_start,
            append_diagnostics=geostatic_resumed,
            initialize_stress=not geostatic_resumed,
        )
    elif args.stage == "settle":
        if args.resume and Path(args.settled_state).exists() and Path(args.settled_tank_state).exists():
            print(f"[Info] Resuming settled MPM state: {Path(args.settled_state).resolve()}")
            solver.load_state(args.settled_state)
            print(f"[Info] Resuming settled tank state: {Path(args.settled_tank_state).resolve()}")
            load_tank_state(args.settled_tank_state, tank)
        else:
            if args.resume:
                print("[Warn] --resume requested but settled state is incomplete; starting settle from geostatic state.")
            print(f"[Info] Loading geostatic state: {Path(args.geostatic_state).resolve()}")
            solver.load_state(args.geostatic_state)
            if solver.moving_window_enabled and not solver.moving_window_template_ready:
                solver.capture_moving_window_template()

    if args.stage in {"settle", "all"}:
        settle_resumed = (
            args.stage == "settle"
            and args.resume
            and Path(args.settled_state).exists()
            and Path(args.settled_tank_state).exists()
        )
        if not settle_resumed:
            global_surface_z = solver.get_soil_surface_z()
            local_surface_z = estimate_soil_surface_under_tracks(solver, tank, args)
            place_tank_on_soil(
                tank,
                soil_zmax=local_surface_z,
                initial_sinkage=args.initial_sinkage,
                clearance=args.settle_initial_clearance,
                args=args,
            )
            surfaces = bottom_track_contact_surfaces_mpm(tank, args)
            if surfaces:
                contact_z = min(float(surface["center"][2]) for surface in surfaces)
                initial_gap = contact_z - local_surface_z
                print(
                    f"[Info] Settle initial placement: global_surface_z={global_surface_z:.6f}, "
                    f"local_surface_z={local_surface_z:.6f}, contact_z={contact_z:.6f}, "
                    f"initial_gap={initial_gap:.6f}, target_clearance={float(args.settle_initial_clearance):.6f}, "
                    f"initial_sinkage={float(args.initial_sinkage):.6f}"
                )
        if track_backend is None:
            track_backend = build_track_backend(args, tank)
        if not settle_resumed:
            align_chrono_track_backend_on_soil(
                track_backend,
                tank,
                args,
                soil_surface_z=local_surface_z,
                label="settle initial",
            )
        run_settle(
            args,
            solver,
            tank,
            out_dir,
            start_step=load_next_stage_step(out_dir, "settle") if settle_resumed else 0,
            append_diagnostics=settle_resumed,
            track_backend=track_backend,
        )
    elif args.stage == "drive":
        if args.resume and Path(args.drive_state).exists() and Path(args.drive_tank_state).exists():
            print(f"[Info] Resuming drive MPM state: {Path(args.drive_state).resolve()}")
            solver.load_state(args.drive_state)
            print(f"[Info] Resuming drive tank state: {Path(args.drive_tank_state).resolve()}")
            load_tank_state(args.drive_tank_state, tank)
        else:
            if args.resume:
                print("[Warn] --resume requested but drive state is incomplete; starting drive from settled state.")
            print(f"[Info] Loading settled MPM state: {Path(args.settled_state).resolve()}")
            solver.load_state(args.settled_state)
            print(f"[Info] Loading settled tank state: {Path(args.settled_tank_state).resolve()}")
            load_tank_state(args.settled_tank_state, tank)

    if args.stage in {"drive", "all"}:
        if track_backend is None:
            track_backend = build_track_backend(args, tank)
        drive_resumed = (
            args.stage == "drive"
            and args.resume
            and Path(args.drive_state).exists()
            and Path(args.drive_tank_state).exists()
        )
        run_drive(
            args,
            solver,
            tank,
            out_dir,
            start_step=load_next_stage_step(out_dir, "drive") if drive_resumed else 0,
            append_diagnostics=drive_resumed,
            track_backend=track_backend,
        )

    print("[Done] Multibody tank + MPM workflow completed.")


if __name__ == "__main__":
    main()
