from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_multibody_tank_mpm import (  # noqa: E402
    build_solver,
    build_tank,
    init_taichi,
    parse_args as parse_mpm_args,
    resolve_geostatic_k0,
    resolve_state_paths,
    resolve_time_steps,
)


def _coerce_config_value(text: str, current: Any) -> Any:
    value = str(text).strip()
    if value == "":
        return current
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if current is None:
        try:
            return int(value) if value.isdigit() else float(value)
        except ValueError:
            return value
    return value


def apply_launcher_config(args: argparse.Namespace, config_path: Path) -> None:
    if not config_path.exists():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in payload.get("fields", {}).items():
        if key in {"python", "script", "model", "out"} or not hasattr(args, key):
            continue
        setattr(args, key, _coerce_config_value(value, getattr(args, key)))
    for key, value in payload.get("checks", {}).items():
        if hasattr(args, key):
            setattr(args, key, bool(value))


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=(
            "Single track-shoe level-set/MPM contact probe. The test keeps Chrono out of "
            "the loop and presses one rigid contact patch into the MPM soil."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "tank_mpm_launcher_config.json")
    parser.add_argument("--probe-out", type=Path, default=None)
    parser.add_argument("--probe-steps", type=int, default=12000)
    parser.add_argument("--probe-hold-steps", type=int, default=0)
    parser.add_argument("--probe-initial-gap", type=float, default=0.08)
    parser.add_argument("--probe-final-penetration", type=float, default=0.03)
    parser.add_argument("--probe-down-speed", type=float, default=None)
    parser.add_argument("--probe-record-every", type=int, default=1)
    parser.add_argument("--probe-side", choices=["left", "right", "center"], default="left")
    parser.add_argument("--probe-load-geostatic-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-geostatic-steps", type=int, default=0)
    parser.add_argument("--probe-save-soil-every", type=int, default=0)
    parser.add_argument(
        "--probe-compact-domain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a local high-resolution soil box around the single shoe instead of the full vehicle soil domain.",
    )
    parser.add_argument("--probe-target-dx", type=float, default=0.035)
    parser.add_argument("--probe-particles-per-cell", type=float, default=2.0)
    parser.add_argument("--probe-x-half-span", type=float, default=0.60)
    parser.add_argument("--probe-y-margin", type=float, default=0.35)
    parser.add_argument("--probe-soil-depth", type=float, default=0.80)
    parser.add_argument("--probe-domain-padding", type=float, default=None)
    probe_args, remaining = parser.parse_known_args()

    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]
        mpm_args = parse_mpm_args()
        apply_launcher_config(mpm_args, probe_args.config)
        sys.argv = [saved_argv[0], *remaining]
        mpm_args = parse_mpm_args(namespace=mpm_args)
    finally:
        sys.argv = saved_argv
    resolve_time_steps(mpm_args)
    resolve_state_paths(mpm_args)
    return probe_args, mpm_args


def _ceil_count(span: float, spacing: float, minimum: int = 2) -> int:
    return max(int(minimum), int(math.ceil(max(float(span), 1.0e-9) / max(float(spacing), 1.0e-9))))


def apply_compact_probe_domain(
    probe_args: argparse.Namespace,
    mpm_args: argparse.Namespace,
    tank,
) -> None:
    if not probe_args.probe_compact_domain:
        return

    if probe_args.probe_side == "left":
        lateral = float(tank.geom.left_track_center_x)
    elif probe_args.probe_side == "right":
        lateral = float(tank.geom.right_track_center_x)
    else:
        lateral = 0.0

    target_dx = max(float(probe_args.probe_target_dx), 1.0e-4)
    ppc = max(float(probe_args.probe_particles_per_cell), 1.0)
    particle_spacing = target_dx / ppc
    half_long = max(0.5 * float(tank.geom.shoe_length), 1.0e-6)
    half_width = max(0.5 * float(tank.geom.track_width), 1.0e-6)
    x_half_span = max(float(probe_args.probe_x_half_span), 4.0 * half_long)
    y_half_span = half_width + max(float(probe_args.probe_y_margin), target_dx)
    soil_top = float(mpm_args.soil_zmax)
    soil_depth = max(float(probe_args.probe_soil_depth), 0.2)
    padding = (
        max(2.0 * target_dx, 0.05)
        if probe_args.probe_domain_padding is None
        else max(float(probe_args.probe_domain_padding), target_dx)
    )

    x_center = float(mpm_args.initial_forward)
    mpm_args.soil_xmin = x_center - x_half_span
    mpm_args.soil_xmax = x_center + x_half_span
    mpm_args.soil_ymin = lateral - y_half_span
    mpm_args.soil_ymax = lateral + y_half_span
    mpm_args.soil_zmin = soil_top - soil_depth
    mpm_args.soil_zmax = soil_top
    mpm_args.domain_padding = padding
    mpm_args.domain_shape = "rect"

    span_x = mpm_args.soil_xmax - mpm_args.soil_xmin
    span_y = mpm_args.soil_ymax - mpm_args.soil_ymin
    span_z = mpm_args.soil_zmax - mpm_args.soil_zmin
    domain_extent = max(span_x, span_y, span_z) + 2.0 * padding
    mpm_args.grid = _ceil_count(domain_extent, target_dx, minimum=8)
    mpm_args.soil_nx = _ceil_count(span_x, particle_spacing, minimum=4)
    mpm_args.soil_ny = _ceil_count(span_y, particle_spacing, minimum=4)
    mpm_args.soil_nz = _ceil_count(span_z, particle_spacing, minimum=4)
    probe_args.probe_load_geostatic_state = False


def make_single_patch(
    *,
    center: np.ndarray,
    velocity: np.ndarray,
    half_extent: np.ndarray,
    shoe_mass: float,
) -> dict[str, np.ndarray]:
    return {
        "center": np.asarray([center], dtype=np.float32),
        "axis_long": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "axis_width": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        "normal": np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        "velocity": np.asarray([velocity], dtype=np.float32),
        "half_extent": np.asarray([half_extent], dtype=np.float32),
        "mass": np.asarray([shoe_mass], dtype=np.float32),
        "shoe_id": np.asarray([0.0], dtype=np.float32),
        "side_id": np.asarray([0], dtype=np.int32),
    }


def prepare_soil(probe_args: argparse.Namespace, mpm_args: argparse.Namespace, solver) -> str:
    state_path = Path(mpm_args.geostatic_state)
    if probe_args.probe_load_geostatic_state and state_path.exists():
        solver.load_state(state_path)
        solver.initialize_contact_history()
        return f"loaded geostatic state: {state_path}"

    k0 = resolve_geostatic_k0(mpm_args)
    solver.initialize_geostatic_stress(k0, float(mpm_args.geostatic_soil_gravity_scale))
    solver.initialize_contact_history()
    if probe_args.probe_geostatic_steps > 0:
        saved_gravity = solver.soil_gravity_scale
        saved_damping = solver.soil_damping
        solver.soil_gravity_scale = float(mpm_args.geostatic_soil_gravity_scale)
        solver.soil_damping = float(max(0.0, min(1.0, mpm_args.geostatic_soil_damping)))
        for _ in range(int(probe_args.probe_geostatic_steps)):
            solver.substep_geostatic()
        solver.soil_gravity_scale = saved_gravity
        solver.soil_damping = saved_damping
        return f"initialized geostatic stress and ran {probe_args.probe_geostatic_steps} geostatic steps"
    return f"initialized geostatic stress only, K0={k0:.6f}"


def write_probe_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    probe_args, mpm_args = parse_args()
    init_taichi(
        mpm_args.arch,
        cpu_threads=mpm_args.cpu_threads,
        offline_cache=mpm_args.taichi_offline_cache,
    )
    tank = build_tank(mpm_args)
    apply_compact_probe_domain(probe_args, mpm_args, tank)
    solver = build_solver(mpm_args, tank)
    soil_note = prepare_soil(probe_args, mpm_args, solver)

    out_dir = probe_args.probe_out or (Path(mpm_args.out) / "single_patch_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "single_patch_contact_probe.csv"

    if probe_args.probe_side == "left":
        lateral = float(tank.geom.left_track_center_x)
    elif probe_args.probe_side == "right":
        lateral = float(tank.geom.right_track_center_x)
    else:
        lateral = 0.0

    half_extent = np.array([0.5 * tank.geom.shoe_length, 0.5 * tank.geom.track_width], dtype=np.float64)
    patch_area = float(4.0 * half_extent[0] * half_extent[1])
    start_z = float(mpm_args.soil_zmax) + float(probe_args.probe_initial_gap)
    travel = float(probe_args.probe_initial_gap) + float(probe_args.probe_final_penetration)
    move_steps = max(1, int(probe_args.probe_steps))
    hold_steps = max(0, int(probe_args.probe_hold_steps))
    total_steps = move_steps + hold_steps
    dt = float(mpm_args.mpm_dt)
    if probe_args.probe_down_speed is None:
        down_speed = travel / max(move_steps * dt, 1.0e-12)
    else:
        down_speed = max(0.0, float(probe_args.probe_down_speed))
    record_every = max(1, int(probe_args.probe_record_every))
    saved_gravity = solver.soil_gravity_scale
    solver.soil_gravity_scale = float(mpm_args.soil_gravity_scale)

    rows: list[dict[str, float]] = []
    for step in range(total_steps):
        current_step = step + 1
        previous_moved = min(travel, min(step, move_steps) * down_speed * dt)
        moved = min(travel, min(current_step, move_steps) * down_speed * dt)
        center_z = start_z - moved
        velocity = np.array([0.0, 0.0, -(moved - previous_moved) / max(dt, 1.0e-12)], dtype=np.float64)
        center = np.array([float(mpm_args.initial_forward), lateral, center_z], dtype=np.float64)
        solver.set_track_patches(
            make_single_patch(
                center=center,
                velocity=velocity,
                half_extent=half_extent,
                shoe_mass=float(tank.track_shoe_mass_value),
            )
        )
        solver.substep()

        if current_step % record_every == 0 or current_step == total_steps:
            forces = solver.contact_forces_by_patch()
            force = forces[0] if forces.shape[0] else np.zeros(3, dtype=np.float64)
            debug = solver.contact_debug_summary()
            rows.append(
                {
                    "step": int(current_step),
                    "time": float(current_step) * dt,
                    "patch_center_x": float(center[0]),
                    "patch_center_y": float(center[1]),
                    "patch_center_z": float(center[2]),
                    "nominal_gap": float(center[2] - float(mpm_args.soil_zmax)),
                    "penetration": float(max(0.0, float(mpm_args.soil_zmax) - center[2])),
                    "patch_velocity_z": float(velocity[2]),
                    "active_patches": int(solver.active_patch_count_value()),
                    "contact_particles": int(solver.contact_particle_count[None]),
                    "patch_force_x": float(force[0]),
                    "patch_force_y": float(force[1]),
                    "patch_force_z": float(force[2]),
                    "patch_force_norm": float(np.linalg.norm(force)),
                    "patch_pressure_z": float(force[2] / max(patch_area, 1.0e-12)),
                    "max_contact_force_norm": debug["max_contact_force_norm"],
                    "max_contact_force_z": debug["max_contact_force_z"],
                    "min_contact_distance": debug["min_contact_distance"],
                    "max_normal_force": debug["max_normal_force"],
                    "max_slip_speed": debug["max_slip_speed"],
                }
            )
            if probe_args.probe_save_soil_every > 0 and current_step % int(probe_args.probe_save_soil_every) == 0:
                solver.export_soil(out_dir / "soil", current_step)

    solver.soil_gravity_scale = saved_gravity
    write_probe_csv(csv_path, rows)
    meta = {
        "soil": soil_note,
        "csv": str(csv_path),
        "grid_shape": solver.grid_shape,
        "dx": solver.dx,
        "particle_count": solver.n_particles,
        "particle_volume": solver.p_vol,
        "particle_mass": solver.p_mass,
        "contact_barrier_stiffness": solver.contact_barrier_stiffness_value,
        "contact_barrier_radius": solver.contact_barrier_radius_value,
        "contact_barrier_min_distance_ratio": solver.contact_barrier_min_distance_ratio_value,
        "contact_slip_smoothing_distance": solver.contact_slip_smoothing_distance_value,
        "compact_domain": bool(probe_args.probe_compact_domain),
        "probe_target_dx": float(probe_args.probe_target_dx),
        "probe_particles_per_cell": float(probe_args.probe_particles_per_cell),
        "soil_bounds": {
            "xmin": float(mpm_args.soil_xmin),
            "xmax": float(mpm_args.soil_xmax),
            "ymin": float(mpm_args.soil_ymin),
            "ymax": float(mpm_args.soil_ymax),
            "zmin": float(mpm_args.soil_zmin),
            "zmax": float(mpm_args.soil_zmax),
        },
        "soil_counts": {
            "nx": int(mpm_args.soil_nx),
            "ny": int(mpm_args.soil_ny),
            "nz": int(mpm_args.soil_nz),
        },
        "shoe_mass": float(tank.track_shoe_mass_value),
        "shoe_length": float(tank.geom.shoe_length),
        "track_width": float(tank.geom.track_width),
        "patch_area": patch_area,
        "initial_gap": float(probe_args.probe_initial_gap),
        "final_penetration": float(probe_args.probe_final_penetration),
        "down_speed": down_speed,
        "mpm_dt": dt,
        "move_steps": move_steps,
        "hold_steps": hold_steps,
    }
    (out_dir / "single_patch_contact_probe_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(f"[Done] single patch probe wrote {csv_path}")
    print(
        f"[Info] dx={solver.dx:.6g}, barrier_radius={solver.contact_barrier_radius_value:.6g}, "
        f"kappa={solver.contact_barrier_stiffness_value:.6g}, rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
