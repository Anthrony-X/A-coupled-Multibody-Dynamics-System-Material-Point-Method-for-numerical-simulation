from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import taichi as ti


CASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tank_mpm.constitutive_models import (  # noqa: E402
    CONSTITUTIVE_MODEL_CHOICES,
    normalize_constitutive_model_name,
)
from tank_mpm.particles import sample_soil_particles  # noqa: E402
from tank_mpm.mpm_solver import TankTrackMpmSolver  # noqa: E402
from validation_cases.cmu_snake_mbd_mpm_3d.config import (  # noqa: E402
    CaseConfig,
    load_case_config,
)
from validation_cases.cmu_snake_mbd_mpm_3d.mbd_model import (  # noqa: E402
    CmuSnakeMBD,
    ContactFeedback,
    gait_envelope,
)
from validation_cases.cmu_snake_mbd_mpm_3d.validation import (  # noqa: E402
    build_summary,
    build_validation_report,
    cycle_statistics,
    write_json,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully coupled 17-body CMU snake robot moving on MPM sand."
    )
    parser.add_argument("--preset", choices=("smoke", "standard", "fine"), default="smoke")
    parser.add_argument("--config", type=Path, default=CASE_DIR / "case_config.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--arch",
        choices=("cuda", "cpu", "vulkan", "gpu"),
        default="cuda",
    )
    parser.add_argument("--max-macro-steps", type=int, default=None)
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--constitutive-model",
        type=normalize_constitutive_model_name,
        default=None,
        metavar="MODEL",
        help="Override soil constitutive model; aliases: " + ", ".join(CONSTITUTIVE_MODEL_CHOICES),
    )
    return parser.parse_args()


def initialize_taichi(arch_name: str) -> None:
    arch_map = {
        "cuda": ti.cuda,
        "cpu": ti.cpu,
        "vulkan": ti.vulkan,
        "gpu": ti.gpu,
    }
    if arch_name == "cuda" and not ti._lib.core.with_cuda():
        raise RuntimeError(
            "CUDA backend requested, but Taichi cannot load the NVIDIA CUDA driver. "
            "Use --arch cpu for the smoke preset or repair the CUDA driver."
        )
    ti.init(
        arch=arch_map[arch_name],
        default_fp=ti.f32,
        offline_cache=False,
        enable_fallback=arch_name != "cuda",
    )


def build_mpm_solver(
    config: CaseConfig,
    constitutive_model: str | None = None,
) -> TankTrackMpmSolver:
    resolution = config.resolution
    soil = config.soil
    selected_model = normalize_constitutive_model_name(
        soil.constitutive_model if constitutive_model is None else constitutive_model
    )
    nx, ny, nz = resolution.soil_particles
    points, particle_volume = sample_soil_particles(
        nx,
        ny,
        nz,
        soil.bounds_lo,
        soil.bounds_hi,
        jitter_ratio=soil.particle_jitter_ratio,
        seed=soil.seed,
    )
    particle_radius = max(
        (3.0 * particle_volume / (4.0 * math.pi)) ** (1.0 / 3.0), 1.0e-9
    )
    barrier_stiffness = (
        soil.contact_barrier_stiffness
        if soil.contact_barrier_stiffness is not None
        else soil.young_modulus
        * particle_radius
        * resolution.contact_barrier_stiffness_scale
    )
    solver = TankTrackMpmSolver(
        soil_points=points,
        soil_p_vol=particle_volume,
        n_grid=resolution.n_grid_x,
        dt=resolution.mpm_dt,
        domain_lo=soil.domain_lo,
        domain_hi=soil.domain_hi,
        soil_bounds_lo=soil.bounds_lo,
        soil_bounds_hi=soil.bounds_hi,
        max_track_patches=config.contact_patch_count,
        soil_density=soil.density,
        soil_E=soil.young_modulus,
        soil_nu=soil.poisson_ratio,
        soil_phi_deg=soil.friction_angle_deg,
        soil_psi_deg=soil.dilation_angle_deg,
        soil_cohesion=soil.cohesion,
        soil_constitutive_model=selected_model,
        soil_gravity_scale=1.0,
        soil_damping=1.0,
        contact_mu=soil.contact_mu,
        contact_barrier_stiffness=barrier_stiffness,
        contact_barrier_min_distance_ratio=soil.contact_barrier_min_distance_ratio,
        track_activation_height=soil.contact_activation_height,
        mpm_precision="f32",
        particle_shape=resolution.soil_particles,
        moving_window_enabled=soil.moving_window_enabled,
        moving_window_template_layers=soil.moving_window_template_layers,
        moving_window_axis=soil.moving_window_axis,
        moving_window_direction=soil.moving_window_direction,
        contact_binning_enabled=True,
        contact_bin_capacity=512,
    )
    k0 = 1.0 - math.sin(math.radians(soil.friction_angle_deg))
    solver.initialize_geostatic_stress(k0=k0, gravity_scale=1.0)
    return solver


def _soil_snapshot(solver: TankTrackMpmSolver) -> tuple[float, float]:
    kinetic, potential = solver.soil_mechanical_energy()
    return float(kinetic), float(potential)


def _write_pvd(path: Path, entries: list[tuple[float, int, str, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time_value, part, group, relative_file in entries:
        lines.append(
            f'    <DataSet timestep="{time_value:.12g}" group="{group}" '
            f'part="{part}" file="{relative_file}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pvd_collections(out_dir: Path, frames: list[tuple[int, float]]) -> None:
    definitions = {
        "soil_series.pvd": ("soil", "soil_{step:06d}.vtk"),
        "snake_modules_series.pvd": ("modules", "snake_modules_{step:06d}.vtk"),
        "snake_topology_series.pvd": ("topology", "snake_topology_{step:06d}.vtk"),
        "snake_contact_series.pvd": ("contact", "snake_contact_{step:06d}.vtk"),
        "snake_envelope_series.pvd": ("envelope", "snake_envelope_{step:06d}.vtk"),
    }
    for filename, (group, pattern) in definitions.items():
        entries = [
            (time_value, 0, group, f"vtk/{pattern.format(step=step)}")
            for step, time_value in frames
        ]
        _write_pvd(out_dir / filename, entries)

    coupled_entries: list[tuple[float, int, str, str]] = []
    for step, time_value in frames:
        coupled_entries.extend(
            [
                (time_value, 0, "soil", f"vtk/soil_{step:06d}.vtk"),
                (time_value, 1, "modules", f"vtk/snake_modules_{step:06d}.vtk"),
                (time_value, 2, "contact", f"vtk/snake_contact_{step:06d}.vtk"),
            ]
        )
    _write_pvd(out_dir / "coupled_scene.pvd", coupled_entries)


def _save_vtk(
    vtk_dir: Path,
    step: int,
    solver: TankTrackMpmSolver,
    mbd: CmuSnakeMBD,
    feedback: ContactFeedback,
    threshold_N: float,
) -> None:
    solver.export_soil(vtk_dir, step)
    mbd.export_vtk(vtk_dir, step, feedback, threshold_N)


def _portable_output_path(out_dir: Path) -> str:
    try:
        return str(out_dir.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(out_dir.resolve())


def run(config: CaseConfig, out_dir: Path, args: argparse.Namespace) -> dict:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    if not args.no_vtk:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    solver = build_mpm_solver(config, args.constitutive_model)
    mbd = CmuSnakeMBD(config)
    initial_patches = mbd.contact_patches()
    solver.set_track_patches(initial_patches)
    feedback = ContactFeedback.zeros(config.contact_patch_count)
    initial_com, _ = mbd.center_of_mass_state()
    if solver.moving_window_enabled:
        solver.capture_moving_window_template()
        solver.configure_moving_window_anchor(
            float(initial_com[solver.moving_window_axis])
        )
    print(
        f"[Topology] bodies={config.model.module_count}, joints={config.model.joint_count}, "
        f"contact_patches={config.contact_patch_count}"
    )
    print(
        f"[MPM] model={solver.soil_constitutive_model_name}, particles={solver.n_particles}, "
        f"grid={solver.grid_shape}, dx={solver.dx:.6g}m, dt={solver.dt:.3e}s"
    )
    print(
        f"[Contact] barrier_radius={solver.contact_barrier_radius_value:.6g}m, "
        f"barrier_stiffness={solver.contact_barrier_stiffness_value:.6g}N/m, "
        f"mu={solver.contact_mu:.3f}, bins={solver.contact_bin_nx}x"
        f"{solver.contact_bin_ny}, bin_size={solver.contact_bin_size_value:.6g}m"
    )
    print(
        f"[MBD] dt={config.resolution.mbd_dt:.3e}s, motor_limit="
        f"{config.model.motor_torque_limit_Nm:.3f}Nm, gait={config.model.gait_frequency_hz:.3f}Hz"
    )
    if solver.moving_window_enabled:
        print(
            f"[MovingWindow] axis={solver.moving_window_axis_name}, "
            f"direction={solver.moving_window_direction:+d}, "
            f"anchor={float(solver.moving_window_anchor):.6g}m, "
            f"layers={solver.particle_layers_moving}, "
            f"template_layers={solver.moving_window_template_layers}"
        )

    planned_steps = int(round(config.resolution.duration / config.resolution.mbd_dt))
    requested_steps = planned_steps
    if args.max_macro_steps is not None:
        requested_steps = min(planned_steps, max(0, int(args.max_macro_steps)))
    save_stride = max(
        1, int(round(config.resolution.save_interval / config.resolution.mbd_dt))
    )
    mpm_substeps = config.resolution.mpm_substeps

    rows: list[dict[str, float | int]] = []
    saved_frames: list[tuple[int, float]] = []
    initial_mbd_energy = mbd.mechanical_energy()
    initial_soil_kinetic, initial_soil_potential = _soil_snapshot(solver)
    cumulative_motor_work = 0.0
    cumulative_mbd_contact_work = 0.0
    cumulative_mpm_rigid_work = 0.0
    cumulative_mpm_soil_work = 0.0
    cumulative_soil_internal_work = 0.0

    if not args.no_vtk:
        _save_vtk(
            vtk_dir,
            0,
            solver,
            mbd,
            feedback,
            config.validation.contact_force_threshold_N,
        )
        saved_frames.append((0, 0.0))
        _write_pvd_collections(out_dir, saved_frames)

    wall_start = time.perf_counter()
    for macro_index in range(requested_steps):
        step = macro_index + 1
        start_patches = mbd.contact_patches()
        motor_power_start = mbd.actuator_power()
        contact_power_start = mbd.patch_contact_power(feedback)

        mbd.step(config.resolution.mbd_dt, feedback)

        end_patches = mbd.contact_patches()
        motor_power_end = mbd.actuator_power()
        contact_power_end = mbd.patch_contact_power(feedback)
        cumulative_motor_work += (
            0.5
            * (motor_power_start + motor_power_end)
            * config.resolution.mbd_dt
        )
        cumulative_mbd_contact_work += (
            0.5
            * (contact_power_start + contact_power_end)
            * config.resolution.mbd_dt
        )

        if solver.moving_window_enabled:
            window_com, _ = mbd.center_of_mass_state()
            solver.advance_moving_window(
                float(window_com[solver.moving_window_axis])
            )
        solver.set_track_patch_keyframes(start_patches, end_patches)
        solver.reset_contact_force_average()
        for substep in range(mpm_substeps):
            solver.update_track_patches_from_keyframes(
                substep / mpm_substeps,
                (substep + 1) / mpm_substeps,
                solver.dt,
            )
            solver.substep()
            solver.accumulate_contact_force_average()

        should_save = step % save_stride == 0 or step == requested_steps
        include_patch_data = not args.no_vtk and should_save
        macro_contact = solver.contact_macro_summary_reduced(
            include_patch_data=include_patch_data
        )
        feedback = ContactFeedback(
            patch_forces=(
                np.asarray(macro_contact["patch_forces"], dtype=np.float32)
                if include_patch_data
                else None
            ),
            patch_moments=(
                np.asarray(macro_contact["patch_moments"], dtype=np.float32)
                if include_patch_data
                else None
            ),
            module_forces=np.asarray(
                macro_contact["module_forces"], dtype=np.float32
            ),
            module_moments_about_origin=np.asarray(
                macro_contact["module_moments_about_origin"], dtype=np.float32
            ),
            axial_segment_normal_loads=np.asarray(
                macro_contact["axial_segment_normal_loads"], dtype=np.float32
            ),
        )
        cumulative_mpm_rigid_work += float(macro_contact["rigid_work"])
        cumulative_mpm_soil_work += float(macro_contact["soil_work"])
        cumulative_soil_internal_work += float(macro_contact["internal_work"])

        time_s = step * config.resolution.mbd_dt
        com, com_velocity = mbd.center_of_mass_state()
        controller = mbd.controller_snapshot(time_s)
        tracking_error_deg = np.degrees(controller.tracking_error_rad)
        constraints = mbd.constraint_errors()
        contact_ratio, active_segments, segment_load = mbd.contact_length_ratio(
            feedback, config.validation.contact_force_threshold_N
        )
        total_contact_force = np.sum(feedback.module_forces, axis=0)
        normal_force = np.asarray(
            feedback.axial_segment_normal_loads, dtype=np.float64
        )
        soil_kinetic, soil_potential = _soil_snapshot(solver)
        mbd_energy = mbd.mechanical_energy()
        envelope_value, _ = gait_envelope(
            time_s, config.model.settle_time_s, config.model.ramp_time_s
        )
        row: dict[str, float | int] = {
            "step": step,
            "time_s": time_s,
            "gait_envelope": envelope_value,
            "gait_phase_rad": mbd.gait_phase_rad(time_s),
            "com_x_m": float(com[0]),
            "com_y_m": float(com[1]),
            "com_z_m": float(com[2]),
            "com_vx_m_s": float(com_velocity[0]),
            "com_vy_m_s": float(com_velocity[1]),
            "com_vz_m_s": float(com_velocity[2]),
            "planar_speed_m_s": float(np.linalg.norm(com_velocity[:2])),
            "displacement_x_m": float(com[0] - initial_com[0]),
            "displacement_y_m": float(com[1] - initial_com[1]),
            "tracking_rms_deg": float(np.sqrt(np.mean(tracking_error_deg**2))),
            "tracking_max_abs_deg": float(np.max(np.abs(tracking_error_deg))),
            "max_motor_torque_Nm": float(np.max(np.abs(controller.torque_command_Nm))),
            "motor_saturated_count": int(np.sum(controller.saturation)),
            "motor_saturation_fraction": float(np.mean(controller.saturation)),
            "motor_power_W": motor_power_end,
            "motor_work_cumulative_J": cumulative_motor_work,
            "max_joint_position_error_m": float(constraints["max_position_error_m"]),
            "max_joint_axis_error_rad": float(constraints["max_axis_error_rad"]),
            "contact_length_ratio": contact_ratio,
            "active_axial_segments": active_segments,
            "max_axial_segment_normal_load_N": float(np.max(segment_load)),
            "normal_contact_force_N": float(np.sum(normal_force)),
            "contact_force_x_N": float(total_contact_force[0]),
            "contact_force_y_N": float(total_contact_force[1]),
            "contact_force_z_N": float(total_contact_force[2]),
            "contact_force_norm_N": float(np.linalg.norm(total_contact_force)),
            "contact_particle_samples": int(macro_contact["contact_particle_samples"]),
            "max_contact_particles": int(macro_contact["max_contact_particles"]),
            "contact_bin_overflow": int(macro_contact["contact_bin_overflow"]),
            "contact_bin_max_occupancy": int(
                macro_contact["contact_bin_max_occupancy"]
            ),
            "action_reaction_absolute_residual_N": float(
                macro_contact["max_action_absolute"]
            ),
            "action_reaction_relative_residual": float(
                macro_contact["max_action_relative"]
            ),
            "mbd_mechanical_energy_J": mbd_energy,
            "mbd_energy_change_J": mbd_energy - initial_mbd_energy,
            "soil_kinetic_energy_J": soil_kinetic,
            "soil_potential_energy_J": soil_potential,
            "soil_energy_change_J": soil_kinetic
            + soil_potential
            - initial_soil_kinetic
            - initial_soil_potential,
            "soil_internal_work_cumulative_J": cumulative_soil_internal_work,
            "mbd_contact_work_cumulative_J": cumulative_mbd_contact_work,
            "mpm_rigid_contact_work_cumulative_J": cumulative_mpm_rigid_work,
            "mpm_soil_contact_work_cumulative_J": cumulative_mpm_soil_work,
            "partitioned_coupling_lag_work_J": cumulative_mbd_contact_work
            - cumulative_mpm_rigid_work,
        }
        rows.append(row)

        if not args.no_vtk and should_save:
            _save_vtk(
                vtk_dir,
                step,
                solver,
                mbd,
                feedback,
                config.validation.contact_force_threshold_N,
            )
            saved_frames.append((step, time_s))
            _write_pvd_collections(out_dir, saved_frames)
        if args.progress_every > 0 and (
            step % args.progress_every == 0 or step == requested_steps
        ):
            elapsed = time.perf_counter() - wall_start
            print(
                f"[Progress] step={step}/{requested_steps}, t={time_s:.4f}s, "
                f"speed={row['planar_speed_m_s']:.5f}m/s, l/L={contact_ratio:.4f}, "
                f"contact={row['contact_force_norm_N']:.3f}N, wall={elapsed:.1f}s"
            )

    wall_time_s = time.perf_counter() - wall_start
    run_complete = requested_steps == planned_steps and len(rows) == planned_steps
    cycles = cycle_statistics(config, rows)
    output_path = _portable_output_path(out_dir)
    summary = build_summary(
        config,
        rows,
        cycles,
        run_complete=run_complete,
        requested_steps=requested_steps,
        planned_steps=planned_steps,
        wall_time_s=wall_time_s,
        output_path=output_path,
    )
    validation = build_validation_report(config, summary)
    write_rows_csv(out_dir / "diagnostics.csv", rows)
    write_rows_csv(
        out_dir / "cycle_statistics.csv",
        cycles,
        fieldnames=[
            "cycle_id",
            "time_start_s",
            "time_end_s",
            "sample_count",
            "displacement_x_m",
            "displacement_y_m",
            "progression_speed_m_s",
            "mean_planar_speed_m_s",
            "mean_contact_length_ratio",
            "std_contact_length_ratio",
            "mean_tracking_rms_deg",
            "motor_work_J",
        ],
    )
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "validation.json", validation)
    resolved = asdict(config)
    resolved["derived"] = {
        "contact_patch_count": config.contact_patch_count,
        "module_pitch_m": config.model.module_pitch_m,
        "module_mass_kg": config.model.module_mass_kg,
        "wavelength_m": config.model.wavelength_m,
        "mpm_substeps_per_mbd": config.resolution.mpm_substeps,
        "actual_grid_shape": list(solver.grid_shape),
        "actual_grid_dx_m": solver.dx,
        "particle_count": solver.n_particles,
        "moving_window_axis": solver.moving_window_axis_name,
        "moving_window_direction": solver.moving_window_direction,
        "moving_window_template_layers": solver.moving_window_template_layers,
        "contact_barrier_radius_m": solver.contact_barrier_radius_value,
        "contact_barrier_stiffness_N_m": solver.contact_barrier_stiffness_value,
    }
    write_json(out_dir / "resolved_config.json", resolved)
    print(
        f"[Done] steps={len(rows)}, simulated={summary.get('simulated_time_s', 0.0):.6g}s, "
        f"wall={wall_time_s:.2f}s, output={output_path}"
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_case_config(args.config, args.preset)
    out_dir = (
        args.out
        if args.out is not None
        else PROJECT_ROOT / "output" / "cmu_snake_mbd_mpm_3d" / args.preset
    )
    initialize_taichi(args.arch)
    run(config, out_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
