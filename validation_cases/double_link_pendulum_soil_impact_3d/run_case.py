from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti


CASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tank_mpm.particles import sample_soil_particles  # noqa: E402
from tank_mpm.constitutive_models import (  # noqa: E402
    CONSTITUTIVE_MODEL_CHOICES,
    normalize_constitutive_model_name,
)
from tank_mpm.mpm_solver import TankTrackMpmSolver  # noqa: E402
from validation_cases.double_link_pendulum_soil_impact_3d.config import (  # noqa: E402
    CaseConfig,
    load_case_config,
)
from validation_cases.double_link_pendulum_soil_impact_3d.mbd_model import (  # noqa: E402
    ContactFeedback,
    DoubleLinkPendulumMBD,
    hammer_contact_patch_count,
)
from validation_cases.double_link_pendulum_soil_impact_3d.validation import (  # noqa: E402
    write_diagnostics_csv,
    write_validation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D double-link pendulum hammer impacting MPM soil validation case."
    )
    parser.add_argument("--preset", choices=("fine", "standard", "smoke"), default="fine")
    parser.add_argument("--config", type=Path, default=CASE_DIR / "case_config.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--arch",
        choices=("cuda", "cpu", "vulkan", "gpu"),
        default="cuda",
        help=(
            "Taichi execution backend. 'cuda' selects NVIDIA CUDA explicitly and "
            "never falls back to CPU; 'gpu' lets Taichi choose an available GPU backend."
        ),
    )
    parser.add_argument("--max-macro-steps", type=int, default=None)
    parser.add_argument(
        "--constitutive-model",
        type=normalize_constitutive_model_name,
        default=None,
        metavar="MODEL",
        help=(
            "Override soil.constitutive_model from the case config. Available aliases: "
            + ", ".join(CONSTITUTIVE_MODEL_CHOICES)
        ),
    )
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
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
            "CUDA backend requested, but Taichi cannot load the NVIDIA CUDA driver "
            "(nvcuda.dll). Install/repair the NVIDIA display driver and verify that "
            "nvidia-smi works before running this case with --arch cuda."
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
    material_density = soil.water_density if selected_model == "pure-water" else soil.density
    mpm_dt = resolution.mpm_dt
    if selected_model == "pure-water":
        domain_extent = max(
            hi - lo for lo, hi in zip(soil.domain_lo, soil.domain_hi)
        )
        grid_dx = domain_extent / resolution.n_grid_x
        acoustic_speed = math.sqrt(soil.water_bulk_modulus / material_density)
        acoustic_dt = 0.4 * grid_dx / acoustic_speed
        water_substeps = max(
            resolution.mpm_substeps,
            int(math.ceil(resolution.mbd_dt / acoustic_dt)),
        )
        mpm_dt = resolution.mbd_dt / water_substeps
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
    solver = TankTrackMpmSolver(
        soil_points=points,
        soil_p_vol=particle_volume,
        n_grid=resolution.n_grid_x,
        dt=mpm_dt,
        domain_lo=soil.domain_lo,
        domain_hi=soil.domain_hi,
        soil_bounds_lo=soil.bounds_lo,
        soil_bounds_hi=soil.bounds_hi,
        max_track_patches=hammer_contact_patch_count(resolution.head_patch_grid),
        soil_density=material_density,
        soil_E=soil.young_modulus,
        soil_nu=soil.poisson_ratio,
        soil_phi_deg=soil.friction_angle_deg,
        soil_psi_deg=soil.dilation_angle_deg,
        soil_cohesion=soil.cohesion,
        soil_constitutive_model=selected_model,
        water_bulk_modulus=soil.water_bulk_modulus,
        water_dynamic_viscosity=soil.water_dynamic_viscosity,
        water_cavitation_pressure=soil.water_cavitation_pressure,
        soil_gravity_scale=1.0,
        soil_damping=1.0,
        contact_mu=soil.contact_mu,
        contact_barrier_stiffness=soil.contact_barrier_stiffness,
        contact_barrier_min_distance_ratio=soil.contact_barrier_min_distance_ratio,
        track_activation_height=soil.contact_activation_height,
        mpm_precision="f32",
        particle_shape=resolution.soil_particles,
    )
    k0 = (
        1.0
        if selected_model == "pure-water"
        else 1.0 - math.sin(math.radians(soil.friction_angle_deg))
    )
    solver.initialize_geostatic_stress(k0=k0, gravity_scale=1.0)
    return solver


def _soil_snapshot(solver: TankTrackMpmSolver) -> tuple[float, float]:
    kinetic, potential = solver.soil_mechanical_energy()
    return float(kinetic), float(potential)


def _save_vtk(
    vtk_dir: Path,
    step: int,
    solver: TankTrackMpmSolver,
    mbd: DoubleLinkPendulumMBD,
    feedback: ContactFeedback,
) -> None:
    solver.export_soil(vtk_dir, step)
    mbd.export_vtk(vtk_dir / f"pendulum_{step:06d}.vtk", feedback)


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
    soil_entries = [
        (time_value, 0, "soil", f"vtk/soil_{step:06d}.vtk")
        for step, time_value in frames
    ]
    pendulum_entries = [
        (time_value, 0, "pendulum", f"vtk/pendulum_{step:06d}.vtk")
        for step, time_value in frames
    ]
    coupled_entries = []
    for step, time_value in frames:
        coupled_entries.append((time_value, 0, "soil", f"vtk/soil_{step:06d}.vtk"))
        coupled_entries.append((time_value, 1, "pendulum", f"vtk/pendulum_{step:06d}.vtk"))
    _write_pvd(out_dir / "soil_series.pvd", soil_entries)
    _write_pvd(out_dir / "pendulum_series.pvd", pendulum_entries)
    _write_pvd(out_dir / "coupled_scene.pvd", coupled_entries)


def _portable_output_path(out_dir: Path) -> str:
    try:
        return str(out_dir.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(out_dir.resolve())


def run(config: CaseConfig, out_dir: Path, args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    if not args.no_vtk:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    solver = build_mpm_solver(config, args.constitutive_model)
    mbd = DoubleLinkPendulumMBD(
        config.model,
        config.resolution.head_patch_grid,
        config.resolution.visual_segments,
    )
    initial_patches = mbd.contact_patches()
    solver.set_track_patches(initial_patches)
    feedback = ContactFeedback.zeros(mbd.contact_patch_count)
    print(
        f"[Contact] mantle_patches={mbd.mantle_contact_patch_count}, "
        f"end_face_patches={mbd.end_face_contact_patch_count} per side, "
        f"total={mbd.contact_patch_count}"
    )

    mpm_substeps = int(round(config.resolution.mbd_dt / solver.dt))
    print(
        f"[MPM] model={solver.soil_constitutive_model_name}, "
        f"density={solver.p_rho:.3f}kg/m^3, actual_dt={solver.dt:.3e}s, "
        f"substeps_per_mbd={mpm_substeps}"
    )
    planned_steps = int(round(config.resolution.duration / config.resolution.mbd_dt))
    requested_steps = planned_steps
    if args.max_macro_steps is not None:
        requested_steps = min(requested_steps, max(0, int(args.max_macro_steps)))
    save_stride = max(1, int(round(config.resolution.save_interval / config.resolution.mbd_dt)))

    initial_mbd_energy = mbd.mechanical_energy()
    initial_soil_kinetic, initial_soil_potential = _soil_snapshot(solver)
    cumulative_mbd_work = 0.0
    cumulative_joint_damping_work = 0.0
    cumulative_mpm_rigid_work = 0.0
    cumulative_mpm_soil_work = 0.0
    cumulative_soil_internal_work = 0.0
    rows: list[dict[str, float | int]] = []
    saved_frames: list[tuple[int, float]] = []

    if not args.no_vtk:
        _save_vtk(vtk_dir, 0, solver, mbd, feedback)
        saved_frames.append((0, 0.0))
        _write_pvd_collections(out_dir, saved_frames)

    run_complete = False
    wall_start = time.perf_counter()
    try:
        for macro_index in range(requested_steps):
            step = macro_index + 1
            start_patches = mbd.contact_patches()
            start_mbd_energy = mbd.mechanical_energy()
            applied_patch_power_start = mbd.patch_contact_power(feedback)
            applied_generalized_power_start = mbd.generalized_contact_power(feedback)
            joint_damping_power_start = mbd.joint_damping_power()

            mbd.step(config.resolution.mbd_dt, feedback)

            end_patches = mbd.contact_patches()
            end_mbd_energy = mbd.mechanical_energy()
            applied_patch_power_end = mbd.patch_contact_power(feedback)
            applied_generalized_power_end = mbd.generalized_contact_power(feedback)
            joint_damping_power_end = mbd.joint_damping_power()
            applied_work_increment = (
                0.5
                * (applied_patch_power_start + applied_patch_power_end)
                * config.resolution.mbd_dt
            )
            cumulative_mbd_work += applied_work_increment
            joint_damping_work_increment = (
                0.5
                * (joint_damping_power_start + joint_damping_power_end)
                * config.resolution.mbd_dt
            )
            cumulative_joint_damping_work += joint_damping_work_increment

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

            macro_contact = solver.contact_macro_summary()
            feedback = ContactFeedback(
                patch_forces=np.asarray(macro_contact["patch_forces"], dtype=np.float32),
                patch_moments=np.asarray(macro_contact["patch_moments"], dtype=np.float32),
            )
            soil_patch_forces = np.asarray(
                macro_contact["soil_patch_forces"], dtype=np.float32
            )
            macro_rigid_work = float(macro_contact["rigid_work"])
            macro_soil_work = float(macro_contact["soil_work"])
            macro_internal_work = float(macro_contact["internal_work"])
            max_action_absolute = float(macro_contact["max_action_absolute"])
            max_action_relative = float(macro_contact["max_action_relative"])
            contact_particle_sum = int(macro_contact["contact_particle_samples"])
            max_contact_particles = int(macro_contact["max_contact_particles"])
            cumulative_mpm_rigid_work += macro_rigid_work
            cumulative_mpm_soil_work += macro_soil_work
            cumulative_soil_internal_work += macro_internal_work

            contact_force = np.sum(feedback.patch_forces, axis=0)
            soil_contact_force = np.sum(soil_patch_forces, axis=0)
            q_contact = mbd.generalized_contact_force(feedback)
            patch_power = mbd.patch_contact_power(feedback)
            generalized_power = mbd.generalized_contact_power(feedback)
            generalized_power_absolute = abs(patch_power - generalized_power)
            generalized_power_scale = max(abs(patch_power), abs(generalized_power), 1.0e-9)
            generalized_power_relative = generalized_power_absolute / generalized_power_scale

            constraints = mbd.constraint_errors()
            reactions = mbd.joint_reactions()
            coordinates, rates = mbd.generalized_state()
            soil_kinetic, soil_potential = _soil_snapshot(solver)
            mbd_energy = mbd.mechanical_energy()
            mbd_energy_change = mbd_energy - initial_mbd_energy
            mbd_energy_residual = (
                mbd_energy_change - cumulative_mbd_work - cumulative_joint_damping_work
            )
            mbd_energy_scale = max(
                abs(mbd_energy_change),
                abs(cumulative_mbd_work) + abs(cumulative_joint_damping_work),
                1.0,
            )
            mbd_energy_relative = abs(mbd_energy_residual) / mbd_energy_scale

            total_energy_change = (
                mbd_energy
                + soil_kinetic
                + soil_potential
                + cumulative_soil_internal_work
                - initial_mbd_energy
                - initial_soil_kinetic
                - initial_soil_potential
            )
            actual_interface_work = (
                cumulative_mbd_work
                + cumulative_mpm_soil_work
                + cumulative_joint_damping_work
            )
            total_energy_residual = total_energy_change - actual_interface_work
            total_energy_scale = max(
                abs(total_energy_change),
                abs(cumulative_mbd_work)
                + abs(cumulative_mpm_soil_work)
                + abs(cumulative_joint_damping_work),
                1.0,
            )
            total_energy_relative = abs(total_energy_residual) / total_energy_scale

            ground_force = reactions["ground_force_world"]
            ground_torque = reactions["ground_torque_world"]
            links_force = reactions["links_force_world"]
            links_torque = reactions["links_torque_world"]
            applied_power_difference = abs(
                0.5
                * (
                    applied_patch_power_start
                    + applied_patch_power_end
                    - applied_generalized_power_start
                    - applied_generalized_power_end
                )
            )
            row: dict[str, float | int] = {
                "step": step,
                "time_s": step * config.resolution.mbd_dt,
                "theta1_rad": float(coordinates[0]),
                "theta2_absolute_rad": float(coordinates[1]),
                "theta2_relative_rad": float(coordinates[1] - coordinates[0]),
                "theta1_rate_rad_s": float(rates[0]),
                "theta2_absolute_rate_rad_s": float(rates[1]),
                "joint_ground_position_error_m": constraints["joint_ground_position_error"],
                "joint_links_position_error_m": constraints["joint_links_position_error"],
                "joint_ground_axis_error_rad": constraints["joint_ground_axis_error"],
                "joint_links_axis_error_rad": constraints["joint_links_axis_error"],
                "chrono_constraint_violation_norm": constraints["chrono_constraint_violation_norm"],
                "contact_particles": max_contact_particles,
                "contact_particle_samples": contact_particle_sum,
                "contact_force_x_N": float(contact_force[0]),
                "contact_force_y_N": float(contact_force[1]),
                "contact_force_z_N": float(contact_force[2]),
                "contact_force_norm_N": float(np.linalg.norm(contact_force)),
                "soil_contact_force_x_N": float(soil_contact_force[0]),
                "soil_contact_force_y_N": float(soil_contact_force[1]),
                "soil_contact_force_z_N": float(soil_contact_force[2]),
                "action_reaction_absolute_residual_N": max_action_absolute,
                "action_reaction_relative_residual": max_action_relative,
                "generalized_contact_force_q1_Nm": float(q_contact[0]),
                "generalized_contact_force_q2_Nm": float(q_contact[1]),
                "patch_contact_power_W": patch_power,
                "generalized_contact_power_W": generalized_power,
                "generalized_power_absolute_residual_W": generalized_power_absolute,
                "generalized_power_relative_residual": generalized_power_relative,
                "applied_power_mapping_absolute_residual_W": applied_power_difference,
                "joint_damping_power_W": joint_damping_power_end,
                "joint_ground_reaction_force_x_N": float(ground_force[0]),
                "joint_ground_reaction_force_y_N": float(ground_force[1]),
                "joint_ground_reaction_force_z_N": float(ground_force[2]),
                "joint_ground_reaction_force_norm_N": float(np.linalg.norm(ground_force)),
                "joint_ground_reaction_torque_x_Nm": float(ground_torque[0]),
                "joint_ground_reaction_torque_y_Nm": float(ground_torque[1]),
                "joint_ground_reaction_torque_z_Nm": float(ground_torque[2]),
                "joint_links_reaction_force_x_N": float(links_force[0]),
                "joint_links_reaction_force_y_N": float(links_force[1]),
                "joint_links_reaction_force_z_N": float(links_force[2]),
                "joint_links_reaction_force_norm_N": float(np.linalg.norm(links_force)),
                "joint_links_reaction_torque_x_Nm": float(links_torque[0]),
                "joint_links_reaction_torque_y_Nm": float(links_torque[1]),
                "joint_links_reaction_torque_z_Nm": float(links_torque[2]),
                "mbd_mechanical_energy_J": mbd_energy,
                "soil_kinetic_energy_J": soil_kinetic,
                "soil_potential_energy_J": soil_potential,
                "soil_internal_work_cumulative_J": cumulative_soil_internal_work,
                "mbd_contact_work_cumulative_J": cumulative_mbd_work,
                "joint_damping_work_cumulative_J": cumulative_joint_damping_work,
                "mpm_rigid_contact_work_cumulative_J": cumulative_mpm_rigid_work,
                "mpm_soil_contact_work_cumulative_J": cumulative_mpm_soil_work,
                "contact_relative_dissipation_cumulative_J": -(
                    cumulative_mpm_rigid_work + cumulative_mpm_soil_work
                ),
                "partitioned_coupling_lag_work_J": cumulative_mbd_work
                - cumulative_mpm_rigid_work,
                "mbd_energy_absolute_residual_J": mbd_energy_residual,
                "mbd_energy_relative_residual": mbd_energy_relative,
                "total_energy_absolute_residual_J": total_energy_residual,
                "total_energy_relative_residual": total_energy_relative,
                "mbd_step_energy_change_J": end_mbd_energy - start_mbd_energy,
                "mbd_step_contact_work_J": applied_work_increment,
                "mbd_step_joint_damping_work_J": joint_damping_work_increment,
            }
            rows.append(row)

            if not args.no_vtk and (step % save_stride == 0 or step == requested_steps):
                _save_vtk(vtk_dir, step, solver, mbd, feedback)
                saved_frames.append((step, step * config.resolution.mbd_dt))
                _write_pvd_collections(out_dir, saved_frames)
            if args.progress_every > 0 and (step % args.progress_every == 0 or step == requested_steps):
                elapsed = time.perf_counter() - wall_start
                print(
                    f"[Progress] {step}/{requested_steps} t={row['time_s']:.5f}s "
                    f"contact={max_contact_particles} |F|={row['contact_force_norm_N']:.3e}N "
                    f"joint={max(constraints['joint_ground_position_error'], constraints['joint_links_position_error']):.3e}m "
                    f"wall={elapsed:.1f}s"
                )
        run_complete = requested_steps == planned_steps
    except KeyboardInterrupt:
        print("[Interrupted] Writing partial diagnostics before exit.")

    if not rows:
        raise RuntimeError("No macro steps were executed; increase --max-macro-steps")
    write_diagnostics_csv(out_dir / "diagnostics.csv", rows)
    summary = write_validation_artifacts(
        out_dir,
        config,
        rows,
        run_complete=run_complete,
    )
    run_metadata = {
        "preset": config.resolution.name,
        "taichi_arch": args.arch,
        "mpm_precision": solver.mpm_precision,
        "constitutive_model": solver.soil_constitutive_model_name,
        "material_density_kg_m3": solver.p_rho,
        "stress_sign_convention": solver.stress_sign_convention,
        "planned_macro_steps": planned_steps,
        "executed_macro_steps": len(rows),
        "mpm_substeps_per_macro": mpm_substeps,
        "configured_mpm_dt_s": config.resolution.mpm_dt,
        "actual_mpm_dt_s": solver.dt,
        "run_complete": run_complete,
        "wall_time_s": time.perf_counter() - wall_start,
        "particle_count": int(np.prod(config.resolution.soil_particles)),
        "contact_patch_count": mbd.contact_patch_count,
        "mantle_contact_patch_count": mbd.mantle_contact_patch_count,
        "end_face_contact_patch_count_per_side": mbd.end_face_contact_patch_count,
        "output_directory": _portable_output_path(out_dir),
    }
    if solver.soil_constitutive_model_name == "pure-water":
        run_metadata.update(
            water_bulk_modulus_Pa=solver.water_bulk_modulus,
            water_dynamic_viscosity_Pa_s=solver.water_dynamic_viscosity,
            water_cavitation_pressure_Pa=solver.water_cavitation_pressure,
            water_acoustic_speed_m_s=solver.water_acoustic_speed,
            water_recommended_dt_s=solver.water_recommended_dt,
        )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_case_config(args.config, args.preset)
    out_dir = args.out or (CASE_DIR / "outputs" / args.preset)
    initialize_taichi(args.arch)
    print(
        f"[Case] preset={args.preset}, particles={np.prod(config.resolution.soil_particles)}, "
        f"patches={hammer_contact_patch_count(config.resolution.head_patch_grid)}, "
        f"configured_mpm_dt={config.resolution.mpm_dt:.3e}s, "
        f"mbd_dt={config.resolution.mbd_dt:.3e}s"
    )
    summary = run(config, out_dir, args)
    print(f"[Done] validation status={summary['overall_status']} output={out_dir}")
    return 0 if summary["overall_status"] in {"PASS", "NOT_EVALUATED", "INCOMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
