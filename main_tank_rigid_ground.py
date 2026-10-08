from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from tank_mpm.vehicle import (
    GroundParams,
    MeshBuilder,
    MultibodyRigidGroundTank,
    VISUAL_DETAIL_CHOICES,
    VISUAL_DETAIL_SIMPLE,
    build_multibody_tank_mesh_with_obstacles,
    build_semicircular_obstacle_in_front,
    write_joint_history_csv,
    write_metrics_csv,
)
from tank_mpm.chrono_vehicle import ProjectChronoVehicleBackend
from tank_mpm.diagnostics import write_constraint_diagnostics


def smooth_ramp(t: float, ramp_time: float) -> float:
    if ramp_time <= 0.0:
        return 1.0
    a = max(0.0, min(1.0, t / ramp_time))
    return a * a * (3.0 - 2.0 * a)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multibody rigid-ground test for an identified tracked vehicle model. "
            "The simulation uses the same generated body/joint tree and default tank parameters "
            "as the MBD-MPM workflow, but replaces the MPM soil interface with rigid-ground contact."
        )
    )
    parser.add_argument("--model", type=str, default="ZTZ_96/multibody/ztz96_multibody_model.json", help="Multibody JSON model")
    parser.add_argument("--out", type=str, default="outputs/output_tank_rigid_ground", help="Output directory")
    parser.add_argument("--steps", type=int, default=1400, help="Number of explicit time steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step in seconds")
    parser.add_argument("--save-every", type=int, default=100, help="VTK export interval")
    parser.add_argument(
        "--tank-vtk-detail",
        choices=VISUAL_DETAIL_CHOICES,
        default=VISUAL_DETAIL_SIMPLE,
        help="Tank VTK visual detail. simple writes a low-face primitive model; full restores the OBJ visual mesh.",
    )
    parser.add_argument(
        "--track-dynamics-model",
        choices=["chrono-vehicle", "project-chrono"],
        default="chrono-vehicle",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sprocket-tooth-count", type=int, default=0, help="Drive sprocket tooth count. Use 0 to infer from sprocket radius and track pitch.")
    parser.add_argument("--chrono-max-substep-dt", type=float, default=5.0e-4)
    parser.add_argument("--chrono-contact-young-modulus", type=float, default=2.0e7)
    parser.add_argument("--chrono-contact-kn", type=float, default=3.0e6)
    parser.add_argument("--chrono-contact-gn", type=float, default=4.0e4)
    parser.add_argument("--chrono-contact-kt", type=float, default=1.0e6)
    parser.add_argument("--chrono-contact-gt", type=float, default=1.0e4)
    parser.add_argument("--chrono-wheel-mass", type=float, default=120.0)
    parser.add_argument("--chrono-wheel-density", type=float, default=900.0)
    parser.add_argument("--chrono-vehicle-asset-dir", type=str, default="", help="Directory for generated Chrono vehicle JSON. Empty uses an ASCII temp directory.")
    parser.add_argument("--chrono-chassis-mass", type=float, default=0.0, help="Chrono vehicle chassis mass, kg. Use <=0 to auto-subtract tracks/running gear from --mass.")
    parser.add_argument("--chrono-roller-mass", type=float, default=0.0, help="Return roller mass, kg. Use <=0 for an estimate from --chrono-wheel-mass.")
    parser.add_argument("--chrono-sprocket-mass", type=float, default=0.0, help="Drive sprocket gear mass, kg. Use <=0 for an estimate from --chrono-wheel-mass.")
    parser.add_argument("--chrono-idler-mass", type=float, default=0.0, help="Idler wheel mass, kg. Use <=0 to use --chrono-wheel-mass.")
    parser.add_argument("--chrono-suspension-arm-mass", type=float, default=75.0)
    parser.add_argument("--chrono-suspension-spring-constant", type=float, default=8.0e4)
    parser.add_argument("--chrono-suspension-damping-coefficient", type=float, default=2.0e3)
    parser.add_argument("--chrono-suspension-preload", type=float, default=-1.0e4)
    parser.add_argument("--chrono-aux-damper-coefficient", type=float, default=1.0e2)
    parser.add_argument("--chrono-tensioner-preload", type=float, default=2.0e4)
    parser.add_argument("--chrono-tensioner-free-length", type=float, default=0.75)
    parser.add_argument("--chrono-tensioner-stiffness", type=float, default=1.0e6)
    parser.add_argument("--chrono-tensioner-damping", type=float, default=1.4e4)
    parser.add_argument("--chrono-track-shoe-count", type=int, default=0, help="Track shoe count per side for Chrono vehicle. Use 0 for model count plus padding.")
    parser.add_argument("--chrono-track-shoe-count-padding", type=int, default=2, help="Extra shoes added in Chrono vehicle assembly when --chrono-track-shoe-count is 0.")
    parser.add_argument("--chrono-track-shoe-pitch", type=float, default=None, help="Optional Chrono assembly pitch override, m.")
    parser.add_argument("--chrono-idler-assembly-retraction", type=float, default=None, help="Signed initial idler shift toward the sprocket, m.")
    parser.add_argument("--chrono-drive-torque-sign", type=float, default=-1.0, help="Sign applied to direct sprocket axle torques in the Chrono vehicle frame.")
    parser.add_argument("--chrono-max-penetration-recovery-speed", type=float, default=0.25)

    parser.add_argument(
        "--drive-torque",
        type=float,
        default=3000.0,
        help="Commanded drive torque per sprocket, N*m. Positive is the default forward-driving direction for the direct tooth-pin contact model.",
    )
    parser.add_argument("--ramp-time", type=float, default=2.0, help="Smooth drive ramp duration, seconds")

    parser.add_argument("--friction-mu", type=float, default=0.58, help="Rigid ground Coulomb friction coefficient")
    parser.add_argument("--rolling-resistance", type=float, default=0.035, help="Rolling resistance coefficient")
    parser.add_argument(
        "--slip-regularization",
        type=float,
        default=0.35,
        help="Velocity scale for regularized track-ground friction, m/s",
    )
    parser.add_argument("--yaw-damping", type=float, default=0.0, help="Optional linear yaw damping, N*m*s/rad")
    parser.add_argument(
        "--mass",
        type=float,
        default=None,
        help="Vehicle mass, kg. Omit to use mass_properties.mass_kg from the selected model JSON.",
    )
    parser.add_argument(
        "--track-pitch",
        type=float,
        default=None,
        help="Track shoe pitch, m. Default matches the MBD-MPM workflow: prefer calibrated model geometry and otherwise match shoe length.",
    )
    parser.add_argument(
        "--track-width",
        type=float,
        default=None,
        help="Track width, m. Default matches the MBD-MPM workflow: prefer calibrated model geometry and otherwise use the identified loop span.",
    )
    parser.add_argument(
        "--shoe-length",
        type=float,
        default=None,
        help="Track shoe length, m. Default matches the MBD-MPM workflow: prefer calibrated model geometry and otherwise use the runtime fallback.",
    )
    parser.add_argument(
        "--track-shoe-mass",
        type=float,
        default=None,
        help="Mass of one track shoe body, kg. Default uses the model JSON calibrated inertial property.",
    )
    parser.add_argument(
        "--track-shoe-pitch-inertia",
        type=float,
        default=None,
        help="Pitch inertia of one track shoe body, kg*m^2. Default uses the model JSON value or a box estimate.",
    )
    parser.add_argument(
        "--suspension-stiffness",
        type=float,
        default=800.0,
        help="Independent suspension angular stiffness, s^-2 equivalent",
    )
    parser.add_argument(
        "--suspension-damping",
        type=float,
        default=220.0,
        help="Independent suspension angular damping, s^-1 equivalent",
    )
    parser.add_argument(
        "--suspension-max-angle",
        type=float,
        default=0.35,
        help="Independent suspension travel limit, rad",
    )
    parser.add_argument(
        "--sprocket-inertia",
        type=float,
        default=220.0,
        help="Equivalent drive sprocket rotational inertia per side, kg*m^2",
    )
    parser.add_argument(
        "--sprocket-damping",
        type=float,
        default=450.0,
        help="Equivalent drive-line viscous damping per side, N*m*s/rad",
    )
    parser.add_argument(
        "--suspension-wave-amplitude",
        type=float,
        default=0.0,
        help="Optional additive suspension excitation amplitude, rad",
    )
    parser.add_argument(
        "--suspension-wave-frequency",
        type=float,
        default=0.0,
        help="Optional kinematic suspension excitation frequency, rad/s",
    )
    parser.add_argument(
        "--obstacle-height",
        type=float,
        default=0.30,
        help="Radius/height of the semicircular obstacle placed in front of the tank, m. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--obstacle-width",
        type=float,
        default=0.0,
        help="Obstacle width across the vehicle, m. Set <= 0 for automatic full-span width.",
    )
    parser.add_argument(
        "--obstacle-clearance",
        type=float,
        default=1.20,
        help="Gap from the initial front-most track edge to the obstacle near face, m",
    )
    return parser.parse_args()


def append_mesh(target: MeshBuilder, source: MeshBuilder) -> None:
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


def current_track_contact_scalars(
    tank: MultibodyRigidGroundTank,
    track_backend: Any | None = None,
) -> dict[int, dict[str, float]]:
    if track_backend is not None:
        return track_backend.track_contact_scalars()
    return tank.track_shoe_contact_scalars()


def write_rigid_ground_vtk(
    tank: MultibodyRigidGroundTank,
    obstacles,
    path: Path,
    *,
    visual_detail: str,
    track_backend: Any | None = None,
) -> None:
    track_contact_scalars = current_track_contact_scalars(tank, track_backend=track_backend)
    mesh = build_multibody_tank_mesh_with_obstacles(
        tank,
        obstacles=obstacles,
        visual_detail=visual_detail,
        track_contact_scalars=track_contact_scalars,
        include_tracks=track_backend is None,
        include_running_gear=track_backend is None,
    )
    if track_backend is not None:
        append_mesh(mesh, track_backend.build_mesh(track_contact_scalars=track_contact_scalars))
    mesh.write_vtk(path)
    tank._clear_track_related_caches()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ground = GroundParams(
        friction_mu=args.friction_mu,
        rolling_resistance=args.rolling_resistance,
        slip_velocity_regularization=args.slip_regularization,
        yaw_damping=args.yaw_damping,
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
        suspension_wave_amplitude=args.suspension_wave_amplitude,
        suspension_wave_frequency=args.suspension_wave_frequency,
    )
    obstacles = []
    if args.obstacle_height > 0.0:
        obstacles.append(
            build_semicircular_obstacle_in_front(
                tank,
                height=args.obstacle_height,
                clearance=args.obstacle_clearance,
                width=args.obstacle_width if args.obstacle_width > 0.0 else None,
            )
        )
    tank.set_scene_obstacles(obstacles)
    solver = ProjectChronoVehicleBackend(args=args, tank=tank, obstacles=obstacles)
    track_backend = solver
    static_summary = solver.initialize()

    metrics_rows = []
    joint_rows = []
    constraint_rows = []
    if track_backend is not None:
        constraint_rows.extend(track_backend.diagnostics_by_side(step=0))

    print(f"[Info] {tank.data.get('model_name', 'tracked vehicle')} multibody rigid-ground model")
    print(f"[Info] model={Path(args.model).resolve()}")
    print(
        "[Info] "
        f"mass={tank.geom.mass:.1f} kg, left_shoes={tank.shoe_count['left']}, right_shoes={tank.shoe_count['right']}, "
        f"track_width={tank.geom.track_width:.5f} m, shoe_length={tank.geom.shoe_length:.5f} m, "
        f"pitch_left={tank.actual_pitch['left']:.4f} m, pitch_right={tank.actual_pitch['right']:.4f} m, "
        f"drive_torque={args.drive_torque:.1f} N*m"
    )
    print(
        f"[Info] track shoe inertia: mass={tank.track_shoe_mass_value:.6g} kg "
        f"({tank.track_shoe_mass_source}), pitch_inertia={tank.track_shoe_pitch_inertia_value:.6g} kg*m^2 "
        f"({tank.track_shoe_pitch_inertia_source}), total_track_shoe_mass={tank.total_track_shoe_mass:.3f} kg"
    )
    print(f"[Info] tank VTK detail={args.tank_vtk_detail}")
    static = static_summary
    print(
        "[Info] Track dynamics model=chrono-vehicle(pychrono.vehicle TrackedVehicle) "
        f"(bodies={static.bodies}, contacts0={static.contacts}, "
        f"left_shoes={static.left_shoes}, right_shoes={static.right_shoes}, "
        f"max_substep_dt={args.chrono_max_substep_dt:.3e}, "
        f"contact_kn={args.chrono_contact_kn:.3e}, contact_gn={args.chrono_contact_gn:.3e}, "
        f"contact_young={args.chrono_contact_young_modulus:.3e}, "
        f"vehicle_json={static.vehicle_file})"
    )
    print(f"[Info] output={out_dir.resolve()}")
    if obstacles:
        obstacle = obstacles[0]
        print(
            "[Info] "
            f"front obstacle center={obstacle.center}, dims={obstacle.dims} m"
        )
    for step in range(args.steps + 1):
        if args.save_every > 0 and step % args.save_every == 0:
            write_rigid_ground_vtk(
                tank,
                obstacles=obstacles,
                path=out_dir / f"tank_{step:06d}.vtk",
                visual_detail=args.tank_vtk_detail,
                track_backend=track_backend,
            )
            print(f"[Output] step={step:06d} t={tank.state.t:.3f}s vtk written")

        if step == args.steps:
            break

        ramp = smooth_ramp(tank.state.t, args.ramp_time)
        metrics = solver.step(
            left_drive_torque=args.drive_torque * ramp,
            right_drive_torque=args.drive_torque * ramp,
            dt=args.dt,
        )
        constraint_rows.extend(track_backend.diagnostics_by_side())
        metrics_rows.append(metrics)
        joint_rows.append(tank.joint_history_row())

    write_metrics_csv(out_dir / "summary.csv", metrics_rows)
    write_joint_history_csv(out_dir / "joint_history.csv", joint_rows)
    if constraint_rows:
        write_constraint_diagnostics(out_dir / "track_constraints.csv", constraint_rows)

    final = tank.last_metrics
    print(
        "[Done] "
        f"t={final.t:.3f}s forward={final.forward_pos:.3f}m lateral={final.lateral_pos:.3f}m "
        f"heave={final.heave:.3f}m pitch={final.pitch:.4f}rad yaw={final.yaw:.4f}rad "
        f"speed={final.forward_speed:.3f}m/s normal_left={final.left_normal_load:.1f}N "
        f"normal_right={final.right_normal_load:.1f}N slip_left={final.left_slip_ratio:.3f} "
        f"slip_right={final.right_slip_ratio:.3f} obstacle_contacts={final.obstacle_contact_shoes}"
    )


if __name__ == "__main__":
    main()
