from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import CaseConfig


def write_rows_csv(
    path: Path,
    rows: list[dict[str, float | int]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    if not rows:
        path.write_text(
            "" if fieldnames is None else ",".join(fieldnames) + "\n",
            encoding="utf-8",
        )
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _values(rows: list[dict[str, float | int]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def cycle_statistics(
    config: CaseConfig,
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    if not rows:
        return []
    start = config.model.settle_time_s + config.model.ramp_time_s
    period = 1.0 / config.model.gait_frequency_hz
    last_time = float(rows[-1]["time_s"])
    complete_cycle_count = max(0, int(math.floor((last_time - start) / period)))
    result: list[dict[str, float | int]] = []
    for cycle_id in range(complete_cycle_count):
        lo = start + cycle_id * period
        hi = lo + period
        selected = [row for row in rows if lo < float(row["time_s"]) <= hi]
        if not selected:
            continue
        first = selected[0]
        last = selected[-1]
        dt = max(float(last["time_s"]) - float(first["time_s"]), 1.0e-12)
        dx = float(last["com_x_m"]) - float(first["com_x_m"])
        dy = float(last["com_y_m"]) - float(first["com_y_m"])
        result.append(
            {
                "cycle_id": cycle_id,
                "time_start_s": lo,
                "time_end_s": hi,
                "sample_count": len(selected),
                "displacement_x_m": dx,
                "displacement_y_m": dy,
                "progression_speed_m_s": math.hypot(dx, dy) / dt,
                "mean_planar_speed_m_s": float(np.mean(_values(selected, "planar_speed_m_s"))),
                "mean_contact_length_ratio": float(
                    np.mean(_values(selected, "contact_length_ratio"))
                ),
                "std_contact_length_ratio": float(
                    np.std(_values(selected, "contact_length_ratio"))
                ),
                "mean_tracking_rms_deg": float(
                    np.mean(_values(selected, "tracking_rms_deg"))
                ),
                "motor_work_J": float(last["motor_work_cumulative_J"])
                - float(first["motor_work_cumulative_J"]),
            }
        )
    return result


def build_summary(
    config: CaseConfig,
    rows: list[dict[str, float | int]],
    cycles: list[dict[str, float | int]],
    *,
    run_complete: bool,
    requested_steps: int,
    planned_steps: int,
    wall_time_s: float,
    output_path: str,
) -> dict:
    if not rows:
        return {
            "preset": config.resolution.name,
            "run_complete": run_complete,
            "requested_steps": requested_steps,
            "planned_steps": planned_steps,
            "completed_steps": 0,
            "wall_time_s": wall_time_s,
            "output_path": output_path,
        }
    steady_start = (
        config.model.settle_time_s
        + config.model.ramp_time_s
        + config.validation.steady_state_delay_s
    )
    steady = [row for row in rows if float(row["time_s"]) >= steady_start]
    if not steady:
        steady = rows
    first = steady[0]
    last = steady[-1]
    dt = max(float(last["time_s"]) - float(first["time_s"]), config.resolution.mbd_dt)
    dx = float(last["com_x_m"]) - float(first["com_x_m"])
    dy = float(last["com_y_m"]) - float(first["com_y_m"])
    contact_force = _values(rows, "contact_force_norm_N")
    summary = {
        "preset": config.resolution.name,
        "run_complete": run_complete,
        "requested_steps": requested_steps,
        "planned_steps": planned_steps,
        "completed_steps": len(rows),
        "simulated_time_s": float(rows[-1]["time_s"]),
        "wall_time_s": wall_time_s,
        "output_path": output_path,
        "topology": {
            "body_count": config.model.module_count,
            "joint_count": config.model.joint_count,
            "contact_patch_count": config.contact_patch_count,
        },
        "steady_window": {
            "requested_start_s": steady_start,
            "actual_start_s": float(first["time_s"]),
            "end_s": float(last["time_s"]),
            "sample_count": len(steady),
            "displacement_x_m": dx,
            "displacement_y_m": dy,
            "progression_speed_m_s": math.hypot(dx, dy) / dt,
            "mean_velocity_x_m_s": float(np.mean(_values(steady, "com_vx_m_s"))),
            "mean_velocity_y_m_s": float(np.mean(_values(steady, "com_vy_m_s"))),
            "mean_planar_speed_m_s": float(np.mean(_values(steady, "planar_speed_m_s"))),
            "mean_contact_length_ratio": float(
                np.mean(_values(steady, "contact_length_ratio"))
            ),
            "std_contact_length_ratio": float(
                np.std(_values(steady, "contact_length_ratio"))
            ),
        },
        "diagnostic_extrema": {
            "max_joint_position_error_m": float(
                np.max(_values(rows, "max_joint_position_error_m"))
            ),
            "max_joint_axis_error_rad": float(
                np.max(_values(rows, "max_joint_axis_error_rad"))
            ),
            "max_action_reaction_relative": float(
                np.max(_values(rows, "action_reaction_relative_residual"))
            ),
            "max_tracking_rms_deg": float(np.max(_values(rows, "tracking_rms_deg"))),
            "max_motor_torque_Nm": float(np.max(_values(rows, "max_motor_torque_Nm"))),
            "max_contact_force_N": float(np.max(contact_force)),
            "max_contact_length_ratio": float(
                np.max(_values(rows, "contact_length_ratio"))
            ),
        },
        "work_and_energy": {
            "motor_work_J": float(rows[-1]["motor_work_cumulative_J"]),
            "mbd_contact_work_J": float(rows[-1]["mbd_contact_work_cumulative_J"]),
            "mpm_rigid_contact_work_J": float(
                rows[-1]["mpm_rigid_contact_work_cumulative_J"]
            ),
            "mpm_soil_contact_work_J": float(
                rows[-1]["mpm_soil_contact_work_cumulative_J"]
            ),
            "soil_internal_work_J": float(rows[-1]["soil_internal_work_cumulative_J"]),
            "final_mbd_mechanical_energy_J": float(rows[-1]["mbd_mechanical_energy_J"]),
        },
        "complete_gait_cycles": len(cycles),
    }
    return summary


def build_validation_report(config: CaseConfig, summary: dict) -> dict:
    extrema = summary.get("diagnostic_extrema", {})
    completed_steps = int(summary.get("completed_steps", 0))
    values = [
        float(value)
        for value in extrema.values()
        if isinstance(value, (float, int))
    ]
    checks = {
        "finite_diagnostics": bool(values) and bool(np.all(np.isfinite(values))),
        "exact_17_body_16_joint_topology": summary.get("topology", {}).get("body_count") == 17
        and summary.get("topology", {}).get("joint_count") == 16,
        "joint_position_closure": float(
            extrema.get("max_joint_position_error_m", math.inf)
        )
        <= config.validation.max_joint_position_error_m,
        "joint_axis_alignment": float(extrema.get("max_joint_axis_error_rad", math.inf))
        <= config.validation.max_joint_axis_error_rad,
        "mpm_action_reaction": float(
            extrema.get("max_action_reaction_relative", math.inf)
        )
        <= config.validation.max_action_reaction_relative,
        "motor_torque_limit": float(extrema.get("max_motor_torque_Nm", math.inf))
        <= config.model.motor_torque_limit_Nm * (1.0 + 1.0e-9),
    }
    gait_started = float(summary.get("simulated_time_s", 0.0)) > (
        config.model.settle_time_s + config.model.ramp_time_s
    )
    tracking_value = float(extrema.get("max_tracking_rms_deg", math.inf))
    tracking_status = (
        "pass"
        if gait_started and tracking_value <= config.validation.max_tracking_rms_deg
        else "fail"
        if gait_started
        else "not_evaluated_before_gait_start"
    )
    structural_pass = completed_steps > 0 and all(checks.values())
    verification_status = (
        "VERIFIED_SMOKE"
        if structural_pass and bool(summary.get("run_complete", False))
        else "VERIFIED_PREFLIGHT"
        if structural_pass
        else "FAILED"
    )
    return {
        "verification_status": verification_status,
        "structural_checks_passed": structural_pass,
        "checks": checks,
        "motor_tracking": {
            "status": tracking_status,
            "observed_max_rms_deg": tracking_value,
            "limit_deg": config.validation.max_tracking_rms_deg,
        },
        "experimental_comparison": {
            "status": "pending_experimental_curve_digitization",
            "required_outputs_present": completed_steps > 0,
            "observables": [
                "cycle-averaged contact length ratio l/L",
                "cycle progression speed",
                "gait frequency and commanded joint amplitudes",
            ],
        },
        "notes": [
            "Smoke validation confirms topology, constraints, coupling bookkeeping and finite output.",
            "A full validation verdict requires at least one complete post-ramp gait cycle.",
            "Experimental agreement is intentionally not claimed until source curves are digitized.",
        ],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
