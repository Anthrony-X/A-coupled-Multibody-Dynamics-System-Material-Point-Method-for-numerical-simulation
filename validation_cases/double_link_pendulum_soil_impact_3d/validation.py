from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import CaseConfig


def write_diagnostics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty diagnostics table")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maximum(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [abs(float(row[key])) for row in rows if np.isfinite(float(row[key]))]
    return max(values) if values else float(default)


def _criterion(value: float | None, limit: float) -> dict[str, Any]:
    if value is None or not np.isfinite(value):
        return {"status": "NOT_EVALUATED", "value": value, "limit": limit}
    return {"status": "PASS" if value <= limit else "FAIL", "value": value, "limit": limit}


def build_validation_summary(
    config: CaseConfig,
    rows: list[dict[str, float | int]],
    *,
    run_complete: bool,
) -> dict[str, Any]:
    contact_rows = [row for row in rows if int(row["contact_particles"]) > 0]
    max_joint_position = max(
        _maximum(rows, "joint_ground_position_error_m"),
        _maximum(rows, "joint_links_position_error_m"),
    )
    max_joint_axis = max(
        _maximum(rows, "joint_ground_axis_error_rad"),
        _maximum(rows, "joint_links_axis_error_rad"),
    )
    max_action_reaction = (
        _maximum(contact_rows, "action_reaction_relative_residual") if contact_rows else None
    )
    max_generalized_power = (
        _maximum(contact_rows, "generalized_power_relative_residual") if contact_rows else None
    )
    final_mbd_energy = float(rows[-1]["mbd_energy_relative_residual"])
    final_total_energy = float(rows[-1]["total_energy_relative_residual"])

    criteria = {
        "joint_position_constraint": _criterion(
            max_joint_position, config.limits.joint_position_error_m
        ),
        "joint_axis_constraint": _criterion(max_joint_axis, config.limits.joint_axis_error_rad),
        "action_reaction": _criterion(
            max_action_reaction, config.limits.action_reaction_relative
        ),
        "contact_generalized_power": _criterion(
            max_generalized_power, config.limits.generalized_power_relative
        ),
        "mbd_energy_balance": _criterion(final_mbd_energy, config.limits.mbd_energy_relative),
        "coupled_total_energy_balance": _criterion(
            final_total_energy, config.limits.total_energy_relative
        ),
    }
    evaluated = [item["status"] for item in criteria.values() if item["status"] != "NOT_EVALUATED"]
    if not run_complete:
        overall = "INCOMPLETE"
    elif not contact_rows:
        overall = "NOT_EVALUATED"
    elif evaluated and all(status == "PASS" for status in evaluated):
        overall = "PASS"
    else:
        overall = "FAIL"

    return {
        "schema": "mbd-mpm-validation-v1",
        "case": "double_link_pendulum_soil_impact_3d",
        "reference_geometry": "3D_MBD-MPM_Validation_Model_Build_Guide_CN.docx / case 1",
        "preset": config.resolution.name,
        "run_complete": bool(run_complete),
        "overall_status": overall,
        "sample_count": len(rows),
        "contact_sample_count": len(contact_rows),
        "peak_contact_force_N": _maximum(rows, "contact_force_norm_N"),
        "peak_joint_reaction_N": max(
            _maximum(rows, "joint_ground_reaction_force_norm_N"),
            _maximum(rows, "joint_links_reaction_force_norm_N"),
        ),
        "criteria": criteria,
        "limits": asdict(config.limits),
    }


def write_validation_artifacts(
    out_dir: Path,
    config: CaseConfig,
    rows: list[dict[str, float | int]],
    *,
    run_complete: bool,
) -> dict[str, Any]:
    summary = build_validation_summary(config, rows, run_complete=run_complete)
    (out_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = {
        "joint_position_constraint": "关节位置约束误差",
        "joint_axis_constraint": "关节轴线约束误差",
        "action_reaction": "作用力–反作用力残差",
        "contact_generalized_power": "接触力–广义力功率一致性",
        "mbd_energy_balance": "刚体子系统能量平衡",
        "coupled_total_energy_balance": "耦合系统总能量平衡",
    }
    lines = [
        "## Material Passport",
        "",
        "- Material Type: Experiment Validation Report",
        "- Verification Status: "
        + ("VERIFIED" if summary["overall_status"] == "PASS" else "ANALYZED"),
        "- Case ID: double_link_pendulum_soil_impact_3d",
        f"- Preset: {config.resolution.name}",
        "",
        "# 双连杆摆锤–三维土体冲击验证报告",
        "",
        f"总体状态：**{summary['overall_status']}**",
        "",
        f"记录 {summary['sample_count']} 个 MBD 宏步，其中 "
        f"{summary['contact_sample_count']} 个宏步发生接触。",
        f"峰值接触合力：{summary['peak_contact_force_N']:.6e} N；"
        f"峰值关节反力：{summary['peak_joint_reaction_N']:.6e} N。",
        "",
        "| 验证项 | 状态 | 数值 | 阈值 |",
        "|---|---:|---:|---:|",
    ]
    for key, item in summary["criteria"].items():
        value = "N/A" if item["value"] is None else f"{float(item['value']):.6e}"
        lines.append(
            f"| {labels[key]} | {item['status']} | {value} | {float(item['limit']):.6e} |"
        )
    lines.extend(
        [
            "",
            "能量口径：刚体机械能包含动能与重力势能；土体项包含粒子动能、"
            "重力势能以及逐步累计的应力功。",
            "刚体能量残差扣除了实际施加到 Chrono 的圆柱接触功与两个转动副的"
            "黏性阻尼功；耦合总残差还计入 MPM 土体侧界面功。CSV 另列接触相对耗散、"
            "分区耦合滞后功和关节阻尼功，以区分物理耗散与分区算法误差。",
            "",
            "详细逐步数据见 `diagnostics.csv`，三维结果见 `vtk/`，ParaView 入口为 "
            "`coupled_scene.pvd`。",
        ]
    )
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
