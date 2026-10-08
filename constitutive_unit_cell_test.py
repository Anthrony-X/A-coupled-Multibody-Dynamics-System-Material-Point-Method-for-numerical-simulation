import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import taichi as ti

from tank_mpm.constitutive_models import (
    CONSTITUTIVE_MODEL_DRUCKER_PRAGER,
    CONSTITUTIVE_MODEL_IDS,
    CONSTITUTIVE_MODEL_MODIFIED_CAM_CLAY,
    CONSTITUTIVE_MODEL_MOHR_COULOMB,
    CONSTITUTIVE_MODEL_PURE_WATER,
    CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP,
    drucker_prager_update,
    make_drucker_prager_parameters,
    make_modified_cam_clay_parameters,
    make_mohr_coulomb_parameters,
    make_pure_water_parameters,
    make_srsh_modified_dp_parameters,
    modified_cam_clay_update,
    mohr_coulomb_update,
    normalize_constitutive_model_name,
    pure_water_update,
    srsh_modified_dp_update,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "tank_mpm_launcher_config.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "constitutive_unit_cell_test"
DEFAULT_MODELS = (
    "drucker-prager",
    "modified-cam-clay",
    "srsh-modified-dp",
)

CSV_COLUMNS = (
    "model",
    "step",
    "time_s",
    "axial_strain",
    "lateral_strain",
    "volumetric_strain",
    "sigma_axial_pa",
    "sigma_lateral_pa",
    "lateral_stress_error_pa",
    "mean_stress_p_pa",
    "deviatoric_stress_q_pa",
    "mcc_preconsolidation_pressure_pa",
    "srsh_equivalent_plastic_strain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-integration-point axial stress-strain tests using the project's "
            "actual Taichi constitutive updates."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all'.",
    )
    parser.add_argument(
        "--mode",
        choices=("triaxial", "oedometer"),
        default="triaxial",
        help=(
            "triaxial keeps both lateral stresses at the confining pressure; "
            "oedometer imposes zero lateral strain."
        ),
    )
    parser.add_argument("--confining-pressure", type=float, default=6.0e3)
    parser.add_argument("--max-axial-strain", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--axial-strain-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--lateral-stress-tolerance",
        type=float,
        default=500.0,
        help="Absolute lateral-stress control tolerance in Pa.",
    )
    parser.add_argument("--max-lateral-iterations", type=int, default=32)
    parser.add_argument(
        "--arch",
        choices=("cpu", "cuda", "gpu", "vulkan"),
        default="cpu",
    )
    return parser.parse_args()


def load_fields(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Launcher configuration not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    fields = payload.get("fields", payload)
    if not isinstance(fields, dict):
        raise ValueError(f"Invalid launcher configuration: {config_path}")
    return {str(key): str(value) for key, value in fields.items()}


def field_float(fields: dict[str, str], key: str, fallback: float) -> float:
    value = fields.get(key, "").strip()
    return fallback if not value else float(value)


def field_optional_float(fields: dict[str, str], key: str) -> float | None:
    value = fields.get(key, "").strip()
    return None if not value else float(value)


def selected_models(value: str) -> list[str]:
    requested = DEFAULT_MODELS if value.strip().lower() == "all" else value.split(",")
    models: list[str] = []
    for item in requested:
        model = normalize_constitutive_model_name(str(item).strip())
        if model not in models:
            models.append(model)
    return models


def resolve_arch(name: str):
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    if name == "vulkan":
        return ti.vulkan
    return ti.gpu


@ti.data_oriented
class ConstitutiveMaterialPoint:
    def __init__(self, model: str, fields: dict[str, str], initial_pressure: float):
        self.model = normalize_constitutive_model_name(model)
        self.model_id = CONSTITUTIVE_MODEL_IDS[self.model]

        soil_E = field_float(fields, "soil_E", 11.5e6)
        soil_nu = field_float(fields, "soil_nu", 0.35)
        soil_phi_deg = field_float(fields, "soil_phi_deg", 23.0)
        soil_psi_deg = field_float(fields, "soil_psi_deg", 2.0)
        soil_cohesion = field_float(fields, "soil_cohesion", 2.2e4)

        dp = make_drucker_prager_parameters(
            soil_E,
            soil_nu,
            soil_phi_deg,
            soil_psi_deg,
            soil_cohesion,
        )
        mc = make_mohr_coulomb_parameters(
            soil_E,
            soil_nu,
            soil_phi_deg,
            soil_psi_deg,
            soil_cohesion,
        )
        mcc = make_modified_cam_clay_parameters(
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_phi_deg=soil_phi_deg,
            critical_state_m=field_optional_float(fields, "soil_mcc_m"),
            virgin_compression_lambda=field_float(fields, "soil_mcc_lambda", 0.12),
            recompression_kappa=field_float(fields, "soil_mcc_kappa", 0.02),
            initial_void_ratio=field_float(fields, "soil_mcc_e0", 0.8),
            initial_preconsolidation_pressure=field_float(fields, "soil_mcc_pc0", 1.0e5),
            min_pressure=field_float(fields, "soil_mcc_min_pressure", 1.0),
        )
        srsh = make_srsh_modified_dp_parameters(
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_phi_deg=soil_phi_deg,
            soil_psi_deg=soil_psi_deg,
            soil_cohesion=soil_cohesion,
            rate_exponent=field_float(fields, "srsh_rate_exponent", 0.185),
            rate_sensitivity=field_float(fields, "srsh_rate_sensitivity", 0.0),
            saturation_increment=field_float(fields, "srsh_saturation_increment", 0.0),
            saturation_plastic_strain=field_float(
                fields,
                "srsh_saturation_plastic_strain",
                0.6,
            ),
        )
        water = make_pure_water_parameters(
            bulk_modulus=field_float(fields, "water_bulk_modulus", 2.2e9),
            dynamic_viscosity=field_float(fields, "water_dynamic_viscosity", 1.002e-3),
            cavitation_pressure=field_float(fields, "water_cavitation_pressure", 0.0),
        )

        self.shear_modulus = dp.shear_modulus
        self.bulk_modulus = dp.bulk_modulus
        self.dp_alpha = dp.alpha
        self.dp_beta = dp.beta
        self.dp_k = dp.k

        self.mc_sin_phi = mc.sin_phi
        self.mc_cos_phi = mc.cos_phi
        self.mc_sin_psi = mc.sin_psi
        self.mc_cohesion = mc.cohesion

        self.mcc_M = mcc.critical_state_m
        self.mcc_lambda = mcc.virgin_compression_lambda
        self.mcc_kappa = mcc.recompression_kappa
        self.mcc_e0 = mcc.initial_void_ratio
        self.mcc_pc0 = mcc.initial_preconsolidation_pressure
        self.mcc_min_pressure = mcc.min_pressure

        self.srsh_friction_slope = srsh.friction_slope
        self.srsh_dilatancy_slope = srsh.dilatancy_slope
        self.srsh_cohesion_intercept = srsh.cohesion_intercept
        self.srsh_rate_exponent = srsh.rate_exponent
        self.srsh_rate_sensitivity = srsh.rate_sensitivity
        self.srsh_saturation_increment = srsh.saturation_increment
        self.srsh_rate0 = srsh.reference_strain_rate
        self.srsh_epnf = srsh.saturation_plastic_strain

        self.water_bulk_modulus = water.bulk_modulus
        self.water_dynamic_viscosity = water.dynamic_viscosity
        self.water_cavitation_pressure = water.cavitation_pressure

        self.stress_old = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
        self.stress_trial = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
        self.pc_old = ti.field(dtype=ti.f64, shape=())
        self.pc_trial = ti.field(dtype=ti.f64, shape=())
        self.srsh_epbar_old = ti.field(dtype=ti.f64, shape=())
        self.srsh_epbar_trial = ti.field(dtype=ti.f64, shape=())

        initial_stress = float(initial_pressure) * np.eye(3, dtype=np.float64)
        self.stress_old[None] = initial_stress
        self.stress_trial[None] = initial_stress
        self.pc_old[None] = self.mcc_pc0
        self.pc_trial[None] = self.mcc_pc0
        self.srsh_epbar_old[None] = 0.0
        self.srsh_epbar_trial[None] = 0.0

    @ti.kernel
    def evaluate_trial(
        self,
        lateral_extension_increment: ti.f64,
        axial_compression_increment: ti.f64,
        dt: ti.f64,
    ):
        identity = ti.Matrix.identity(ti.f64, 3)
        deps = ti.Matrix.zero(ti.f64, 3, 3)
        deps[0, 0] = lateral_extension_increment
        deps[1, 1] = lateral_extension_increment
        deps[2, 2] = -axial_compression_increment
        eps = 1.0e-12

        stress_new = self.stress_old[None]
        pc_new = self.pc_old[None]
        epbar_new = self.srsh_epbar_old[None]

        if ti.static(self.model_id == CONSTITUTIVE_MODEL_DRUCKER_PRAGER):
            stress_new = drucker_prager_update(
                self.stress_old[None],
                -deps,
                self.shear_modulus,
                self.bulk_modulus,
                self.dp_alpha,
                self.dp_beta,
                self.dp_k,
                -1.0,
                identity,
                eps,
            )
        elif ti.static(self.model_id == CONSTITUTIVE_MODEL_MOHR_COULOMB):
            tension_positive_stress = mohr_coulomb_update(
                -self.stress_old[None],
                deps,
                self.shear_modulus,
                self.bulk_modulus,
                self.mc_sin_phi,
                self.mc_cos_phi,
                self.mc_sin_psi,
                self.mc_cohesion,
                identity,
                eps,
            )
            stress_new = -tension_positive_stress
        elif ti.static(self.model_id == CONSTITUTIVE_MODEL_MODIFIED_CAM_CLAY):
            tension_positive_stress, pc_new = modified_cam_clay_update(
                -self.stress_old[None],
                deps,
                self.pc_old[None],
                self.shear_modulus,
                self.bulk_modulus,
                self.mcc_M,
                self.mcc_lambda,
                self.mcc_kappa,
                self.mcc_e0,
                self.mcc_min_pressure,
                identity,
                eps,
            )
            stress_new = -tension_positive_stress
        elif ti.static(self.model_id == CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP):
            stress_new, epbar_new, _plastic_increment = (
                srsh_modified_dp_update(
                    self.stress_old[None],
                    -deps,
                    self.srsh_epbar_old[None],
                    dt,
                    self.shear_modulus,
                    self.bulk_modulus,
                    self.srsh_friction_slope,
                    self.srsh_dilatancy_slope,
                    self.srsh_cohesion_intercept,
                    self.srsh_rate_exponent,
                    self.srsh_rate_sensitivity,
                    self.srsh_saturation_increment,
                    self.srsh_rate0,
                    self.srsh_epnf,
                    identity,
                    eps,
                )
            )
        elif ti.static(self.model_id == CONSTITUTIVE_MODEL_PURE_WATER):
            stress_new = pure_water_update(
                self.stress_old[None],
                deps,
                dt,
                self.water_bulk_modulus,
                self.water_dynamic_viscosity,
                self.water_cavitation_pressure,
                identity,
                eps,
            )

        self.stress_trial[None] = stress_new
        self.pc_trial[None] = pc_new
        self.srsh_epbar_trial[None] = epbar_new

    @ti.kernel
    def commit_trial(self):
        self.stress_old[None] = self.stress_trial[None]
        self.pc_old[None] = self.pc_trial[None]
        self.srsh_epbar_old[None] = self.srsh_epbar_trial[None]

    def lateral_stress(self) -> float:
        stress = np.asarray(self.stress_trial[None], dtype=np.float64)
        return 0.5 * float(stress[0, 0] + stress[1, 1])

    def current_record(
        self,
        step: int,
        time_s: float,
        axial_strain: float,
        lateral_strain: float,
        target_lateral_stress: float,
    ) -> dict[str, float | int | str]:
        stress = np.asarray(self.stress_old[None], dtype=np.float64)
        mean_stress = float(np.trace(stress)) / 3.0
        deviatoric = stress - mean_stress * np.eye(3)
        q = math.sqrt(max(0.0, 1.5 * float(np.sum(deviatoric * deviatoric))))
        return {
            "model": self.model,
            "step": step,
            "time_s": time_s,
            "axial_strain": axial_strain,
            "lateral_strain": lateral_strain,
            "volumetric_strain": 2.0 * lateral_strain - axial_strain,
            "sigma_axial_pa": float(stress[2, 2]),
            "sigma_lateral_pa": 0.5 * float(stress[0, 0] + stress[1, 1]),
            "lateral_stress_error_pa": (
                0.5 * float(stress[0, 0] + stress[1, 1])
                - target_lateral_stress
            ),
            "mean_stress_p_pa": mean_stress,
            "deviatoric_stress_q_pa": q,
            "mcc_preconsolidation_pressure_pa": float(self.pc_old[None]),
            "srsh_equivalent_plastic_strain": float(self.srsh_epbar_old[None]),
        }


def solve_lateral_increment(
    material: ConstitutiveMaterialPoint,
    axial_increment: float,
    dt: float,
    target_lateral_stress: float,
    tolerance: float,
    max_iterations: int,
    predicted_increment: float,
) -> tuple[float, float]:
    material.evaluate_trial(predicted_increment, axial_increment, dt)
    predicted_residual = material.lateral_stress() - target_lateral_stress
    if abs(predicted_residual) <= max(1.0e-6, tolerance * 1.0e-8):
        return predicted_increment, predicted_residual

    best_increment = predicted_increment
    best_residual = predicted_residual
    span = max(0.05 * axial_increment, 1.0e-10)
    lower = predicted_increment - span
    upper = predicted_increment + span
    residual_lower = predicted_residual
    residual_upper = predicted_residual
    for _ in range(20):
        lower = predicted_increment - span
        upper = predicted_increment + span
        material.evaluate_trial(lower, axial_increment, dt)
        residual_lower = material.lateral_stress() - target_lateral_stress
        if abs(residual_lower) < abs(best_residual):
            best_increment = lower
            best_residual = residual_lower
        material.evaluate_trial(upper, axial_increment, dt)
        residual_upper = material.lateral_stress() - target_lateral_stress
        if abs(residual_upper) < abs(best_residual):
            best_increment = upper
            best_residual = residual_upper
        if residual_lower * residual_upper <= 0.0:
            break
        span *= 2.0
    else:
        material.evaluate_trial(best_increment, axial_increment, dt)
        if abs(best_residual) <= tolerance:
            return best_increment, best_residual
        raise RuntimeError(
            f"{material.model}: failed to bracket the local lateral-strain root; "
            f"best residual={best_residual:.6e} Pa"
        )

    for _ in range(max_iterations):
        candidate = 0.5 * (lower + upper)
        material.evaluate_trial(candidate, axial_increment, dt)
        residual = material.lateral_stress() - target_lateral_stress
        if abs(residual) < abs(best_residual):
            best_increment = candidate
            best_residual = residual
        if abs(residual) <= max(1.0e-6, tolerance * 1.0e-8):
            break
        if residual * residual_lower > 0.0:
            lower = candidate
            residual_lower = residual
        else:
            upper = candidate
            residual_upper = residual

    material.evaluate_trial(best_increment, axial_increment, dt)
    return best_increment, best_residual


def run_model(
    model: str,
    fields: dict[str, str],
    mode: str,
    confining_pressure: float,
    max_axial_strain: float,
    steps: int,
    axial_strain_rate: float,
    lateral_tolerance: float,
    max_lateral_iterations: int,
) -> list[dict[str, float | int | str]]:
    material = ConstitutiveMaterialPoint(model, fields, confining_pressure)
    axial_increment = max_axial_strain / steps
    dt = axial_increment / axial_strain_rate
    axial_strain = 0.0
    lateral_strain = 0.0
    if model == "pure-water":
        predicted_lateral_increment = 0.5 * axial_increment
    else:
        predicted_lateral_increment = (
            field_float(fields, "soil_nu", 0.30) * axial_increment
        )
    records = [
        material.current_record(
            0,
            0.0,
            axial_strain,
            lateral_strain,
            confining_pressure,
        )
    ]

    for step in range(1, steps + 1):
        if mode == "triaxial":
            lateral_increment, _lateral_residual = solve_lateral_increment(
                material,
                axial_increment,
                dt,
                confining_pressure,
                lateral_tolerance,
                max_lateral_iterations,
                predicted_lateral_increment,
            )
            predicted_lateral_increment = lateral_increment
        else:
            lateral_increment = 0.0
            material.evaluate_trial(lateral_increment, axial_increment, dt)

        material.commit_trial()
        axial_strain += axial_increment
        lateral_strain += lateral_increment
        records.append(
            material.current_record(
                step,
                step * dt,
                axial_strain,
                lateral_strain,
                confining_pressure,
            )
        )
    return records


def write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def write_svg_plot(
    output_dir: Path,
    records_by_model: dict[str, list[dict[str, float | int | str]]],
    mode: str,
    confining_pressure: float,
) -> None:
    colors = {
        "drucker-prager": "#0072B2",
        "mohr-coulomb": "#009E73",
        "modified-cam-clay": "#D55E00",
        "srsh-modified-dp": "#CC79A7",
        "pure-water": "#56B4E9",
    }
    width = 1200
    height = 575
    panel_width = 510
    panel_height = 350
    panel_top = 95
    panel_lefts = (75, 665)
    x_max = max(
        float(row["axial_strain"]) * 100.0
        for records in records_by_model.values()
        for row in records
    )
    axial_y_max = max(
        float(row["sigma_axial_pa"]) / 1.0e3
        for records in records_by_model.values()
        for row in records
    )
    q_y_max = max(
        float(row["deviatoric_stress_q_pa"]) / 1.0e3
        for records in records_by_model.values()
        for row in records
    )
    x_max = max(x_max, 1.0e-12)
    axial_y_max = max(1.0, axial_y_max * 1.08)
    q_y_max = max(1.0, q_y_max * 1.08)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}.legend{font-size:12px}</style>',
        (
            '<text class="title" x="600" y="30" text-anchor="middle">'
            f'Constitutive unit-cell test: {escape(mode)}, confining pressure '
            f'{confining_pressure / 1.0e3:.3f} kPa</text>'
        ),
    ]

    panel_specs = (
        (panel_lefts[0], axial_y_max, "Axial stress-strain", "Axial compressive stress (kPa)", "sigma_axial_pa"),
        (panel_lefts[1], q_y_max, "Deviatoric response", "Deviatoric stress q (kPa)", "deviatoric_stress_q_pa"),
    )
    for left, y_max, title, y_label, value_key in panel_specs:
        bottom = panel_top + panel_height
        lines.append(
            f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" fill="#fafafa" stroke="#333"/>'
        )
        for tick in range(6):
            x_value = x_max * tick / 5.0
            x = left + panel_width * tick / 5.0
            y_value = y_max * tick / 5.0
            y = bottom - panel_height * tick / 5.0
            lines.append(
                f'<line x1="{x:.3f}" y1="{panel_top}" x2="{x:.3f}" y2="{bottom}" stroke="#e1e1e1"/>'
            )
            lines.append(
                f'<line x1="{left}" y1="{y:.3f}" x2="{left + panel_width}" y2="{y:.3f}" stroke="#e1e1e1"/>'
            )
            lines.append(
                f'<text class="tick" x="{x:.3f}" y="{bottom + 20}" text-anchor="middle">{x_value:.2f}</text>'
            )
            lines.append(
                f'<text class="tick" x="{left - 10}" y="{y + 4:.3f}" text-anchor="end">{y_value:.1f}</text>'
            )
        lines.append(
            f'<text class="title" x="{left + panel_width / 2:.3f}" y="{panel_top - 18}" text-anchor="middle">{escape(title)}</text>'
        )
        lines.append(
            f'<text class="label" x="{left + panel_width / 2:.3f}" y="{bottom + 48}" text-anchor="middle">Axial compressive strain (%)</text>'
        )
        lines.append(
            f'<text class="label" transform="translate({left - 54},{panel_top + panel_height / 2}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'
        )

        for model, records in records_by_model.items():
            points = []
            for row in records:
                x_value = float(row["axial_strain"]) * 100.0
                y_value = float(row[value_key]) / 1.0e3
                x = left + panel_width * x_value / x_max
                y = bottom - panel_height * y_value / y_max
                points.append(f"{x:.3f},{y:.3f}")
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors.get(model, "#333333")}" stroke-width="2.2"/>'
            )

    legend_y = 545
    legend_spacing = 205
    legend_start = (width - legend_spacing * len(records_by_model)) / 2.0
    for index, model in enumerate(records_by_model):
        x = legend_start + index * legend_spacing
        color = colors.get(model, "#333333")
        lines.append(
            f'<line x1="{x:.3f}" y1="{legend_y}" x2="{x + 28:.3f}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text class="legend" x="{x + 36:.3f}" y="{legend_y + 4}">{escape(model)}</text>'
        )
    lines.append("</svg>")
    (output_dir / "constitutive_stress_strain.svg").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_plots(
    output_dir: Path,
    records_by_model: dict[str, list[dict[str, float | int | str]]],
    mode: str,
    confining_pressure: float,
) -> None:
    write_svg_plot(output_dir, records_by_model, mode, confining_pressure)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "[Unit cell] Matplotlib is unavailable; wrote the dependency-free SVG plot.",
            flush=True,
        )
        return

    colors = {
        "drucker-prager": "#0072B2",
        "mohr-coulomb": "#009E73",
        "modified-cam-clay": "#D55E00",
        "srsh-modified-dp": "#CC79A7",
        "pure-water": "#56B4E9",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for model, records in records_by_model.items():
        axial_strain_percent = np.array(
            [float(row["axial_strain"]) for row in records]
        ) * 100.0
        axial_stress_kpa = np.array(
            [float(row["sigma_axial_pa"]) for row in records]
        ) / 1.0e3
        q_kpa = np.array(
            [float(row["deviatoric_stress_q_pa"]) for row in records]
        ) / 1.0e3
        axes[0].plot(
            axial_strain_percent,
            axial_stress_kpa,
            label=model,
            color=colors.get(model),
            linewidth=2.0,
        )
        axes[1].plot(
            axial_strain_percent,
            q_kpa,
            label=model,
            color=colors.get(model),
            linewidth=2.0,
        )

    axes[0].set_xlabel("Axial compressive strain (%)")
    axes[0].set_ylabel("Axial compressive stress (kPa)")
    axes[0].set_title("Axial stress-strain")
    axes[1].set_xlabel("Axial compressive strain (%)")
    axes[1].set_ylabel("Deviatoric stress q (kPa)")
    axes[1].set_title("Deviatoric response")
    for axis in axes:
        axis.grid(True, color="#d8d8d8", linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Constitutive unit-cell test: {mode}, confining pressure "
        f"{confining_pressure / 1.0e3:.3f} kPa"
    )
    fig.savefig(output_dir / "constitutive_stress_strain.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.confining_pressure < 0.0:
        raise ValueError("Confining pressure must be nonnegative")
    if args.max_axial_strain <= 0.0:
        raise ValueError("Maximum axial strain must be positive")
    if args.steps <= 0:
        raise ValueError("Number of steps must be positive")
    if args.axial_strain_rate <= 0.0:
        raise ValueError("Axial strain rate must be positive")
    if args.lateral_stress_tolerance <= 0.0:
        raise ValueError("Lateral stress tolerance must be positive")
    if args.max_lateral_iterations <= 0:
        raise ValueError("Maximum lateral iterations must be positive")

    fields = load_fields(args.config)
    models = selected_models(args.models)
    args.out.mkdir(parents=True, exist_ok=True)
    ti.init(
        arch=resolve_arch(args.arch),
        default_fp=ti.f64,
        offline_cache=False,
    )

    all_records: list[dict[str, float | int | str]] = []
    records_by_model: dict[str, list[dict[str, float | int | str]]] = {}
    model_diagnostics: dict[str, dict[str, float]] = {}
    for model in models:
        print(f"[Unit cell] running {model} ...", flush=True)
        records = run_model(
            model=model,
            fields=fields,
            mode=args.mode,
            confining_pressure=args.confining_pressure,
            max_axial_strain=args.max_axial_strain,
            steps=args.steps,
            axial_strain_rate=args.axial_strain_rate,
            lateral_tolerance=args.lateral_stress_tolerance,
            max_lateral_iterations=args.max_lateral_iterations,
        )
        records_by_model[model] = records
        all_records.extend(records)
        write_csv(args.out / f"{model}_stress_strain.csv", records)
        final = records[-1]
        max_lateral_error = max(
            abs(float(row["lateral_stress_error_pa"])) for row in records
        )
        axial_stresses = np.array(
            [float(row["sigma_axial_pa"]) for row in records],
            dtype=np.float64,
        )
        max_axial_stress_step = float(np.max(np.abs(np.diff(axial_stresses))))
        max_abs_axial_stress = float(np.max(np.abs(axial_stresses)))
        model_diagnostics[model] = {
            "max_lateral_stress_error_pa": max_lateral_error,
            "max_abs_axial_stress_pa": max_abs_axial_stress,
            "max_abs_axial_stress_step_pa": max_axial_stress_step,
        }
        print(
            f"[Unit cell] {model}: sigma_axial="
            f"{float(final['sigma_axial_pa']) / 1.0e3:.6f} kPa, "
            f"q={float(final['deviatoric_stress_q_pa']) / 1.0e3:.6f} kPa, "
            f"max lateral error={max_lateral_error:.6f} Pa",
            flush=True,
        )
        if args.mode == "triaxial" and max_lateral_error > args.lateral_stress_tolerance:
            print(
                f"[Unit cell] warning: {model} has a discontinuous stress-control "
                f"step exceeding the {args.lateral_stress_tolerance:.6f} Pa tolerance.",
                flush=True,
            )
        if (
            max_abs_axial_stress > 0.0
            and max_axial_stress_step > 0.10 * max_abs_axial_stress
        ):
            print(
                f"[Unit cell] warning: {model} has a large step-to-step axial "
                f"stress jump ({max_axial_stress_step / 1.0e3:.6f} kPa).",
                flush=True,
            )

    write_csv(args.out / "all_models_stress_strain.csv", all_records)
    metadata = {
        "config": str(args.config.resolve()),
        "models": models,
        "mode": args.mode,
        "confining_pressure_pa": args.confining_pressure,
        "max_axial_strain": args.max_axial_strain,
        "steps": args.steps,
        "axial_strain_rate_1_per_s": args.axial_strain_rate,
        "lateral_stress_tolerance_pa": args.lateral_stress_tolerance,
        "max_lateral_iterations": args.max_lateral_iterations,
        "dt_s": args.max_axial_strain / args.steps / args.axial_strain_rate,
        "stress_convention": "compression-positive",
        "strain_convention": "axial compression positive in output",
        "model_diagnostics": model_diagnostics,
        "launcher_fields": fields,
    }
    (args.out / "test_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_plots(args.out, records_by_model, args.mode, args.confining_pressure)
    print(f"[Unit cell] results: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
