from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(r"C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe")
MAIN_SCRIPT = ROOT / "main_multibody_tank_mpm.py"
ICON_PATH = ROOT / "assets" / "launcher.ico"


try:
    from PySide6 import QtCore, QtGui, QtWidgets

    QT_BINDING = "PySide6"
    QT_IMPORT_ERROR = None
except ImportError as pyside_error:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets

        QT_BINDING = "PyQt6"
        QT_IMPORT_ERROR = None
    except ImportError as pyqt_error:
        QT_BINDING = None
        QT_IMPORT_ERROR = (pyside_error, pyqt_error)

        class _MissingQtWidgets:
            class QMainWindow:
                pass

        QtCore = None
        QtGui = None
        QtWidgets = _MissingQtWidgets()


FONT_FAMILY = "Microsoft YaHei UI"
MONO_FAMILY = "Cascadia Mono"

COLORS = {
    "bg": "#21192f",
    "sidebar": "#28233f",
    "sidebar_active": "#17162a",
    "panel": "#302a4b",
    "panel_alt": "#383257",
    "field": "#403960",
    "line": "#5a517d",
    "line_soft": "#433d61",
    "text": "#f7f3ff",
    "muted": "#bdb4da",
    "muted_deep": "#9188b1",
    "cyan": "#61d8ff",
    "blue": "#78a7ff",
    "purple": "#b486ff",
    "pink": "#d883ff",
    "success": "#6df0c2",
    "danger": "#ff6f91",
    "warning": "#ffd166",
    "terminal": "#171426",
}

CONSTITUTIVE_MODEL_OPTIONS = (
    "drucker-prager",
    "mohr-coulomb",
    "modified-cam-clay",
    "srsh-modified-dp",
    "pure-water",
)

CONSTITUTIVE_PARAMETER_DEFINITIONS = (
    ("弹性模量 E (Pa)", "soil_E"),
    ("泊松比 nu", "soil_nu"),
    ("内摩擦角 phi (deg)", "soil_phi_deg"),
    ("剪胀角 psi (deg)", "soil_psi_deg"),
    ("黏聚力 c (Pa)", "soil_cohesion"),
    ("SRSH beta", "srsh_rate_exponent"),
    ("SRSH eta", "srsh_rate_sensitivity"),
    ("SRSH delta", "srsh_saturation_increment"),
    ("SRSH ep95", "srsh_saturation_plastic_strain"),
    ("MCC M (可空)", "soil_mcc_m"),
    ("MCC lambda", "soil_mcc_lambda"),
    ("MCC kappa", "soil_mcc_kappa"),
    ("MCC e0", "soil_mcc_e0"),
    ("MCC pc0 (Pa)", "soil_mcc_pc0"),
    ("MCC p min (Pa)", "soil_mcc_min_pressure"),
    ("水体积模量 (Pa)", "water_bulk_modulus"),
    ("水动力黏度 (Pa s)", "water_dynamic_viscosity"),
    ("水空化表压 (Pa)", "water_cavitation_pressure"),
)

CONSTITUTIVE_MODEL_PARAMETER_KEYS = {
    "drucker-prager": (
        "soil_E",
        "soil_nu",
        "soil_phi_deg",
        "soil_psi_deg",
        "soil_cohesion",
    ),
    "mohr-coulomb": (
        "soil_E",
        "soil_nu",
        "soil_phi_deg",
        "soil_psi_deg",
        "soil_cohesion",
    ),
    "modified-cam-clay": (
        "soil_E",
        "soil_nu",
        "soil_phi_deg",
        "soil_mcc_m",
        "soil_mcc_lambda",
        "soil_mcc_kappa",
        "soil_mcc_e0",
        "soil_mcc_pc0",
        "soil_mcc_min_pressure",
    ),
    "srsh-modified-dp": (
        "soil_E",
        "soil_nu",
        "soil_phi_deg",
        "soil_psi_deg",
        "soil_cohesion",
        "srsh_rate_exponent",
        "srsh_rate_sensitivity",
        "srsh_saturation_increment",
        "srsh_saturation_plastic_strain",
    ),
    "pure-water": (
        "water_bulk_modulus",
        "water_dynamic_viscosity",
        "water_cavitation_pressure",
    ),
}

CONSTITUTIVE_PARAMETER_KEYS = frozenset(
    key
    for keys in CONSTITUTIVE_MODEL_PARAMETER_KEYS.values()
    for key in keys
)

NAV_ITEMS = (
    ("基本", "运行环境"),
    ("阶段", "流程控制"),
    ("土体/网格", "颗粒与材料"),
    ("接触/坦克", "车辆接触"),
    ("驱动/反馈", "动力反馈"),
)


def _missing_qt_message() -> str:
    return (
        "缺少 Qt 运行库。\n\n"
        "请先安装 PySide6，推荐在当前 Taichi 环境中执行：\n"
        f"  {DEFAULT_PYTHON} -m pip install PySide6\n\n"
        "也可以安装 PyQt6，启动器会自动兼容。"
    )


class TankMpmLauncher(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        if QT_BINDING is None:
            raise SystemExit(_missing_qt_message())

        super().__init__()
        self.setWindowTitle("Multibody Tank + MPM 启动器")
        if ICON_PATH.exists():
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
        self.resize(1360, 900)
        self.setMinimumSize(1120, 760)

        self.process: QtCore.QProcess | None = None
        self.fields: dict[str, QtWidgets.QWidget] = {}
        self.checks: dict[str, QtWidgets.QCheckBox] = {}
        self.nav_buttons: list[QtWidgets.QPushButton] = []
        self.metric_labels: dict[str, QtWidgets.QLabel] = {}
        self.constitutive_parameter_widgets: dict[str, QtWidgets.QWidget] = {}
        self.constitutive_parameter_layout: QtWidgets.QGridLayout | None = None
        self.process_output_buffer = ""
        self.progress_stage_order: list[str] = []
        self.progress_stage_offsets: dict[str, int] = {}
        self.progress_stage_starts: dict[str, int] = {}
        self.progress_stage_totals: dict[str, int] = {}
        self.progress_total_steps = 1

        self.defaults = self._build_defaults()
        self._apply_style()
        self._build_ui()
        self._select_page(0)
        self.refresh_command()

    def _build_defaults(self) -> dict[str, str]:
        return {
            "python": str(DEFAULT_PYTHON),
            "script": str(MAIN_SCRIPT),
            "stage": "all",
            "out": str(ROOT / "outputs" / "output_multibody_tank_mpm"),
            "model": str(ROOT / "ZTZ_96" / "multibody" / "ztz96_multibody_model.json"),
            "arch": "cuda",
            "cpu_threads": "0",
            "dt": "1.0e-5",
            "mpm_dt": "",
            "mbd_dt": "5.0e-4",
            "save_every": "2000",
            "seed": "7",
            "geostatic_steps": "5000",
            "settle_steps": "10000",
            "drive_steps": "15000",
            "geostatic_state": "",
            "settled_state": "",
            "settled_tank_state": "",
            "drive_state": "",
            "drive_tank_state": "",
            "grid": "152",
            "soil_nx": "144",
            "soil_ny": "60",
            "soil_nz": "32",
            "soil_jitter": "0.15",
            "soil_xmin": "-1.5",
            "soil_xmax": "10.5",
            "soil_ymin": "-2.5",
            "soil_ymax": "2.5",
            "soil_zmin": "-1.5",
            "soil_zmax": "0.5",
            "domain_padding": "0.3333333333333333",
            "domain_shape": "rect",
            "mpm_precision": "f32",
            "moving_window_template_layers": "8",
            "soil_density": "1700.0",
            "soil_E": "1.5e6",
            "soil_nu": "0.30",
            "soil_phi_deg": "18.0",
            "soil_psi_deg": "2.0",
            "soil_cohesion": "4.0e3",
            "soil_constitutive_model": "drucker-prager",
            "srsh_rate_exponent": "0.3",
            "srsh_rate_sensitivity": "0.1",
            "srsh_saturation_increment": "1.1",
            "srsh_saturation_plastic_strain": "0.5",
            "soil_mcc_m": "",
            "soil_mcc_lambda": "0.12",
            "soil_mcc_kappa": "0.02",
            "soil_mcc_e0": "0.8",
            "soil_mcc_pc0": "1.0e5",
            "soil_mcc_min_pressure": "1.0",
            "water_bulk_modulus": "2.2e9",
            "water_dynamic_viscosity": "1.002e-3",
            "water_cavitation_pressure": "0.0",
            "soil_gravity_scale": "1.0",
            "soil_damping": "1.0",
            "geostatic_soil_gravity_scale": "1.0",
            "geostatic_soil_damping": "1.0",
            "geostatic_k0": "",
            "contact_mu": "0.50",
            "patch_update_every": "1",
            "surface_estimator": "gpu",
            "track_update_mode": "gpu",
            "chrono_max_substep_dt": "5.0e-4",
            "chrono_contact_young_modulus": "2.0e7",
            "chrono_contact_kn": "3.0e6",
            "chrono_contact_gn": "4.0e4",
            "chrono_contact_kt": "1.0e6",
            "chrono_contact_gt": "1.0e4",
            "chrono_wheel_mass": "120.0",
            "chrono_vehicle_asset_dir": "",
            "chrono_chassis_mass": "0.0",
            "chrono_roller_mass": "0.0",
            "chrono_sprocket_mass": "0.0",
            "chrono_idler_mass": "0.0",
            "chrono_suspension_arm_mass": "75.0",
            "chrono_suspension_spring_constant": "8.0e4",
            "chrono_suspension_damping_coefficient": "2.0e3",
            "chrono_suspension_preload": "-1.0e4",
            "chrono_aux_damper_coefficient": "1.0e2",
            "chrono_tensioner_preload": "2.0e4",
            "chrono_tensioner_free_length": "0.75",
            "chrono_tensioner_stiffness": "1.0e6",
            "chrono_tensioner_damping": "1.4e4",
            "chrono_track_shoe_count": "0",
            "chrono_track_shoe_count_padding": "2",
            "chrono_drive_torque_sign": "-1.0",
            "chrono_max_penetration_recovery_speed": "0.25",
            "sprocket_tooth_count": "0",
            "contact_barrier_stiffness": "",
            "contact_barrier_radius": "",
            "contact_barrier_min_distance_ratio": "0.15",
            "contact_slip_smoothing_distance": "",
            "mass": "42000.0",
            "track_pitch": "",
            "track_width": "",
            "shoe_length": "",
            "track_shoe_mass": "",
            "track_shoe_pitch_inertia": "",
            "initial_forward": "4.0",
            "initial_lateral": "0.0",
            "initial_sinkage": "0.000",
            "suspension_stiffness": "800.0",
            "suspension_damping": "220.0",
            "suspension_max_angle": "0.35",
            "sprocket_inertia": "220.0",
            "sprocket_damping": "450.0",
            "max_sprocket_omega": "90.0",
            "drive_torque": "-12000.0",
            "left_drive_torque": "",
            "right_drive_torque": "",
            "ramp_time": "1.0",
            "yaw_damping": "0.0",
            "body_drag_coefficient_area": "0.0",
            "air_density": "1.225",
            "body_rolling_resistance": "0.0",
            "slip_regularization": "0.35",
            "vertical_damping": "0.0",
            "pitch_rate_damping": "0.0",
            "max_feedback_normal_factor": "4.0",
            "max_feedback_traction_factor": "2.0",
            "settle_initial_clearance": "0.025",
            "settle_surface_percentile": "95.0",
        }

    def _apply_style(self) -> None:
        QtWidgets.QApplication.setStyle("Fusion")
        self.setFont(QtGui.QFont(FONT_FAMILY, 10))
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {COLORS["bg"]};
                color: {COLORS["text"]};
            }}
            QWidget#root {{
                background: {COLORS["bg"]};
                color: {COLORS["text"]};
            }}
            QWidget#sidebar {{
                background: {COLORS["sidebar"]};
                border-top-left-radius: 28px;
                border-bottom-left-radius: 28px;
            }}
            QWidget#mainArea {{
                background: {COLORS["bg"]};
                border-top-right-radius: 28px;
                border-bottom-right-radius: 28px;
            }}
            QLabel {{
                color: {COLORS["text"]};
                background: transparent;
            }}
            QLabel#mutedLabel, QLabel#fieldLabel {{
                color: {COLORS["muted"]};
            }}
            QLabel#titleLabel {{
                color: {COLORS["text"]};
                font-size: 28px;
                font-weight: 800;
            }}
            QLabel#pageTitle {{
                color: {COLORS["text"]};
                font-size: 22px;
                font-weight: 800;
            }}
            QLabel#sectionTitle {{
                color: {COLORS["text"]};
                font-size: 16px;
                font-weight: 800;
            }}
            QLabel#brand {{
                color: {COLORS["text"]};
                font-size: 21px;
                font-weight: 800;
            }}
            QLabel#logoMark {{
                background: white;
                color: {COLORS["sidebar_active"]};
                border-radius: 18px;
                font-size: 22px;
                font-weight: 900;
            }}
            QLabel#statusPill {{
                color: {COLORS["text"]};
                background: rgba(26, 22, 43, 0.55);
                border-radius: 16px;
                padding: 8px 14px;
                font-weight: 700;
            }}
            QLabel#metricValue {{
                color: white;
                font-size: 19px;
                font-weight: 800;
            }}
            QLabel#metricName {{
                color: rgba(255, 255, 255, 0.78);
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0px;
            }}
            QFrame#heroCard {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #9a7cff,
                    stop: 0.52 #78a7ff,
                    stop: 1 #59d5f1
                );
                border-radius: 30px;
            }}
            QFrame#controlPanel, QFrame#card, QFrame#sideCard {{
                background: {COLORS["panel"]};
                border: 1px solid {COLORS["line_soft"]};
                border-radius: 24px;
            }}
            QFrame#sideCard {{
                background: {COLORS["panel_alt"]};
            }}
            QPushButton {{
                border: none;
                color: {COLORS["text"]};
                font-weight: 700;
                min-height: 40px;
            }}
            QPushButton#navButton {{
                background: transparent;
                color: {COLORS["muted"]};
                border-radius: 20px;
                padding: 13px 18px;
                text-align: left;
                font-size: 14px;
            }}
            QPushButton#navButton:hover {{
                background: rgba(255, 255, 255, 0.055);
                color: {COLORS["text"]};
            }}
            QPushButton#navButton:checked {{
                background: {COLORS["sidebar_active"]};
                color: white;
                border: 1px solid #6d63ff;
            }}
            QPushButton#primaryButton {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #7d8cff,
                    stop: 1 #5bd7ff
                );
                color: white;
                border-radius: 19px;
                padding: 8px 24px;
            }}
            QPushButton#primaryButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #918fff,
                    stop: 1 #70e0ff
                );
            }}
            QPushButton#primaryButton:disabled {{
                background: #4c5f89;
                color: rgba(255, 255, 255, 0.58);
            }}
            QPushButton#ghostButton, QPushButton#smallButton {{
                background: rgba(255, 255, 255, 0.07);
                color: {COLORS["text"]};
                border-radius: 18px;
                padding: 8px 18px;
            }}
            QPushButton#ghostButton:hover, QPushButton#smallButton:hover {{
                background: rgba(255, 255, 255, 0.12);
            }}
            QPushButton#dangerButton {{
                background: rgba(255, 111, 145, 0.14);
                color: {COLORS["danger"]};
                border-radius: 18px;
                padding: 8px 22px;
            }}
            QPushButton#dangerButton:hover {{
                background: rgba(255, 111, 145, 0.22);
            }}
            QPushButton#dangerButton:disabled {{
                color: {COLORS["muted_deep"]};
                background: rgba(255, 255, 255, 0.04);
            }}
            QLineEdit, QComboBox {{
                background: {COLORS["field"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["line"]};
                border-radius: 15px;
                padding: 10px 13px;
                selection-background-color: {COLORS["blue"]};
                min-height: 24px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {COLORS["cyan"]};
                background: #463f69;
            }}
            QComboBox::drop-down {{
                width: 28px;
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background: {COLORS["panel_alt"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["line"]};
                selection-background-color: {COLORS["sidebar_active"]};
                outline: 0;
            }}
            QCheckBox {{
                color: {COLORS["text"]};
                spacing: 10px;
                font-weight: 650;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 7px;
                border: 1px solid {COLORS["line"]};
                background: {COLORS["field"]};
            }}
            QCheckBox::indicator:checked {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {COLORS["blue"]},
                    stop: 1 {COLORS["cyan"]}
                );
                border: 1px solid {COLORS["cyan"]};
            }}
            QTextEdit {{
                background: {COLORS["terminal"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["line_soft"]};
                border-radius: 18px;
                padding: 10px;
                selection-background-color: #514b83;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QSplitter::handle {{
                background: transparent;
                width: 12px;
            }}
            QMenuBar {{
                background: {COLORS["bg"]};
                color: {COLORS["text"]};
            }}
            QMenuBar::item:selected {{
                background: {COLORS["panel_alt"]};
            }}
            QMenu {{
                background: {COLORS["panel"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["line"]};
            }}
            QMenu::item:selected {{
                background: {COLORS["sidebar_active"]};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 4px 1px 4px 1px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.20);
                border-radius: 5px;
                min-height: 42px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.33);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 10px;
                margin: 1px 4px 1px 4px;
            }}
            QScrollBar::handle:horizontal {{
                background: rgba(255, 255, 255, 0.20);
                border-radius: 5px;
                min-width: 42px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QProgressBar {{
                background: {COLORS["terminal"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["line_soft"]};
                border-radius: 15px;
                min-height: 30px;
                text-align: center;
                font-weight: 800;
            }}
            QProgressBar::chunk {{
                border-radius: 15px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #7d8cff,
                    stop: 0.55 #8f8bff,
                    stop: 1 #5bd7ff
                );
            }}
            """
        )

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        main = QtWidgets.QWidget()
        main.setObjectName("mainArea")
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(30, 26, 26, 26)
        main_layout.setSpacing(22)

        main_layout.addWidget(self._build_top_area())
        main_layout.addWidget(self._build_workspace(), 1)
        layout.addWidget(main, 1)

        self.setCentralWidget(root)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(270)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(30, 34, 24, 26)
        layout.setSpacing(16)

        brand_row = QtWidgets.QHBoxLayout()
        brand_row.setSpacing(13)
        logo = QtWidgets.QLabel("✓")
        logo.setObjectName("logoMark")
        logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(38, 38)
        brand = QtWidgets.QLabel("MPM Lab")
        brand.setObjectName("brand")
        brand_row.addWidget(logo)
        brand_row.addWidget(brand, 1)
        layout.addLayout(brand_row)
        layout.addSpacing(26)

        for index, (title, subtitle) in enumerate(NAV_ITEMS):
            button = QtWidgets.QPushButton(f"{title}\n{subtitle}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, i=index: self._select_page(i))
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)

        save_button = QtWidgets.QPushButton("保存配置")
        save_button.setObjectName("ghostButton")
        save_button.clicked.connect(self._save_config_dialog)
        layout.addWidget(save_button)

        load_button = QtWidgets.QPushButton("加载配置")
        load_button.setObjectName("ghostButton")
        load_button.clicked.connect(self._load_config_dialog)
        layout.addWidget(load_button)
        return sidebar

    def _build_top_area(self) -> QtWidgets.QWidget:
        top = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(top)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(22)

        hero = QtWidgets.QFrame()
        hero.setObjectName("heroCard")
        hero.setMinimumHeight(150)
        hero.setMinimumWidth(460)
        self._add_shadow(hero, blur=42, offset_y=18, alpha=130)

        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(30, 24, 30, 24)
        hero_layout.setSpacing(14)

        hero_top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Multibody Tank + MPM")
        title.setObjectName("titleLabel")
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("statusPill")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hero_top.addWidget(title, 1)
        hero_top.addWidget(self.status_label)
        hero_layout.addLayout(hero_top)

        subtitle = QtWidgets.QLabel("深色参数控制台 · 地应力、沉降与驱动阶段统一配置")
        subtitle.setObjectName("mutedLabel")
        hero_layout.addWidget(subtitle)

        metrics = QtWidgets.QHBoxLayout()
        metrics.setSpacing(28)
        for key, label in (("stage", "STAGE"), ("grid", "GRID"), ("particles", "PARTICLES"), ("arch", "ARCH")):
            item = QtWidgets.QVBoxLayout()
            item.setSpacing(2)
            value = QtWidgets.QLabel("-")
            value.setObjectName("metricValue")
            name = QtWidgets.QLabel(label)
            name.setObjectName("metricName")
            item.addWidget(value)
            item.addWidget(name)
            metrics.addLayout(item)
            self.metric_labels[key] = value
        hero_layout.addLayout(metrics)

        controls = QtWidgets.QFrame()
        controls.setObjectName("controlPanel")
        controls.setMinimumHeight(150)
        self._add_shadow(controls, blur=34, offset_y=15, alpha=115)
        control_layout = QtWidgets.QGridLayout(controls)
        control_layout.setContentsMargins(24, 20, 24, 20)
        control_layout.setHorizontalSpacing(14)
        control_layout.setVerticalSpacing(12)

        heading = QtWidgets.QLabel("运行控制")
        heading.setObjectName("sectionTitle")
        control_layout.addWidget(heading, 0, 0, 1, 4)

        control_layout.addWidget(self._field_label("阶段"), 1, 0)
        stage = self._combo_field("stage", ("geostatic", "settle", "drive", "all"))
        control_layout.addWidget(stage, 1, 1)
        resume = self._check_field("resume", "续算 --resume", checked=False)
        control_layout.addWidget(resume, 1, 2)

        refresh = QtWidgets.QPushButton("生成命令")
        refresh.setObjectName("ghostButton")
        refresh.clicked.connect(self.refresh_command)
        control_layout.addWidget(refresh, 2, 0)

        self.run_button = QtWidgets.QPushButton("开始计算")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.start_run)
        control_layout.addWidget(self.run_button, 2, 1)

        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_run)
        control_layout.addWidget(self.stop_button, 2, 2)
        control_layout.setColumnStretch(3, 1)

        layout.addWidget(hero, 3)
        layout.addWidget(controls, 2)
        return top

    def _build_workspace(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        page_header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        self.page_title = QtWidgets.QLabel("基本")
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QtWidgets.QLabel("运行环境")
        self.page_subtitle.setObjectName("mutedLabel")
        title_box.addWidget(self.page_title)
        title_box.addWidget(self.page_subtitle)
        page_header.addLayout(title_box)
        page_header.addStretch(1)
        left_layout.addLayout(page_header)

        self.pages = QtWidgets.QStackedWidget()
        self._build_pages()
        left_layout.addWidget(self.pages, 1)

        splitter.addWidget(left)
        splitter.addWidget(self._build_command_and_progress())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 5)
        return splitter

    def _build_pages(self) -> None:
        builders = (
            self._build_general_page,
            self._build_steps_page,
            self._build_soil_page,
            self._build_contact_tank_page,
            self._build_drive_page,
        )
        for builder in builders:
            scroll, layout = self._scroll_page()
            builder(layout)
            layout.addStretch(1)
            self.pages.addWidget(scroll)

    def _scroll_page(self) -> tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 12, 18)
        layout.setSpacing(16)
        scroll.setWidget(content)
        return scroll, layout

    def _build_general_page(self, layout: QtWidgets.QVBoxLayout) -> None:
        env = self._section("运行环境")
        env_layout = QtWidgets.QGridLayout()
        env_layout.setHorizontalSpacing(12)
        env_layout.setVerticalSpacing(12)
        self._add_path_row(env_layout, 0, "Python 解释器", "python", file=True)
        self._add_path_row(env_layout, 1, "主程序", "script", file=True)
        self._add_path_row(env_layout, 2, "输出目录", "out", directory=True)
        self._add_path_row(env_layout, 3, "模型 JSON", "model", file=True)
        env.layout().addLayout(env_layout)
        layout.addWidget(env)

        numeric = self._section("基础数值")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self._add_combo_row(grid, 0, "Taichi 后端", "arch", ("cpu", "cuda", "vulkan", "gpu"))
        self._add_entry_row(grid, 1, "CPU threads", "cpu_threads")
        self._add_entry_row(grid, 2, "时间步 dt", "dt")
        self._add_entry_row(grid, 3, "MPM dt 可空", "mpm_dt")
        self._add_entry_row(grid, 4, "MBD dt", "mbd_dt")
        self._add_entry_row(grid, 5, "保存间隔", "save_every")
        self._add_entry_row(grid, 6, "随机种子", "seed")
        numeric.layout().addWidget(self._check_field("taichi_offline_cache", "Taichi offline cache", checked=False))
        numeric.layout().addLayout(grid)
        layout.addWidget(numeric)

    def _build_steps_page(self, layout: QtWidgets.QVBoxLayout) -> None:
        steps = self._section("阶段步数")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self._add_entry_row(grid, 0, "地应力步数", "geostatic_steps")
        self._add_entry_row(grid, 1, "Settle 步数", "settle_steps")
        self._add_entry_row(grid, 2, "Drive 步数", "drive_steps")
        steps.layout().addLayout(grid)
        layout.addWidget(steps)

        states = self._section("状态文件路径")
        state_grid = QtWidgets.QGridLayout()
        state_grid.setHorizontalSpacing(12)
        state_grid.setVerticalSpacing(12)
        self._add_path_row(state_grid, 0, "地应力状态 可空", "geostatic_state", file=True)
        self._add_path_row(state_grid, 1, "Settle MPM 状态 可空", "settled_state", file=True)
        self._add_path_row(state_grid, 2, "Settle 坦克状态 可空", "settled_tank_state", file=True)
        self._add_path_row(state_grid, 3, "Drive MPM 状态 可空", "drive_state", file=True)
        self._add_path_row(state_grid, 4, "Drive 坦克状态 可空", "drive_tank_state", file=True)
        states.layout().addLayout(state_grid)
        layout.addWidget(states)

        options = self._section("阶段选项")
        option_layout = QtWidgets.QVBoxLayout()
        option_layout.setSpacing(12)
        option_layout.addWidget(self._check_field("geostatic_initial_stress", "启用地应力初始应力场", checked=True))
        option_layout.addWidget(self._check_field("settle_lock_planar", "Settle 阶段约束水平位移/偏航", checked=True))
        options.layout().addLayout(option_layout)
        layout.addWidget(options)

        surface = self._section("Settle 土面估计")
        surface_grid = QtWidgets.QGridLayout()
        surface_grid.setHorizontalSpacing(12)
        surface_grid.setVerticalSpacing(12)
        self._add_entry_row(surface_grid, 0, "土面分位数", "settle_surface_percentile")
        self._add_entry_row(surface_grid, 1, "初始离地间隙 m", "settle_initial_clearance")
        surface.layout().addLayout(surface_grid)
        layout.addWidget(surface)

    def _build_soil_page(self, layout: QtWidgets.QVBoxLayout) -> None:
        grid_rows = [
            ("背景网格", "grid"),
            ("粒子 nx", "soil_nx"),
            ("粒子 ny", "soil_ny"),
            ("粒子 nz", "soil_nz"),
            ("随机扰动", "soil_jitter"),
            ("x min", "soil_xmin"),
            ("x max", "soil_xmax"),
            ("y min", "soil_ymin"),
            ("y max", "soil_ymax"),
            ("z min", "soil_zmin"),
            ("z max", "soil_zmax"),
            ("域外 padding", "domain_padding"),
            ("domain shape", "domain_shape"),
            ("MPM precision", "mpm_precision"),
            ("moving window template layers", "moving_window_template_layers"),
            ("密度", "soil_density"),
        ]
        stage_rows = [
            ("土体重力系数", "soil_gravity_scale"),
            ("土体阻尼", "soil_damping"),
            ("地应力重力系数", "geostatic_soil_gravity_scale"),
            ("地应力阻尼", "geostatic_soil_damping"),
            ("K0", "geostatic_k0"),
        ]

        grid_section = self._section("颗粒与背景网格")
        grid_section.layout().addWidget(
            self._check_field("moving_window", "Drive 阶段启用 x 向 MPM 移动窗口", checked=True)
        )
        grid_section.layout().addLayout(self._two_column_grid(grid_rows))
        layout.addWidget(grid_section)

        model_section = self._section("本构模型")
        selector_layout = QtWidgets.QGridLayout()
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(12)
        selector_layout.addWidget(self._field_label("模型"), 0, 0)
        model_selector = self._combo_field(
            "soil_constitutive_model",
            CONSTITUTIVE_MODEL_OPTIONS,
        )
        selector_layout.addWidget(model_selector, 0, 1)
        selector_layout.setColumnStretch(1, 1)
        model_section.layout().addLayout(selector_layout)

        self.constitutive_parameter_layout = QtWidgets.QGridLayout()
        self.constitutive_parameter_layout.setHorizontalSpacing(18)
        self.constitutive_parameter_layout.setVerticalSpacing(12)
        for label, key in CONSTITUTIVE_PARAMETER_DEFINITIONS:
            self.constitutive_parameter_widgets[key] = self._constitutive_parameter_row(
                label,
                key,
            )
        model_section.layout().addLayout(self.constitutive_parameter_layout)
        model_selector.currentTextChanged.connect(self._update_constitutive_fields)
        self._update_constitutive_fields(model_selector.currentText())
        layout.addWidget(model_section)

        stage_section = self._section("土体阶段控制")
        stage_section.layout().addLayout(self._two_column_grid(stage_rows))
        layout.addWidget(stage_section)

    def _build_contact_tank_page(self, layout: QtWidgets.QVBoxLayout) -> None:
        rows = [
            ("接触摩擦系数", "contact_mu"),
            ("surface estimator", "surface_estimator"),
            ("track update mode", "track_update_mode"),
            ("Chrono substep dt", "chrono_max_substep_dt"),
            ("Chrono Young", "chrono_contact_young_modulus"),
            ("Chrono kn", "chrono_contact_kn"),
            ("Chrono gn", "chrono_contact_gn"),
            ("Chrono kt", "chrono_contact_kt"),
            ("Chrono gt", "chrono_contact_gt"),
            ("Chrono wheel mass", "chrono_wheel_mass"),
            ("Chrono asset dir", "chrono_vehicle_asset_dir"),
            ("Chrono chassis mass", "chrono_chassis_mass"),
            ("Chrono roller mass", "chrono_roller_mass"),
            ("Chrono sprocket mass", "chrono_sprocket_mass"),
            ("Chrono idler mass", "chrono_idler_mass"),
            ("Chrono arm mass", "chrono_suspension_arm_mass"),
            ("Chrono susp k", "chrono_suspension_spring_constant"),
            ("Chrono susp c", "chrono_suspension_damping_coefficient"),
            ("Chrono susp preload", "chrono_suspension_preload"),
            ("Chrono aux damper", "chrono_aux_damper_coefficient"),
            ("Chrono tension preload", "chrono_tensioner_preload"),
            ("Chrono tension free L", "chrono_tensioner_free_length"),
            ("Chrono tension k", "chrono_tensioner_stiffness"),
            ("Chrono tension c", "chrono_tensioner_damping"),
            ("Chrono shoe count", "chrono_track_shoe_count"),
            ("Chrono shoe padding", "chrono_track_shoe_count_padding"),
            ("Chrono torque sign", "chrono_drive_torque_sign"),
            ("Chrono max recovery v", "chrono_max_penetration_recovery_speed"),
            ("sprocket tooth count", "sprocket_tooth_count"),
            ("patch 更新子步", "patch_update_every"),
            ("barrier kappa 可空", "contact_barrier_stiffness"),
            ("barrier 半径 可空", "contact_barrier_radius"),
            ("barrier d_min/r", "contact_barrier_min_distance_ratio"),
            ("滑移平滑距离 可空", "contact_slip_smoothing_distance"),
            ("质量", "mass"),
            ("履带节距 留空自动", "track_pitch"),
            ("履带宽度 留空自动", "track_width"),
            ("履带板长 留空自动", "shoe_length"),
            ("履带板质量 kg 留空模型", "track_shoe_mass"),
            ("履带板俯仰惯量 留空模型", "track_shoe_pitch_inertia"),
            ("初始 x", "initial_forward"),
            ("初始 y", "initial_lateral"),
            ("初始沉陷", "initial_sinkage"),
            ("悬挂刚度", "suspension_stiffness"),
            ("悬挂阻尼", "suspension_damping"),
            ("悬挂最大角", "suspension_max_angle"),
            ("主动轮惯量", "sprocket_inertia"),
            ("主动轮阻尼", "sprocket_damping"),
            ("最大主动轮角速度", "max_sprocket_omega"),
        ]
        section = self._section("接触与坦克参数")
        section.layout().addLayout(self._two_column_grid(rows))
        section.layout().addSpacing(8)
        section.layout().addWidget(self._check_field("include_grousers", "计入外侧履齿", checked=False))
        layout.addWidget(section)

    def _build_drive_page(self, layout: QtWidgets.QVBoxLayout) -> None:
        rows = [
            ("默认驱动扭矩", "drive_torque"),
            ("左侧扭矩 可空", "left_drive_torque"),
            ("右侧扭矩 可空", "right_drive_torque"),
            ("扭矩 ramp 时间", "ramp_time"),
            ("偏航阻尼", "yaw_damping"),
            ("阻力 CdA", "body_drag_coefficient_area"),
            ("空气密度", "air_density"),
            ("滚阻系数", "body_rolling_resistance"),
            ("滑移正则速度", "slip_regularization"),
            ("垂向阻尼", "vertical_damping"),
            ("俯仰速率阻尼", "pitch_rate_damping"),
            ("法向反馈限幅倍数", "max_feedback_normal_factor"),
            ("牵引反馈限幅倍数", "max_feedback_traction_factor"),
        ]
        section = self._section("驱动与反馈")
        section.layout().addLayout(self._two_column_grid(rows))
        layout.addWidget(section)

    def _build_command_and_progress(self) -> QtWidgets.QWidget:
        side = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(side)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        command_card = self._side_card("命令预览")
        command_layout = command_card.layout()
        self.command_text = QtWidgets.QTextEdit()
        self.command_text.setReadOnly(True)
        self.command_text.setMinimumHeight(118)
        self.command_text.setFont(QtGui.QFont(MONO_FAMILY, 10))
        command_layout.addWidget(self.command_text)

        command_buttons = QtWidgets.QHBoxLayout()
        command_buttons.addStretch(1)
        copy_button = QtWidgets.QPushButton("复制命令")
        copy_button.setObjectName("ghostButton")
        copy_button.clicked.connect(self.copy_command)
        command_buttons.addWidget(copy_button)
        command_layout.addLayout(command_buttons)
        layout.addWidget(command_card)

        progress_card = self._side_card("运行进度")
        progress_layout = progress_card.layout()

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        progress_layout.addWidget(self.progress_bar)

        status_grid = QtWidgets.QGridLayout()
        status_grid.setHorizontalSpacing(16)
        status_grid.setVerticalSpacing(12)
        status_grid.addWidget(self._field_label("当前阶段"), 0, 0)
        self.progress_stage_label = QtWidgets.QLabel("未开始")
        self.progress_stage_label.setObjectName("metricValue")
        status_grid.addWidget(self.progress_stage_label, 0, 1)
        status_grid.addWidget(self._field_label("阶段步数"), 1, 0)
        self.progress_step_label = QtWidgets.QLabel("-")
        self.progress_step_label.setObjectName("metricValue")
        status_grid.addWidget(self.progress_step_label, 1, 1)
        status_grid.addWidget(self._field_label("总体进度"), 2, 0)
        self.progress_percent_label = QtWidgets.QLabel("0%")
        self.progress_percent_label.setObjectName("metricValue")
        status_grid.addWidget(self.progress_percent_label, 2, 1)
        status_grid.setColumnStretch(1, 1)
        progress_layout.addLayout(status_grid)

        self.progress_detail_label = QtWidgets.QLabel("等待启动")
        self.progress_detail_label.setObjectName("mutedLabel")
        self.progress_detail_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_detail_label)
        progress_layout.addStretch(1)
        layout.addWidget(progress_card, 1)
        return side

    def _section(self, title: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        self._add_shadow(frame)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(22, 20, 22, 22)
        layout.setSpacing(16)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return frame

    def _side_card(self, title: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("sideCard")
        self._add_shadow(frame, blur=30, offset_y=12, alpha=105)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return frame

    def _two_column_grid(self, rows: list[tuple[str, str]]) -> QtWidgets.QGridLayout:
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        split = (len(rows) + 1) // 2
        for index, (label, key) in enumerate(rows):
            row = index if index < split else index - split
            col = 0 if index < split else 2
            grid.addWidget(self._field_label(label), row, col)
            grid.addWidget(self._line_field(key), row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    def _constitutive_parameter_row(self, label: str, key: str) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)
        row_layout.addWidget(self._field_label(label))
        row_layout.addWidget(self._line_field(key), 1)
        return row_widget

    def _update_constitutive_fields(self, model: str | None = None) -> None:
        if self.constitutive_parameter_layout is None:
            return
        selected_model = model or self._field_value("soil_constitutive_model")
        active_keys = CONSTITUTIVE_MODEL_PARAMETER_KEYS.get(selected_model, ())

        for widget in self.constitutive_parameter_widgets.values():
            self.constitutive_parameter_layout.removeWidget(widget)
            widget.hide()

        for index, key in enumerate(active_keys):
            widget = self.constitutive_parameter_widgets[key]
            row = index // 2
            column = index % 2
            self.constitutive_parameter_layout.addWidget(widget, row, column)
            widget.show()
        self.constitutive_parameter_layout.setColumnStretch(0, 1)
        self.constitutive_parameter_layout.setColumnStretch(1, 1)

    def _add_entry_row(self, grid: QtWidgets.QGridLayout, row: int, label: str, key: str) -> None:
        grid.addWidget(self._field_label(label), row, 0)
        grid.addWidget(self._line_field(key), row, 1)
        grid.setColumnStretch(1, 1)

    def _add_combo_row(
        self,
        grid: QtWidgets.QGridLayout,
        row: int,
        label: str,
        key: str,
        values: tuple[str, ...],
    ) -> None:
        grid.addWidget(self._field_label(label), row, 0)
        grid.addWidget(self._combo_field(key, values), row, 1)
        grid.setColumnStretch(1, 1)

    def _add_path_row(
        self,
        grid: QtWidgets.QGridLayout,
        row: int,
        label: str,
        key: str,
        *,
        file: bool = False,
        directory: bool = False,
    ) -> None:
        grid.addWidget(self._field_label(label), row, 0)
        grid.addWidget(self._line_field(key), row, 1)
        browse = QtWidgets.QPushButton("浏览")
        browse.setObjectName("smallButton")
        browse.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        browse.clicked.connect(lambda _checked=False, k=key, f=file, d=directory: self._browse_path(k, file=f, directory=d))
        grid.addWidget(browse, row, 2)
        grid.setColumnStretch(1, 1)

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("fieldLabel")
        label.setMinimumWidth(118)
        return label

    def _line_field(self, key: str) -> QtWidgets.QLineEdit:
        field = QtWidgets.QLineEdit(self.defaults.get(key, ""))
        field.textChanged.connect(self.refresh_command)
        self.fields[key] = field
        return field

    def _combo_field(self, key: str, values: tuple[str, ...]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        value = self.defaults.get(key, values[0] if values else "")
        if value in values:
            combo.setCurrentText(value)
        combo.currentTextChanged.connect(self.refresh_command)
        self.fields[key] = combo
        return combo

    def _check_field(self, key: str, text: str, *, checked: bool) -> QtWidgets.QCheckBox:
        check = QtWidgets.QCheckBox(text)
        check.setChecked(checked)
        check.stateChanged.connect(self.refresh_command)
        self.checks[key] = check
        return check

    def _add_shadow(
        self,
        widget: QtWidgets.QWidget,
        *,
        blur: int = 32,
        offset_y: int = 14,
        alpha: int = 120,
    ) -> None:
        shadow = QtWidgets.QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, offset_y)
        shadow.setColor(QtGui.QColor(8, 5, 18, alpha))
        widget.setGraphicsEffect(shadow)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        save_action = file_menu.addAction("保存配置")
        save_action.triggered.connect(self._save_config_dialog)
        load_action = file_menu.addAction("加载配置")
        load_action.triggered.connect(self._load_config_dialog)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("退出")
        quit_action.triggered.connect(self.close)

    def _select_page(self, index: int) -> None:
        if hasattr(self, "pages"):
            self.pages.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        title, subtitle = NAV_ITEMS[index]
        if hasattr(self, "page_title"):
            self.page_title.setText(title)
            self.page_subtitle.setText(subtitle)

    def _field_value(self, key: str) -> str:
        widget = self.fields[key]
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        return ""

    def _set_field_value(self, key: str, value: str) -> None:
        widget = self.fields.get(key)
        if widget is None:
            return
        if isinstance(widget, QtWidgets.QComboBox):
            index = widget.findText(value)
            if index >= 0:
                widget.setCurrentIndex(index)
            else:
                widget.setEditText(value)
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(value)

    def _check_value(self, key: str) -> bool:
        return self.checks[key].isChecked()

    def _browse_path(self, key: str, *, file: bool = False, directory: bool = False) -> None:
        current = self._field_value(key)
        initial = str(Path(current).parent if current else ROOT)
        if directory:
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目录", initial)
        elif file:
            value, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件", initial)
        else:
            value = ""
        if value:
            self._set_field_value(key, value)

    def build_command(self) -> list[str]:
        command = [
            self._field_value("python"),
            self._field_value("script"),
            "--stage",
            self._field_value("stage"),
            "--out",
            self._field_value("out"),
            "--model",
            self._field_value("model"),
            "--arch",
            self._field_value("arch"),
        ]
        if self._check_value("resume"):
            command.append("--resume")

        option_map = {
            "dt": "--dt",
            "cpu_threads": "--cpu-threads",
            "mpm_dt": "--mpm-dt",
            "mbd_dt": "--mbd-dt",
            "save_every": "--save-every",
            "seed": "--seed",
            "geostatic_steps": "--geostatic-steps",
            "settle_steps": "--settle-steps",
            "drive_steps": "--drive-steps",
            "geostatic_state": "--geostatic-state",
            "settled_state": "--settled-state",
            "settled_tank_state": "--settled-tank-state",
            "drive_state": "--drive-state",
            "drive_tank_state": "--drive-tank-state",
            "grid": "--grid",
            "soil_nx": "--soil-nx",
            "soil_ny": "--soil-ny",
            "soil_nz": "--soil-nz",
            "soil_jitter": "--soil-jitter",
            "soil_xmin": "--soil-xmin",
            "soil_xmax": "--soil-xmax",
            "soil_ymin": "--soil-ymin",
            "soil_ymax": "--soil-ymax",
            "soil_zmin": "--soil-zmin",
            "soil_zmax": "--soil-zmax",
            "domain_padding": "--domain-padding",
            "domain_shape": "--domain-shape",
            "mpm_precision": "--mpm-precision",
            "moving_window_template_layers": "--moving-window-template-layers",
            "soil_density": "--soil-density",
            "soil_E": "--soil-E",
            "soil_nu": "--soil-nu",
            "soil_phi_deg": "--soil-phi-deg",
            "soil_psi_deg": "--soil-psi-deg",
            "soil_cohesion": "--soil-cohesion",
            "soil_constitutive_model": "--soil-constitutive-model",
            "srsh_rate_exponent": "--srsh-rate-exponent",
            "srsh_rate_sensitivity": "--srsh-rate-sensitivity",
            "srsh_saturation_increment": "--srsh-saturation-increment",
            "srsh_saturation_plastic_strain": "--srsh-saturation-plastic-strain",
            "soil_mcc_m": "--soil-mcc-m",
            "soil_mcc_lambda": "--soil-mcc-lambda",
            "soil_mcc_kappa": "--soil-mcc-kappa",
            "soil_mcc_e0": "--soil-mcc-e0",
            "soil_mcc_pc0": "--soil-mcc-pc0",
            "soil_mcc_min_pressure": "--soil-mcc-min-pressure",
            "water_bulk_modulus": "--water-bulk-modulus",
            "water_dynamic_viscosity": "--water-dynamic-viscosity",
            "water_cavitation_pressure": "--water-cavitation-pressure",
            "soil_gravity_scale": "--soil-gravity-scale",
            "soil_damping": "--soil-damping",
            "geostatic_soil_gravity_scale": "--geostatic-soil-gravity-scale",
            "geostatic_soil_damping": "--geostatic-soil-damping",
            "geostatic_k0": "--geostatic-k0",
            "contact_mu": "--contact-mu",
            "patch_update_every": "--patch-update-every",
            "surface_estimator": "--surface-estimator",
            "track_update_mode": "--track-update-mode",
            "chrono_max_substep_dt": "--chrono-max-substep-dt",
            "chrono_contact_young_modulus": "--chrono-contact-young-modulus",
            "chrono_contact_kn": "--chrono-contact-kn",
            "chrono_contact_gn": "--chrono-contact-gn",
            "chrono_contact_kt": "--chrono-contact-kt",
            "chrono_contact_gt": "--chrono-contact-gt",
            "chrono_wheel_mass": "--chrono-wheel-mass",
            "chrono_vehicle_asset_dir": "--chrono-vehicle-asset-dir",
            "chrono_chassis_mass": "--chrono-chassis-mass",
            "chrono_roller_mass": "--chrono-roller-mass",
            "chrono_sprocket_mass": "--chrono-sprocket-mass",
            "chrono_idler_mass": "--chrono-idler-mass",
            "chrono_suspension_arm_mass": "--chrono-suspension-arm-mass",
            "chrono_suspension_spring_constant": "--chrono-suspension-spring-constant",
            "chrono_suspension_damping_coefficient": "--chrono-suspension-damping-coefficient",
            "chrono_suspension_preload": "--chrono-suspension-preload",
            "chrono_aux_damper_coefficient": "--chrono-aux-damper-coefficient",
            "chrono_tensioner_preload": "--chrono-tensioner-preload",
            "chrono_tensioner_free_length": "--chrono-tensioner-free-length",
            "chrono_tensioner_stiffness": "--chrono-tensioner-stiffness",
            "chrono_tensioner_damping": "--chrono-tensioner-damping",
            "chrono_track_shoe_count": "--chrono-track-shoe-count",
            "chrono_track_shoe_count_padding": "--chrono-track-shoe-count-padding",
            "chrono_drive_torque_sign": "--chrono-drive-torque-sign",
            "chrono_max_penetration_recovery_speed": "--chrono-max-penetration-recovery-speed",
            "sprocket_tooth_count": "--sprocket-tooth-count",
            "contact_barrier_stiffness": "--contact-barrier-stiffness",
            "contact_barrier_radius": "--contact-barrier-radius",
            "contact_barrier_min_distance_ratio": "--contact-barrier-min-distance-ratio",
            "contact_slip_smoothing_distance": "--contact-slip-smoothing-distance",
            "mass": "--mass",
            "track_pitch": "--track-pitch",
            "track_width": "--track-width",
            "shoe_length": "--shoe-length",
            "track_shoe_mass": "--track-shoe-mass",
            "track_shoe_pitch_inertia": "--track-shoe-pitch-inertia",
            "initial_forward": "--initial-forward",
            "initial_lateral": "--initial-lateral",
            "initial_sinkage": "--initial-sinkage",
            "suspension_stiffness": "--suspension-stiffness",
            "suspension_damping": "--suspension-damping",
            "suspension_max_angle": "--suspension-max-angle",
            "sprocket_inertia": "--sprocket-inertia",
            "sprocket_damping": "--sprocket-damping",
            "max_sprocket_omega": "--max-sprocket-omega",
            "drive_torque": "--drive-torque",
            "left_drive_torque": "--left-drive-torque",
            "right_drive_torque": "--right-drive-torque",
            "ramp_time": "--ramp-time",
            "yaw_damping": "--yaw-damping",
            "body_drag_coefficient_area": "--body-drag-coefficient-area",
            "air_density": "--air-density",
            "body_rolling_resistance": "--body-rolling-resistance",
            "slip_regularization": "--slip-regularization",
            "vertical_damping": "--vertical-damping",
            "pitch_rate_damping": "--pitch-rate-damping",
            "max_feedback_normal_factor": "--max-feedback-normal-factor",
            "max_feedback_traction_factor": "--max-feedback-traction-factor",
            "settle_initial_clearance": "--settle-initial-clearance",
            "settle_surface_percentile": "--settle-surface-percentile",
        }
        selected_constitutive_model = self._field_value("soil_constitutive_model")
        active_constitutive_keys = set(
            CONSTITUTIVE_MODEL_PARAMETER_KEYS.get(selected_constitutive_model, ())
        )
        for key, flag in option_map.items():
            if (
                key in CONSTITUTIVE_PARAMETER_KEYS
                and key not in active_constitutive_keys
            ):
                continue
            value = self._field_value(key).strip()
            if value:
                if value.startswith("-"):
                    command.append(f"{flag}={value}")
                else:
                    command.extend([flag, value])

        command.append(
            "--geostatic-initial-stress"
            if self._check_value("geostatic_initial_stress")
            else "--no-geostatic-initial-stress"
        )
        command.append("--moving-window" if self._check_value("moving_window") else "--no-moving-window")
        command.append("--include-grousers" if self._check_value("include_grousers") else "--no-include-grousers")
        command.append(
            "--settle-lock-planar"
            if self._check_value("settle_lock_planar")
            else "--no-settle-lock-planar"
        )
        command.append(
            "--taichi-offline-cache"
            if self._check_value("taichi_offline_cache")
            else "--no-taichi-offline-cache"
        )
        return command

    def command_string(self) -> str:
        return subprocess.list2cmdline(self.build_command())

    def refresh_command(self) -> None:
        if hasattr(self, "command_text"):
            self.command_text.setPlainText(self.command_string())
        self._update_metrics()

    def _update_metrics(self) -> None:
        if not self.metric_labels:
            return
        stage = self._field_value("stage") if "stage" in self.fields else "-"
        grid = self._field_value("grid") if "grid" in self.fields else "-"
        arch = self._field_value("arch") if "arch" in self.fields else "-"
        soil_nx = self._field_value("soil_nx") if "soil_nx" in self.fields else "-"
        soil_ny = self._field_value("soil_ny") if "soil_ny" in self.fields else "-"
        soil_nz = self._field_value("soil_nz") if "soil_nz" in self.fields else "-"
        self.metric_labels["stage"].setText(stage.upper())
        self.metric_labels["grid"].setText(grid)
        self.metric_labels["particles"].setText(f"{soil_nx}×{soil_ny}×{soil_nz}")
        self.metric_labels["arch"].setText(arch.upper())

    def copy_command(self) -> None:
        self.refresh_command()
        QtWidgets.QApplication.clipboard().setText(self.command_string())
        self._set_progress_message("命令已复制到剪贴板。")

    def _safe_int_field(self, key: str) -> int:
        try:
            return max(0, int(float(self._field_value(key).strip())))
        except Exception:
            return 0

    def _selected_stage_order(self) -> list[str]:
        stage = self._field_value("stage")
        if stage == "all":
            return ["geostatic", "settle", "drive"]
        return [stage]

    def _configured_stage_steps(self, stage: str) -> int:
        key = {
            "geostatic": "geostatic_steps",
            "settle": "settle_steps",
            "drive": "drive_steps",
        }.get(stage)
        return self._safe_int_field(key) if key else 0

    def _stage_span(self, stage: str) -> int:
        return max(1, int(self.progress_stage_totals.get(stage, self._configured_stage_steps(stage))))

    def _prepare_progress_run(self) -> None:
        self.process_output_buffer = ""
        self.progress_stage_order = self._selected_stage_order()
        self.progress_stage_totals = {
            stage: self._configured_stage_steps(stage)
            for stage in self.progress_stage_order
        }
        self.progress_stage_starts = {stage: 0 for stage in self.progress_stage_order}
        self.progress_stage_offsets = {}
        offset = 0
        for stage in self.progress_stage_order:
            self.progress_stage_offsets[stage] = offset
            offset += self._stage_span(stage)
        self.progress_total_steps = max(1, offset)
        self._set_progress(0, stage="准备启动", step_text="-", detail="正在启动计算进程...")

    def _set_progress(
        self,
        value: int,
        *,
        stage: str | None = None,
        step_text: str | None = None,
        detail: str | None = None,
    ) -> None:
        if not hasattr(self, "progress_bar"):
            return
        value = max(0, min(1000, int(value)))
        percent = value / 10.0
        percent_text = f"{percent:.1f}%" if percent < 99.95 and percent % 1 else f"{percent:.0f}%"
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(percent_text)
        self.progress_percent_label.setText(percent_text)
        if stage is not None:
            self.progress_stage_label.setText(stage)
        if step_text is not None:
            self.progress_step_label.setText(step_text)
        if detail is not None:
            self.progress_detail_label.setText(detail)

    def _set_progress_message(self, detail: str) -> None:
        if hasattr(self, "progress_detail_label"):
            self.progress_detail_label.setText(detail)

    def _update_progress_from_step(self, stage: str, step: int) -> None:
        if stage not in self.progress_stage_order:
            return
        start_step = int(self.progress_stage_starts.get(stage, 0))
        total = int(self.progress_stage_totals.get(stage, self._configured_stage_steps(stage)))
        span = max(1, total)
        if total <= 0 and step >= start_step:
            local = span
        else:
            local = max(0, min(span, int(step) - start_step))
        overall = self.progress_stage_offsets.get(stage, 0) + local
        value = round(overall / self.progress_total_steps * 1000)
        self._set_progress(
            value,
            stage=stage,
            step_text=f"{local}/{total}",
            detail=f"{stage} 阶段进度：step {step}",
        )

    def _handle_process_output(self, text: str) -> None:
        self.process_output_buffer += text
        while "\n" in self.process_output_buffer:
            line, self.process_output_buffer = self.process_output_buffer.split("\n", 1)
            self._handle_process_line(line.strip())

    def _handle_process_line(self, line: str) -> None:
        if not line:
            return

        stage_line = re.search(r"Stage\s+(\d+).*start_step=(\d+),\s*steps=(\d+)", line)
        if stage_line:
            stage_index = stage_line.group(1)
            stage = {"1": "geostatic", "2": "settle", "3": "drive"}.get(stage_index)
            if stage in self.progress_stage_order:
                self.progress_stage_starts[stage] = int(stage_line.group(2))
                self.progress_stage_totals[stage] = int(stage_line.group(3))
                self._update_progress_from_step(stage, self.progress_stage_starts[stage])
            return

        output_line = re.search(r"\[Output\]\s+(geostatic|settle|drive)\s+step=(\d+)", line)
        if output_line:
            self._update_progress_from_step(output_line.group(1), int(output_line.group(2)))
            return

        lowered = line.lower()
        if "[done]" in lowered:
            self._set_progress(1000, stage="完成", step_text="-", detail=line)
        elif "[warn]" in lowered:
            self._set_progress_message(line)
        elif "[error]" in lowered or "traceback" in lowered or "exception" in lowered:
            self._set_progress_message(line)

    def start_run(self) -> None:
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            QtWidgets.QMessageBox.information(self, "正在运行", "当前已经有计算任务在运行。")
            return

        self.refresh_command()
        command = self.build_command()
        if not Path(command[0]).exists():
            QtWidgets.QMessageBox.critical(self, "Python 不存在", f"找不到 Python:\n{command[0]}")
            return
        if not Path(command[1]).exists():
            QtWidgets.QMessageBox.critical(self, "主程序不存在", f"找不到主程序:\n{command[1]}")
            return

        self._prepare_progress_run()

        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.start()

        if not self.process.waitForStarted(2500):
            error = self.process.errorString()
            self._set_progress(0, stage="启动失败", step_text="-", detail=error)
            QtWidgets.QMessageBox.critical(self, "启动失败", error)
            self.process = None
            self._set_running(False)
            return

        self._set_running(True)

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._handle_process_output(data)

    def _process_finished(self, code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        if self.process_output_buffer.strip():
            self._handle_process_line(self.process_output_buffer.strip())
            self.process_output_buffer = ""
        if code == 0:
            self._set_progress(1000, stage="完成", step_text="-", detail="计算完成。")
        else:
            self._set_progress_message(f"进程结束，退出码 {code}")
        self.process = None
        self._set_running(False)

    def _process_error(self, _error: QtCore.QProcess.ProcessError) -> None:
        if self.process is not None:
            self._set_progress_message(self.process.errorString())

    def stop_run(self) -> None:
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self.status_label.setText("正在停止")
            self._set_progress_message("正在停止进程...")
            self.process.terminate()

    def _set_running(self, running: bool) -> None:
        self.status_label.setText("运行中" if running else "就绪")
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def log(self, text: str) -> None:
        self._handle_process_output(text)

    def save_config(self, path: Path) -> None:
        payload = {
            "fields": {key: self._field_value(key) for key in self.fields},
            "checks": {key: check.isChecked() for key, check in self.checks.items()},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_config(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.get("fields", {}).items():
            self._set_field_value(key, str(value))
        for key, value in payload.get("checks", {}).items():
            if key in self.checks:
                self.checks[key].setChecked(bool(value))
        self.refresh_command()

    def _save_config_dialog(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存配置",
            str(ROOT / "tank_mpm_launcher_config.json"),
            "JSON (*.json);;All files (*.*)",
        )
        if path:
            self.save_config(Path(path))
            self._set_progress_message(f"配置已保存: {path}")

    def _load_config_dialog(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载配置",
            str(ROOT),
            "JSON (*.json);;All files (*.*)",
        )
        if path:
            self.load_config(Path(path))
            self._set_progress_message(f"配置已加载: {path}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            answer = QtWidgets.QMessageBox.question(
                self,
                "计算仍在运行",
                "计算任务仍在运行。是否停止任务并退出？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.terminate()
        event.accept()


def main() -> None:
    if QT_BINDING is None:
        raise SystemExit(_missing_qt_message())

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Multibody Tank + MPM Launcher")
    app.setFont(QtGui.QFont(FONT_FAMILY, 10))
    window = TankMpmLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
