# 三维双连杆摆锤–土体冲击验证算例

本目录实现 `docs/references/3D_MBD-MPM_Validation_Model_Build_Guide_CN.docx` 的案例 1。MBD 侧由两个三维刚体和两个理想转动副组成；第二刚体是铝制连杆与钢制圆柱锤头的组合体。两个转轴均沿全局 `y` 方向，机构运动位于 `x-z` 平面，但质量、转动惯量、关节反力、接触力和 VTK 几何均保持三维。

## 指南对齐后的几何

- 土体：`x ∈ [-0.80, 0.80] m`、`y ∈ [-0.30, 0.30] m`、`z ∈ [-0.40, 0] m`，自由面为 `z = 0`。
- 固定铰点：`O = (0, 0, 0.650) m`。
- 连杆 1：中心距 `0.450 m`，宽 `0.050 m`，厚 `0.040 m`，质量 `2.430 kg`。
- 连杆 2：铰点 A 至圆柱中心 `0.350 m`，杆宽 `0.045 m`，杆厚 `0.035 m`。
- 圆柱锤头：轴线沿刚体局部 `y` 轴，半径 `0.050 m`，宽 `0.100 m`，质量 `6.1653756 kg`。
- 第二刚体总质量 `7.6537506 kg`，质心距 A 为 `0.3159689 m`。
- 初始绝对角：`α1 = 150°`、`α2 = 170°`；角度正方向定义为从 `+x` 转向 `+z`。
- 初始位置：`A = (-0.389711, 0, 0.875000) m`，锤头中心 `C = (-0.734394, 0, 0.935777) m`，锤头最低点距土面约 `0.8858 m`。

圆柱外周面和轴向两端的圆形侧面均被离散为随刚体运动的接触片。外周面采用周向–轴向切平面网格；两个圆形侧面采用径向环带网格，各环周向片数随半径增加。`fine` 预设包括外周面 `64 × 20 = 1280` 片和两个侧面各 `320` 片，共 `1920` 片，对应约 `5 mm` 的表面尺度。接触搜索带有锤头包围盒预筛选和最低表面邻域激活，避免所有土粒遍历全部圆柱面片。

## CUDA 性能配置

- Taichi 默认浮点类型和 MPM 粒子、本构、网格、接触及能量账本均使用 `f32`。
- Project Chrono 的两个刚体仍使用其库内部固定的双精度计算；该部分在 CPU 上执行且规模很小。
- 圆柱接触面片的局部模板只创建一次，同一刚体状态下复用缓存，并通过单个旋转矩阵批量变换。
- 每个 MBD 宏步只上传一次接触面首末 keyframe；各 MPM 子步在 GPU 上插值。
- 五个 MPM 子步的刚体侧力、独立土体侧力、力矩、接触功、应力功和残差均在设备端累计，宏步结束后统一回读一次。
- `run_metadata.json` 中的 `mpm_precision` 应为 `f32`，可用于确认运行精度。

## 分辨率预设

| 预设 | 土粒数 | 背景网格尺度 | 圆柱接触片 | MPM 步长 | 用途 |
|---|---:|---:|---:|---:|---|
| `fine` | `160 × 60 × 40 = 384000` | 约 `20 mm` | 外周 `1280` + 侧面 `640` = `1920` | `2e-5 s` | 正式验证 |
| `standard` | `128 × 48 × 32 = 196608` | 约 `25 mm` | 外周 `320` + 侧面 `160` = `480` | `4e-5 s` | 参数调试/收敛对照 |
| `smoke` | `32 × 12 × 8 = 3072` | 约 `50 mm` | 外周 `80` + 侧面 `48` = `128` | `1e-4 s` | 接口和输出检查 |

正式与标准预设的总时长均为 `1.20 s`。无土自由摆预检中，圆柱最低点约在 `0.439 s` 首次到达 `z = 0`，所以正式时窗覆盖冲击及冲击后的机构–土体响应。`smoke` 只有 `0.020 s`，不会形成正式的冲击验证结论。

## 运行

正式算例：

```powershell
& 'C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe' `
  'D:\MPM履带\validation_cases\double_link_pendulum_soil_impact_3d\run_case.py' `
  --preset fine --arch cuda
```

快速检查并生成 VTK：

```powershell
& 'C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe' `
  'D:\MPM履带\validation_cases\double_link_pendulum_soil_impact_3d\run_case.py' `
  --preset smoke --max-macro-steps 3 --arch cuda
```

`cuda` 是默认后端，并且显式映射到 `ti.cuda`。程序关闭了 CUDA 到 CPU 的自动回退：如果 NVIDIA 驱动不可用会直接报错，避免把 CPU 运行误认为 CUDA 运行。可用的后端参数为：

- `--arch cuda`：NVIDIA CUDA，正式计算推荐；
- `--arch cpu`：显式 CPU 计算；
- `--arch vulkan`：显式 Vulkan 后端；
- `--arch gpu`：由 Taichi 自动选择可用 GPU 后端。

运行 CUDA 前应先确认 `nvidia-smi` 可正常显示显卡和驱动信息。使用 `--max-macro-steps` 截断的运行会标记为 `INCOMPLETE`。

## 本构模型选择

默认模型由 `case_config.json` 中的 `soil.constitutive_model` 指定，也可在命令行临时覆盖：

```powershell
--constitutive-model drucker-prager
--constitutive-model mohr-coulomb
--constitutive-model modified-cam-clay
--constitutive-model pure-water
```

`water` 是 `pure-water` 的别名。纯水模型采用配置中的
`water_density`、`water_bulk_modulus`、`water_dynamic_viscosity` 和
`water_cavitation_pressure`；默认分别为 `1000 kg/m³`、`2.2 GPa`、
`1.002e-3 Pa·s` 和 `0 Pa` 表压。选择纯水时程序自动采用静水
`K0 = 1`，并把 MPM 子步细分到声学 CFL 限制以内。

## 四项验证量

1. **关节约束误差**：机架铰点闭合误差、两连杆铰点闭合误差、两转轴夹角误差以及 Chrono 原生约束违反量。
2. **作用力–反作用力残差**：MPM 核独立累计土体侧 `F_soil` 和刚体侧 `F_body`，报告 `||F_body + F_soil|| / max(||F_body||, ||F_soil||)`。
3. **接触力与广义力一致性**：按 `Q_c = J_c^T W_c` 把所有圆柱面片的力–力矩映射到两个绝对转角，核对 `W_c · V_c` 与 `Q_c · qdot`。
4. **能量平衡**：分别记录刚体机械能、土体动能/势能/应力功、两侧接触功、接触相对耗散、分区耦合滞后功和两个转动副的黏性阻尼功。

两个理想转动副的反力和约束力矩也写入 `diagnostics.csv`。

## VTK/ParaView 输出

默认输出目录为 `outputs/<preset>/`：

- `diagnostics.csv`：每个 MBD 宏步的关节、接触、广义力、关节反力和能量账本；
- `validation_summary.json`、`validation_report.md`：门槛判定和中文报告；
- `vtk/soil_*.vtk`：MPM 物质点；
- `vtk/pendulum_*.vtk`：双连杆、圆柱锤头、销轴和圆柱接触分片；
- `soil_series.pvd`、`pendulum_series.pvd`：两个独立时间序列；
- `coupled_scene.pvd`：土体与摆锤组合后的 ParaView 时间序列入口。

在 ParaView 中直接打开 `coupled_scene.pvd`。土体文件包含 `velocity`、`displacement`、`stress` 和 `von_mises`；其中 `stress` 全程采用压应力为正、拉应力为负的 MBD–MPM 统一约定，因此自重场中的 `stress_ZZ` 为正值。摆锤文件包含 `body_id`、`contact_patch_id`、`contact_force_N` 和 `contact_pressure_Pa`，可用 `Color By` 显示接触力或接触压力云图。接触片的 `part_id=6/7/8` 分别表示圆柱外周面、局部 `-y` 侧面和局部 `+y` 侧面。

## 单元检查

```powershell
& 'C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe' -m unittest `
  validation_cases.double_link_pendulum_soil_impact_3d.tests.test_kinematics `
  validation_cases.double_link_pendulum_soil_impact_3d.tests.test_contact_ledger -v
```

检查覆盖初始指南坐标、关节闭合、圆柱接触面积、三维接触功率与二维广义功率解析等价性，以及 MPM 接触的独立作用–反作用账本。
