# CMU 17 模块蛇形机器人：几何与 MPM 接触拓扑预览

本目录实现一个面向 MBD–MPM 耦合的几何预览模型。固定拓扑为：

- 17 个刚体：`Head + 15 Middle + Tail`；
- 16 个理想转动副：从头到尾按 `Yaw / Pitch` 交替；
- 总弧长 `L = 0.94 m`，总质量 `m = 3.15 kg`，接触外径 `D = 0.05 m`；
- 由于参考文献没有给出逐模块 CAD 和质量表，当前采用等节距、等质量的显式暂定值。

## 为什么输出两层表面

`*_modules.vtk` 是 17 个分离的可视化刚体，模块间留有小间隙，以便检查刚体编号和关节拓扑。它不参与 MPM 接触。

`*_contact_envelope.vtk` 是连续、封闭的虚拟外包络，用来避免砂粒从关节缝进入或在相互重叠的模块端盖上重复受力。当前网格的边界边数和非流形边数均为 0。

`*_contact_patches.vtk` 是与仓库现有 MPM 接口一致的矩形 patch 集。每个 cell 都含有：

- `patch_id`、`module_id`、`module_type`、`surface_id`；
- `normal`、`axis_long`、`axis_width`、`patch_center`；
- `half_length_m`、`half_width_m`、`patch_area_m2`、`patch_mass_kg`；
- `initial_near_ground`：patch 中心距初始砂面不超过 4 mm 时为 1。

连续包络是随整条中心线重建的虚拟关节蒙皮。正式耦合时应在每个 MBD 宏步根据 17 个刚体姿态重新生成 patch，并把每个 patch 的力和力矩回传给其 `module_id`，不能把整层包络当作第 18 个刚体。

## 生成

在仓库根目录运行：

```powershell
& 'C:\Users\90522\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m validation_cases.cmu_snake_geometry_3d.generate_preview `
  --out output\cmu_snake_geometry_preview `
  --frames 12
```

## ParaView 检查顺序

先同时打开以下五个文件并点击 **Apply**：

1. `sidewinding_phase_000_ground.vtk`
2. `sidewinding_phase_000_modules.vtk`
3. `sidewinding_phase_000_topology.vtk`
4. `sidewinding_phase_000_contact_envelope.vtk`
5. `sidewinding_phase_000_contact_patches.vtk`

推荐显示设置：

- `modules`：按 cell data `module_id` 着色；
- `topology`：应用 **Tube** 过滤器，半径取 `0.002 m`，按 `joint_type` 着色；
- `contact_envelope`：Opacity 设为 `0.15–0.25`，按 `module_id` 着色；
- `contact_patches`：Representation 选 **Surface With Edges**，按 `initial_near_ground` 着色；
- `ground`：灰色，Opacity 约 `0.35`。

打开 `snake_shape_cycle_*.pvd` 可播放 12 帧关节波形动画。这里的时间值是波相位，不是秒；该动画只是运动学形状循环，不是 MBD–MPM 求解轨迹。

## 当前接触分辨率建议

默认共有 864 个矩形 patch。最小完整 patch 尺寸约 `4.91 mm`，圆柱侧面 patch 的典型尺寸约为 `18.43 mm × 9.82 mm`。建议第一轮 MPM 网格取 `dx = 4 mm`，并用 `5 / 4 / 3 mm` 做网格收敛检查。

`initial_near_ground` 只用于初始几何检查。后续试验对比中的接触长度比应定义为：在一个步态周期内，满足接触压力或法向力阈值的中心线弧长之和除以 `L`，不能直接使用“接近地面”的几何标记。
