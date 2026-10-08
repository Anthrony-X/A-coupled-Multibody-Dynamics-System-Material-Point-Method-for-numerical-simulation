# CMU 模块化蛇形机器人在砂面侧绕：MBD–MPM 完整算例

本算例把 17 模块蛇形机器人建模为 17 个独立刚体、16 个交替偏航/俯仰转动副，并与三维 Drucker–Prager MPM 砂床双向耦合。

## 1. 模型组成

### 1.1 多体动力学

- 刚体拓扑：`Head + 15 Middle + Tail`；
- 转动副序列：`Yaw, Pitch, ..., Yaw, Pitch`，共 16 个；
- 总弧长：`0.94 m`；
- 总质量：`3.15 kg`；
- 接触外径：`0.05 m`；
- 当前逐模块质量：`3.15/17 kg`；
- 当前惯量：按沿局部 x 轴的均匀实心圆柱计算；
- Chrono 刚体自身不启用砂面碰撞，所有地形力均由 MPM 返回。

参考文献没有给出逐模块 CAD、逐模块质量和惯量表，因此等节距、等质量和圆柱惯量均是可替换的显式假设。

### 1.2 关节驱动

关节不是直接指定刚体位姿，而是由有扭矩上限的 PD 电机驱动：

```text
tau_i = clip[Kp(q_target_i - q_i) + Kd(qdot_target_i - qdot_i), -3, +3] Nm
```

目标角只由连续骨干曲线拟合产生，不再把水平/竖直身体波直接当作交替电机关节波。连续目标中心线为：

```text
y(s,t) = a_h sin(2 pi s/lambda - 2 pi f t)
z(s,t) = a_v sin(2 pi s/lambda - 2 pi f t + phi)
```

在 17 个模块中心计算骨干切向，使用 Bishop/parallel-transport 构造无扭转目标标架；随后依次将相邻目标标架的相对转动投影到实际的局部 yaw/pitch 单轴上，得到 16 个 `q_target`。MBD 与 VTK 几何预览共用这一个拟合实现。

默认参数：

- `a_h = 0.035 m`；
- `a_v = 0.016 m`；
- `lambda = 0.5 L = 0.47 m`；
- `phi = pi/2`；
- `f = 0.5 Hz`；
- 静置阶段 `0.25 s`；
- 增幅阶段 `0.50 s`，使用三次 smoothstep，角度和角速度连续；
- 单关节扭矩上限 `3 Nm`。

### 1.3 MPM 砂床

- 本构：Drucker–Prager；
- 密度：`1600 kg/m³`；
- 杨氏模量：`1 MPa`；
- 泊松比：`0.25`；
- 内摩擦角：`32°`；
- 剪胀角：`0°`；
- 黏聚力：`0 Pa`；
- 蛇体—砂接触摩擦系数：`0.55`。

这些砂参数是基线值，正式对比前应由目标试验砂的密度、直剪/三轴试验和贯入试验重新标定。

### 1.4 沿全局 y 轴的粒子回收

本算例启用有限 MPM 移动窗口。当前步态的推进轴为全局 `y`，已有计算结果的净位移符号为 `-y`，因此配置采用 `moving_window_axis="y"`、`moving_window_direction=-1`。每前进一个背景网格间距，就在 GPU 上把尾端一层粒子槽位回收到前端，并用初始地应力状态下捕获的 8 层原状砂模板重置位置、速度、APIC 仿射速度、应力、硬化变量和接触历史。`x` 与 `z` 方向的土体边界不随蛇体移动。

为保证窗口平移后重叠区的背景网格节点仍位于相同世界坐标，`y` 向粒子层间距严格等于背景网格间距：`smoke / standard / fine` 分别为 `0.02 / 0.01 / 0.005 m`。回收只处理配置的 `-y` 单向推进，不会因蛇体周期性横摆而反向擦除已经扰动的砂床；如果后续通过相位或初始姿态把推进符号反转，只需把 `moving_window_direction` 改为 `+1`。

## 2. MBD–MPM 双向耦合

每个 MBD 宏步执行：

1. 保存 864/592/152 个接触 patch 的起始位置和方向；
2. 将上一宏步 MPM 返回的逐 patch 力和力矩按 `module_id` 汇总到 17 个刚体；
3. Chrono 前进一步，同时由 16 个扭矩电机驱动关节；
4. 重建连续关节外包络和末态 patch；
5. MPM 在 5 个子步内插值 patch 运动并计算砂体接触；
6. 返回逐 patch 力、力矩、接触功和作用反作用残差。

连续外包络是虚拟关节蒙皮，不是第 18 个刚体。每个 patch 仍归属于一个真实模块，力矩按 patch 中心到模块质心的力臂计算。

## 3. 三档分辨率

| Preset | 用途 | MBD dt | MPM dt | Patch 数 | 预计砂粒数 |
|---|---:|---:|---:|---:|---:|
| `smoke` | CPU 代码路径/接触预检 | `5e-4 s` | `1e-4 s` | 152 | 16,008 |
| `standard` | 参数调试和初步趋势 | `2e-4 s` | `4e-5 s` | 592 | 192,096 |
| `fine` | 最终曲线与网格收敛 | `1e-4 s` | `2e-5 s` | 864 | 1,280,640 |

`fine` 计算量很大，应先完成 `standard` 标定和短时稳定性检查。

粗网格 smoke 粒子的等效半径接近蛇体半径，不能使用正式分辨率的接触势垒刚度，否则会产生非物理冲击。因此三档势垒刚度缩放分别为 `0.01 / 0.50 / 1.00`；smoke 结果只用于接口验证，不能用于速度或接触比结论。

## 4. 运行命令

CPU 烟雾测试：

```powershell
& 'C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe' `
  run_cmu_snake_mbd_mpm.py `
  --preset smoke `
  --arch cpu `
  --out output\cmu_snake_mbd_mpm_3d\smoke
```

GPU 标准计算：

```powershell
& 'C:\Users\90522\miniconda3\envs\mpm_taichi\python.exe' `
  run_cmu_snake_mbd_mpm.py `
  --preset standard `
  --arch cuda `
  --out output\cmu_snake_mbd_mpm_3d\standard
```

当前工作站在 2026-07-27 的预检中无法加载 `nvcuda.dll`，因此 CUDA standard 尚未执行。修复 NVIDIA 驱动后可直接重试上述命令；短时调试也可显式使用 `--arch cpu --max-macro-steps N`，但完整 standard/fine 不建议在 CPU 上运行。

可用 `--max-macro-steps N` 做短时预检，用 `--no-vtk` 关闭场输出。

## 5. 输出文件

- `coupled_scene.pvd`：ParaView 联合场景；
- `soil_series.pvd`：砂粒；
- `snake_modules_series.pvd`：17 个刚体；
- `snake_topology_series.pvd`：中心线、关节轴、目标角、实际角和电机扭矩；
- `snake_contact_series.pvd`：接触 patch、压力和力；
- `snake_envelope_series.pvd`：连续接触外包络；
- `diagnostics.csv`：每个 MBD 宏步的速度、接触比、约束、电机和能量诊断；
- `cycle_statistics.csv`：完整步态周期的平均速度和平均 `l/L`；
- `summary.json`：稳态指标和极值；
- `validation.json`：结构、约束、作用反作用和控制检查；
- `resolved_config.json`：完整可复现实验配置。

## 6. 接触长度比定义

每个模块沿轴向分为若干小段。一个轴向小段上所有周向 patch 的法向力之和超过 `0.02 N` 时，该小段视为接触：

```text
l/L = 有效接触轴向段数 / 总轴向段数
```

端盖 patch 不计入弧长，避免重复计算。正式与试验图像对比时，应对力阈值做敏感性分析，并保持试验和数值的时间平均窗口一致。

## 7. 与试验数据的后续验证矩阵

每个工况至少输出：

1. 完整周期平均 `l/L`；
2. 质心周期净位移除以周期时长得到的推进速度；
3. 关节跟踪误差、峰值扭矩和饱和比例；
4. 法向接触力分布和砂面沉陷；
5. `standard/fine` 网格差异。

建议首先扫描步态频率和竖直波幅，再标定接触摩擦与砂体参数。不要用接触摩擦系数直接拟合所有速度误差，否则会掩盖竖直波幅、扭矩饱和和砂体强度参数的影响。
