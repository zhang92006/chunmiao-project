# SHRP2 行驶轨迹复现指标

## 1. 计算范围

本次计算针对 SHRP2 `Crash` 类别中的训练事件 `131785457`，目标车辆 `22038`，比较窗口为碰撞前 `0–4 s`，共 41 个 `0.1 s` 采样点。

这不是完整数据集的复现率统计，而是对当前已经完成桥接和故障语义验证的一个高质量追尾事件做可审计基准。指标分成两段：

1. **SHRP2 原始轨迹 → 运动学种子**：衡量用恒定航向、恒定速度和碰撞时刻条件替代原始轨迹造成的信息损失。
2. **运动学桥接参考 → SUMO 闭环**：衡量 SUMO 路网、IDM 控制器和碰撞故障语义相对桥接参考的偏离，不等于原始 SHRP2 回放误差。

原始轨迹以碰撞时刻附近的 CAV 状态平移并按其航向旋转；目标车中心按 SHRP2 目标车前保险杠和车长换算。SUMO 轨迹取 v3 故障接口验证 episode，目标车从 `av_obs` 中按车辆 ID 对齐。

## 2. 数值结果

结果文件：[`shrp2_trajectory_metrics_event_131785457.json`](../results/shrp2_trajectory_metrics_event_131785457.json)

### 2.1 原始 SHRP2 与运动学种子

| 车辆 | ADE (m) | FDE (m) | 位置 RMSE (m) | 速度 MAE (m/s) | 速度 RMSE (m/s) | 航向 MAE (rad) | 最大位置误差 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAV | 1.147 | 0.010 | 1.599 | 0.794 | 0.924 | 0.009 | 3.175 |
| BV_primary | 2.792 | 0.740 | 3.100 | 1.310 | 1.773 | 1.777 | 6.137 |

解释：CAV 的碰撞末状态被条件化得较好，因此 FDE 很小；但全窗口 ADE 和速度误差仍较大，说明恒速种子不能复现真实制动/加速过程。目标车误差更大，尤其航向误差约 `1.78 rad`，说明恒航向种子没有保留源轨迹中的大幅航向变化；第 6 节进一步检查了这些变化本身的可用性。

### 2.2 SUMO 闭环与桥接运动学参考

| 车辆 | ADE (m) | FDE (m) | 位置 RMSE (m) | 速度 MAE (m/s) | 速度 RMSE (m/s) | 航向 MAE (rad) | 最大位置误差 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAV | 0.695 | 1.129 | 0.728 | 0.182 | 0.182 | 0.011 | 1.129 |
| BV_primary | 0.871 | 1.898 | 1.054 | 0.485 | 0.497 | 0.000 | 1.898 |

该 episode 确认发生碰撞，碰撞 ID 为 `BV_primary` 和 `CAV`。碰撞前最小 TTC 约 `0.0055 s`，最小中心距离约 `0.0182 m`。这里的 TTC/距离是 SUMO 故障语义运行结果，不应解释为 SHRP2 原始碰撞时间误差。

## 3. 结论

- 当前管线已经能够对一个事件给出位置、速度、航向、TTC 和距离等可复核指标，但还不能声称“完整 SHRP2 行驶数据已复现”。
- 主要误差来源在 **SHRP2 → 种子**：当前只使用碰撞附近状态和恒速/恒航向外推，没有回放真实加速度、制动、横向运动和车辆控制。
- **种子 → SUMO** 的闭环误差相对较小，但这只说明当前 2Lane 桥接模板与 SUMO 控制器的一致性尚可，不代表 SUMO 已复现真实道路轨迹。
- 下一步应使用质量门控后的分段速度/加速度和分段航向作为软约束；不应在检查目标车重构一致性之前直接逐点回放。

## 4. 复现实验命令

基础 Python 环境需要安装 `pandas`、`PyTables`、`numpy`。在仓库根目录执行：

```powershell
python -m scenario_reconstruction.shrp2_trajectory_metrics `
  --source_root 'path/to/SHRP2_Public' `
  --seed 'data_analysis/raw_data/shrp2_collision_pilot_v7/scenarios/shrp2_131785457_rear_end_source_speed.json' `
  --sumo_episode 'data_analysis/raw_data/shrp2_rear_end_fault_v3_interface_smoke/episode/crash/0.json' `
  --sumo_template 'data_analysis/raw_data/shrp2_rear_end_fault_v3_interface_smoke/template.json' `
  --output 'results/shrp2_trajectory_metrics_event_131785457.json'
```

## 5. 下一步指标扩展

1. 用 `116591908` 的质量合格参考轨迹拟合 SUMO 的分段加速度、反应时间和控制延迟参数。
2. 用 `2934487` 做独立备选复核，确认参数不是单一事件过拟合。
3. 按 `train/validation/test` 分开报告 ADE/FDE、碰撞成功率和初始碰撞率。
4. 加入真实轨迹的加速度、横向偏移、车道保持和冲突前 TTC 曲线误差。
5. 把“原始轨迹回放误差”和“SUMO 控制器闭环误差”作为两个独立实验，不合并为一个分数。

## 6. 第一阶段：软约束参考轨迹

已实现 `scenario_reconstruction.shrp2_reference_trajectory`，从原始 HDF5 导出碰撞前 `4 s`、10 Hz 的 CAV 与目标车轨迹，包括位置、报告速度、路径速度、报告航向、路径切向航向和速度差分加速度。完整逐点数据保存在被 Git 排除的 `data_analysis/raw_data/`；Git 只保存配置、代码和精简质量摘要。

SHRP2 数据字典明确说明：`psi_ego/psi_sur` 都是相对重构坐标系 x 轴的航向，`x_sur/y_sur` 是目标车**前保险杠**坐标。因此导出器同时保留前保险杠原始轨迹和按报告航向换算的车心轨迹，避免混淆两种参考点。

质量摘要：[`shrp2_reference_trajectory_event_131785457_summary.json`](../results/shrp2_reference_trajectory_event_131785457_summary.json)

| 检查项 | CAV | BV_primary | 判定 |
|---|---:|---:|---|
| 报告速度 vs 路径速度 RMSE | 0.069 m/s | 2.489 m/s（车心） | CAV 通过，BV 不通过 |
| 报告航向 vs 路径航向 MAE | 0.00008 rad | 0.754 rad（车心） | CAV 通过，BV 不通过 |
| 目标车前保险杠速度 RMSE | — | 0.874 m/s | 换算车心放大了航向噪声 |
| 目标车前保险杠航向 MAE | — | 0.476 rad | 仍超过 0.35 rad 门槛 |
| 原始参考窗口首次采样接触 | 3.9 s | 3.9 s | 初态无碰撞，窗口内有碰撞 |

这意味着第一阶段已经完成，但不能立刻生成可信的“双车分段控制轨迹”：CAV 位置、速度和航向可进入软拟合目标；目标车的前保险杠位置可以作为软参考，报告速度单独作为软参考，目标车航向与车心换算目前只能用于诊断。下一步应先批量审计候选追尾事件的目标轨迹一致性，再选择满足质量门槛的事件进入 SUMO 参数拟合。

导出命令：

```powershell
python -m scenario_reconstruction.shrp2_reference_trajectory `
  --source_root 'path/to/SHRP2_Public' `
  --seed 'data_analysis/raw_data/shrp2_collision_pilot_v7/scenarios/shrp2_131785457_rear_end_source_speed.json' `
  --output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_131785457_reference.json' `
  --summary_output 'results/shrp2_reference_trajectory_event_131785457_summary.json'
```

## 7. 追尾候选质量审计

已对训练划分中全部 `leading` Crash 候选进行一次 HDF5 单次读取审计：67 个候选中 39 个具有完整的 4 秒参考窗口，只有 2 个同时满足轨迹质量和事故事件门槛：

| 事件 | 目标车 | BV 前保险杠速度 RMSE | BV 前保险杠航向 MAE | 首次采样接触 | 状态 |
|---:|---:|---:|---:|---:|---|
| 116591908 | 21258 | 0.076 m/s | 0.126 rad | 3.6 s | 推荐首选 |
| 2934487 | 13483 | 0.317 m/s | 0.033 rad | 3.6 s | 推荐备选 |

审计硬门槛是：初态无碰撞、参考窗口内发生碰撞、碰撞时间误差不超过 `0.5 s`、CAV 和 BV 的位置—速度—航向一致性通过。`0.5 s` 是源数据 10 Hz 重采样和碰撞标注/几何换算的审计容差，不等同于后续 SUMO 校准的 `±0.3 s` 目标。

审计结果：[`shrp2_leading_reference_audit_train.json`](../results/shrp2_leading_reference_audit_train.json)。排名第一的 `116591908` 已导出完整参考轨迹：[`shrp2_reference_trajectory_event_116591908_summary.json`](../results/shrp2_reference_trajectory_event_116591908_summary.json)。因此后续 SUMO 参数拟合应优先使用 `116591908`，保留 `2934487` 作为独立备选，不再围绕 `131785457` 这个目标车质量不合格的事件继续调参。

批量审计命令：

```powershell
python -m scenario_reconstruction.shrp2_reference_audit `
  --source_root 'path/to/SHRP2_Public' `
  --output 'results/shrp2_leading_reference_audit_train.json'
```

## 8. 进入 SUMO 参数拟合

事件 `116591908` 已转换为桥接初始化模板：初始纵向位置差 `15.087 m`，CAV 初速 `5.972 m/s`，BV 初速 `0.898 m/s`。模板只使用参考轨迹的初始状态，不会把后续每一帧坐标强行写入 SUMO。

校准器接受历史 `source_quality=high` 模板，以及本轮批量轨迹审计产生的 `source_quality=audited` 模板。`audited` 表示该来源已通过位置、速度、航向和碰撞时刻门槛；它的使用范围仍严格限于**校准**，不能作为 D2RL 的训练事件。

本地生成的模板位于被 Git 排除的 `data_analysis/raw_data/shrp2_reference_trajectory_v1/`。生成命令如下：

```powershell
python -m scenario_reconstruction.shrp2_reference_to_seed `
  --reference 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_116591908_reference.json' `
  --output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_116591908_initial_state_seed.json' `
  --bridge_config 'configs/shrp2_sumo_bridge_pilot.json' `
  --bridge_output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/sumo_shrp2_116591908_rear_end_reference_initial_state.json'
```

此前 v2 网格在该事件上已经完成 45/45 条目标车碰撞，但全部发生在 `1.4–2.4 s`，早于来源审计的首次接触 `3.6 s`；因此 `selected_count=0` 不是 SUMO 接口故障，而是“前车强制急刹”的干预假设与该来源轨迹不符。后续使用源轨迹对齐的 v3 协议，不再重跑 v2。详见 [SHRP2 源轨迹对齐校准 v3](SHRP2源轨迹对齐校准V3.md)。

v3 会顺序执行 48 个 SUMO 校准候选，属于长时间运行任务，请由用户在本地执行。先重新生成桥接模板，使其目标碰撞时刻来自审计的 `3.6 s`：

```powershell
python -m scenario_reconstruction.shrp2_reference_to_seed `
  --reference 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_116591908_reference.json' `
  --output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_116591908_initial_state_seed.json' `
  --bridge_config 'configs/shrp2_sumo_bridge_pilot.json' `
  --bridge_output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/sumo_shrp2_116591908_rear_end_reference_initial_state.json'
```

```powershell
python -m scenario_reconstruction.shrp2_collision_calibration `
  'data_analysis/raw_data/shrp2_reference_trajectory_v1/sumo_shrp2_116591908_rear_end_reference_initial_state.json' `
  --config 'configs/shrp2_rear_end_calibration_116591908_v3.json' `
  --output 'data_analysis/raw_data/shrp2_reference_116591908_calibration_v3' `
  --run
```

运行结束后只看 `calibration_summary.json` 的 `executed_count`、`selected_count`、碰撞时间误差和最小间距；不要仅根据 PowerShell 的 SUMO stderr 警告判断失败。

v3 已完成：47/48 条目标追尾，3 条满足时间窗。已补算相对实测 SHRP2 的共同窗口指标：CAV ADE 约 `0.41–0.43 m`，BV 前保险杠 ADE 约 `2.89–3.70 m`。窗口为 `0–3.2 s`，不能当作完整 4 秒拟合结果。全类别数据盘点、参考点修正与下一条条件扩散数据导出命令见 [SHRP2 条件扩散数据准备](SHRP2条件扩散数据准备.md)。
