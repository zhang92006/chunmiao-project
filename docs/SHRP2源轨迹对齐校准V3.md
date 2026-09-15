# SHRP2 源轨迹对齐校准 v3

## 目的与边界

v3 是事件 `116591908` 的**源轨迹对齐校准**：验证当前 SUMO 桥接能否在不逐点写入真实坐标的前提下，使碰撞对象和首次碰撞时刻接近审计后的 SHRP2 参考。它不是精确事故回放，也不是逐事件手工调参后直接作为训练样本的流程。

该协议只使用训练划分，所有候选均带有 `calibration_only=true` 和 `not_for_d2rl_training=true`。它的结果只能用于确定后续的共享参数化与验证方案。

## v2 诊断

事件 `116591908` 的 v2 运行完成了 45/45 个候选，且所有候选均发生 `CAV–BV_primary` 碰撞；`selected_count=0` 的唯一原因是碰撞过早：首次碰撞范围为 `1.4–2.4 s`，而配置目标为 `4.0±0.3 s`。

这排除了 SUMO 碰撞接口、目标车 ID 和候选生成失败。问题来自 v2 的干预假设：前车从 `0 s` 起固定以 `-2.0 m/s²` 制动 5 秒，而来源参考中 BV_primary 近似匀速，CAV 才表现为持续减速。

此外，参考轨迹的首次 10 Hz 采样接触为 `3.6 s`。此前桥接把参考窗口末端 `4.0 s` 误写为目标碰撞时刻；v3 已改为传递 `collision_audit.first_sampled_contact_time_s`。

## v3 网格

配置：[`shrp2_rear_end_calibration_116591908_v3.json`](../configs/shrp2_rear_end_calibration_116591908_v3.json)。

| 参数 | 值 | 数据依据 |
|---|---|---|
| 目标首次碰撞 | `3.6±0.3 s` | SHRP2 审计首次采样接触 |
| 初始位置差偏移 | `-1, 0, +1, +2 m` | 仅覆盖小幅映射/车辆几何不确定性 |
| BV 主车动作 | `0, -0.25 m/s²` | `0` 为显式速度保持；不再假设强制急刹 |
| CAV 动作 | `-0.5, -0.75, -1.0 m/s²` | 覆盖来源 CAV 的约 `-0.88 m/s²` 平均减速趋势 |
| CAV 动作窗口 | `3.6, 4.0 s` | 碰撞前窗口的有限干预 |
| 候选总数 | `48` | `4×2×3×2` |

`braking_accelerations_mps2=0` 在接口中明确表示“前车速度保持”，不表示无约束或恢复 IDM；负值才表示制动。两种动作都只是 SUMO 校准干预，不能解释为真实驾驶员的动作估计。

## 执行顺序

先重新生成本地种子与桥接模板，令其携带来源审计的 `3.6 s` 目标。以下命令覆盖的文件都在 Git 排除的 `data_analysis/raw_data/` 下：

```powershell
python -m scenario_reconstruction.shrp2_reference_to_seed `
  --reference 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_116591908_reference.json' `
  --output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/event_116591908_initial_state_seed.json' `
  --bridge_config 'configs/shrp2_sumo_bridge_pilot.json' `
  --bridge_output 'data_analysis/raw_data/shrp2_reference_trajectory_v1/sumo_shrp2_116591908_rear_end_reference_initial_state.json'
```

再执行 48 条候选。该命令会启动 SUMO，属于长时间运行任务：

```powershell
New-Item -ItemType Directory -Force 'data_analysis/logs' | Out-Null
python -m scenario_reconstruction.shrp2_collision_calibration `
  'data_analysis/raw_data/shrp2_reference_trajectory_v1/sumo_shrp2_116591908_rear_end_reference_initial_state.json' `
  --config 'configs/shrp2_rear_end_calibration_116591908_v3.json' `
  --output 'data_analysis/raw_data/shrp2_reference_116591908_calibration_v3' `
  --run 2>&1 | Tee-Object -FilePath 'data_analysis/logs/shrp2_reference_116591908_calibration_v3.log'
```

## 结果解释与下一关

只接受同时满足以下条件的候选：初始位置差不小于 `7 m`、碰撞对象为 `CAV–BV_primary`、首次碰撞位于 `3.6±0.3 s`。即使得到候选，也只能称为“对真实轨迹约束一致的 SUMO 校准结果”。

v3 成功后，不为每一条事故人工复制一套网格。下一步是对训练集所有质量合格追尾事件自动提取初态、相对速度、CAV/BV 加速度统计与碰撞时刻，并学习按事件类型共享的参数范围；验证/测试事件只用于冻结后的评估。
