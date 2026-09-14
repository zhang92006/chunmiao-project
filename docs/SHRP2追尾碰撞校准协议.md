# SHRP2 追尾碰撞校准协议

## 目标与边界

本协议只用于把高可信、训练划分的 SHRP2 追尾初始化候选校准为可控的 SUMO 碰撞压力测试。它不是事故动力学重放，也不从 SHRP2 直接估计制动行为概率。

校准输入必须来自 `shrp2_sumo_bridge` 生成的高可信 `rear_end` JSON 模板。低可信种子、验证集和测试集不得参与本阶段参数搜索。

## 搜索参数与固定规则

配置文件 `configs/shrp2_rear_end_calibration.json` 固定以下网格：

| 参数 | 候选值 |
|---|---|
| 初始间距偏移 | -8、-6、-4、-2、0 m |
| 主 BV 制动加速度 | -2、-4、-6、-8 m/s² |
| 制动触发时刻 | 0、0.5、1.0 s |
| 制动持续时间 | 3 s |
| 最小初始间距 | 7 m |
| 目标首次碰撞时刻 | 4 s |
| 时间容差 | 0.3 s |

总计 60 个候选。候选初始间距小于 7 m、或事件超出模板时长时会在运行前被拒绝。每个候选中的制动事件带有 `calibration_only=true` 和 `not_for_d2rl_training=true` 标记，不能直接进入训练数据目录。

## 选择规则

一个候选只有同时满足以下条件才被标为 `selected`：

1. 初始间距不小于 7 m；
2. SUMO 报告碰撞；
3. 碰撞对象包含 `CAV` 与 `BV_primary`；
4. 碰撞结束记录相对 4 s 的误差绝对值不超过 0.3 s。

输出还会保存最小 TTC、最小间距、候选参数、碰撞 ID、时间误差和未选中原因。`selected` 的含义仅为“满足当前 SUMO 校准准则”，不是“精确复现 SHRP2 事故”。

## 接口冒烟结果

完整网格尚未运行。已对候选 0 进行一次短接口验证：初始间距 8.355 m、BV 制动 -2 m/s²、在 0 s 触发、持续 3 s。该候选没有碰撞，最小 TTC 为 1.356 s、最小间距为 1.074 m，因而被正确标为 `not_selected`。

这验证了“候选生成 → SUMO 执行 → episode 解析 → 选择规则”的工程链路；它不代表其余 59 条候选的结果。机器可读记录为 `results/shrp2_rear_end_calibration_interface_summary.json`。

## 命令

先只生成 60 个候选、检查清单，不运行 SUMO：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'

& 'D:\Anaconda3\envs\D2RL\python.exe' -m scenario_reconstruction.shrp2_collision_calibration `
  data_analysis\raw_data\shrp2_sumo_bridge_pilot\templates\sumo_shrp2_131785457_rear_end_source_speed.json `
  --config configs\shrp2_rear_end_calibration.json `
  --output data_analysis\raw_data\shrp2_rear_end_calibration_grid
```

运行完整网格会顺序执行 60 条 SUMO episode，预计数分钟；请由用户在 PowerShell 中运行，并保存日志：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'

New-Item -ItemType Directory -Force data_analysis\logs | Out-Null

$log = 'data_analysis\logs\shrp2_rear_end_calibration_grid.log'

& 'D:\Anaconda3\envs\D2RL\python.exe' -m scenario_reconstruction.shrp2_collision_calibration `
  data_analysis\raw_data\shrp2_sumo_bridge_pilot\templates\sumo_shrp2_131785457_rear_end_source_speed.json `
  --config configs\shrp2_rear_end_calibration.json `
  --output data_analysis\raw_data\shrp2_rear_end_calibration_grid_run `
  --run 2>&1 | Tee-Object -FilePath $log
```

测试接口时只运行第一个候选：

```powershell
& 'D:\Anaconda3\envs\D2RL\python.exe' -m scenario_reconstruction.shrp2_collision_calibration `
  data_analysis\raw_data\shrp2_sumo_bridge_pilot\templates\sumo_shrp2_131785457_rear_end_source_speed.json `
  --config configs\shrp2_rear_end_calibration.json `
  --output data_analysis\raw_data\shrp2_rear_end_calibration_one `
  --run --max_candidates 1
```

完整运行完成后，把 `calibration_summary.json` 的路径或日志发给我；我会生成筛选结果、失败类型统计和下一阶段的验证集冻结方案。
