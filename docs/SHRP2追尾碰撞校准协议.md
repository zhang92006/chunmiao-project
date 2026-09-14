# SHRP2 追尾碰撞校准协议

## 结论

零碰撞不是 SUMO 碰撞检测失效，而是当前项目控制层与校准事件共同造成的：

1. 项目全局采用 `high_speed`，`Vehicle.act()` 会把车辆动作裁剪到 20–40 m/s；SHRP2 样本的 CAV/BV 初速仅为 3.117/0.522 m/s，因此 v1 的低速 BV 制动没有按预期执行。
2. v1 的 BV 制动只持续到 3.0 s，而目标碰撞时刻是 4.0 s；BV 随后恢复 IDM 并加速，在最小间距约 2.53 m 时脱离冲突。
3. 没有覆盖 CAV 响应时，项目中的安全跟驰控制会主动降速或换道。

因此 v1 的 60 条结果只保留为工程诊断，不能用于论文中的制动参数结论。v2 绕过低速动作的 20 m/s 下限，将 BV 制动延长到 5 s，并增加明确标记为“仅校准”的 CAV 响应保持窗口。

## v2 的科学边界

`calibration_cav_action` 是碰撞可达性干预，不是感知延迟、控制延迟或人类驾驶员反应模型。它只回答：“在安全初态下，如果 CAV 在限定窗口内保持当前响应，SUMO 能否到达并记录目标追尾？”

该事件必须满足：

- 只能作用于 `CAV`；
- `calibration_only=true`；
- `not_for_d2rl_training=true`；
- 不写入 D2RL 训练事件权重与观测；
- 不能把结果称为 SHRP2 事故的精确重放。

配置文件为 `configs/shrp2_rear_end_calibration_v2.json`。网格共 45 条，由 5 个初始位置差、3 个 CAV 加速度和 3 个保持时长组成。这里的 `initial_gap_m` 是 BV 与 CAV 的纵向位置差，不是保险杠净距。

## 验证结果

v2 短验证只运行了前三条候选，3/3 均由 SUMO 报告 `CAV` 与 `BV_primary` 在 3.8 s 发生碰撞，目标时刻误差为 -0.2 s，满足 ±0.3 s 规则。最小记录 TTC 为 0.00595 s，最小记录距离为 0.0196 m。

这证明：

- SUMO 的物理接触检测接口可用；
- 低速动作绕过修复有效；
- 该 SHRP2 初态在声明的干预条件下具有追尾可达性；
- 尚未证明真实驾驶员或真实 ADS 会以该方式响应。

机器可读摘要位于 `results/shrp2_rear_end_calibration_v2_interface_summary.json`。

## 是否更换仿真器

SUMO 官方说明，碰撞可被检测和记录；在 `--collision.mingap-factor 0` 下判据是车辆物理外形接触，TraCI 的 `speedMode` 可关闭安全速度约束。因此当前问题不要求更换 SUMO：项目已经使用 `collision.mingap-factor=0`、`collision.action=warn`，v2 也已实测记录碰撞。

CARLA 能提供三维刚体动力学、碰撞传感器、摄像头/雷达和更真实的车辆响应，适合后期做传感器与动力学复核；但 CARLA Traffic Manager 本身也有碰撞风险检测与制动阶段，直接换成默认自动驾驶同样可能避免碰撞。换软件不能替代“故障/响应模型”的定义。

推荐采用两级方案：SUMO 负责大规模场景搜索、概率估计和消融实验；冻结选中的代表性碰撞后，再把少量场景迁移到 CARLA 做动力学和传感器层验证。现阶段直接整体迁移会增加地图、坐标、控制器和复现成本，却不会自动解决零碰撞。

参考：

- [SUMO Safety](https://sumo.dlr.de/docs/Simulation/Safety.html)
- [SUMO TraCI vehicle state](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html)
- [SUMO ACC model](https://sumo.dlr.de/docs/Car-Following-Models/ACC.html)
- [CARLA Traffic Manager](https://carla.readthedocs.io/en/latest/adv_traffic_manager/)
- [CARLA collision sensor](https://carla.readthedocs.io/en/latest/ref_sensors/)

## 完整 v2 网格命令

完整运行会顺序执行 45 条 SUMO episode，请在 PowerShell 中运行：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'

New-Item -ItemType Directory -Force data_analysis\logs | Out-Null

$calibrationLog = 'data_analysis\logs\shrp2_rear_end_calibration_v2_grid.log'

& 'D:\Anaconda3\envs\D2RL\python.exe' -m scenario_reconstruction.shrp2_collision_calibration `
  data_analysis\raw_data\shrp2_sumo_bridge_pilot\templates\sumo_shrp2_131785457_rear_end_source_speed.json `
  --config configs\shrp2_rear_end_calibration_v2.json `
  --output data_analysis\raw_data\shrp2_rear_end_calibration_v2_grid_run `
  --run 2>&1 | Tee-Object -FilePath $calibrationLog
```

完成后检查：

```powershell
$summaryPath = 'data_analysis\raw_data\shrp2_rear_end_calibration_v2_grid_run\calibration_summary.json'
$summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
$summary | Select-Object executed_count, selected_count
$summary.selected | Select-Object candidate_index, initial_gap_m, cav_override_acceleration_mps2, cav_override_duration_s, end_time_s, collision_time_error_s
```

完整结果出来后，再冻结一个最接近 4.0 s 的候选，并在验证集上只执行冻结协议，不重新调参。

完整网格现已完成；结果、数值边界重评分和冻结候选见 [SHRP2 追尾碰撞校准 v2：完整网格结果](SHRP2追尾碰撞校准V2网格结果.md)。
