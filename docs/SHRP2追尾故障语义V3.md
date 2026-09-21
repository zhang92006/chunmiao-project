# SHRP2 追尾故障语义 v3

## 目标

v2 的 `calibration_cav_action` 已完成“碰撞是否可达”的任务，但它是动作覆盖，不能解释为真实驾驶员或 ADS 故障。v3 将故障接入 CAV 的控制边界：观测先经过感知故障，再由 IDM 决策，最后经过控制延迟才执行。

已实现的事件与含义如下：

| 事件 | 注入位置 | 可审计输出 |
|---|---|---|
| `perception_delay` | 将指定目标的观测替换为历史帧 | 延迟值、目标 ID |
| `perception_dropout` | 在观测中移除指定目标 | 被移除的观测槽位 |
| `perception_position_bias` | 修改目标报告的纵向位置与距离 | 偏置值、受影响槽位 |
| `control_delay` | 将控制器当前决策替换为延迟历史动作 | 期望动作与实际执行动作 |

每个 CAV 故障步骤写入 episode 的 `cav_fault_step_info`，因此可以复查每一时刻故障是否处于活动状态，以及控制延迟具体替换了哪一个动作。

## 低速执行约束

SHRP2 当前种子的速度低于项目原有的 20 m/s 动作下限。v3 仅在模板声明 CAV 故障事件时启用 `FaultAwareVehicle`，使 IDM 的 `[-4, 2] m/s²` 动作能以低速语义执行；普通 D2RL 训练和无故障 IDM 路径不变。

## v3 短验证

基于 v2 冻结候选 19，移除了 `calibration_cav_action`，只保留：

- BV：`-2.0 m/s²` 制动，0.0–5.0 s；
- CAV：`control_delay=4.0 s`，0.0–4.0 s；当尚无历史控制动作可重放时，按声明保持 `central, 0.0 m/s²`。

结果：`CAV–BV_primary` 在 4.1 s 碰撞，误差 +0.1 s，满足 4.0±0.3 s；记录了 41 个故障审计步。该结果说明 v2 冻结基线可以由明确的控制延迟机制达到，而不依赖动作覆盖。

机器可读摘要：`results/shrp2_rear_end_fault_validation_v3_interface_summary.json`。

## 科学边界与下一步

这不是“SHRP2 中真实存在 4 s 控制延迟”的结论。4 s 是为了匹配冻结时刻而进行的机制化验证参数，尚未估计其经验分布。

下一阶段应固定 BV 与初态，对 `perception_delay`、`control_delay` 和 `perception_position_bias` 分别做小网格消融；随后只在未参与调参的样本上验证。长时间网格运行前，将单独提供命令与参数表，避免混入本次训练集校准。
