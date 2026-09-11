# Qwen 真实调用测试记录

本次测试目标：

```text
从 Autoware metadata.yaml + db3 摘要出发
真实调用 Qwen 生成事故模板
核对模板真实性和当前仿真可用性
通过后进入变体生成与碰撞评分
```

## 1. 输入数据

```text
metadata.yaml
2019_07_9_cuted_run01_0.db3
```

当前已经可以读取 topic 级摘要，并且已经完成 Autoware 自定义 topic 的 CDR 反序列化验证。

关键 topic 统计：

```text
duration_s = 51.763542645
message_count = 5100
control_command.bag_rate_hz = 15.165
ego_pose.bag_rate_hz = 11.514
perception_objects.bag_rate_hz = 7.051
ego_velocity.bag_rate_hz = 12.692
```

当前预计算的 cut-in 事件持续时间：

```text
forced_bv_action_duration_estimate_s = 1.135
```

## 1.1 ROS2 CDR 自定义 topic 解码状态

已安装：

```text
rosbags = 0.11.3
```

已下载 Autoware 消息定义：

```text
external/autoware_msgs
```

已注册自定义消息类型数量：

```text
75
```

已验证可批量解码的 topic：

```text
/vehicle/status/velocity_status
/planning/scenario_planning/trajectory
/perception/object_recognition/objects
/localization/kinematic_state
/control/command/control_cmd
```

解码结果：

```text
/vehicle/status/velocity_status decoded=657 errors=0
/planning/scenario_planning/trajectory decoded=269 errors=0
/perception/object_recognition/objects decoded=365 errors=0
/localization/kinematic_state decoded=596 errors=0
/control/command/control_cmd decoded=785 errors=0
```

已生成紧凑解码摘要：

```text
data_analysis/raw_data/AutowareDecoded/autoware_decoded_summary.json
```

其中已经包含：

```text
ego localization first/last pose
vehicle longitudinal/lateral velocity range
control target velocity/acceleration/steering range
perception object count and first non-empty object sample
planning trajectory point count and sample trajectory points
```

计算方式：

```text
duration = max(0.5, min(2.0, 8 / min(perception_objects.bag_rate_hz, control_command.bag_rate_hz)))
duration = max(0.5, min(2.0, 8 / min(7.051, 15.165)))
duration = 1.135 s
```

## 2. 第一次可校验 Qwen 模板的问题

Qwen 曾生成 5 个可校验模板，但没有产生 crash。

主要问题：

```text
速度偏低，约 5 到 12 m/s，更像城市低速场景
position 从 0 到几十米开始，不符合当前 D2RL 高速 2Lane 已跑通坐标尺度
apply_once=True，当前 forced cut-in 更适合在事件窗口内持续触发
```

结论：

```text
这些模板语义合理，但不适合直接作为当前 D2RL 高速 SUMO 环境的事故种子。
```

## 3. 高风险投影后 Qwen 生成结果

随后 prompt 增加了当前工程约束：

```text
CAV position around 350 to 420 m
CAV speed around 28 to 36 m/s
BV_cut_in ahead by about 10 to 25 m
BV_cut_in speed around 18 to 28 m/s
event start_time around 0 to 2 s
apply_once=false
```

Qwen 生成 5 个模板，全部通过校验。

```text
cut_in_001
gap = 20.0 m
CAV speed = 32.0 m/s
BV speed = 24.0 m/s
CAV lane = 1
BV lane = 0
lateral = left
start_time = 1.2 s
duration = 1.135 s
apply_once = false

cut_in_002
gap = 10.0 m
CAV speed = 33.0 m/s
BV speed = 20.0 m/s
CAV lane = 1
BV lane = 0
lateral = left
start_time = 0.5 s
duration = 1.135 s
apply_once = false

cut_in_003
gap = 25.0 m
CAV speed = 34.0 m/s
BV speed = 26.0 m/s
CAV lane = 1
BV lane = 0
lateral = left
start_time = 1.8 s
duration = 1.135 s
apply_once = false

cut_in_004
gap = 35.0 m
CAV speed = 31.0 m/s
BV speed = 18.0 m/s
CAV lane = 1
BV lane = 0
lateral = left
start_time = 2.5 s
duration = 1.135 s
apply_once = false

cut_in_005
gap = 10.0 m
CAV speed = 30.0 m/s
BV speed = 22.0 m/s
CAV lane = 1
BV lane = 0
lateral = left
start_time = 1.0 s
duration = 1.135 s
apply_once = false
```

## 4. 真实性核对

当前能确认：

```text
模板结构符合 ScenarioTemplate schema
map 和 route 可被当前仿真环境消费
lane_index 合法
BV_cut_in 的 lateral left 会从 lane 0 切向 CAV 所在 lane 1
BV_cut_in 位于 CAV 前方，gap 大多在 10 到 25 m
CAV 速度高于 BV_cut_in，具备追尾/cut-in 风险
duration 使用 Autoware topic rate 预计算值 1.135 s
apply_once=false，适合当前 forced cut-in 持续触发逻辑
```

当前不能确认：

```text
这些位置和速度是否与真实 Autoware CDR 轨迹逐点一致
真实故障发生时刻是否就是这些 start_time
真实碰撞瞬间车辆状态是否完全对应
```

旧版本原因是当时尚未完成 ROS2 CDR 反序列化，只能使用 topic 级摘要。
当前已经完成自定义 topic 解码，见第 7 节。

结论：

```text
这批模板作为“从 Autoware 摘要推理并投影到当前 D2RL 高速 SUMO 环境”的事故种子是可用的。
它不是严格的 Autoware 轨迹复现。
```

## 5. 变体生成与碰撞评分结果

每个模板生成 6 个变体，共 30 次仿真。

```text
cut_in_001
successful_runs = 6
failed_runs = 0
training_ready_crashes = 4
best_end_time = 2.0
best_closeness_score = 1.1513
reward_at_epsilon_0.99 = 70.5757

cut_in_002
successful_runs = 6
failed_runs = 0
training_ready_crashes = 1
best_end_time = 1.2
best_closeness_score = 1.4991
reward_at_epsilon_0.99 = 56.3459

cut_in_003
successful_runs = 6
failed_runs = 0
training_ready_crashes = 4
best_end_time = 2.8
best_closeness_score = 0.9786
reward_at_epsilon_0.99 = 68.7218

cut_in_004
successful_runs = 6
failed_runs = 0
training_ready_crashes = 5
best_end_time = 3.0
best_closeness_score = 1.5812
reward_at_epsilon_0.99 = 87.2886

cut_in_005
successful_runs = 6
failed_runs = 0
training_ready_crashes = 5
best_end_time = 1.5
best_closeness_score = 1.4612
reward_at_epsilon_0.99 = 52.1495
```

总计：

```text
total_simulations = 30
successful_runs = 30
failed_runs = 0
training_ready_crashes = 19
```

## 6. 输出位置

Qwen 模板生成结果：

```text
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/templates/template_generation_manifest.json
```

变体生成与评分结果：

```text
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/variant_batches/variant_batch_summary.json
```

每个模板的完整 batch 结果：

```text
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/variant_batches/cut_in_001/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/variant_batches/cut_in_002/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/variant_batches/cut_in_003/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/variant_batches/cut_in_004/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenRetestHighRiskProjection/variant_batches/cut_in_005/batch_summary.json
```

## 7. 使用 ROS2 CDR 解码状态量后的真实状态全流程复测

本轮目标：

```text
尽量使用 Autoware 原始数据中解码出的车辆状态量
不再人为指定高速 D2RL 速度
让 Qwen 基于 decoded_signal_summary 推理事故模板
随后自动进入变体生成与碰撞评分
```

运行输入：

```text
metadata.yaml
2019_07_9_cuted_run01_0.db3
data_analysis/raw_data/AutowareDecoded/autoware_decoded_summary_latest.json
```

本轮全流程输出目录：

```text
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates
```

关键解码状态量：

```text
/vehicle/status/velocity_status decoded=657 errors=0
/planning/scenario_planning/trajectory decoded=269 errors=0
/perception/object_recognition/objects decoded=365 errors=0
/localization/kinematic_state decoded=596 errors=0
/control/command/control_cmd decoded=785 errors=0
```

可用于推理的真实状态范围：

```text
vehicle longitudinal velocity min = -0.0504 m/s
vehicle longitudinal velocity max = 4.2648 m/s
control target velocity min = 0.0 m/s
control target velocity max = 4.2008 m/s
control target acceleration min = -2.5 m/s^2
control target acceleration max = 1.218 m/s^2
control steering tire angle min = -0.00283 rad
control steering tire angle max = 0.06878 rad
perception messages with objects = 50
perception max_object_count = 1
first perceived object class = CAR
first perceived object position approx = (-88.514, -27.653, 0.873)
first perceived object linear velocity x approx = 0.25 m/s
localization first pose approx = (-101.793, 48.381)
localization last pose approx = (-110.669, -38.023)
planning trajectory max point count = 167
```

运行结果汇总：

```text
generation_source = llm
model = qwen-plus
llm_error = null
requested_template_count = 5
raw_candidate_count = 5
usable_template_count = 4
unusable_template_count = 1
simulation_batch_count = 4
successful_runs = 24
failed_runs = 0
training_ready_crashes = 0
```

不可用模板原因：

```text
cut_in_004: BV_cut_in.position must be non-negative.
```

Qwen 生成的可用模板要点：

```text
cut_in_001
ego speed = 0.25 m/s
BV_cut_in speed = 0.25 m/s
BV_cut_in position = 21.7 m
event start_time = 3.0 s
event duration = 1.135 s
依据 = 首个非空感知目标、ego 初始位姿、低速轨迹

cut_in_002
ego speed = 4.26 m/s
BV_cut_in speed = 3.8 m/s
BV_cut_in position = 12.0 m
event start_time = 2.0 s
event duration = 1.135 s
依据 = 解码出的最大 ego velocity，用作较高风险低速变体

cut_in_003
ego speed = 2.0 m/s
BV_cut_in speed = 2.0 m/s
BV_cut_in position = 25.0 m
event start_time = 5.0 s
event duration = 1.135 s
依据 = 由 0 到 4.26 m/s 速度范围取中间运行状态

cut_in_005
ego speed = 0.0026 m/s
BV_cut_in speed = 0.0 m/s
BV_cut_in position = 30.0 m
event start_time = 2.5 s
event duration = 1.135 s
依据 = 近停止状态和首个轨迹样本
```

变体仿真结果：

```text
cut_in_001
successful_runs = 6
failed_runs = 0
training_ready_crashes = 0
best_outcome = null

cut_in_002
successful_runs = 6
failed_runs = 0
training_ready_crashes = 0
best_end_time = 28.7
min_ttc = 33.254
min_distance = 5.637
collision_result = 0

cut_in_003
successful_runs = 6
failed_runs = 0
training_ready_crashes = 0
best_end_time = 29.1
min_ttc = 1.620
min_distance = 10.591
collision_result = 0

cut_in_005
successful_runs = 6
failed_runs = 0
training_ready_crashes = 0
best_outcome = null
```

真实性核对：

```text
Qwen 这次确实使用了解码后的原始状态量。
生成速度主要来自 decoded_signal_summary，而不是人为高速假设。
0.0026, 0.25, 2.0, 4.26 m/s 均与解码出的低速行驶状态一致。
event duration = 1.135 s 仍来自 topic rate 估算公式。
感知目标位置和 ego localization 被投影到当前 2Lane 纵向坐标中使用。
```

主要问题：

```text
当前真实 Autoware 片段整体速度偏低，直接放入当前 2Lane D2RL 仿真环境后没有形成碰撞。
当前 2Lane 环境只接受非负纵向 position，因此真实相对后方目标会被判为不可用。
Qwen 对 perturbations.low/high 的理解不稳定，部分模板把 low/high 当成绝对取值范围。
当前变体生成器实际把 low/high 当作 additive delta，因此已经加强 prompt 约束。
```

原始数据碰撞核对：

```text
当前 metadata.yaml 中没有 collision、crash、contact、diagnostics accident result 等明确碰撞真值 topic。
已解码 topic 主要是 ego velocity、ego localization、control command、planning trajectory、perception objects。
对 /localization/kinematic_state 与 /perception/object_recognition/objects 做最近时间戳对齐后：
localization_count = 596
object_records = 50
nearest_time_delta_s = 0.0020
nearest_xy_distance_m = 12.9923
ego speed at nearest pair = 3.5044 m/s
object velocity x at nearest pair = 7.9287 m/s
object dimensions approx = 5.0870 m x 2.0681 m
```

因此：

```text
这份 Autoware bag 可以确认是一个含有低速车辆状态、规划轨迹、感知目标和控制指令的故障/场景片段。
但仅凭当前已解码 topic，不能确认原始 Autoware 数据已经发生碰撞。
最近距离约 13 m，明显大于两车包围盒接触尺度；当前证据更像“有风险/故障场景”，不是“碰撞真值样本”。
如果需要确认碰撞，必须额外有仿真器 collision/contact topic、事故标注文件，或车辆几何轮廓逐帧重叠检测结果。
```

当前数据定位：

```text
metadata.yaml + 2019_07_9_cuted_run01_0.db3 更适合被定义为 Autoware 故障/风险场景数据。
它可以作为后续自动模板生成、风险放大、变体生成和 D2RL 训练样本筛选的种子数据。
它不应直接被标记为“已碰撞事故样本”或“碰撞真值样本”。
```

原因：

```text
已有数据能提供 ego 速度、定位轨迹、控制指令、规划轨迹、感知目标状态。
这些信息足够支持大模型推理故障场景结构和参与车辆状态。
但当前没有碰撞结果 topic，也没有人工事故标注文件。
最近距离核对没有显示车辆包围盒接触。
```

后续使用方式：

```text
先把该数据作为真实故障场景种子。
再在当前 SUMO/NADE/D2RL 仿真环境中进行 simulation_projection 和风险放大。
最后通过碰撞结果、碰撞瞬间状态、碰撞后状态、min TTC、min distance 等指标筛选真正可用于 D2RL 训练的 crash 样本。
```

结论：

```text
如果目标是“真实性复现”，本轮结果更可信，但还不足以产生 D2RL 训练用 crash 样本。
如果目标是“训练补充事故数据”，需要在真实状态基础上增加一个仿真投影或风险放大步骤。
更合理的后续流程是：
1. 用 CDR 解码状态量建立真实事件模板。
2. 保留真实速度、目标、姿态、对象类别作为证据。
3. 单独生成 simulation_projection 字段，把真实场景投影到当前 2Lane 可碰撞参数域。
4. 变体生成只扰动 delta，不直接覆盖真实证据。
5. 用碰撞瞬间状态、碰撞后状态、min TTC、min distance 综合排序。
```

本轮结果文件：

```text
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates/auto_analysis_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates/templates/template_generation_manifest.json
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates/batches/cut_in_001/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates/batches/cut_in_002/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates/batches/cut_in_003/batch_summary.json
data_analysis/raw_data/AutowareAutoAnalysisQwenDecodedOriginalStates/batches/cut_in_005/batch_summary.json
```
