# SHRP2 多智能体关键性模型修正

## 触发证据

在 `TTC <= 5 s` 的 4 条 SHRP2 高速训练模板上，使用 `epsilon=0.001` 各执行 50 次，共 200 次 SUMO/NADE rollout：

- 200 次均成功，均为安全结束；
- 联合控制覆盖 23,800 个决策步，权重可低至约 `4.4e-137`；
- CAV 在每次首步均以约 `-4 m/s²` 制动；
- 主 BV 首步却以约 `+0.89` 至 `+1.82 m/s²` 加速；
- 最小车间净距仍为 8.03 至 12.40 m。

因此问题不是采样次数或 importance weight 阈值，而是原双 BV 风险代理把几何接近等同于碰撞风险，未正确区分前车加速与急刹。

## 修正内容

`scenario_reconstruction.joint_criticality` 现在对每个 BV 动作对进行 4 秒、0.1 秒分辨率的向量化前向预测：

1. 采用有符号的纵向车辆顺序和车身净间距；
2. 将 BV 动作映射为纵向加速度或进入相邻车道；
3. 当同车道前车的 TTC 不大于 5 秒时，预测 CAV 以 `-4 m/s²` 紧急制动；
4. 以预测期内最小净间距为 challenge：发生接触为 1；无接触且净距大于 2 m 为 0；
5. 在碰撞威胁形成 1 秒后，同时预测 CAV 的相邻车道逃逸；只有第二辆 BV 保持在相邻车道 5 m 净距内时才计为封堵；
6. 仍将联合 challenge 边缘化为每辆 BV 的 factorised proposal，保持既有 importance-weight 契约。

输出日志新增每辆候选 BV 的 `sampled_action_ids`。pair debug 还包括最优动作、最大 challenge 动作对、其最小净距、可导致接触的动作对数量和可封堵换道逃逸的动作对数量。

## 边界与验收

该模型不是 SHRP2 原驾驶行为回放，也不强制插入碰撞。它仅改变 NADE 对可采样动作的风险排序；是否发生碰撞仍由 SUMO 和 CAV 控制器决定。

下一轮只运行原 4 条 `TTC <= 5 s` 模板进行小批量回归。验收应同时检查：

- `training_ready_crashes > 0`；
- 碰撞前的 `sampled_action_ids` 包含预测的高风险动作，而非只包含前车加速；
- `pair_criticality` 的 `collision_action_pair_count` 为正；
- crash episode 保留 `joint`、`per_agent`、`drl_epsilon` 和 importance-weight 字段。

若仍然零碰撞，结论将是当前无故障 IDM CAV 在这四个真实初始化下可避免风险。后续应把 CAV 反应延迟作为显式、独立标注的实验条件，而不能把故障样本混入默认 D2RL 训练池。

## V3 小批量验收结果

使用 TTC 不大于 5 秒的 4 条训练模板、每条 10 次、`epsilon=0.001` 进行 40 次 rollout 后：

- 40 次均成功；
- 3 条为 `training_ready_crashes`，17 条为 `training_ready_safe`；
- 三条碰撞均来自独立源事件 `NearCrash_151578944`，发生在 3.6 至 4.1 秒；
- 碰撞对为 `CAV` 与 `BV_context`，不是初始重叠：主 BV 先在 CAV 当前车道形成纵向制约，context BV 保持相邻车道封堵，CAV 换道时发生接触；
- 三条碰撞的 episode importance weight 分别约为 `6.73e-171`、`4.67e-143` 和 `1.46e-151`。

该结果证明双 BV 风险模型、SUMO 模板和训练数据接口已端到端贯通，但尚不能启动正式训练：3 条 crash 只来自一个源场景。下一阶段应扩大到更多独立 SHRP2 种子，并按源事件而非 rollout 重复次数统计样本多样性。
