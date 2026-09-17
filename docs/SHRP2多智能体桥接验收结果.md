# SHRP2 多智能体桥接验收结果

## 当前批次

`data_analysis/raw_data/shrp2_multibv_sumo_templates_v1/bridge_summary.json` 的结果为：

| 项目 | 数量 |
|---|---:|
| 质量合格的双车事件种子 | 2988 |
| 成功投影到当前 2Lane 路网的模板 | 1748 |
| 被阻断的种子 | 1240 |
| 模板是否已经是 D2RL 训练样本 | 否 |

阻断原因主要有三类：

1. `primary_gap_out_of_range`：源场景纵向间距不满足当前 2Lane 初始化的无重叠约束；
2. `BV_context_position_out_of_map`：上下文车辆投影后超出当前短路网范围；
3. 不支持的拓扑：行人、骑行者、对向车流、交叉穿越和转向冲突不能用当前同向两车道模板表达。

## 为什么模板运行后没有联合训练字段

模板的 `events` 为空是有意设计：不回放 SHRP2 的动作，而是让 SUMO/NADE 自主采样。只有在同一个决策时刻满足以下条件，才会写入联合训练记录：

- 至少 `K=2` 个 BV 同时通过 NADE 临界度筛选；
- 两个 BV 都得到有效的 NDD 概率和重要性采样权重；
- 联合权重确实改变了采样分布（`joint_weight < 0.999` 时才进入训练样本）。

对 `shrp2_multibv_Crash_116168844_train.json` 的单例结果：

```text
t=0.0: BV_primary criticality=1.0, BV_context criticality=0.0
联合选中数=1，不生成 K=2 联合样本
SUMO 正常结束，collision_result=0
```

这表示日志接口已经能区分“候选车辆”和“实际联合控制车辆”，但该样本本身不是双 BV 联合训练样本。不能为了凑数量把第二辆车强行写入 `drl_obs`，否则权重与动作概率不一致，会污染 D2RL 训练。

每个多 BV 决策步现在会保留 `multibv_selection_debug_step_info`，可用于统计候选数、各车临界度、原始 NDD 概率和最终选中数。

## 下一步验收顺序

此前 20 个 train 模板的 pilot 结果是 19 次完成、1 次失败，且联合训练样本为 0。19 个输出中的最大实际选中数为 0 或 1，没有出现两个 BV 同时通过 NADE 临界度筛选。失败样本 `116168907` 是上下文车辆在投影后与 CAV 同车道重叠，已在桥接器中增加成对间距检查；首步碰撞日志也已修复。旧批次模板需要重新生成后再统计。

1. 重新运行桥接器，排除所有同车道初始化重叠；
2. 先对少量模板做自主 rollout，统计 `selected_count=2` 的比例，而不是立即运行全部模板；
3. 确认输出 JSON 同时出现：
   - `drl_obs_step_info[t].joint` 长度 14；
   - `drl_obs_step_info[t].per_agent` 为 2 个长度 10 的观测；
   - `weight_step_info[t].per_agent` 和 `ndd_step_info[t].per_agent` 长度均为 2；
   - `controlled_bv_ids_step_info[t]` 长度为 2；
3. 再按 train/validation/test 分开批量运行，并用 `prepare_training_data --multi_bv --agent_num 2` 生成训练索引；
4. 如果联合样本比例过低，优先扩大路网、延长 rollout 或增加“两个风险 BV 同时存在”的源事件筛选，不修改日志协议去制造伪联合样本。

## 结论

当前桥接批次已经完成“SHRP2 多车状态 → SUMO 可执行初始场景”的数据供给阶段，但尚未证明能提供足够密度的多智能体 D2RL 训练样本。下一项实验目标是测量联合采样覆盖率，并据此决定是否需要扩大场景时长、路网或风险事件筛选范围。

## BV 对联合临界度接口

原始 NADE 仅计算“CAV 与一辆 BV”的临界度，因此即使场景中有两辆 BV，也通常只会控制其中一辆。`joint_criticality.py` 新增了受限的两秒动作对代理：对每个 BV 动作对预测其与 CAV 的车道和纵向间距，得到联合 challenge，再分别对另一辆 BV 的 NDD 分布边缘化，形成两个 NDD 加权临界度数组。

这样仍保持每辆 BV 的 proposal 可分解、importance weight 可解释，并能够在两车的联合作用影响风险时同时采样两辆 BV；它不是把上下文车辆强行记为第二个动作标签。

在 SHRP2 v2 模板单例 `151553145` 上，联合接口已经写出 119 个包含两辆 BV 的联合记录。默认 `epsilon=0.99` 基本等于自然驾驶采样，权重接近 1；数据采集 pilot 使用 `epsilon=0.1`，以获得足够的非单位 importance weight。该 epsilon 是采样超参数，后续论文实验必须报告并做敏感性分析。

## 最新 50 模板联合 rollout（`epsilon=0.01`）

`shrp2_multibv_joint_rollout_epsilon001_train50` 运行了 50 个 train 模板：49 个成功、1 个失败；其中 27 个 episode 含有 K=2 联合控制记录，共 2016 个联合决策步，21 个安全 episode 已通过联合字段校验。接口与日志链路因此可用。

但碰撞训练池仍为 0。输出中有 5 个碰撞 episode，其累计 importance weight 介于 0.869 和 1.225，均未低于当前 `prepare_crash_weight_dict` 的默认阈值 0.1。它们表示自然或接近自然驾驶即可发生的碰撞，不能替代低权重反事实碰撞训练样本。

唯一失败模板 `Crash-Crash_10858441` 的 `BV_primary` 初始速度为 43.507 m/s，超过当前高速 D2RL/SUMO 域的 40 m/s 上限。桥接器现已增加速度域校验：这类记录会在模板生成阶段以 `*_speed_exceeds_d2rl_domain` 被阻断，不再在长时间 rollout 中失败。

下一轮应先重新生成 v3 桥接模板，再使用 runner 的 `--repeats` 对每个模板做多次独立 rollout；只降低 epsilon 而不增加重复次数，不能保证得到低权重碰撞。

## v3 重复采样诊断与高速域隔离

v3 对前 50 个 train 模板各重复 5 次，250 次全部完成。共保存 25 个碰撞 episode，但它们只来自 5 个模板，而且每个模板在 5 次重复中都于几乎相同的时刻碰撞；权重范围为 0.8758～1.1717，未产生小于 0.1 的训练碰撞。这说明重复次数不是当前主要瓶颈。

进一步检查发现，这 5 个模板的 CAV 初始速度只有 0～4.6 m/s，而现有 D2RL、NDD 状态表及 `Vehicle.act` 使用 20～40 m/s 高速域。车辆进入 NADE 控制后会受到 20 m/s 下界约束，因此这些结果不能视为可信的高速域反事实碰撞。

v3 的 1658 个模板中，只有 295 个模板的 CAV、主风险 BV 和上下文 BV 全部位于 20～40 m/s；train/validation/test 分别为 201/48/46。其余 1363 个低速或混合速度模板继续保留在 SHRP2 种子层，但不再送入当前高速 D2RL rollout。

桥接配置现同时要求 `minimum_initial_speed_mps=20` 和 `maximum_initial_speed_mps=40`。v4 汇总中的阻断原因按稳定代码聚合，例如 `cav_speed_below_d2rl_domain`，详细原始速度仍保留在每条 record 的 `reason` 字段。下一步只对 v4 高速域模板做小批量重复采样，再决定是否需要调整联合临界度模型；不得用放宽碰撞权重阈值替代速度域校准。

## 关键时刻前移初始化（v5）

v4 高速域 rollout 的唯一碰撞模板在 0.3 s 内确定性接触，说明将 `anchor_only` 种子的关键时刻（4.0 s）直接作为 SUMO 初态没有留下策略干预空间。导出器现增加 `initialization_offset_s`：v5 使用 `2.0 s`，即用原始 4 秒风险窗口的第 2 秒帧启动 SUMO，同时仍保留 `critical_time_s=4.0` 作为事件参考。该修改只改变后续新目录中的种子和模板，不重写原始 SHRP2、anchor v1 或 v4 结果。

v5 的首轮验收不以“立即得到低权重碰撞”为标准，而先检查碰撞是否在 1 s 后发生、同一模板的重复结果是否随采样动作改变，以及 `source_initialization_offset_before_critical_s=2.0` 是否贯穿种子、模板和 episode 元数据。只有通过这三项，才评估联合临界度对碰撞密度的贡献。
