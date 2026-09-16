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

1. 先对少量模板做自主 rollout，统计 `selected_count=2` 的比例，而不是立即运行全部 1748 个模板；
2. 确认输出 JSON 同时出现：
   - `drl_obs_step_info[t].joint` 长度 14；
   - `drl_obs_step_info[t].per_agent` 为 2 个长度 10 的观测；
   - `weight_step_info[t].per_agent` 和 `ndd_step_info[t].per_agent` 长度均为 2；
   - `controlled_bv_ids_step_info[t]` 长度为 2；
3. 再按 train/validation/test 分开批量运行，并用 `prepare_training_data --multi_bv --agent_num 2` 生成训练索引；
4. 如果联合样本比例过低，优先扩大路网、延长 rollout 或增加“两个风险 BV 同时存在”的源事件筛选，不修改日志协议去制造伪联合样本。

## 结论

当前桥接批次已经完成“SHRP2 多车状态 → SUMO 可执行初始场景”的数据供给阶段，但尚未证明能提供足够密度的多智能体 D2RL 训练样本。下一项实验目标是测量联合采样覆盖率，并据此决定是否需要扩大场景时长、路网或风险事件筛选范围。
