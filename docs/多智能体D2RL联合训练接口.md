# 多智能体 D2RL 联合训练接口

## 目标和边界

本接口将已有的 K-BV 日志真正交给一个**集中式联合策略**：策略输入一个 CAV 和 K 辆受控 BV 的联合状态，输出 K 个 epsilon 值。它不是把原始 SHRP2 轨迹直接监督成动作，也不是将 K 辆车分别交给 K 个独立学习器。

SHRP2 的职责是提供经质量筛选的初始几何、速度、冲突类别及留出测试场景；SUMO/NADE 的职责是从这些场景种子生成可追溯的反事实交互和 D2RL 权重记录；D2RL 的职责才是学习联合 epsilon 决策。

```text
SHRP2 多车事件
  -> 多 BV 场景种子（只含观测事实）
  -> SUMO / NADE 交互采样
  -> joint + per_agent 训练 episode
  -> 集中式 K-BV D2RL 策略
```

校准候选带有 `calibration_only=true` 或 `not_for_d2rl_training=true`，不得进入此链路。

## 训练记录契约

对 `K=2`，一个可训练决策步必须包含：

| 字段 | 形状 | 用途 |
|---|---:|---|
| `drl_obs_step_info[t].joint` | `6 + 4K = 14` | CAV 公共状态与每辆 BV 相对状态，供联合策略输入 |
| `drl_obs_step_info[t].per_agent` | `K x 10` | 保留诊断和旧接口兼容，不作为联合训练输入 |
| `drl_epsilon_step_info[t]` | `K = 2` | 策略对每辆 BV 给出的 epsilon |
| `weight_step_info[t].per_agent` | `K` | 每辆车的采样权重 |
| `ndd_step_info[t].per_agent` | `K` | 每辆车的自然驾驶概率 |
| `controlled_bv_ids_step_info[t]` | `K` | 动作位置与车辆 ID 的可追溯映射 |

联合动作的 importance 项按车辆相乘：对于每一辆 BV，`weight > 1` 时使用 `1 / epsilon`；`weight < 0.999` 时使用 `ndd_probability / (1 - epsilon)`。这避免了旧代码只取列表中第一个 epsilon 的错误。

## 代码开关

默认仍兼容旧训练：`multi_bv_training=false` 使用 10 维观测和 1 维动作。

在 D2RL 配置中设置：

```yaml
multi_bv_training: true
multi_bv_num: 2
```

则 `D2RLTrainingEnv` 的空间变为：

```text
observation_space: (14,)
action_space: (2,)
```

RLlib/PPO 的模型输入输出层也必须随之设为 14 与 2；仅改 JSON 日志或仅运行 `run_batch --multi_bv_num 2` 不会自动完成这一点。

## 数据集结果与下一步

公开 SHRP2 全类别扫描得到 3,043 个质量合格的双车风险窗口（训练 2,130、验证 449、测试 464）。它们是场景来源，不是训练 episode。已观测到至少两辆周边 BV 的合格事件有 2,988 个，因此下一项工作是从原始多目标轨迹导出“主风险 BV + 上下文 BV”的多车场景种子，并保持 event-level train/validation/test 划分不泄漏。

在该导出器完成前，不应把 `windows.jsonl` 直接放入 `crash_weight_dict.json`，因为它缺少 NADE 采样动作、自然驾驶概率、importance 权重和 SUMO 可执行路网映射。

当前已增加 `shrp2_multibv_seed_export`。它从已完成审计的窗口回到原始 HDF5，按事件级 split 选择主风险 BV 外最近的上下文 BV，输出 `shrp2_measured_multibv_seed_v1`。输出明确标记 `drl_training_ready=false`，下一阶段才进行 2Lane/目标路网映射和 SUMO 验证。

导出器有两个数据层：`context_mode=full` 要求上下文 BV 也有完整 4 秒历史，适合高保真验证；`context_mode=anchor_only` 只要求上下文 BV 在关键时刻有状态，输出是一帧关键时刻初始化种子，适合扩大 SUMO 场景池，但不得与完整历史样本合并统计。

`shrp2_multibv_sumo_bridge` 将 `anchor_only` 种子投影到当前 `2Lane` 路网。它只支持 `leading`、`adjacent_lane`、`merging` 和 `none` 这类可近似为同向两车道初始状态的事件；横穿、行人、动物和对向转弯会进入 blocked 清单。生成模板的 `events` 为空，确保后续行为由 NADE/D2RL 自主采样，而不是把 SHRP2 观测动作写成标签。
