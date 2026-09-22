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

联合 importance weight 按车辆相乘。一般 factorized proposal 对每辆 BV 使用
`q(epsilon) = epsilon * p + (1 - epsilon) * c`，其中 `p` 是自然驾驶动作概率、`c` 是
关键动作提议概率，因此新权重必须精确计算为 `p/q(epsilon)`。生成日志中的旧权重满足
`weight_generation=p/q_generation`，结合生成 epsilon 可重建 `c`。车辆 ID、生成 epsilon、
自然概率与权重必须在同一顺序对齐。

旧兼容模式曾按 `weight > 1` 使用 `1/epsilon`、按 `weight < 0.999` 使用
`p/(1-epsilon)`；它只适用于特殊退化分布，不适用于一般 factorized 或冻结 CEM proposal，
不得用于新的正式实验。

## 代码开关

默认仍兼容旧训练：`multi_bv_training=false` 使用 10 维观测和 1 维动作。

在 D2RL 配置中设置：

```yaml
multi_bv_training: true
multi_bv_num: 2
multi_bv_factorized_reward_mode: "exact_mixture"
multi_bv_reward_mode: "bounded_log_weight"
multi_bv_log_reward_scale: 5.0
log_episode_rewards: false
num_workers: 0
direct_training: true
action_distribution: "bounded_beta"
```

`log_episode_rewards` 默认为 `false`，正式训练时不再逐 episode 打印动作和奖励。
只有诊断单个 episode 时才应临时设为 `true`；该开关不改变奖励公式或训练样本。

`exact_mixture` 会拒绝缺少精确概率重建字段的 episode；`bounded_log_weight` 使用
`-R*tanh(log(weight)/scale)`，避免旧线性奖励在极端权重下大量硬截断到 ±R。
保留 `legacy` / `legacy_linear` 仅用于复现实验，不与新奖励的数值直接横向比较。

在约 16 GiB 内存的 Windows 主机上，`num_workers=0` 配合 `direct_training=true` 可让
PPO 在驱动进程中本地采样和训练，避免 Ray Tune 的独立 Trainer actor 触发 95% 内存保护。
该选项只影响执行方式和并行度，不改变训练样本、奖励或策略结构。

`bounded_beta` 在 RLlib 的归一化动作域 [-1,1] 内使用独立 Beta 分布，再由框架映射到
epsilon [0.001,0.999]。与默认无界对角高斯相比，它不会依赖事后裁剪制造边界动作，
同时提供 PPO 需要的 log-probability、entropy 与 KL。旧无界高斯配置保留为对照。

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

`anchor_only` 的历史默认版本以关键时刻初始化，适合场景清单统计，但不适合作为反事实碰撞 rollout 的起点。导出器现支持 `--initialization_offset_s 2.0`：从 `critical_time_s=4.0` 向前取 `initialization_time_s=2.0` 的 CAV、主风险 BV 和上下文 BV 状态。种子会记录 `initialization_offset_before_critical_s`，桥接模板会传递 `source_state_time_s`、`source_critical_time_s` 和该 offset，保证训练数据能审计“从碰撞前多长时间开始”。上下文 BV 若在前移时刻没有 `maximum_target_alignment_dt_s` 内的测量状态，会被排除，而不会用关键时刻状态冒充前移状态。
