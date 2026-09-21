# 多智能体概率权重与 Proposal 消融

## 目的

本阶段保证 SHRP2 条件场景中的多智能体稀有事件采样可以被审计、稳定存储，并能在相同场景与预算下比较不同 proposal。

它不改变 SHRP2 的真实初始状态，也不强制注入碰撞；只改变或记录 BV 动作的采样分布 `q(a|s)`，并保留自然驾驶分布 `p(a|s)`。

## 日志契约

每个执行了联合 BV 动作的时间步写入：

```text
naturalistic_probability       = p(a|s)
proposal_probability           = q(a|s)
log_naturalistic_probability   = log p(a|s)
log_proposal_probability       = log q(a|s)
log_importance_weight          = log p(a|s) - log q(a|s)
importance_weight              = p(a|s) / q(a|s)
proposal_type
```

episode 级别累积 `log_naturalistic_probability`、`log_proposal_probability` 和 `log_importance_weight`。原有 `weight_episode` 保留以兼容旧训练代码，但发生浮点下溢时不再作为概率审计的唯一依据。

每个 episode 还写入：

```text
scenario_metadata.source_event_id
scenario_metadata.source_split
scenario_metadata.source_conflict
scenario_metadata.template_id
scenario_metadata.multibv_proposal_mode
```

这保证训练、验证、测试能够按源事件检查泄漏。

## Proposal 模式

| 模式 | 含义 | 用途 |
|---|---|---|
| `naturalistic` | 令 `q=p`，固定 `epsilon=1` | 自然驾驶碰撞率基线 |
| `factorized` | 两辆 BV 按各自边缘 proposal 独立采样 | 原始多 BV NADE 基线 |
| `joint_pair` | 直接从相关的二维动作对 proposal 采样 | 当前联合风险发生器 |

三种模式必须使用相同的：

- SHRP2 源事件；
- 初始化模板；
- SUMO 版本与地图；
- rollout 数量；
- CAV 控制器；
- 碰撞判定规则。

除自然驾驶模式外，`epsilon` 保持项目既有定义：它是自然驾驶混合质量；较小值表示更侧重风险条件 proposal。

## ESS 诊断

`prepare_training_data` 对实际进入训练池的 crash episode 生成：

```text
importance_weight_diagnostics.json
```

其中包含：

- episode 数量；
- effective sample size（ESS）；
- ESS 比率；
- 最大归一化权重；
- 最小和最大 log-weight；
- 按 `source_event_id` 的 ESS；
- proposal 类型计数。

ESS 使用 log-sum-exp 形式计算，避免极小权重先下溢为零。

## 建议消融

先使用已证实可发生碰撞的三个训练源事件：

```text
151569125
151578944
151894403
```

对每种 proposal 运行相同次数，比较：

1. 每 100 次 rollout 的碰撞数；
2. 独立源事件覆盖；
3. 碰撞车辆组合和碰撞时刻；
4. 训练可用 joint episode 数；
5. crash 权重的 ESS；
6. 最大单样本权重占比；
7. SUMO 成功率。

不能只比较碰撞数量：一种 proposal 即使生成更多碰撞，如果所有权重集中于一条轨迹，仍不适合正式 D2RL 训练。

## 验收条件

进入 D2RL 冒烟训练前，至少应确认：

- 每条新 episode 都有源事件元数据；
- 每个联合动作步都有正且有限的 `p`、`q` 与 log-weight；
- episode raw weight 即使下溢，log-weight 仍完整可用；
- `joint_pair` 的 `p/q` 与动作对概率一致；
- `factorized` 的联合 `p/q` 与两个边缘项的乘积一致；
- ESS 诊断自动产生；
- naturalistic、factorized、joint_pair 可以通过命令行选择。
