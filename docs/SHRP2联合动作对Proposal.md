# SHRP2 多智能体联合动作对 Proposal

## 问题与证据

在 10 个具有相邻车道潜在封堵的训练源事件、每事件 10 次 rollout 的试验中，90 次 episode 形成 K=2 联合训练结构，但 5 次碰撞均来自 `NearCrash_151578944`。其余模板的 `pair_criticality.max_challenge` 为 1，且均存在可导致接触的动作对；因此问题不再是缺少联合场景，而是原先把动作对风险边缘化后分别采样，可能错过需要两辆 BV 同时执行的动作组合。

## 联合 proposal

对两个已选择的 BV，令自然驾驶动作分布为 `p1(a1)`、`p2(a2)`，联合临界度网格为 `c(a1,a2)`。实现使用：

\[
p(a_1,a_2)=p_1(a_1)p_2(a_2),\qquad
r(a_1,a_2)=\frac{p(a_1,a_2)c(a_1,a_2)}{\sum p(a_1,a_2)c(a_1,a_2)}
\]

并沿用项目中 `epsilon` 表示自然驾驶混合质量的约定：

\[
q(a_1,a_2)=(1-\epsilon)r(a_1,a_2)+\epsilon p(a_1,a_2).
\]

直接从 `q` 采样动作对，单步 importance weight 为：

\[
w=\frac{p(a_1,a_2)}{q(a_1,a_2)}.
\]

这不是把权重近似为两个边缘权重的乘积。日志中的 `weight_record` 和 `ndd_record` 带有 `proposal_type="joint_pair"`，并保存 `joint_naturalistic_probability`、`joint_proposal_probability`、动作对及 `critical_mass`，使该权重可复算。

## 兼容性与验收

- 只在恰好选择两个 BV 且联合临界度质量为正时启用；其他情形保持原有因子化 NADE 采样。
- `epsilon=0.0001` 时主要探索联合高风险动作，但仍保留自然驾驶分布的完整支持。
- D2RL 训练端识别 `joint_pair`，直接使用记录的 `p/q`，不会再将关联 proposal 错当作两个独立 epsilon 事件。
- 单模板 SUMO smoke test 已验证动作对、`p`、`q`、`p/q` 和 episode 权重均能贯通。下一步以 10 个独立封堵源事件进行相同 100 次预算的对照 rollout，比较碰撞的独立 `source_event_id` 数，而非只比较碰撞总次数。
