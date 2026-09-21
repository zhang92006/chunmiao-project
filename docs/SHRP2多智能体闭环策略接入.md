# SHRP2 多智能体闭环策略接入

## 状态与实验边界

2026-09-21：Seed19 的有界 Beta checkpoint-20 已通过导出一致性验证并接入 SUMO。
训练、验证三 seed 的离线结果见《SHRP2五来源Seed19训练与验证结果》。此次工作验证的是
在线执行和概率日志契约；尚未证明学习策略在闭环中提高了碰撞产出或估计效率。

Beta 的动作支持域本来就有界，零越界部分来自分布设计，不能单独证明策略最优或泛化。
阶段验收包含：102 项全量测试通过；新增完整分母和累计权重审计测试后，相关 12 项
定向测试通过；三组真实 SUMO 冒烟审计通过，PowerShell 脚本语法和 14 模板预检通过。

离线训练每条 crash 只选择一个事后关键时刻。当前在线策略在每个满足关键性条件且有
两辆受控 BV 的时刻调用，属于新的连续决策评估，不能把离线 reward 当作闭环碰撞率。
不依据未来轨迹选择在线决策时刻。缺少第二辆 BV 时显式记录自然驾驶回退。

## 实现

- `d2rl_export_online_policy` 在训练环境恢复 RLlib checkpoint，将网络和 Beta 均值映射
  一起导出为 TorchScript，输出直接为两个 epsilon。SUMO 环境无需导入 Ray。
- 导出限制为 14 维观测、双动作、NoFilter、非循环、有界 Beta 模型；保存模型及
  checkpoint 的 SHA256。运行器检查模型校验和。
- 车辆顺序采用训练日志的“候选列表中的选中子集顺序”，策略输出以 BV ID 回填，避免
  按关键性排序后误交换两辆车的动作。
- `--frozen_epsilon_source runtime` 保留冻结 CEM 的动作分布和生效时间窗，epsilon 则
  取运行器或在线策略的值。默认 `template` 保持历史实验行为；在线模式要求 runtime。
- 推理使用确定性 Beta 均值；实际 BV 动作仍从 `q=epsilon*p+(1-epsilon)*c` 采样，
  逐步累计实际 `p/q`。策略 epsilon 不是加速度，也不是直接回放训练动作。
- 日志保存推理观测、BV ID、策略 epsilon、采样动作及对应 p/q/c。全量保存安全 episode。
- `--simulation_seed` 同时初始化 Python、NumPy 和 SUMO；每个 episode 使用基础种子
  加 episode ID。各组使用相同起始种子，但策略不同后随机数消耗和轨迹仍会分歧。

## 已完成冒烟

导出检查使用 86 个训练观测和 32 个 [-5,5] 范围探测观测，共 118 个；与 RLlib
`explore=False` 的最大动作误差为 0。

| 检查 | 两个 validation 模板 | 一个冻结 CEM train 模板 |
|---|---:|---:|
| 完整执行 | 2/2 | 1/1 |
| CAV 碰撞 | 0 | 1 |
| 在线推理次数 | 54 | 11 |
| 审计 BV 动作数 | 108 | 22 |
| q 重建最大绝对误差 | 5.83e-15 | 2.94e-15 |
| p/q 权重重建误差 | 0 | 0 |
| 在线/训练日志观测差异 | 0 | 0 |

固定 epsilon=0.5 的两个相同 validation 模板也已执行。冻结 train 场景只用于验证
覆盖 epsilon 的接口是否正常，不混入 validation 结果。冒烟规模不足以比较性能。

本地结果目录：

- `data_analysis/raw_data/shrp2_online_seed19_smoke_v3`
- `data_analysis/raw_data/shrp2_online_seed19_frozen_smoke_v2`
- `data_analysis/raw_data/shrp2_fixed05_smoke_v1`
- 模型：`data_analysis/raw_data/shrp2_online_policy_seed19_v1/policy.pt`（附同名 JSON）

## 下一步：同预算 validation pilot

从仓库根目录运行 `configs/run_multibv_closed_loop_pilot.ps1`，通过 `-SimulationPython`
指定安装 SUMO 的 D2RL 环境解释器。脚本按顺序运行下面三组，每组固定同一 14 个
validation 模板、每模板 10 次、基础仿真种子 52000，共 420 次。使用 Seed19 是延续
首个固定候选，未按 validation 挑选最佳训练 seed。

1. 在线 Seed19 策略，因子分解 proposal。
2. 固定 epsilon=0.5，同一 proposal。
3. 固定 epsilon=0.0001，复现早期 validation 的强干预提议设置。

这批 validation 模板没有冻结 CEM 分布，因此第三组是现有关键性 proposal 基线，
不能把它命名为“CEM 生成器”。未对 validation 进行 CEM 搜索。test 保持锁定。
脚本在输出目录已存在、仿真失败、场景数量不符或审计失败时停止；不要覆盖失败结果。

每组生成 `manifest_run_summary.json` 和 `closed_loop_audit.json`。后者检查完整分母、
动作/观测对齐、q 及 p/q 重建、episode 原始权重与逐步 log 权重一致性，并报告：

- 原始 CAV 碰撞数及比例（用全部执行 episode，而非 training_ready 子集）；
- 普通重要性加权碰撞贡献均值、其 log 值；
- 碰撞贡献 ESS（零碰撞时为零，不能解释为零风险）；
- 实际在线调用与回退次数、概率审计误差。

重要性加权结果的目标仅是这些固定初始模板组成的场景混合下、当前 NDD 基准行为的
条件碰撞概率；没有 SHRP2 初始状态选择概率的校正，不能声称估计真实道路总体碰撞率。
普通重要性估计有限样本时可以大于 1（冻结单样本冒烟为 1.80），不能截断后冒充精确概率。
同理，ESS 仅是权重诊断，不是独立场景数量，也不能单独证明估计效率更高。

pilot 完成后再按源事件比较采样效率、失败、权重集中程度，补齐加权估计的不确定性。
若出现长时域观测漂移或权重恶化，先诊断单关键步训练与连续在线执行的差异，再决定
扩大独立训练源或训练多步策略。当前三 seed 的小标准差只说明固定数据上的优化随机性小。
