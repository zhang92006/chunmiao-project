# SHRP2 多智能体干预预算与训练池审计

## 结论

在固定 Seed19 checkpoint、14 个 validation 模板、每模板 10 次重复和相同基础随机种子的
条件下，`online_intervention_budget=10` 是当前推荐配置。它与不限预算得到完全相同的
8 个碰撞 episode 和两个碰撞来源，但把在线策略推理步数从 3295 降到 775，减少
76.5%，并消除了 raw weight 下溢。

该结果证明前 10 个有效关键决策已经覆盖当前策略产生碰撞所需的干预，但没有证明统计
效率提高。预算 10 的碰撞贡献 ESS 仍为 1.00，按当前 140 次样本计算的普通重要性采样
相对 95% 半宽约为 196%，远未达到原 D2RL 工作采用的 30% 相对半宽停止标准。

## 干预预算消融结果

| 每个 episode 的干预预算 | CAV 碰撞 | 原始碰撞率 | 条件加权均值 | 碰撞贡献 ESS | inferred 步数 | budget exhausted 步数 | raw 权重状态 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1/140 | 0.71% | 2.37e-4 | 1.00 | 110 | 1229 | 140 normal |
| 5 | 1/140 | 0.71% | 1.96e-5 | 1.00 | 455 | 1298 | 140 normal |
| 10 | 8/140 | 5.71% | 1.55e-5 | 1.00 | 775 | 1056 | 140 normal |
| 不限 | 8/140 | 5.71% | 1.58e-5 | 1.00 | 3295 | 不适用 | 129 normal，11 underflow |

预算 10 与不限预算的 8 个碰撞逐 episode 完全一致，其中 1 个来自事件 `137038878`，
7 个来自事件 `138608096`。这不是两个样本中碰巧得到相近的总数，而是配对运行中
`132` 个 episode 同为安全、`8` 个 episode 同为碰撞。

预算 1 和预算 5 都只保留事件 `137038878` 的同一个碰撞，说明前 5 次干预不足以触发
事件 `138608096` 中的碰撞链。预算从 10 增加到不限没有增加碰撞来源或碰撞 episode，
只增加了后续动作干预和权重乘积长度。

## 如何解释 ESS 仍为 1

8 个碰撞并没有形成 8 个等贡献样本。预算 10 的碰撞 log importance weight 分布跨度很大，
其中事件 `137038878` 的一个碰撞权重远高于其余 7 个碰撞，因此加权风险均值几乎由它
独自决定。减少后续干预避免了浮点下溢，却没有改变这个主导样本，所以 ESS 没有提高。

因此本次实验改善的是计算成本、闭环一致性和数值稳定性，不是风险估计精度。后续不能
用“8 个原始碰撞”直接声称统计效率提高，必须继续报告归一化权重集中程度、ESS 和相对
置信区间半宽。

## 历史 86 条训练碰撞账本审计

对五来源训练池中的 86 条碰撞逐条执行了三路重算：

1. 训练池索引中的 `log_weight`；
2. episode 顶层 `log_importance_weight` 与逐步 `log_probability_step_info` 求和；
3. `weight_step_info` 中每步实际联合权重取对数后求和。

索引、episode 总 log 权重和逐步 log 概率账本三者为 86/86 一致；但是第三条 raw 权重
路径只在 66 条上可重建且一致，最大绝对误差为 `1.07e-13`。其余 20 条包含某一步
自然驾驶概率 `p=0`、联合权重 `0`，该步没有进入旧 log 账本，导致 raw episode 权重为
0、记录的总 log 权重却仍为有限值。这不是正常的长乘积浮点下溢，而是 proposal 采到了
自然分布支持集以外的动作。

| 历史批次 | 原碰撞数 | 有效数 | 无支持动作数 |
|---|---:|---:|---:|
| `shrp2_multibv_train49_x50_factorized_epsilon0001` | 15 | 15 | 0 |
| `shrp2_frozen_collision_proposal_epsilon09_confirm_v1` | 27 | 10 | 17 |
| `shrp2_frozen_collision_epsilon09_source151570590_extra_v1` | 44 | 41 | 3 |

20 条问题 episode 中，16 条来自事件 `61385336`，4 条来自事件 `151570590`。按当前
`single_critical` 规则回放，12 条训练 episode 选中的恰好是零权重决策时刻，其中事件
`61385336` 占 10 条。因此现有 checkpoint 可保留为工程回归基线，但不应作为最终论文
模型；正式模型必须在清洗或重新生成后的训练池上重训。

该检查已经固化为可重复命令。每次组装或扩充训练池后运行：

```powershell
& $python -m scenario_reconstruction.training_pool_ledger_audit `
  'data_analysis\raw_data\shrp2_multibv_frozen5_source_balanced_train_v1\crash_log_weight_dict.json' `
  --output 'data_analysis\raw_data\shrp2_multibv_frozen5_source_balanced_train_v1\training_pool_ledger_audit.json'
```

命令在任一路径不一致、动作没有自然分布支持或 episode 缺失时返回非零状态，适合放到
后续训练数据生成流程中。

已生成不复制 episode 的清洗索引
`shrp2_multibv_probability_audited_train_v2`：保留 66 条、4 个来源，剔除 20 条；事件
`61385336` 因 16/16 均无自然分布支持而暂时退出正式训练池。清洗池整体权重 ESS 为
1.72，最大归一化权重占 73.0%，说明有效样本仍高度集中。

代码入口已经加入两道防线：新的 multi-BV training-ready 索引会拒绝非正逐步权重，
训练池合并默认遇到此类 episode 直接失败。冻结碰撞 proposal 在采样前会按当前自然驾驶
动作支持集掩码并重新归一化，从源头避免再次生成 `p=0, q>0` 的训练动作。

事件 `61385336` 的修正后短仿真已通过：2/2 成功运行，均为安全 episode；每条记录
15 个实际权重步，最小联合权重分别为 `6.86e-9` 和 `2.29e-6`，零权重步均为 0，累计
log 权重为有限值。短测试只验证接口和概率支持，不用于判断碰撞率。

## 下一步

1. 冻结 `online_intervention_budget=10`，后续闭环 validation 均使用该配置。
2. 用支持集掩码后的碰撞生成器重新生成事件 `61385336`，并扩展新的独立事件来源。
3. 每个新增来源先通过概率账本审计，再记录碰撞数、log weight 分布和来源内 ESS。
4. 恢复至少 5 个有效训练来源后重新训练；随后训练与闭环预算 10 一致的连续多步策略。
5. 用等仿真预算比较单 BV、双 BV 固定 proposal 和双 BV 学习 proposal；validation 规则冻结后，
   最后只运行一次保留 test。

事件 `61385336` 的下一轮 100 次独立重采样为长命令，由用户终端运行：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'
$python = 'D:\Anaconda3\envs\D2RL\python.exe'
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

& $python -m scenario_reconstruction.run_template_manifest `
  'data_analysis\raw_data\shrp2_frozen_collision_proposals_v1\frozen_collision_proposal_manifest.json' `
  --experiment_path 'data_analysis\raw_data\shrp2_frozen_collision_support_mask_61385336_train100_v2' `
  --split train `
  --source_event_id 61385336 `
  --repeats 100 `
  --proposal_mode factorized `
  --epsilon 0.99 `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_frozen_collision_support_mask_61385336_train100_v2.log'
```

验收顺序是：运行成功数、training-ready 碰撞数、零支持拒绝数、碰撞 log 权重和来源内
ESS。若修正后碰撞降为 0，说明旧事件的碰撞主要依赖自然模型不支持的动作，应放弃该
proposal 并重新进行受支持动作的 CEM 搜索，而不是放宽账本规则。
