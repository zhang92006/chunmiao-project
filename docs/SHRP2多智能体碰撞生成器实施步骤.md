# SHRP2 多智能体碰撞生成器实施步骤

## 当前结论

已有 49 个训练模板、每个 50 次 rollout，共 2,450 次，其中 15 次碰撞只来自 3 个独立源事件。它证明现有链路可以生成碰撞，但场景覆盖和有效样本量不足，不能靠继续重复同一个 epsilon 解决。

碰撞生成器的第一阶段不修改 SHRP2 原始事实，也不强制写入碰撞。它在固定场景上搜索 NADE 的采样分布，使“本来可避免但具有风险的动作组合”更容易被采到，同时继续记录实际执行动作下的精确自然概率 `p`、提议概率 `q` 和 `p/q`。

## 已完成的接口改动

1. `BV_primary` 和 `BV_context` 支持独立 epsilon，并按车辆 ID 而不是候选数组顺序赋值。
2. factorized 模式继续使用每车概率乘积形成联合 `p/q`，日志增加 `epsilon_by_bv_id`，可以检查实际动作对应的参数。
3. 从已有 2,450 次结果生成逐模板诊断，固定选择 10 个 train 场景：3 个已碰撞正对照和 7 个高潜力/冲突类型补充场景。
4. 搜索输出自动写入 `SEARCH_ONLY_DO_NOT_TRAIN.json`。这些样本参与了参数选择，禁止直接进入正式 D2RL 训练池。

固定 pilot 的事件 ID 为：

```text
151578944, 151569125, 151586341,
61385336, 29712180, 151894403, 151570590,
151550414, 134902016, 138353215
```

其中前三个是正对照。pilot 文件位于：

```text
data_analysis/raw_data/shrp2_collision_generator_pilot_v1/
  per_template_diagnostics.json
  collision_generator_pilot_manifest.json
  collision_generator_pilot_summary.json
```

## 下一步：运行双 epsilon 小网格

在 `scenario_reconstruction` 目录执行：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'
$python = 'D:\Anaconda3\envs\D2RL\python.exe'
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

& $python -m scenario_reconstruction.shrp2_collision_generator_search `
  'data_analysis\raw_data\shrp2_collision_generator_pilot_v1\collision_generator_pilot_manifest.json' `
  --output 'data_analysis\raw_data\shrp2_collision_generator_epsilon_search_v1' `
  --epsilon_primary 0.000001 --epsilon_primary 0.0001 --epsilon_primary 0.01 `
  --epsilon_context 0.000001 --epsilon_context 0.0001 --epsilon_context 0.01 `
  --repeats 10 `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_collision_generator_epsilon_search_v1.log'
```

总预算是 `10 场景 × 9 组参数 × 10 次 = 900` 次 SUMO rollout。完成后主要查看：

```text
data_analysis/raw_data/shrp2_collision_generator_epsilon_search_v1/epsilon_search_summary.json
```

## 验收与分支判断

- 先看原始碰撞率和发生碰撞的独立源事件数，不只看 `training_ready_crashes`。
- 再看每组 proposal 的 log weight/ESS，不能用极端退化权重换取表面碰撞率。
- 若 `1e-6` 与 `1e-4` 几乎无差别，说明当前 proposal 已接近纯关键性分布；继续压低 epsilon 没有意义，下一步应搜索关键动作分布或动作时序（例如 CEM），而不是扩大 epsilon 网格。
- 若某组明显更好，先冻结该组参数，再换新目录和新随机样本正式重采样。搜索阶段的 900 条不能并入训练集。
- 正式重采样后才运行 `prepare_training_data`，再接入 source-balanced D2RL；validation/test 不参与搜索。

## 后续关键节点

1. **P1：参数接口验收**：检查 episode 内 `epsilon_by_bv_id`、联合 `p/q` 与执行动作一致。
2. **P2：900 次 pilot 完成**：决定 epsilon 是否仍是有效搜索变量。
3. **P3：动作级碰撞生成器**：若 epsilon 饱和，使用受限 CEM 搜索主车/上下文车动作及动作时刻，并为冻结后的新 proposal 建立可计算的 `q(a|s)`。
4. **P4：冻结后重采样**：生成独立、未参与调参的 train episodes；验证集只做最终泛化评估。
5. **P5：接入多智能体 D2RL**：采用 source-balanced 采样，报告 source-macro、micro、碰撞覆盖数和 ESS。

注意：CEM 搜索轨迹本身通常没有可直接用于无偏估计的解析 `q`，所以不能把“最优搜索轨迹”直接当正式 D2RL 数据。必须把搜索结果固化成可采样、可计算概率的 proposal 后重新生成。

## epsilon pilot 结果与动作级 CEM

9 组 epsilon 共 900 次 rollout 得到 32 次碰撞，但全部来自 3 个正对照源事件；7 个搜索场景共 630 次仍为零碰撞。因此停止扩大 epsilon 网格，转入动作与时刻搜索。

动作级 CEM 对每个零碰撞源事件独立搜索以下离散变量：

- 两车共同的基本干预开始时刻；
- context BV 相对 primary BV 的动作延迟；
- 两车各自的动作持续时间；
- primary BV 的纵向动作 ID；
- context BV 的纵向或换道动作 ID。

搜索评分优先级为目标 CAV 碰撞，其次为最小 TTC 和最小距离。每代只用高分 elite 更新下一代分类分布。所有候选模板及 episode 都明确标记：

```text
collision_search_only=true
not_for_d2rl_training=true
```

训练数据准备器会硬性排除这些记录，即使它们包含碰撞或看起来具有完整联合字段。

全量第一轮共 `7 × 4 × 16 = 448` 次 SUMO rollout：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'
$python = 'D:\Anaconda3\envs\D2RL\python.exe'
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

& $python -m scenario_reconstruction.shrp2_collision_action_cem `
  'data_analysis\raw_data\shrp2_collision_generator_pilot_v1\collision_generator_pilot_manifest.json' `
  --output 'data_analysis\raw_data\shrp2_collision_action_cem_v1' `
  --generations 4 `
  --population 16 `
  --elite_fraction 0.25 `
  --smoothing 0.25 `
  --rollouts_per_candidate 1 `
  --seed 20260919 `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_collision_action_cem_v1.log'
```

程序默认跳过 3 个正对照，只搜索 7 个零碰撞场景，并在每个 generation 后增量保存结果。最终查看：

```text
data_analysis/raw_data/shrp2_collision_action_cem_v1/cem_search_summary.json
```

若至少一个新源事件被搜索到目标碰撞，下一步不是直接训练，而是把各源事件的 elite 分布拟合成显式 categorical proposal，并冻结后重新采样。若仍为零，则应扩大动作时序表达能力或加入有边界的 CAV 反应延迟实验，不应直接把故障轨迹混入默认训练池。
