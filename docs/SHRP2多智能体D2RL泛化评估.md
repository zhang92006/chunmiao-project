# SHRP2 多智能体 D2RL 泛化评估

> 2026-09-21 更新：精确混合权重、有界对数奖励和有界 Beta 动作分布的 Seed19
> 已完成 20 轮训练。独立验证相对固定 epsilon 基线为 +0.52，29/38 逐条更优，
> 且动作越界/边界裁剪为 0/38。下一步固定协议补跑 seed 7、29；下文早期命令和
> legacy 结果仅用于保留实验演进记录，当前正式配置以
> `d2rl_train_shrp2_multibv_frozen5_exact_log_bounded_beta_seed19.yaml` 为准。

## 2026-09-19 五来源训练协议

执行状态更新：seed=19 的 20 轮训练及 checkpoint-20 的 train/validation 确定性评估
已完成。训练集平均奖励 55.04，验证集 -68.94（29/38 下限截断），验证固定 epsilon=0.5
基线为 12.60。多 seed 扩跑暂缓，先检查动作边界与概率/奖励契约。
详见 [五来源训练与验证结果](SHRP2五来源Seed19训练与验证结果.md)。下文命令保留供复现。

当前正式候选训练池为 `shrp2_multibv_frozen5_source_balanced_train_v1`，包含 86 条
crash episode、5 个互不相同的 train 源事件。训练按源事件均衡抽样；该分布用于学习
多样化策略，不作为自然碰撞率的无偏估计器。

逐 episode 的动作和奖励输出现由 `log_episode_rewards` 控制，默认关闭。94 项单元测试
和 2 轮安静模式冒烟训练均已通过；相同 seed 下第 2 轮 reward 仍为 `-0.8095`，证明
关闭日志没有改变训练数值。

下一项只运行 seed=19 的 20 轮训练：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'
$trainPython = 'D:\Anaconda3\envs\D2RLTrain39\python.exe'
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

& $trainPython -m scenario_reconstruction.d2rl_smoke_train `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_frozen5_source_balanced_seed19.yaml' `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_multibv_frozen5_source_balanced_seed19_20iter.log'
```

程序名保留 `d2rl_smoke_train` 是为了兼容已有入口；本配置和独立实验目录明确标记为
20 轮候选训练，不再与 2 轮 smoke 结果混放。此任务耗时较长，应在用户终端运行。

训练完成后才能执行下一项：固定 `checkpoint_000020`，在已有
`shrp2_multibv_validation14_x50_factorized_epsilon0001` 上关闭探索进行确定性评估。
该 validation 池有 38 条 crash，但只来自 3 个独立 validation 源事件，所以可用于
第一轮事件外泛化检查，不能作为最终稳定性能结论。seed=19 验证通过后，再建立另外
两个显式 seed 配置；在此之前不批量训练，避免在协议尚未验证时浪费计算。

## 当前结论

K=2 `single_trainable_critical` 已完成优化可行性验证：seed=7 的 20 轮 PPO
训练从平均 reward 50.12 提升到 95.20，最后五轮不再出现 -100 下限截断。
该结果只证明训练目标可优化，不代表对未见 SHRP2 事件具有泛化能力。

随后对训练池 14 条 crash 做逐条等权、关闭探索的 checkpoint 评估，均值仅
-31.59，9/14 被截断。其中源事件 151569125 的 4 条平均 89.48，而源事件
151578944 的 10 条平均 -80.02。原因是两组 log importance weight 约为 -61
与 -225，legacy importance 抽样几乎只会看到前一事件。训练日志的 95.20 因而是
对高抽样质量区域的有效优化，但不是训练事件覆盖率。

现有训练池来自 3 个 train 事件的重复仿真。验证必须使用桥接清单中独立的
14 个 validation 事件重新运行 SUMO，不能从这 3 个事件的 rollout 中随机切分。
test 的 11 个事件在模型与阈值冻结前保持锁定。

## 1. 生成 validation rollout

以下命令运行时间较长，在用于 SUMO 的 `D2RL` 环境中执行。proposal、epsilon
和每模板重复次数与当前 train 消融保持一致：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'
$python = 'D:\Anaconda3\envs\D2RL\python.exe'
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

& $python -m scenario_reconstruction.run_template_manifest `
  'data_analysis\raw_data\shrp2_multibv_sumo_templates_adaptivecritical_context_v2\bridge_summary.json' `
  --split validation `
  --repeats 50 `
  --epsilon 0.0001 `
  --proposal_mode factorized `
  --experiment_path 'data_analysis\raw_data\shrp2_multibv_validation14_x50_factorized_epsilon0001' `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_multibv_validation14_x50_factorized_epsilon0001.log'
```

运行完成后应有 `attempted=700`。碰撞数可以低于训练池，不能为了提高验证成绩
反复筛选 validation 模板或改变 epsilon。

为碰撞与安全 episode 生成审计索引：

```powershell
& $python -m scenario_reconstruction.prepare_training_data `
  'data_analysis\raw_data\shrp2_multibv_validation14_x50_factorized_epsilon0001' `
  --multi_bv --agent_num 2 --include_safe_weight_dict
```

## 2. 固定 checkpoint 确定性评估

评估器逐条读取 validation crash，每条只评估一次，不按 importance weight 重采样；
策略关闭探索（`explore=False`），并记录二维动作、reward、源事件和被拒绝原因。

```powershell
$trainPython = 'D:\Anaconda3\envs\D2RLTrain39\python.exe'
$trial = Get-ChildItem 'ray_results\SHRP2_MultiBV_K2_TrainableCritical_Smoke' -Directory |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$checkpoint = Join-Path $trial.FullName 'checkpoint_000020'

& $trainPython -m scenario_reconstruction.d2rl_checkpoint_evaluate `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_trainable_critical_smoke.yaml' `
  --checkpoint $checkpoint `
  --episode_pool 'data_analysis\raw_data\shrp2_multibv_validation14_x50_factorized_epsilon0001' `
  --expected_split validation `
  --output 'results\shrp2_multibv_seed7_validation.json'
```

评估器会拒绝混入 train/test 的目录。开发阶段也会主动拒绝 `--expected_split test`。

## 3. 来源均衡抽样消融

先运行来源均衡消融：每个 SHRP2 源事件获得相同总抽样概率，事件内部再均匀抽取
episode。该模式用于判断来源塌缩是否可改善，不能替代 legacy importance 估计器，
两者必须在论文中作为不同抽样目标并列报告。

```powershell
& $trainPython -m scenario_reconstruction.d2rl_smoke_train `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_source_balanced_smoke.yaml' `
  --stop_iterations 20 `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_multibv_k2_source_balanced_seed7_20iter.log'
```

完成后先在原 14 条 train crash 上逐条确定性评估。只有两个源事件的结果都不再
塌缩，才值得继续多 seed；否则需扩大 train 源事件及 collision pool，而不是继续
增加同一事件的 rollout 数量。

## 4. 三随机种子

seed=7 的结果已经存在。validation 流程确认可用后，再运行 seed=17 和 seed=29；
显式实验名避免覆盖或混淆：

```powershell
& $trainPython -m scenario_reconstruction.d2rl_smoke_train `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_trainable_critical_smoke.yaml' `
  --stop_iterations 20 --seed 17 `
  --experiment_name 'SHRP2_MultiBV_K2_TrainableCritical_Seed17' `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_multibv_k2_seed17_20iter.log'

& $trainPython -m scenario_reconstruction.d2rl_smoke_train `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_trainable_critical_smoke.yaml' `
  --stop_iterations 20 --seed 29 `
  --experiment_name 'SHRP2_MultiBV_K2_TrainableCritical_Seed29' `
  2>&1 | Tee-Object -FilePath 'data_analysis\logs\shrp2_multibv_k2_seed29_20iter.log'
```

每个 seed 使用其 `checkpoint_000020` 在同一 validation pool 上确定性评估。最终
至少报告三 seed 的 validation reward 均值与标准差、-100 截断比例、有效 episode
数量、独立源事件数量以及两个 BV epsilon 的分布。只有这些结果稳定后，才冻结模型、
阈值和 checkpoint，并解锁一次 test 评估。
