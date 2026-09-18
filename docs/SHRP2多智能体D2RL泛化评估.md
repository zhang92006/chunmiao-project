# SHRP2 多智能体 D2RL 泛化评估

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
