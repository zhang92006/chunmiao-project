# SHRP2 多智能体 K=2 冒烟训练

## 目的与边界

本阶段只验证 **factorized proposal** 生成的 SHRP2 K=2 episode 能被当前
`D2RLTrainingEnv` 和 PPO 读取：观测为 14 维、动作为 2 维、重放的 reward
可计算。它不是论文中的正式训练，也不用于报告泛化性能。

本次消融中，factorized 池有 14 条碰撞、136 条安全 episode；相同预算下
joint-pair 池有 10 条碰撞、140 条安全 episode。前者作为这次接口冒烟池，
因为碰撞覆盖更多，并包含 `CAV--BV_context` 的实际多车碰撞机制。

## 稳定抽样索引

`crash_weight_dict.json` 保持旧格式，用于兼容已有脚本。与它同时生成的
`crash_log_weight_dict.json` 保存每条 crash 的 `log_importance_weight`。
训练环境在该文件存在时使用：

```text
sampling_weight_i = exp(log_weight_i - max_j(log_weight_j))
```

这只去掉共同的数值尺度，不改变任意两条样本的权重比例，避免极小的原始
`p/q` 在 long rollout 中下溢后使抽样失真。正式实验必须保存这两个索引以及
`importance_weight_diagnostics.json`。

## 运行顺序

先为现有 factorized rollout 补写稳定索引（这一步会改写 ignored 的运行目录）：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'
$python = 'D:\Anaconda3\envs\D2RL\python.exe'
& $python -m scenario_reconstruction.prepare_training_data `
  'data_analysis\raw_data\shrp2_multibv_ablation_factorized_train3_x50_epsilon0001' `
  --multi_bv --agent_num 2 --include_safe_weight_dict
```

验证 K=2 数据接口：

```powershell
& $python -m scenario_reconstruction.validate_training_env `
  'data_analysis\raw_data\shrp2_multibv_ablation_factorized_train3_x50_epsilon0001' `
  --multi_bv_training --multi_bv_num 2
```

安装完整训练依赖后，运行两次 iteration 的 PPO 冒烟：

```powershell
& $python -m scenario_reconstruction.d2rl_smoke_train `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_factorized_smoke.yaml' `
  --stop_iterations 2
```

当前 `D2RL` 环境尚未安装 `ray[rllib]`，因此最后一条命令由本机完整训练环境
执行。日志应显示 K=2 环境；PPO 默认全连接网络会根据 Gym space 自动接收 14
维输入并输出 2 维连续动作。

## joint-pair 的限制

`joint_pair` 当前是**相关动作的碰撞生成 / 重要性采样 proposal**。它的
`p(a_1,a_2|s)/q(a_1,a_2|s)` 已被正确记录，但现有 PPO 动作是两个独立的
epsilon，不能从任意 PPO 输出反推出相同的二维相关 proposal。因此不能把
joint-pair 池直接当作当前 epsilon-policy 的 PPO 训练池，否则 reward 会与
策略动作脱钩。它保留为生成器消融和离线重要性分析；若要训练它，需要下一
阶段把策略输出改为 joint-pair proposal 的参数并重新定义对应的策略概率。
