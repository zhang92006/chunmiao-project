# SHRP2 多智能体 K=2 冒烟训练

## 目的与边界

本阶段先验证 **factorized proposal** 生成的 SHRP2 K=2 episode 能被当前
`D2RLTrainingEnv` 和 PPO 读取：观测为 14 维、动作为 2 维、重放的 reward
可计算。`single_critical` 基线在每个 episode 中选择联合关键性最高的一个
时刻，继续使用原始 D2RL 的单决策 reward；源 episode 文件不会被改写。它不是
论文中的正式训练，也不用于报告泛化性能。

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

原有 D2RL PPO 代码使用 `ray.rllib.agents` 和 `PPOTrainer`，对应 Ray 1.11。
该版本在 Windows 上不支持当前的 Python 3.10 运行时，因此不要在用于 SUMO 的
`D2RL` 环境中安装 Ray 2.x；应建立独立的 Python 3.9 训练环境：

```powershell
conda create -n D2RLTrain39 python=3.9 pip -y
conda run -n D2RLTrain39 python -m pip install --force-reinstall `
  "pip==23.2.1" "setuptools==65.5.0" "wheel==0.38.4"
conda run -n D2RLTrain39 python -m pip install --no-build-isolation `
  "gym==0.21.0"
conda run -n D2RLTrain39 python -m pip install `
  "PyYAML==6.0.1" "ray[rllib]==1.11.0" `
  "torch==1.11.0" "numpy==1.23.1" "protobuf==3.20.3"
```

必须按以上三条安装命令的顺序执行。`gym==0.21.0` 的旧安装元数据与新版本
`pip/setuptools/wheel` 不兼容；`--no-build-isolation` 保证它使用刚刚固定的构建
工具，而不是在临时构建环境中重新安装最新版。
Ray 1.11 的生成 protobuf 文件也只能与 `protobuf==3.20.3` 兼容；若已安装
过新版 protobuf，重新执行第三条命令即可降级修复。

安装完成后，运行两次 iteration 的 PPO 冒烟：

```powershell
$trainPython = 'D:\Anaconda3\envs\D2RLTrain39\python.exe'
& $trainPython -m scenario_reconstruction.d2rl_smoke_train `
  --yaml_conf 'd2rl_training\d2rl_train_shrp2_multibv_single_critical_smoke.yaml' `
  --stop_iterations 2
```

日志应显示 K=2 环境；PPO 默认全连接网络会根据 Gym space 自动接收 14
维输入并输出 2 维连续动作。该启动器固定 CPU 模式（`num_gpus=0`），因此不要求
安装 NVIDIA 驱动或提供 `nvidia-smi` 命令。

`legacy_all_steps` 仍是默认模式，用于复现旧行为；它会拒绝多次对抗动作并返回
零 reward。`single_critical` 是新的 K=2 基线：在可训练时间步中按最大联合
criticality 选择一个动作，同分时选更晚的仿真时刻。它避免直接删除旧保护而造成
多步 importance weight 连乘下溢和 reward 饱和。

## joint-pair 的限制

`joint_pair` 当前是**相关动作的碰撞生成 / 重要性采样 proposal**。它的
`p(a_1,a_2|s)/q(a_1,a_2|s)` 已被正确记录，但现有 PPO 动作是两个独立的
epsilon，不能从任意 PPO 输出反推出相同的二维相关 proposal。因此不能把
joint-pair 池直接当作当前 epsilon-policy 的 PPO 训练池，否则 reward 会与
策略动作脱钩。它保留为生成器消融和离线重要性分析；若要训练它，需要下一
阶段把策略输出改为 joint-pair proposal 的参数并重新定义对应的策略概率。
