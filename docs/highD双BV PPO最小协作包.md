# highD 双 BV 强化学习：最小协作包

本包让合作者可以阅读和调整双 epsilon PPO 策略、模型结构和优化参数，并在相同关键序列上运行训练。代码发布在 GitHub 分支 `feature/highd-ppo-training-share`，包含新策略和旧版策略选项。

## 训练入口

PowerShell 包装器是 `configs/run_highd_cav_first_sequence_train.ps1`。运行前需要准备兼容的 `sequence_manifest.json`、Python 3.9 训练环境和一个全新输出目录。包装器可通过 `SequenceManifest`、`OutputRoot`、`Iterations`、`Seed`、`PpoConfig` 参数指定这些项目。

PPO 配置模板是 `configs/highd_dual_bv_meanprecision_ppo_v1.json`。合作者可调整 `model.fcnet_hiddens`、`model.fcnet_activation`、`model.vf_share_layers`，以及 `ppo` 中的 batch 大小、学习率、SGD 次数、折扣因子、熵系数、PPO clip、KL 和价值损失等字段。输出目录会保留训练配置、种子、数据/代码校验值、每轮指标和 checkpoint；同一输出目录不会被覆盖。

策略选择为 `mean_precision` 或 `legacy_beta`。建议先保留默认的 `initial_epsilon=0.1` 与 `initial_precision=80`，让未训练策略能与固定基线直接核对，再单独改动 PPO 超参数。每轮的 episode reward 是训练时随机动作获得的回报；策略比较应使用记录的确定性经验二阶矩，且不能把训练池结果当作独立验证。

## 训练数据

训练需要符合 `highd_first_collision_critical_sequence_v1` 契约的 80 条行为序列：42 条含关键决策、38 条没有可训练决策；80 条都保留在风险分母中。参考训练文件 SHA-256 为 `08ba49fc06262f9bb969fa234158f1468ff827f3e61113f32bf5a276a15ad56f`，大小约 3.8 MiB。

轨迹/序列数据不放入这个源码包。团队成员须从团队获准的数据存储获取同一份 `sequences.json`，或使用相同格式和目标契约的数据；不能只拿 manifest。使用 `configs/highd_cav_first_sequences_train80_manifest.template.json` 建立本机 manifest，将 `data_path` 改为本机序列文件路径。训练入口会核对文件 SHA-256 和数据契约。新数据若内容不同，必须相应更新 SHA 和数据摘要，并按新的实验记录管理。

数据包含 SUMO 运行派生的交通轨迹与动作概率记录，因此本包只发布代码、配置、schema 摘要与校验值；不会把原始 highD 数据、派生轨迹、训练输出或 checkpoint 推到 GitHub。

## 代码构成

| 路径 | 用途 |
|---|---|
| `scenario_reconstruction/highd_critical_sequence_train.py` | PPO 序列训练、配置读取和逐轮记录 |
| `scenario_reconstruction/highd_mean_precision_policy.py` | 可调的 Beta 均值/集中度策略分布与固定基线初始化 |
| `scenario_reconstruction/highd_sequence_policy_diagnostic.py` | checkpoint 输出和训练池目标复核 |
| `scenario_reconstruction/highd_sequence_io.py` | 配置、摘要和校验值读写 |
| `d2rl_training/highd_critical_sequence_env.py` | 关键序列回放环境与原二阶矩目标 |
| `d2rl_training/conditional_chain.py` | 保持两辆 BV 条件联合 P/Q 的概率重放 |
| `d2rl_training/highd_event_contract.py` | CAV 首次碰撞事件语义 |
| `scenario_reconstruction/highd_mean_precision_policy.py` | 新策略和可复现的旧 Beta 策略对照 |
| `tests/test_highd_mean_precision_policy.py` | 新策略分布、初始化和梯度检查 |

新策略让 epsilon=0.1 可表示，并从这一固定策略初始化。训练结果显示最终模型仍略差于固定基线，故当前 checkpoint 供实验调整使用，不应称为已验证的性能提升或闭环策略。

## 环境

本训练接口使用 Windows、Python 3.9、Ray/RLlib 1.11、PyTorch 1.11、Gym 0.21、NumPy 1.23.1 和 protobuf 3.20.3。精确版本记录在 `configs/requirements-highd-sequence-ppo.txt`。Gym 0.21 对较新的 setuptools 构建工具敏感，安装时需遵循该文件的 setuptools/wheel 约束。这个离线关键序列训练不需要启动 SUMO；若重采行为数据，需另行配置项目仿真环境。
