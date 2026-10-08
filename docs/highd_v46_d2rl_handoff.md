# V46 single-BV baseline 与双车跟驰 D2RL 交接

当前自然背景冻结在 annotated tag `nde-v46-frozen-20261003`（commit `178df83`）。
训练入口在分支 `feature/highd-v46-d2rl-handoff`；**需要 checkout 此分支，冻结标签本身不含后续训练代码**。
冻结模型说明见 [highd_v46_freeze.md](highd_v46_freeze.md)，实际首轮训练记录见
[highd_v46_single_baseline_run.json](../configs/highd_v46_single_baseline_run.json)。

## 拉取与安装

本入口直接运行 native V46，不需要 SUMO、Autoware submodule 或原始 highD tracks。
必需的拟合表已放进 Git LFS archive，解压后约 140 MB。
下面的 clone 先跳过历史 LFS 大文件，再只取本实验需要的资产：

```bash
git lfs install
git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false clone --single-branch --branch feature/highd-v46-d2rl-handoff https://github.com/zhang92006/chunmiao-project.git
cd chunmiao-project
git lfs pull --include="source_data/highd_paper_v46/runtime.zip" --exclude=""
conda create -n highd-v46-d2rl python=3.9 -y
conda activate highd-v46-d2rl
python -m pip install pip==23.0.1 setuptools==65.5.0 wheel==0.37.1
python -m pip install -r configs/requirements-highd-v46-d2rl.txt
python -m scenario_reconstruction.highd_v46_workflow verify
```

Python 3.9 / Ray 1.11 / Gym 0.21 / Torch 1.11 为当前兼容组合；Gym 0.21 需要上面的旧版打包工具。
命令均从仓库根目录执行。Ray 需要能够创建本地进程与 IPC，受限沙箱中可能停在初始化；在正常终端运行。
`verify` 校验冻结的代码、配置及 archive，默认解压至 `outputs/highd_paper_v46_assets`。
如果 checkout 后仍是 LFS pointer，会在这里报错；先完成 LFS 拉取。
解压目录已有不同内容时不会覆盖，请使用新的目录或处理本地冲突。

## 另一位研究者：直接开始双车训练

```bash
python -m scenario_reconstruction.highd_v46_workflow start --mode dual --episodes 512 --seed 1000 --iterations 100 --training-seed 7 --output outputs/v46_dual_seed7
```

这条命令依次校验 V46、采集双车干预轨迹、压缩为完整 critical sequences、训练两维 epsilon PPO，
最后在 `outputs/v46_dual_seed7/training` 保存 Ray checkpoint 与可移植 NumPy policy。
默认 CPU 单进程，网络为两个 64 单元的 tanh 隐层。输出目录必须是新目录，避免覆盖研究结果。
双车指 V46 当前自然跟驰配对中的两辆 BV，每个 maneuver 最多干预一对；配对不含 CAV。
仅有不满足支持/历史条件的车辆时，该步沿用自然动作。

复现 single-BV 的相同初始状态范围和 PPO 设置，只需要改模式：

```bash
python -m scenario_reconstruction.highd_v46_workflow start --mode single --episodes 512 --seed 1000 --iterations 100 --training-seed 7 --output outputs/v46_single_seed7
```

碰撞是稀有事件，512 场景不保证足够正样本。若采集没有 CAV 碰撞，训练会明确停止并保留数据。
继续收集预先确定的不重叠 seed 范围，并合并**所有**正负场景；不要只挑碰撞数据：

```bash
python -m scenario_reconstruction.highd_v46_d2rl --mode dual --episodes 512 --seed 1512 --output outputs/v46_dual_more
python -m scenario_reconstruction.highd_v46_workflow merge --sources outputs/v46_dual_seed7/collection outputs/v46_dual_more --output outputs/v46_dual_combined
python -m scenario_reconstruction.highd_v46_workflow start --mode dual --sequence-manifest outputs/v46_dual_combined/sequence_manifest.json --iterations 100 --training-seed 7 --output outputs/v46_dual_retry
```

`merge` 拒绝重复 seed 和不同目标/模式的数据；`--sequence-manifest` 从已有轨迹重新训练，
不是从旧 checkpoint 续训。单车数据不能换个标记拿来训练双车。

## 两组实验的共同定义

- 自然背景：冻结 V46 单车边缘分布和有支持的动态双车跟驰联合分布，按 1 s 选择 maneuver；
  换道固定 1 s、换道车纵向加速度为 0，物理步长 1/15 s。D2RL 不改冻结拟合表。
- CAV：用户确认的固定 IDM，不主动换道；accel=2、decel=4、emergency_decel=9、tau=0.2、min_gap=2、desired_speed=40。
  在同一初始化车流中，从有 115 m 内当前车道前车的候选车辆中均匀选取 CAV。
  这是使用仓库现有参数的 native IDM 适配，不声称与 SUMO IDM 数值逐步相同。
- 共同实验目标 P：在该固定 CAV 和初始化下的 V46 BV 过程。CAV 不作为自然配对的随机行动者，
  但仍是 BV 观察到的前车。这个 CAV 条件实验与全 NDE 车流评估有不同的目标身份。
- 事件：60 s 内**第一次任意车辆碰撞涉及 CAV**。任意碰撞即终止；BV-only 碰撞记为 competing negative，
  不能解释为 CAV 在剩余 60 s 内安全。两组采用完全相同事件定义。
- single：每步最多倾斜一个 BV 的条件分解因子。若其属于自然相关车对，搭档仍从原始条件分布采样，
  不把自然联合律改为独立；因此搭档的无条件边缘可能通过相关性随之变化。
- dual：每步最多倾斜一对的两个因子，
  `Q(a,b) = [eps1 P(a)+(1-eps1) H(a)] [eps2 P(b|a)+(1-eps2) H(b|a)]`。
  零 critical-mass 行回退至自然条件分布。
- H：决策抽样前的 4 s 确定性延拓 clearance proxy；候选车辆保持所枚举加速度，其他 BV 用当前均值，CAV 用 IDM 响应。
  它是挑选 critical actions 的启发式，未经碰撞概率校准。每步最多评估最近的 4 个候选，epsilon 不参与候选选择。
- epsilon 下界 0.05；不增加 P 的动作支持、不在抽样后修正动作。按实际一个/两个因子累积精确 log(P/Q)。
- 观测合同 `native_cav6_bv4x2_v1`：14 维，单车补零第二 BV；策略输出分别为 1/2 维。
  不加载旧 SUMO / 10 维 / 18 维策略权重。

完整轨迹只删去 Q=P 的非 critical 决策。离线训练最小化完整采集分布下的二阶矩：

`E_Qb[I_event * (P/Qb) * (P/Q_epsilon)]`。

奖励做固定正比例缩放，无 clipping。为避免稀有碰撞在绝大多数 batch 中缺席，
默认正负 critical sequences 各 50% replay；每条轨迹乘 `1/(N * replay_probability)`，
保留所有采集场景的原始分母 N。训练集上的目标下降不是独立测试效率结论。

自然模型身份：`106a9fbc795c75c9d4f60a880da7264cb626f9163e5ec073b720e6b48c6f9db5`。
共同 CAV 实验身份：`e5828a9740d9fa8f6270f7a80edb3b34cd28896f853e9f479779df7f970b8d0b`。
修改 CAV、事件、初始化或 collector 源码会产生新实验身份，不应混用旧数据或权重。

## 已完成的 single-BV 初始 baseline

采集 seed 1000–1511 共 512 个完整场景：491 个到达时限、19 个 BV-only competing collision、
2 个 CAV first collision，共 27,836 个 critical decisions。所有场景均保留。
PPO seed=7，100 iterations，110,042 replay timesteps；checkpoint 和 NumPy policy 已实际生成。
固定 epsilon=0.1 的 scaled empirical second moment 为 0.001953125060，
训练后为 0.001878026932，比值 0.96154976，训练集下降约 3.85%。
NumPy export 与训练策略输出最大绝对误差为 2.24e-8。

这只是首轮 baseline：正例只有 2 个，不能据此宣布收敛或在线方差降低。
在线运行检查使用新 seed 20000–20031：学习策略 32 个场景均无碰撞；固定 epsilon 组 31 个无碰撞、
1 个 BV-only competing collision，两组均未观察到 CAV 碰撞。这批样本不足以比较 IS 效率。
双车已完成实际双因子轨迹采集验证；完整双车 PPO 留给合作者按上面的命令运行。
模型 checkpoint、完整轨迹和运行日志保留在本地 ignored 目录；Git 提供代码、拟合资产、配置、seed、摘要，
合作者可以自行重新生成数据和训练。

## 新 seed 在线评估

```bash
python -m scenario_reconstruction.highd_v46_workflow evaluate --policy outputs/v46_dual_seed7/training --episodes 512 --seed 20000 --output outputs/v46_dual_evaluation
```

评估自动检查权重 checksum 和实验身份，仅需要 NumPy 运行推断。
结果含 first-CAV-event 的 IS 均值、二阶矩、标准误、CAV/BV-only/无碰撞计数。
比较固定 epsilon 时，在相同新 seed 范围用 `highd_v46_d2rl --mode dual` 采集；
纯自然抽样使用 `--mode natural`。报告效率还需要足够事件和独立样本；零观测事件不是零风险证据。

检查新干预算子的最小数学回归可运行：

```bash
python -m unittest discover -s tests -p test_highd_v46_d2rl.py
```
