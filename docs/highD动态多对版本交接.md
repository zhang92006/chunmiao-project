# 动态多对 NDD 与双 BV D2RL 交接

日期：2026-09-27。分支：feature/highd-dynamic-pairs。本文不放命令行，操作参数见脚本和交接聊天。

## 现在可以交给训练同伴什么

本交接固定当前动态多对自然驾驶研究基线，不继续拟合 NDD。每步允许多个不重叠的自然双车对，关系随时序更新；最多对其中一对执行有偏采样。双车道、20–40 m/s 支持域、已有拟合核、动作编码和执行器保持不变。

冻结契约为 configs/highd_dynamic_pairs_freeze_v1.json；自然目标 SHA256 为 c2942781b96d901823fd291f7167bfc1ba8f6d0129eff47404f963f7c876cb64，四配置场景的实验目标为 24743c972160a9b57f76fd630ed4017573d3425ed6eae335030aa0d946237629。

这是可用于研究训练的冻结参考，不是实证自然驾驶真实性验收。前车、并排车及身份不匹配的换道后车关系仍有明确独立回退。不要为了提高训练奖励直接修改 P 或关闭校验。

## 交付文件及放置方式

接收者需要三份文件：

1. 动态版本源码 ZIP，或对应 Git 分支。
2. 原 highd_frozen_assets.zip，提供不变的 NDD 模型和原校验资产；已有正确版本可复用。
3. 新 highd_dynamic_assets.zip，提供新增仿真、序列和冒烟 checkpoint。

先解压源码，再把两个资产 ZIP 解压到源码根目录。最终 HANDOFF.md、portable_assets/、dynamic_assets/、scenario_reconstruction/ 应在同一层。Git 分支不包含私有资产；只拉代码不够。源码包不需要原始 highD/SHRP2 CSV 或旧 NDD 大表。

原资产有 222 个文件；新资产白名单有 94 个文件，约 351 MB 解压数据，包含 40 次仿真 episode 及摘要、采集协议、20 条行为序列及审计、两轮训练的记录和 checkpoint。不收集其他工作目录，不包含原始驾驶数据。资产仅走团队获准渠道，不推 GitHub；只加载可信 checkpoint。

公开资产锁文件记录 ZIP 和内部清单 SHA256。路径转换在内存中完成，不重写数据、模型或 checkpoint 的原始字节。接收者不需要 Windows G 盘目录。

## 操作入口

统一入口为 scenario_reconstruction.highd_dynamic_handoff。

| 子命令 | 用途 |
|---|---|
| verify | 核对两套资产和冻结源码、验证当前动态目标；可检查 SUMO |
| collect | 当前动态目标下采集自然与固定双 epsilon 行为数据 |
| prepare | 审计配对历史、单对干预、P/Q，导出 CAV-first 关键序列 |
| train | 从固定 epsilon 初始化训练，默认读取新 20 条行为序列，可指定新序列清单 |
| evaluate | 恢复同目标 checkpoint，在线执行两个 epsilon，审计配对、概率和策略输出 |

Windows 脚本是 configs/run_highd_dynamic_handoff.ps1，Mac/Linux 是 configs/run_highd_dynamic_handoff.sh。两者有 verify、smoke、train、collect、prepare、evaluate、compare 任务。

smoke 会先校验、使用新数据从头训练两轮，再加载本机新 checkpoint 做一个相邻接近场景的闭环检查。compare 默认对四场景使用 seed2000 比较自然、固定干预和附带的两轮 checkpoint；它只是技术比较，不能作为正式策略效果结论。自定义训练后的模型通过 evaluate 的 training-root 指定。同一个输出目录不可重复覆盖。

保持 Python 3.9、Ray 1.11、Torch 1.11、Gym 0.21 和 SUMO 1.27.0；沿用已有 requirements 与 Mac 环境准备脚本，无需重新拟合或安装新的机器学习框架。M4 尚未实机验证，接收者先跑 smoke。不要在同一个解释器内并发调用路径适配和在线入口；独立进程可分开运行。

## 新旧目标与 checkpoint 不得混用

原 highd_portable 服务旧“最多一个自然对”的目标，新 highd_dynamic_handoff 服务动态多对目标。evaluate 在加载 Ray 或运行 SUMO 前检查目标，不允许旧 checkpoint 因为输入同样是 14 维就进入新评估。

自然目标、首次碰撞事件定义和初始场景混合都需匹配。仅跑一个场景必须显式标记 smoke；不能把子集结果冒充四场景总体。更换随机种子不等于获得独立真实场景来源。

新默认 checkpoint 是 outputs/dynamic_pairs_ppo_smoke_seed7_v1 的第 2 轮。它只证明可加载、可执行，不是推荐最终策略。train 是从头训练，不是恢复该 checkpoint 接着优化。原第 200 轮单对模型仍在原资产中供旧入口使用，新入口不会默认选择它。

## 数据口径与当前证据

训练数据来自四个配置场景、seed1000–1004，每组 20 条，已用于本地训练冒烟，因此不能再当作未见验证集。20 条行为轨迹中 10 条含关键决策，共 345 个关键状态；其余 10 条作为暴露保留，非空抽样修正为 0.5。不要删除空序列后不修正分母。

事件仍为 8 秒内 SUMO 首次碰撞时刻是否涉及 CAV。BV-only 是竞争事件，不等于全时域安全。自然组 CAV 首次碰撞 0/20、行为组 7/20；碰撞贡献 ESS 约 1.0003，一条轨迹贡献约 99.984%，尚未证明统计效率提高。

新数据训练两轮后，训练池缩放经验二阶矩从 0.0500000012 变为 0.0499233031。仅为极小训练池上的回代，不代表收敛或闭环收益。接口验证细节记录于 results/highd_dynamic_handoff_v1.json。

交接验证已通过 38 项相关测试、原资产及动态资产校验、默认新 checkpoint 恢复，以及 seed2000 相邻接近场景的自然/固定/PPO 短闭环。三组均无碰撞跑满 8 秒，PPO 在 2 个关键决策输出 epsilon，复核误差为零。另在独立解压目录重新通过 38 项测试、从头两轮训练和本机新 checkpoint 的在线闭环，概率审计通过。这些仅证明可移植链路正常；Mac 尚未实机运行。

## 交接后按什么顺序工作

1. 在接收机器上完成 verify 和 smoke；检查读取的是新目标和 20 条新行为序列。
2. 扩充同一目标的行为训练池，同时保留全部尝试、竞争事件和空序列。新增记录先 prepare，不能直接拼接旧单对轨迹。
3. 同伴在固定 NDD 与完整路径二阶矩目标下研究 PPO，保存中间 checkpoint，保留固定 epsilon 对照。
4. 用训练未用过的种子做自然、固定干预、新策略闭环对照；模型选择与最终测试应另分批次，当前 seed2000 短测试只算开发冒烟。
5. 对概率估计的方差、置信区间和同等精度成本作正式比较，不只比较碰撞数或训练奖励。若需要调整 NDD，另开新自然目标版本，不覆盖本次冻结基线。
