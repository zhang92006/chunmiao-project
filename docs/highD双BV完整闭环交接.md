# highD 双 BV D2RL：冻结环境与训练闭环交接

日期：2026-09-27。主线仍为当前冻结双车道实验；不包含新三车道开发。

## 交付内容

交接由源码和私有资产两部分组成，缺一不可。源码可从 `feature/highd-ppo-training-share` 分支取得，也可使用独立源码 ZIP。独立包不需要原仓库的旧 NDD 大表、Autoware 子模块、highD 原始 CSV 或 SHRP2 数据。

私有包 `highd_frozen_assets.zip` 内含冻结 NDD 拟合文件及其来源清单、80 条关键序列、训练审计摘要、第 200 轮完整 checkpoint 及其训练记录，共 222 个文件。解压到源码根目录后，应出现 `portable_assets/manifest.json` 和 `portable_assets/files/`。不要额外嵌套一层同名目录。文件校验值记录在公开的 `configs/highd_portable_assets.lock.json`。

私有包只通过团队获准的文件渠道交付，不推送 GitHub，也不表示获得 highD 原始数据的再分发许可。仅加载可信的团队 checkpoint，RLlib 的序列化格式不能用于执行不可信文件。私有模型清单保留原始机器路径以保持历史哈希；这些路径在运行时映射到接收者本地，不要求创建 Windows 盘符目录。

## 已具备的功能

| 入口子命令 | 功能 |
|---|---|
| verify | 校验资产、冻结代码、NDD 模型加载；可检查 SUMO 版本 |
| collect | 在同一冻结环境收集自然策略与固定双 epsilon 策略的新轨迹 |
| prepare | 审计全部采集尝试，并按 CAV-first 契约生成关键序列 |
| train | 从固定基线初始化离线 PPO，可指定新序列清单和超参数配置 |
| diagnose | 恢复附带第 200 轮模型或指定训练目录，复核训练池指标 |
| evaluate | 从 checkpoint 在线输出两个 epsilon，执行 SUMO 并复核实际条件 P/Q 与在线策略输出 |

统一 Python 入口为 `scenario_reconstruction.highd_portable`。Mac 操作脚本为 `configs/run_highd_portable.sh`，提供 verify、smoke、train、compare 四种任务。compare 对自然策略、固定干预、训练后策略使用同样四场景、种子和重复次数；默认仅每场景一次，不构成正式效率实验。长命令保存在脚本或聊天中，本文不放命令行。

训练默认读取附带 80 条行为序列，42 条非空、38 条无可训练决策。全部 80 次仍保留在风险分母，非空抽样修正为 42/80。修改采样策略时也必须保留相应修正。

`train` 是从头训练，不是接着第 200 轮优化；本次实现的是恢复推理与闭环验证，没有增加断点续训。研究 PPO 缺陷时优先保留固定 epsilon=(0.1,0.1) 初始化对照。

## 冻结目标与概率

NDD 为 `research_reference_frozen_not_behavior_accepted`，不是行为真实性验收通过。保留原双车道地图、20–40 m/s 范围、31 个纵向加速度与 3 个横向选择、已有自然配对规则和最大一个不重叠双车对。其他 BV 继续自然采样，每辆车每个决策仅采样一次。

两车自然概率为 p1(a1|h) p2(a2|a1,h)。在线策略只改变两个 epsilon 对自然分布和关键性提议的混合，不改变 P、已有关键性判定和横向执行核。epsilon 表示自然概率质量。

终点为 8 秒内 SUMO 首次报告的碰撞时刻是否涉及 CAV。BV-only 是竞争终点，该首次事件指标为零，但不是全时域安全样本。支持域退出和执行错误保留原始结果，不能丢掉后宣布风险审计通过。初始分布是四个配置诊断场景的等权混合，不是真实 highD 暴露分布。

`evaluate` 默认要求训练时同样的四场景混合。单场景测试必须显式标为 smoke，结果不能冒充原混合分布的正式评估。新随机种子减少轨迹重复，但不意味着换了独立场景来源。

冻结代码与模型保持原始字节，Git 属性避免跨平台换行转换破坏哈希。路径适配仅发生在独立入口进程中：先校验原始文件，再映射 JSON 内的资源位置。不会禁用模型校验、裁剪状态或换回旧 NDD。不要在同一个解释器内并发运行多个适配入口。

## 环境与 M4 Mac

继续使用 Python 3.9、Ray/RLlib 1.11、Torch 1.11、Gym 0.21、NumPy 1.23.1、protobuf 3.20.3；新增 pandas 2.3.3、SciPy 1.13.1 和 SUMO/TraCI 1.27.0。完整要求见 `configs/requirements-highd-portable.txt`。

Mac 安装脚本为 `configs/setup_highd_macos.sh`，在用户已经激活的独立 ARM64 Python 3.9 环境内运行，不修改系统 Python。脚本先固定旧 Gym 所需构建工具，再装 Python 依赖和 SUMO。当前任务使用 CPU，不需要 CUDA 或 MPS。

SUMO 1.27.0 官方提供 macOS 14+ ARM64 wheel；Ray 1.11 和 Torch 1.11 也提供 Python 3.9 ARM64 wheel。安装包可用不等于本工程已在 M4 实测。接收者需先完成 smoke，再运行批量比较。不同操作系统可能存在浮点差异，不要求每一条随机轨迹逐位一致，也不能无记录地混合不同 SUMO 版本。

资料：[SUMO 1.27.0 安装包](https://pypi.org/project/eclipse-sumo/1.27.0/#files)、[Ray 1.11 安装包](https://pypi.org/project/ray/1.11.0/#files)、[Torch 1.11 安装包](https://pypi.org/project/torch/1.11.0/#files)。本次实际短测试在 Windows、Python 3.9、SUMO 1.27.0 完成，M4 安装脚本尚未实机执行。

## checkpoint 兼容性与实验边界

早先发布最小源码包时，将旧版 Beta 的独立注册函数合并进了策略模块，因此发布源码哈希和训练时哈希不同。此次不伪造原哈希：`configs/highd_checkpoint_compatibility_v1.json` 只允许明确记录的这一对源码版本迁移，且恢复后必须复现原训练池指标才能运行。冻结 NDD 的校验仍逐字节执行。未知策略代码变化不会自动放行。

原第 200 轮模型训练池经验二阶矩为 0.06268816573932408，固定策略为 0.06262043527758177，尚未证明稳定优于固定策略。本次恢复得到相同数值。第 55 轮只有历史指标，没有保存的 checkpoint。

本次已通过 24 项相关单元测试、两轮 PPO 冒烟、冻结模型迁移加载、原 checkpoint 恢复、相邻接近场景自然/固定策略短仿真，以及训练后策略的一个闭环冒烟。该策略冒烟在 8 个关键时刻推理，在线 epsilon 复核误差为零，概率审计通过。另外在独立解压目录、仅保留 SUMO 自身 Python 工具路径的条件下，重新通过 24 项测试、模型加载和同一策略冒烟；Git 暂存内容也逐文件核对了 29 项冻结源码哈希。这些证明交接链路可运行，不证明 NDD 真实性、策略泛化或统计效率改善。精简证据见 `results/highd_portable_handoff_validation_v1.json`。

## 合作者接下来的工作

先复现固定基线和冒烟结果，再研究 PPO 优势信号、价值估计与训练后期回退，保存中间 checkpoint。完整路径二阶矩奖励、自然目标和首次事件定义保持不变。新的对照实验在同一场景混合、仿真版本和预算下比较，不把训练最低点或单次碰撞当作效率证明。

源码 ZIP 或 Git checkout 与私有包绑定校验。源码修改后，PPO 训练模块可以继续开发；如需改冻结 NDD、动作编码或事件契约，应建立新版本与新行为池，而不是更新哈希后沿用旧权重。
