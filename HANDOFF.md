# 当前 highD 动态多对 NDD / 双 BV D2RL 交接入口

本版本请先读 [动态多对版本交接](docs/highD动态多对版本交接.md)。新的统一入口为 highd_dynamic_handoff；需要源码、原冻结资产包和新增动态数据包。Windows 与 Mac 均提供操作脚本。当前附带的新 checkpoint 只有两轮训练，仅作接口冒烟。

下面保留旧单对版本说明，旧 highd_portable 入口不应代替新的动态入口。

请优先阅读 [完整闭环交接说明](docs/highD双BV完整闭环交接.md)，不要按仓库历史 README 的旧 NDD/Autoware 环境重新安装。

需要源码以及团队提供的 `highd_frozen_assets.zip`。将后者解压到本文件所在目录，得到 `portable_assets/`。该目录不入 Git。独立源码包不含原始 highD/SHRP2 数据，不需要旧 NDD LFS 表。

Mac 安装与操作入口分别为 `configs/setup_highd_macos.sh` 和 `configs/run_highd_portable.sh`；先准备独立 ARM64 Python 3.9 环境。统一 Python 入口为 `scenario_reconstruction.highd_portable`。命令行见脚本或交接聊天。

当前仅承诺已完成 Windows 端短测试；M4 需先运行 smoke。NDD 是冻结研究参考，尚无稳定优于固定策略的 PPO 性能证据。
