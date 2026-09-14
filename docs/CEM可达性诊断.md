# CEM 碰撞可达性诊断

本阶段在高冲突潜力标签之后，使用 Cross-Entropy Method（CEM）对当前四参数定时动作空间做大预算搜索。它是“在当前动作空间、物理筛选和轻量仿真模型下是否能找到风险边界”的上界诊断；没有找到碰撞不等于已证明全局不可达，找到碰撞也不等于真实事故率或 D2RL 加速验证结果。

## 搜索空间与规则

受控对象仍只有换道 BV，参数为纵/横向加速度修正、相对 1 s 观察前缀的起效延迟和保持时长。范围与 `configs/trajectory_baselines.json` 一致：纵向加速度不超过 3 m/s²，横向不超过 1.5 m/s²，延迟为 0–1 s，保持为 0.4–2 s。动作以 0.2 s 五次 smoothstep 平滑进入和退出。

每个场景沿同一条 CEM 随机序列在 24、100、500、2000 次候选预算处保存精确快照。候选必须通过原有的速度、加速度、jerk、道路边界、初始重叠和背景 BV 碰撞筛选。CEM 的内部标量只用于采样更新；最终候选仍按公开的字典序规则选择：可行目标碰撞优先，其次可行近失，最后才是更小的目标未来间距。

命令只接受 `train` 或 `validation`，没有解锁 test 的参数。高冲突组必须读取带 manifest 哈希的精简诊断摘要，防止与其他导出版本混用。

```bash
python -m scenario_reconstruction.reachability_cem --manifest data/processed/highd_stratified_v2/manifest.json --trajectory_config configs/trajectory_baselines.json --cem_config configs/reachability_cem.json --split validation --group high_conflict --diagnostic_summary results/conflict_diagnostics_summary.json --output data_analysis/raw_data/reachability_cem_validation_high_v1
```

## 首轮结果

| 组别 | 场景数 | 24 次 | 100 次 | 500 次 | 2000 次 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 高冲突 | 5 | 1 碰撞、1 近失 | 2 碰撞 | 2 碰撞 | 2 碰撞、1 近失 |
| validation 高冲突 | 2 | 0 碰撞、0 近失 | 0、0 | 0、0 | 0、0 |
| validation 自然全体 | 60 | 0 碰撞、0 近失 | 0、0 | 0、0 | 0、0 |

高冲突训练组最早的可行碰撞分别在第 9 和第 31 个候选被发现；这说明当前物理边界并非对所有高冲突自然场景都不可达。验证高冲突组的平均目标未来间距则从 24 次预算的 19.967 m 降到 2000 次预算的 14.679 m，仍未到达 1 m 近失。自然验证全体对应值从 30.045 m 降到 28.179 m。

因此，不能以训练组的 2/5 成功率声称方法有效，也不能以验证组 0/2 证明安全。严谨结论仅为：高冲突标签具有可达性区分作用；当前四参数动作、3 秒未来窗口和开环背景车模型不足以在验证集上稳定生成风险边界。原始逐场景记录保存在被忽略的 `data_analysis/raw_data/`；可提交摘要在 `results/reachability_cem_summary.json`。

下一步应扩展动作表示为分段动作和可控换道时序，并在不放宽物理上限的前提下先做几何上界诊断。随后必须把候选接入 highD—SUMO 闭环复核；不应据此开始 D2RL 训练或报告真实碰撞概率。
