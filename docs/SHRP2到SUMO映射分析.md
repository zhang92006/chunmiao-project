# SHRP2 到 SUMO 映射：第一轮分析

## 本轮结论

已完成 SHRP2 运动学碰撞种子到当前 SUMO 工程的第一次映射与真实运行验证。高可信追尾种子可以被投影为现有 `2Lane` 路网中的可执行模板；但直接保留初始相对位置和速度后，SUMO 的 IDM 跟驰控制会避撞，因此尚未实现“SHRP2 条件化碰撞在 SUMO 中重现”。

这个结果应作为一次失败复现实验记录，而不是把模板可启动误写成 SUMO 已复现真实事故。

机器可读摘要见 `results/shrp2_sumo_bridge_pilot_summary.json`。

## 映射覆盖范围

| SHRP2 种子类型 | 可信度 | 数量 | 本轮状态 | 原因 |
|---|---|---:|---|---|
| 追尾 | 高 | 3 | 已生成 `2Lane` 模板 | 与当前同向双车道路网相容 |
| 交叉冲突 | 中 | 3 | 阻塞 | 缺少交叉口路网与横穿路线 |
| 对向转弯 | 中 | 3 | 阻塞 | 缺少双向交叉口路网与转弯路线 |
| 侧向擦碰 | 低 | 3 | 排除 | 事故时位置条件化需平移 5.187 m，不进入主实验 |

桥接程序是 `scenario_reconstruction/shrp2_sumo_bridge.py`，配置是 `configs/shrp2_sumo_bridge_pilot.json`。它只允许高/中可信记录进入决策；当前仅将 `rear_end` 映射为模板。其余类型会写明阻塞或排除原因，而不是生成语义错误的直路场景。

## 追尾映射规则

对于每条高可信追尾种子：

1. 读取 SHRP2 条件化运动学轨迹第 0 帧的 CAV 和主 BV 纵向坐标。
2. 计算源初始纵向距离 `x_BV - x_CAV`。
3. 将 CAV 放到 `2Lane` 的车道 1、位置 400 m；将主 BV 放到同一车道，并保留该相对距离。
4. 保留两车的初始速度；车道 0 放置一个不参与事故的上下文 BV，保证当前 MultiBV 运行接口可初始化。
5. 不凭空添加制动、切入或攻击动作。因而生成的模板是“SUMO 初始化候选”，不是“保证碰撞的脚本”。

例如基础速度追尾种子 `shrp2_131785457_rear_end_source_speed` 被映射为：初始纵向距离 16.355 m，CAV 速度 3.117 m/s，主 BV 速度 0.522 m/s。

## SUMO 冒烟验证

对上述基础速度模板执行了一次无 GUI 的 SUMO/libsumo 运行：

| 指标 | 结果 |
|---|---:|
| 模板解析与 SUMO 初始化 | 成功 |
| 碰撞结果 | 否 |
| 碰撞车辆 | 无 |
| 仿真结束时间 | 5.9 s |
| 最小 TTC | 4.318 s |
| 最小间距 | 8.846 m |

因此当前直接映射没有发生碰撞。最合理的解释是：SHRP2 种子中的碰撞是在“碰撞时几何位置条件化”的恒速运动学模型中产生，而 SUMO 中的 IDM 控制器看到前车后会调整速度、留出安全距离。两者不是同一个动力学/控制系统，不能期待相同的初态自动产生相同的结局。

本轮同时修复了时长语义：`ScenarioNADE._terminate_check()` 现在在保留碰撞、车辆离网等安全退出条件的前提下，按 `ScenarioTemplate.duration` 终止场景。6 s 模板的记录结束时间为 5.9 s，这是现有 episode 日志在 0.1 s 步长下记录最后一个完整步的约定。时长问题已经解决，但碰撞校准问题仍然存在。

## 对论文表述的影响

当前可以表述为：

> 我们从 SHRP2 公共事故事件中构造了具有来源、坐标转换、目标关联和几何条件化审计信息的碰撞种子；并实现了其向 SUMO 同向追尾初始化的可执行映射。

当前不能表述为：

> 我们已在 SUMO 中精确复现 SHRP2 事故。

也不能把一次未碰撞运行解释为原 SHRP2 事件不危险，或把一次未来的强制碰撞解释为事故动力学重放。

## 下一步的技术顺序

1. 只在训练划分的高可信追尾种子上，定义可审计的“碰撞校准”协议：待搜索参数必须包含初始间距、BV 制动曲线、CAV 控制策略和碰撞时间误差；不能直接手调到碰撞后便宣称复现。
2. 用验证划分选择阈值，记录每个候选的初态重叠、首次碰撞时间、最小 TTC/间距、碰撞对象和 SUMO 日志；冻结协议后才触碰测试划分。
3. 用 `netconvert` 建立交叉口与双向交叉口网络，并把模板模型扩展为“地图文件、路线、车道集合”可配置，之后才能处理交叉冲突和对向转弯的 6 条中可信种子。
4. 侧向擦碰在重新改善目标关联和位置条件化之前，保持在低可信诊断集，不计入主要论文指标或 D2RL 主训练集。

## 后续批量运行命令

当第 1、2 步完成后，再由用户在 PowerShell 中启动长时间批量实验。预期命令格式为：

```powershell
Set-Location 'G:\chunmiao\d2rl\Dense-Deep-Reinforcement-Learning\scenario_reconstruction'

$log = 'data_analysis\logs\shrp2_rear_end_calibration.log'

& 'D:\Anaconda3\envs\D2RL\python.exe' -m scenario_reconstruction.run_batch `
  data_analysis\raw_data\shrp2_sumo_bridge_pilot\templates\sumo_shrp2_131785457_rear_end_source_speed.json `
  --count 100 --seed 20260914 `
  --output_root data_analysis\raw_data\shrp2_rear_end_calibration `
  2>&1 | Tee-Object -FilePath $log
```

这条命令暂时不要运行：当前没有校准协议，且 `run_batch` 的默认目标碰撞对象仍是旧的 cut-in 设定。完成校准接口后，再给出与新接口精确一致的最终批量命令。
