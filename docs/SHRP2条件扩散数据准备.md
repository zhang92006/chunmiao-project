# SHRP2 条件扩散数据准备

## 本轮完成与目的

已实现全类别元数据盘点、可选的 HDF5 轨迹质量扫描、测量双车窗口导出，以及已完成 SUMO 候选相对真实 SHRP2 参考的误差计算。本轮没有训练扩散模型；先确认可用独立事件数量、标签口径和轨迹参考点，再决定模型容量与训练方案。

## 公开版实际规模

元数据结果：[`audit_summary.json`](../results/shrp2_public_metadata_inventory_v1/audit_summary.json)。

| 范围 | 独立事件数 |
|---|---:|
| 全部 14 类 | 6,664 |
| train | 4,663 |
| validation | 991 |
| test | 1,010 |
| 单事件 Crash 类 | 1,242 |
| 单事件 NearCrash 类 | 4,648 |
| Crash / leading（全部划分） | 90 |
| NearCrash / leading（全部划分） | 2,055 |

当前各类别间没有重复 event_id。数量不等同于可训练窗口：Crash 类中 1,019 个事件的 conflict 为 `none`，不能凭类别名称就认定存在可重建的双车事故轨迹。复合与 Secondary 类别的前后事件也不能直接作为同一个目标车的碰撞标签。

## v3 轨迹指标

结果：[`shrp2_reference_116591908_v3_metrics.json`](../results/shrp2_reference_116591908_v3_metrics.json)。48 条运行中 47 条发生目标追尾，3 条满足 `3.6±0.3 s`。

指标使用共同观测窗口 `0–3.2 s`，33 个采样点；FDE 是 `3.2 s` 的误差。日志没有保存早终止候选的完整 4 秒状态，因此不能报告 4 秒 FDE 或完整窗口拟合率。

| 候选 | 碰撞时刻 | CAV ADE | CAV 速度 RMSE | BV 前保险杠 ADE | BV 前保险杠 FDE |
|---|---:|---:|---:|---:|---:|
| 28 | 3.3 s | 0.409 m | 0.339 m/s | 2.888 m | 3.264 m |
| 29 | 3.3 s | 0.409 m | 0.339 m/s | 2.888 m | 3.264 m |
| 41 | 3.9 s | 0.426 m | 0.327 m/s | 3.700 m | 4.185 m |

28/29 的主车与 CAV 共同窗口指标完全相同，不能作为两个独立真实事故。当前证据支持“CAV 减速趋势与目标碰撞可达”，尚不支持“完整双车轨迹准确重建”。

SUMO `getPosition` 是前保险杠中心，SHRP2 CAV 是车心，目标车 `x_sur/y_sur` 是前保险杠。新指标将 SUMO CAV 转为车心，并直接比较目标车前保险杠；SUMO 航向按北向零度、顺时针转换到笛卡尔角度。依据：[SUMO 官方位置定义](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)。

当前地图车辆长度为 5 m；指标明确记录这一假设。t=0 日志中 CAV 已比模板初始位置前移约 0.605 m，日志与物理时间口径尚需进一步核对。SHRP2 初始目标前保险杠相对 CAV 有约 2 m 横向偏移，而当前桥接把双方投影到同一车道，横向信息损失是前车位置误差的重要候选来源。指标没有单独移动目标车或搜索时间偏移来改善分数。

## 窗口接口与筛选

配置：[`shrp2_diffusion_data_audit.json`](../configs/shrp2_diffusion_data_audit.json)。窗口为关键时刻前 4 秒、10 Hz，每条记录的 `states` 和 `state_mask` 形状为 `[41, 2, 4]`，通道为 `x, y, speed, heading`。

- CAV 位置使用车心，BV 使用实测前保险杠；在进入统一车辆中心的多车模型前，必须按质量门槛处理目标车心转换。
- 条件包含初态及掩码、关键时刻、源事件类别、conflict 和尺寸。该公开接口不伪造车道或道路几何条件。
- `derived_tangential_acceleration_mps2` 是报告速度的差分，不是 SUMO 控制器动作或真实油门/制动标签；不能直接当作可执行控制真值。
- 基础训练窗口要求位置—速度 RMSE 不超过配置阈值。低速或不一致的航向使用掩码；未碰撞的 NearCrash 窗口同样可被保留。
- 几何碰撞诊断独立于训练窗口筛选；最近几何目标不等同于已验证的事故参与者。
- 不外推缺失历史，不跨越超过 0.2 秒的源数据空洞进行插值。各排除原因写入审计结果，阈值调整应另建配置并记录消融。

同一 event_id 在所有类别、目标与未来仿真变体中使用相同划分。train 才能拟合归一化、参数分布或模型权重；此处扫描 validation/test 仅做固定规则的数据检查。若日后获得 trip/driver 标识，还应升级为相应层级的分组划分。

## 下一条长任务

在仓库根目录，用含 `pandas` 与 `tables` 的 Python 环境执行；当前本地 Anaconda 基础环境可用，D2RL 环境缺少 PyTables。配置与依赖范围由仓库 requirements.txt 记录。

```powershell
New-Item -ItemType Directory -Force 'data_analysis/logs' | Out-Null
python -m scenario_reconstruction.shrp2_diffusion_data_audit `
  --source_root 'path/to/SHRP2_Public' `
  --config 'configs/shrp2_diffusion_data_audit.json' `
  --output 'data_analysis/raw_data/shrp2_diffusion_windows_v1' `
  --scan_trajectories --export_windows `
  2>&1 | Tee-Object -FilePath 'data_analysis/logs/shrp2_diffusion_windows_v1.log'
```

程序逐类别读取 HDF5 并校验摘要，可能消耗较多内存和时间。输出包含 `audit_summary.json`、每类别的配对审计 JSONL，以及 `train/validation/test/windows.jsonl`。目录必须是新建或空目录；重复实验使用新目录。

快速元数据盘点可省略 `--scan_trajectories --export_windows`；接口抽样验证可加 `--max_events_per_category 3`，但抽样报告不能代表全数据规模。

## 运行结束后的决策

先看独立 quality-pass 事件数、各划分数量、类别/冲突覆盖和排除原因，再选择小模型容量。初版建议从追尾双车开始；对正常驾驶数据预训练，然后研究事故条件引导与可执行性约束。真正的多车联合训练仍需要多个目标在同一坐标系、时间窗口与身份关联下组装，当前配对接口只是第一阶段。

扩散模型评估至少包含：未参与训练事件上的分布一致性、固定初态下的轨迹多样性、目标时刻/终态满足率、SUMO 可执行率、速度/加速度/急动度，以及替换为闭环 CAV 后的风险。不得把网格的 47/48 碰撞率解释为真实道路事故概率。
