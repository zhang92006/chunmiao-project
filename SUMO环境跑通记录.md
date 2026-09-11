# SUMO 环境跑通记录

本文档记录在本机 conda 环境 `D2RL` 中跑通模板场景、启动 SUMO/libsumo、并输出项目标准 crash JSON 的过程。

## 1. 环境检查结果

本机存在 conda 环境：

```text
D2RL -> D:\Anaconda3\envs\D2RL
```

该环境中已经安装：

```text
traci 1.14.1
sumolib 1.14.1
libsumo 1.14.1
```

`sumo` 和 `sumo-gui` 命令没有出现在系统 PATH 中，`SUMO_HOME` 也没有设置。但当前项目在 `gui_flag=False` 时会使用 `libsumo`，因此这次不依赖外部 `sumo.exe` 命令也能运行。

## 2. PyYAML 缺失处理

`D2RL` 环境中没有安装 `PyYAML`。直接运行模板入口时最初报错：

```text
ModuleNotFoundError: No module named 'yaml'
```

尝试联网安装 `PyYAML==6.0.1` 时，pip 被 SSL 问题拦住。因此当前实现已调整为：

```text
优先使用 PyYAML
如果没有 PyYAML，则使用 templates.py 内置的轻量 YAML 解析器
```

这个轻量解析器只覆盖当前事故模板需要的 YAML 格式，包括普通 key/value、嵌套 mapping、list、折叠文本 `>`、数字、布尔、空值和字符串。

## 3. 运行命令

在项目根目录运行：

```bash
conda run -n D2RL python -m scenario_reconstruction.run_template scenario_reconstruction/templates/autoware_cut_in.yaml --episode 0 --experiment_path ./data_analysis/raw_data/ScenarioReconstructionDebug
```

## 4. 运行结果

运行成功，输出：

```text
Scenario finished. weight_result=0.9999999999996793
```

SUMO/libsumo 同时报告碰撞：

```text
Warning: Vehicle 'CAV'; collision with vehicle 'BV_cut_in', lane='0to1_1', gap=-0.41', latGap=-1.80, time=1.40 stage=move.
```

这说明模板中的 `BV_cut_in` 已经被放入 SUMO，并且 `forced_bv_action` 触发的 cut-in 行为确实导致了 CAV 与 `BV_cut_in` 的碰撞。

## 5. 输出文件

本次输出目录：

```text
data_analysis/raw_data/ScenarioReconstructionDebug/
```

生成了：

```text
crash/0.json
tested_and_safe/
rejected/
```

其中 `crash/0.json` 包含：

```json
"collision_result": 1,
"collision_id": [
    "CAV",
    "BV_cut_in"
],
"episode_info": {
    "id": 0,
    "start_time": 0.0,
    "end_time": 1.4
}
```

这说明 `NADEInfoExtractor` 已经按项目原有格式写出了 crash episode JSON。

## 6. 当前训练数据问题

虽然 SUMO 场景已经跑通并产生 crash JSON，但当前这条 JSON 还不是理想的 D2RL 训练样本。

主要问题是：

```json
"weight_step_info": {},
"drl_obs_step_info": {}
```

这两个字段为空。

原因是当前碰撞主要由模板事件 `forced_bv_action` 直接触发，而不是由原 NADE/D2RL 行为采样机制自然选中并记录。因此虽然 `criticality_step_info`、`ndd_step_info`、`ttc_step_info`、`distance_step_info` 已经有值，但训练最关键的 `drl_obs_step_info` 和 `weight_step_info` 还没有形成有效记录。

下一步需要处理的是：

```text
让模板重建场景不仅能撞，还要能产生 D2RL 训练环境真正会采样的 weight_step_info 和 drl_obs_step_info
```

可选实现方向：

```text
方向 1：让 forced_bv_action 通过 NADE 采样/权重机制执行，而不是直接绕过它
方向 2：在模板事件激活时，显式记录 discriminator_input、criticality、ndd_possi 和合理的 step weight
方向 3：把模板场景作为初始化种子，再交给原 NADE 控制器自动选择高 criticality 动作
```

## 7. 训练字段补齐与 D2RLTrainingEnv 校验

在后续修改中，已经让模板触发的 `forced_bv_action` 事件补齐 D2RL 训练环境需要的关键字段。

### 7.1 修改内容

修改文件：

```text
scenario_reconstruction/environment.py
scenario_reconstruction/templates/autoware_cut_in.yaml
scenario_reconstruction/prepare_training_data.py
scenario_reconstruction/validate_training_env.py
d2rl_training/d2rl_training_env.py
```

核心策略是：当 `forced_bv_action` 第一次触发时，额外记录一个独立训练时间键，例如：

```text
forced_0.100000
```

这样可以避免后续 `NADEInfoExtractor` 的普通 snapshot 覆盖模板事件写入的 `ndd_possi`、`criticality` 等训练字段。

### 7.2 模板新增训练参数

当前 `autoware_cut_in.yaml` 中，`forced_bv_action` 增加了：

```yaml
training_weight: 0.05
epsilon: 0.99
ndd_possi: 0.00001
criticality: 1.0
```

含义如下：

```text
training_weight
```

写入 `weight_step_info`，并乘入 `weight_episode`。当前值 `0.05` 可以通过训练前 `weight_episode < 0.1` 的筛选。

```text
epsilon
```

写入 `drl_epsilon_step_info` 和 `real_epsilon_step_info`，表示这个强制事件对应的 adversarial probability。

```text
ndd_possi
```

写入 `ndd_step_info`。这里用 `0.00001` 表示该强制 cut-in 故障动作是低概率事件。这个值会直接影响 D2RL 训练环境中的 reward。

```text
criticality
```

写入 `criticality_step_info`。当原控制器当前步 criticality 为 0 时，使用模板值作为强制事故事件的风险标记。

### 7.3 新增训练数据准备脚本

新增：

```text
scenario_reconstruction/prepare_training_data.py
```

作用是扫描：

```text
experiment_path/crash/*.json
```

并生成：

```text
experiment_path/crash_weight_dict.json
```

该文件是 `D2RLTrainingEnv` 读取 crash 样本的必要索引。

运行命令：

```bash
conda run -n D2RL python -m scenario_reconstruction.prepare_training_data data_analysis/raw_data/ScenarioReconstructionDebug --max_ndd_possi 0.01
```

输出：

```text
Prepared 1 crash episodes.
Wrote data_analysis\raw_data\ScenarioReconstructionDebug\crash_weight_dict.json
```

`--max_ndd_possi 0.01` 用来过滤掉早期调试样本，只保留低 NDD 概率的训练样本。

### 7.4 新增训练环境校验脚本

新增：

```text
scenario_reconstruction/validate_training_env.py
```

它会直接实例化：

```python
D2RLTrainingEnv
```

并从重建数据中执行：

```text
reset
step
reward 计算
done 判断
```

运行命令：

```bash
conda run -n D2RL python -m scenario_reconstruction.validate_training_env data_analysis/raw_data/ScenarioReconstructionDebug
```

输出：

```text
D2RLTrainingEnv validation OK
steps=1
final_reward=49.99995231623869
```

这说明当前重建 crash JSON 已经能被 D2RL 训练环境消费，并产生正的训练 reward。

### 7.5 gym 缺失兼容

本机 `D2RL` 环境中只有：

```text
gym-0.21.0.dist-info
```

但缺少真正的 `gym` Python 包目录，导致：

```text
ModuleNotFoundError: No module named 'gym'
```

因此 `d2rl_training/d2rl_training_env.py` 增加了一个很窄的 fallback：

```text
如果 gym 存在，继续使用原 gym.spaces.Box 和 gym.core.Env
如果 gym 不存在，只提供当前环境需要的 Box 和 Env 最小实现
```

这只用于让当前项目的训练环境可以被本机 `D2RL` 环境直接校验。后续如果要跑完整 RLlib PPO 训练，仍建议安装完整训练依赖，尤其是 `gym` 和 `ray[rllib]`。

### 7.6 当前状态

现在链路已经推进到：

```text
模板 YAML
-> SUMO/libsumo 仿真
-> CAV 与 BV_cut_in 碰撞
-> NADEInfoExtractor 输出 crash JSON
-> prepare_training_data 生成 crash_weight_dict.json
-> D2RLTrainingEnv 成功读取并计算 reward
```

这表示当前单个模板样本已经具备进入 D2RL 训练数据池的基本条件。

## 8. 扰动扩增批量生成

已新增模板扰动扩增和批量运行工具，用于把一个事故模板扩展成多条可训练样本。

### 8.1 新增文件

```text
scenario_reconstruction/augment_templates.py
scenario_reconstruction/run_batch.py
```

`augment_templates.py` 负责读取模板中的 `perturbations`，按随机种子采样变体。

`run_batch.py` 负责：

```text
生成模板变体
逐条运行 SUMO/NADE 场景
收集 crash/tested_and_safe/rejected 输出
生成 crash_weight_dict.json
写出 batch_summary.json
```

### 8.2 扰动字段

当前示例模板中配置了 4 个扰动：

```yaml
perturbations:
  - field: ego.speed
    distribution: uniform
    low: -2.0
    high: 2.0
  - field: actors.BV_cut_in.position
    distribution: uniform
    low: -5.0
    high: 5.0
  - field: actors.BV_cut_in.speed
    distribution: uniform
    low: -2.0
    high: 2.0
  - field: events.forced_bv_action.start_time
    distribution: uniform
    low: 0.0
    high: 1.0
```

含义是：每个变体都会在原始模板基础上，对 CAV 速度、cut-in 车位置、cut-in 车速度、cut-in 开始时间做均匀随机扰动。

### 8.3 生成模板变体

单独生成变体可运行：

```bash
conda run -n D2RL python -m scenario_reconstruction.augment_templates scenario_reconstruction/templates/autoware_cut_in.yaml --count 3 --seed 42 --output_dir scenario_reconstruction/generated_templates_test
```

输出：

```text
Generated 3 variants in scenario_reconstruction\generated_templates_test
```

变体以 JSON 文件保存。`load_template()` 已支持直接读取这些 JSON 变体。

### 8.4 批量运行

批量运行命令：

```bash
conda run -n D2RL python -m scenario_reconstruction.run_batch scenario_reconstruction/templates/autoware_cut_in.yaml --count 3 --seed 42 --output_root data_analysis/raw_data/ScenarioReconstructionBatchTest2 --max_ndd_possi 0.01
```

运行结果：

```text
Batch finished.
successful_runs=3
failed_runs=0
training_ready_crashes=3
summary=data_analysis\raw_data\ScenarioReconstructionBatchTest2\batch_summary.json
```

SUMO/libsumo 报告三条变体均发生了 CAV 与 `BV_cut_in` 的碰撞。

### 8.5 批量输出结构

输出目录结构：

```text
data_analysis/raw_data/ScenarioReconstructionBatchTest2/
  variants/
    autoware_cut_in_run01_var_0000.json
    autoware_cut_in_run01_var_0001.json
    autoware_cut_in_run01_var_0002.json
  episodes/
    crash/
      0.json
      1.json
      2.json
    tested_and_safe/
    rejected/
    crash_weight_dict.json
  batch_summary.json
```

其中 `episodes/` 是后续 D2RL 训练环境读取的数据目录。

### 8.6 批量统计

`batch_summary.json` 当前记录：

```json
{
    "count": 3,
    "seed": 42,
    "successful_runs": 3,
    "failed_runs": 0,
    "training_ready_crashes": 3
}
```

每个 episode 还会记录对应的 variant 文件路径和 `weight_result`。

### 8.7 批量训练环境校验

对批量输出执行：

```bash
conda run -n D2RL python -m scenario_reconstruction.validate_training_env data_analysis/raw_data/ScenarioReconstructionBatchTest2/episodes
```

输出：

```text
D2RLTrainingEnv validation OK
steps=1
final_reward=49.99995231623869
```

说明批量生成的数据目录也能被 `D2RLTrainingEnv` 消费。

### 8.8 当前边界

当前批量生成已经完成：

```text
模板扰动采样
变体文件输出
批量 SUMO/NADE 运行
批量 crash JSON 输出
crash_weight_dict.json 自动生成
D2RLTrainingEnv 消费校验
```

当前还没有完成：

```text
near-miss 样本主动筛选
批量规模性能优化
自动跳过已有 episode 的断点续跑
从 Autoware db3 自动生成初始模板
完整 RLlib PPO 训练依赖安装与训练启动
```
