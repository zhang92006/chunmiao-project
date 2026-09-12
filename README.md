# Scenario Reconstruction

This repository contains the code and runtime assets needed to reconstruct and
perturb traffic scenarios in the D2RL/NADE SUMO environment. It packages the
scenario layer together with the D2RL modules, SUMO map, and probability tables
that the runtime imports.

The D2RL core is derived from
[`michigan-traffic-lab/Dense-Deep-Reinforcement-Learning`](https://github.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning)
at commit `087921a994b3f5176fb0d75667a0a11fa357cd2b`, with the local integration
changes required by scenario reconstruction. See `LICENSE` for the upstream
license.

## Clone and install

Git LFS is required for the runtime probability tables, and the Autoware message
definitions are pinned as a submodule.

```bash
git lfs install
git clone --recurse-submodules https://github.com/zhang92006/chunmiao-project.git
cd chunmiao-project
git lfs pull
conda env create -f environment.yml
conda activate scenario-reconstruction
```

If the repository was cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## Verify the installation

Run all commands from the repository root:

```bash
python -m scenario_reconstruction.validate_template scenario_reconstruction/templates/autoware_cut_in.yaml
python -m scenario_reconstruction.run_template scenario_reconstruction/templates/autoware_cut_in.yaml --episode 0 --experiment_path data_analysis/raw_data/smoke_test
```

The first command validates the template without starting SUMO. The second runs
one headless SUMO episode and writes generated data under the ignored
`data_analysis/raw_data/` directory.

The example is a synthetic cut-in demonstration, not validated Autoware accident
ground truth. Perception/control fault descriptions without an implemented runtime
are rejected. Runs fix Python/NumPy/PyTorch/SUMO seeds (`--seed`, default 0), honor
the template duration, and retain all safe outcomes. Use a fresh output directory
for each experiment; existing episode files are not overwritten.

Template initialization and scripted interventions have no derived importance
likelihood ratio. Their returned `weight_result` is `null`; upstream weights in
logs are diagnostics, not valid D2RL training weights or road crash probabilities.
The legacy single-BV trainer now rejects MultiBV joint records rather than silently
using only the first agent.

## Research plan

See [科研提升路线图](docs/科研提升路线图.md) for the literature comparison,
implementation order and publication requirements. Historical notes under the
package directory may describe superseded probability/training behavior.

The [trajectory benchmark guide](docs/轨迹基准使用说明.md) documents local highD
export and replay/uniform/constrained-search screening, plus masked-trajectory
interpolation baselines. See [implementation results](docs/实施记录.md) for tested
behavior, pilot numbers, and limitations. These baselines use a lightweight
lane-fixed IDM response, not a complete SUMO/Autoware planner evaluation.

See [零碰撞修正记录](docs/零碰撞修正记录.md) for the future target-pair objective,
timed action pulse, between-frame collision checks, and latest negative-result pilot.

## Batch reconstruction

```bash
python -m scenario_reconstruction.run_batch scenario_reconstruction/templates/autoware_cut_in.yaml --count 5 --seed 7 --output_root data_analysis/raw_data/scenario_batch
```

Generated templates, simulation outputs, logs, and checkpoints are intentionally
ignored. Commit experiment configurations and concise result summaries instead.

## Autoware input

Raw ROS2 bag files are not stored in normal Git history. Place `metadata.yaml`
and its `.db3` file in a local input directory, then run:

```bash
python -m scenario_reconstruction.autoware_cdr --bag_dir path/to/bag --msg_root external/autoware_msgs --output data_analysis/raw_data/autoware_decoded_summary.json --max_samples 2
```

Automatic Qwen template generation reads `DASHSCOPE_API_KEY` or `QWEN_API_KEY`
from the environment. Never commit either value. The pipeline can be exercised
without an API key using `--no_llm`.

Detailed implementation and experiment notes are under
`scenario_reconstruction/`.
