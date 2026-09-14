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

## SHRP2 collision seeds

The SHRP2 importer creates deterministic, impact-conditioned kinematic
collision seeds from the public bird's-eye trajectory reconstruction. Keep the
multi-gigabyte source data outside Git and pass its extracted root explicitly:

```bash
python -m scenario_reconstruction.shrp2_collision --source_root path/to/SHRP2_Public --config configs/shrp2_collision_pilot.json --output data_analysis/raw_data/shrp2_collision_pilot
```

The generated trajectories are collision seeds, not exact crash replays or
closed-loop SUMO results. See
[`docs/SHRP2碰撞数据处理.md`](docs/SHRP2碰撞数据处理.md) for provenance,
quality tiers, limitations, and the SUMO integration plan.

The first SUMO bridge maps only high-confidence same-direction rear-end seeds;
it does not claim closed-loop crash reproduction. See
[`docs/SHRP2到SUMO映射分析.md`](docs/SHRP2到SUMO映射分析.md) for the executed
mapping result, blocked road topologies, and calibration requirements.

Detailed implementation and experiment notes are under
`scenario_reconstruction/`.

