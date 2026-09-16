# Scenario Reconstruction

This package defines the accident-template, perturbation, simulation, and
Autoware-analysis layers used to rebuild cases in the project SUMO/NADE
environment. Repository installation instructions and dependency information
are in the root `README.md`.

Main components:

- `templates.py` defines the machine-readable template schema.
- `templates/autoware_cut_in.yaml` is a starter accident template.
- `validate_template.py` checks that a template is structurally usable before it
  is connected to SUMO initialization.
- `environment.py` defines `ScenarioNADE`, a NADE environment that initializes
  CAV/BV vehicles from a template instead of random NDD traffic.
- `run_template.py` runs one reconstructed scenario and keeps the standard
  `NADEInfoExtractor` logging path.

The template describes:

- `ego`: CAV initial lane, position, speed, and controller.
- `actors`: background vehicles that matter for the reconstructed accident.
- `events`: fault or maneuver events to replay in the simulator.
- `perturbations`: parameter ranges used to expand one Autoware accident into
  many training episodes.

Run:

```bash
python -m scenario_reconstruction.validate_template scenario_reconstruction/templates/autoware_cut_in.yaml
```

Generate MultiBV K=2 training data from variants:

```bash
python -m scenario_reconstruction.run_batch scenario_reconstruction/templates/autoware_cut_in.yaml \
  --count 100 \
  --output_root data_analysis/raw_data/ScenarioReconstructionMultiBV \
  --multi_bv_num 2
```

The generated `episodes/` folder contains `crash/`, `tested_and_safe/`,
`crash_weight_dict.json`, and `safe_weight_dict.json`. It can be checked with:

```bash
python -m scenario_reconstruction.validate_training_env \
  data_analysis/raw_data/ScenarioReconstructionMultiBV/episodes
```

To train a centralized two-BV policy rather than use the legacy first-BV
projection, set `multi_bv_training: true` and `multi_bv_num: 2` in the D2RL
training configuration. The policy must then accept the 14-D joint observation
and emit two epsilon values. Validate a generated K=2 episode set with:

```bash
python -m scenario_reconstruction.validate_training_env \
  data_analysis/raw_data/ScenarioReconstructionMultiBV/episodes \
  --multi_bv_training --multi_bv_num 2
```

Measured SHRP2 windows are only sources for multi-BV scenario seeds; they are
not D2RL episodes until SUMO/NADE produces the joint observations, actions,
NDD probabilities, and importance weights described in
[`docs/多智能体D2RL联合训练接口.md`](../docs/多智能体D2RL联合训练接口.md).

Export context-augmented K=2 source-frame seeds from a completed SHRP2 audit:

```bash
python -m scenario_reconstruction.shrp2_multibv_seed_export \
  --source_root path/to/SHRP2_Public \
  --audit_root data_analysis/raw_data/shrp2_diffusion_windows_v1 \
  --output data_analysis/raw_data/shrp2_multibv_seeds_v1 \
  --bv_count 2
```

This is a long HDF5 scan. The output is still `drl_training_ready=false` until
the source-frame seed is mapped to a valid SUMO route and simulated by NADE.

For the larger context-anchor pool, use the explicitly weaker mode below. It
keeps the complete CAV/primary-BV history but only requires a context BV state
at the critical timestamp; these records must remain separate from the strict
full-history validation pool:

```bash
python -m scenario_reconstruction.shrp2_multibv_seed_export \
  --source_root path/to/SHRP2_Public \
  --audit_root data_analysis/raw_data/shrp2_diffusion_windows_v1 \
  --output data_analysis/raw_data/shrp2_multibv_seeds_anchor_v1 \
  --bv_count 2 \
  --context_mode anchor_only
```

Map the anchor-only seeds to autonomous 2Lane templates (no forced actions):

```bash
python -m scenario_reconstruction.shrp2_multibv_sumo_bridge \
  --seed_root data_analysis/raw_data/shrp2_multibv_seeds_anchor_v1 \
  --output data_analysis/raw_data/shrp2_multibv_sumo_templates_v4 \
  --config configs/shrp2_multibv_sumo_bridge.json
```

The bridge blocks unsupported topologies, projected same-lane overlap, and
initial speeds outside the configured D2RL domain (20--40 m/s on the current
high-speed model), then writes a `bridge_summary.json`. It blocks rather than
silently clips speed, preserving a traceable boundary between SHRP2
observations and SUMO-executable initial states. Rejected low-speed seeds remain
available for a future low-speed NDD/controller pipeline.
Templates are only initialization candidates; run them through autonomous
NADE/D2RL and require `joint`, `per_agent`, NDD, and weight fields before using
the resulting episodes for learning.

Run a bounded pilot directly from that bridge summary before launching a full
split. The runner accepts both the older `path` manifest field and the bridge's
`template_path` field:

```bash
python -m scenario_reconstruction.run_template_manifest \
  data_analysis/raw_data/shrp2_multibv_sumo_templates_v1/bridge_summary.json \
  --split train --start 0 --limit 20 \
  --epsilon 0.1 \
  --experiment_path data_analysis/raw_data/shrp2_multibv_rollout_pilot_train20
```

Inspect `manifest_run_summary.json` and the episode JSON files. A joint sample
must contain two controlled IDs and the `joint`/`per_agent` fields described in
the MultiBV contract; a safe episode without those fields is not a training
sample.

`--epsilon` is the fixed naturalistic-mixture probability used while collecting
NADE importance-sampling episodes. The default `0.99` is close to ordinary NDD
sampling and is useful for safety smoke tests. Use a documented lower pilot
value such as `0.1` when collecting candidate D2RL training records, then run
an epsilon sensitivity study before treating the data as a final experiment.

`--repeats N` runs each selected template `N` independent times. Each run has a
unique episode ID and records its zero-based `repeat` number in
`manifest_run_summary.json`; use this for importance-sampling data collection,
not for a GUI run. For example, after rebuilding a fresh bridge directory:

```bash
python -m scenario_reconstruction.run_template_manifest \
  data_analysis/raw_data/shrp2_multibv_sumo_templates_v4/bridge_summary.json \
  --split train --start 0 --limit 50 --repeats 5 --epsilon 0.01 \
  --experiment_path data_analysis/raw_data/shrp2_multibv_highspeed_rollout_v4_train50_x5_epsilon001
```
