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

## Frozen V46 NDE and D2RL handoff

For a ready-to-train dual-BV experiment, use the
[conditional PPO baseline](docs/highd_v46_conditional_baseline.md): explicit
source-weighted replay, fresh validation, fixed-policy comparators, native
held-out evaluation and training curves. Its first version controls one
decision followed by four seconds of native continuation; its conditional
target is distinct from population-initialized testing.

For the current native highD experiment, start with the
[V46 single-BV baseline and dual-following training guide](docs/highd_v46_d2rl_handoff.md).
For the dual extension with single fallback, causal context observations,
independent checkpoint selection and paper-style statistics, use the
[V46 dual validation experiment](docs/highd_v46_dual_validation.md).
Before further PPO tuning, the [pair-synergy diagnostic](docs/highd_v46_pair_synergy.md)
audits joint-action influence and a natural-rollout high-risk state pool.
The opt-in [cross-unit sampler](docs/highd_v46_cross_unit.md) preserves natural
partners' conditional law while intervening on a pair from different units.
The [state-discovery audit](docs/highd_v46_state_discovery.md) compares cheap
per-second screening, the five-second grid and random state-bank controls.
Its training-only [coverage audit](docs/highd_v46_discovery_coverage.md) separates
missed states from missed pairs by exhaustively checking a bounded pair universe.
The [discovery v2 pilot](docs/highd_v46_discovery_v2.md) tests mixed temporal
sampling and rank-1/rank-2/audit screening on a fresh validation cohort.
The [multi-state real-Q pilot](docs/highd_v46_multistate_q.md) compares actual
natural/single-factor/dual samples under a declared conditional state law.
The [training-only native calibration](docs/highd_v46_native_calibration.md)
builds endpoint-labelled discovery cases without changing NDE, Q or PPO.
The [fresh source-selector validation](docs/highd_v46_source_selector_validation.md)
compares absolute-risk ranking with and without the dual gate, using paired
source-episode uncertainty and per-source second-moment contributions.
Its [collision-action coverage audit](docs/highd_v46_source_selector_validation.md#post-hoc-collision-action-coverage-audit-2026-10-05)
decomposes observed likelihood tails and reconstructs the interaction score
of the dominant collision action before changing the proposal.
The opt-in [fixed-bank proposal comparison](docs/highd_v46_proposal_comparison.md)
tests normalized absolute-risk/interaction H scores on locked training cases,
then on a separate fresh matched-source validation cohort. Native action-coverage
repair is observed, but moment concentration and source diversity still fail
readiness; no proposal replaces production or launches PPO.
The [V47 probability-allocation audit](docs/highd_v47_probability_allocation.md)
adds source-held-out collision calibration and constrained second-moment
allocation with explicit Q factors on the frozen training bank.
The [V47 event-source diagnosis and action teacher](docs/highd_v47_event_source_diagnosis.md)
separates strict cooperative testing eligibility from collision-label access.
Risk-bin exploration expands labelled event sources, without claiming that
natural-risk concentration or probability-model generalization is solved.
The [V47 state-allocation experiment](docs/highd_v47_state_allocation.md)
adds Q_S with complete state/action likelihood correction while keeping the
conditional target and action proposal fixed. Its guarded update has mixed
fixed-bank variance results and is not a validated production policy.
The [fresh-source Q_S transfer protocol](docs/highd_v47_source_transfer.md)
uses new, disjoint training/validation cohorts, outcome-blind features and
source-grouped fitting. It tests prospective same-target objective benefit,
without refining the old fixed-bank certificate or automatically launching PPO.
Its first complete new-source training check did not improve held-out M2;
the declared review keeps fresh validation unopened and production unchanged.
The [nested state-model revision](docs/highd_v47_state_model_revision.md)
compares richer context and direct M2 learning without new native outcomes.
Its fixed primary candidate also fails held-source review; fitting gains and
neutral optimizer fallbacks are not promoted as transfer success.
The frozen NDE tag is `nde-v46-frozen-20261003`; training code is on
`feature/highd-v46-d2rl-handoff`. This path needs Python 3.9 and the V46 LFS
archive, with no SUMO or raw highD tracks. The fixed IDM CAV does not change lanes.

```bash
python -m scenario_reconstruction.highd_v46_workflow start --mode dual --episodes 512 --seed 1000 --iterations 100 --training-seed 7 --output outputs/v46_dual_seed7
```

Single-BV PPO has completed 100 iterations on 512 collected scenarios (two CAV
first collisions). See the [run record](configs/highd_v46_single_baseline_run.json)
for configuration, results and the limited statistical scope. The guide includes
installation, exact intervention/event definitions and fresh-seed evaluation.

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

Detailed implementation and experiment notes are under
`scenario_reconstruction/`.
