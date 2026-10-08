# V46 dual intervention with independent validation

This extension builds on `feature/highd-v46-d2rl-handoff` at `142cea3` and keeps
the frozen V46 natural model and all twenty checksum-locked files intact.
The physical experiment remains the fixed IDM CAV, random initialized fleet,
60 s horizon, and CAV involvement in the first collision of any actors.
BV-only collisions are competing negatives, not evidence of CAV safety for the
remaining horizon. The native experiment is an adaptation of the Nature D2RL
method, not a replication of its 400 m SUMO experiment or its 2100x result.

## Intervention and observations

The original 14-dimensional observation and strict following-pair mode remain
available through `configs/highd_v46_d2rl_experiment.json`. The opt-in v2 config
is `configs/highd_v46_dual_context_v2.json`.

Both single and dual examine the same nearest four actor anchors. Dual also
evaluates supported natural following pairs incident to these actors and uses
the most critical eligible pair. If no pair passes the pre-draw support and
action-contrast gates, it uses the best single candidate. Single partners still
follow the natural conditional law. An inactive second factor is logged as
P=H=Q=1 with mask=0, so changing its epsilon cannot change actions or weights.
Each maneuver tilts at most one natural unit, with at most two active actors.
PPO also masks the inactive output's log-probability, entropy and KL. It does
not inject a score-function gradient from an unused second epsilon during a
single-car fallback. This mask follows the recorded pre-draw actor selection.

The v2 policy has 22 observations, in this order:

| Indices | Meaning |
| --- | --- |
| 0-13 | Existing CAV6 + BV4x2 geometry, single candidates pad the second BV |
| 14 | log10 pre-draw critical mass / 16 |
| 15 | Supported action-score range / maximum supported action score |
| 16 | Past cumulative log(P/Q) / 30 |
| 17 | Remaining fraction of the 60 s horizon |
| 18-19 | Active factor mask |
| 20-21 | Following gap / 115 and lead-minus-follow speed / 20 |

All features are bounded to [-5,5]. The weight feature is reconstructed from
the current replay policy's prefix at every step and objective evaluation.
It never contains a future collision label or the behavior policy's terminal
weight. Old 14-dimensional weights are rejected for the v2 config.

The risk proxy and original thresholds are retained for diagnosis. Reports
include critical-step fraction, available-pair fraction, one/two-factor counts,
H-versus-P factor TV, action contrast and collision-actor coverage. The proxy
is not a calibrated crash probability. A high retained-step fraction should
prompt validation of the gate; tuning a threshold requires a new proposal and
fresh collection rather than deleting difficult or negative episodes.

## Reproducible collection and training

Use Python 3.9 / Ray 1.11 / Torch 1.11 with the handoff requirements, then verify:

```bash
python -m scenario_reconstruction.highd_v46_workflow verify
python -m pip install -r configs/requirements-highd-v46-tests.txt
python -m pytest tests/test_highd_v46_d2rl.py tests/test_highd_v46_validation.py tests/test_highd_paper_discrete_pairs.py tests/test_highd_mean_precision_policy.py tests/test_highd_dynamic_pairs.py -q
```

The predeclared pilot protocol is
`configs/highd_v46_dual_comparison_protocol_v2.json`: 1024 training episodes
from seed 1000, 1024 validation episodes from 50000, and 1024 final-test episodes
from 100000, with PPO seeds 7/19/29. For a first implementation check, collect
the first 512 episodes of each training/validation cohort. All collected
positive and negative episodes stay in the dataset. The remaining 512-episode
shards are predetermined, not collision-picked.

```bash
python -m scenario_reconstruction.highd_v46_workflow start --mode dual --config configs/highd_v46_dual_context_v2.json --training-config configs/highd_v46_validation_ppo_v2.json --episodes 1024 --seed 1000 --validation-episodes 1024 --validation-seed 50000 --iterations 100 --training-seed 7 --output outputs/v46_dual_context_seed7
```

Or collect independent cohorts separately and train using both manifests:

```bash
python -m scenario_reconstruction.highd_v46_workflow start --mode dual --config configs/highd_v46_dual_context_v2.json --training-config configs/highd_v46_validation_ppo_v2.json --sequence-manifest outputs/v46_v2_train512/sequence_manifest.json --validation-manifest outputs/v46_v2_validation512/sequence_manifest.json --iterations 100 --training-seed 7 --output outputs/v46_v2_ppo_seed7
python -m scenario_reconstruction.highd_v46_diagnostics --collection outputs/v46_v2_train512 --output outputs/v46_v2_diagnostics
```

The trainer rejects overlapping seeds, mixed targets, mixed observation/proxy
contracts and validation without CAV events. A single validation event permits
exploratory selection only; it does not support a significance claim. Each
validation interval saves a checkpoint. The initial fixed policy is included
in selection, so worsening validation cannot replace it. The root export is
the validation-selected model; `last_iteration/` keeps the final training
candidate. Final testing refuses seeds seen in training or validation.

If a cohort lacks events, preserve it and finish the predetermined additional
cohort using `collect` and `merge` from the handoff guide. Merge preserves every
episode and rejects duplicate seeds and different proxy/candidate structures.
Different behavior epsilons can share P; their exact logged Qb factors remain
separate. Checkpoints and datasets are ignored runtime outputs, not Git files.

## Fixed baseline and independent closed-loop comparison

Select constant epsilons on validation data, using the same grid for each arm;
dual evaluates the Cartesian product of the grid. Then freeze it for testing:
the default minimum 0.0501 is inside the PPO deterministic mean's numerical
interior (approximately 0.050095 to 0.999905).

```bash
python -m scenario_reconstruction.highd_v46_fixed_baseline --validation-manifest outputs/v46_v2_validation512/sequence_manifest.json --grid 0.0501 0.1 0.5 0.99 --output outputs/v46_v2_fixed_dual
python -m scenario_reconstruction.highd_v46_workflow evaluate --config configs/highd_v46_dual_context_v2.json --policy outputs/v46_v2_ppo_seed7/training --episodes 1024 --seed 100000 --output outputs/v46_v2_test_learned_dual
python -m scenario_reconstruction.highd_v46_workflow evaluate --config configs/highd_v46_dual_context_v2.json --policy outputs/v46_v2_fixed_dual --episodes 1024 --seed 100000 --output outputs/v46_v2_test_fixed_dual
python -m scenario_reconstruction.highd_v46_workflow collect --mode natural --config configs/highd_v46_dual_context_v2.json --episodes 1024 --seed 100000 --output outputs/v46_v2_test_natural
python -m scenario_reconstruction.highd_v46_statistics --collections natural=outputs/v46_v2_test_natural fixed_dual=outputs/v46_v2_test_fixed_dual learned_dual=outputs/v46_v2_test_learned_dual --bootstrap 1000 --output outputs/v46_v2_comparison
```

Repeat single collection/training with `--mode single`, using the same context
config, backbone and seed ranges, and add `fixed_single`/`learned_single` arms.
Train additional PPO seeds before making a learned-method comparison.

P's identity contains the natural freeze, physical CAV/horizon, event and
runtime identity. Q's identity separately contains epsilon, proxy, observation
and candidate settings. Changing epsilon does not create a different natural
risk target. A proxy or candidate change is not interchangeable replay data.

## Statistics and limits

For every attempted test use Y=I_event*exp(log(P/Q)), including zero outcomes.
Reports contain the ordinary IS mean, second moment, sample variance, standard
error, 90% normal interval, RHW, raw event rate and event-contribution ESS.
The RHW target is 0.3. At least 20 positive tests are required before declaring
precision reached; this guard does not guarantee normality for heavy-tailed
weights. A zero-event interval and RHW are undefined and reported as null.
Floats outside the supported moment range raise rather than clipping weights.

Observed required tests are the first prefix after the last failed prefix,
provided the final prefix meets the RHW and event requirements. This is only
finite-budget evidence. Bootstrap resamples episodes within each arm; replicates
that never reach precision remain censored. Conditional quantiles are explicitly
labelled and do not imply all replicates converged. Acceleration is reported
only when both reference and method reach precision. Plug-in projections are
separate fields and cannot be presented as measured speedups. The current
report does not claim a bootstrap confidence interval for the acceleration
ratio. Record wall-clock time because dual evaluates up to 33x33 action pairs.

Impact states, longitudinal speed differences, minimum TTC and bumper gap are
logged. `record_physics_trace=true` additionally stores per-physics-step actor
states in episode files for later near-miss/PET analysis. These trajectory
minima are not automatically the paper's exact event-distribution definitions;
crash typing and PET postprocessing remain separate work.

## Pilot evidence, 2026-10-03

The complete predeclared training and validation cohorts have been collected.
The compact machine-readable record is
[`results/highd_v46_dual_context_v2_pilot.json`](../results/highd_v46_dual_context_v2_pilot.json).

| Cohort | Seeds | Episodes | CAV first collisions | BV-only competing collisions |
| --- | --- | ---: | ---: | ---: |
| Training | 1000-2023 | 1024 | 2 | 41 |
| Validation | 50000-51023 | 1024 | 0 | 43 |

Training retains 92.63% of decision steps as critical. Among its 26,576
two-factor steps, 86.57% have conditional second-factor H-versus-P TV below
0.001. This suggests that the current pair selector/proxy often provides
little second-factor leverage; it does not establish that a two-car method
cannot help. The two event contributions have ESS approximately 1.00002,
and the 90% RHW is 1.645, well above 0.3.

Strict independent-validation training correctly refuses the zero-event
validation cohort. No validated checkpoint or validation-selected fixed
baseline is available from this pilot. A separately labelled, unvalidated
five-iteration PPO interface check (seed 7, 5509 timesteps) completed with the
inactive-factor gradient mask. Its training second-moment ratio against the
initial epsilon=0.1 baseline was 1.009749 (lower is better), so this check
shows no training improvement. Portable inference differed from RLlib by at
most 2.24e-8 in epsilon.

Natural, fixed-dual epsilon=0.1 and the unvalidated learned dual policy each
completed 16 fresh closed-loop episodes, seeds 110000-110015. Every arm had
zero CAV events and two BV-only competing events. CI/RHW/acceleration remain
undefined; all 30 bootstrap replicates per arm were budget-censored. These
are interface checks, not the protocol's final test. Seeds 100000-101023
remain unused. Thirty-eight unit/regression tests and frozen asset/code
verification passed.
Checkpoint selection, restoration and both policy exports also completed on
an explicitly synthetic one-iteration fixture; that fixture is software test
evidence only and is excluded from all traffic results above.

Reproduce the unvalidated interface check explicitly:

```bash
python -m scenario_reconstruction.highd_v46_train --sequence-manifest outputs/v46_v2_train1024/sequence_manifest.json --config configs/highd_v46_d2rl_ppo.json --iterations 5 --seed 7 --output outputs/v46_v2_masked_ppo_interface_smoke
python -m scenario_reconstruction.highd_v46_workflow evaluate --config configs/highd_v46_dual_context_v2.json --policy outputs/v46_v2_masked_ppo_interface_smoke --episodes 16 --seed 110000 --output outputs/v46_v2_interface_learned16
```

The next research experiment should first audit the risk proxy and dual
candidate criterion on training-only scenarios: distinguish genuinely
two-factor action influence from incidental natural following pairs. The
fallback mask already removes gradients for an absent second actor; it does
not remove a selected actor whose H is almost P. Predeclare any changed gate
and fresh training/validation seed cohorts, retain every outcome, then select
checkpoints only when validation has informative events. Increasing PPO
iterations alone is not supported by this pilot. No 2100x replication or
statistically significant advantage is claimed.

Paper references: [Nature article](https://doi.org/10.1038/s41586-023-05732-2),
[author training workflow](https://github.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning/blob/main/README_D2RL_Train.md),
[author RHW analysis](https://github.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning/blob/main/data_analysis/analysis_and_figures/crash_rate/crash_rate_bootstrap_plot.ipynb).
