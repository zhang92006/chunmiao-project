# Trainable conditional dual-BV PPO baseline

This baseline starts actual PPO on the frozen native V46 dynamics. It uses
the completed training calibration as an explicit conditional state bank,
fresh source-disjoint validation for checkpoint/fixed-policy selection, and
a separately locked test bank for native rollouts. The configuration is
[highd_v46_conditional_baseline_v1.json](../configs/highd_v46_conditional_baseline_v1.json).

## Ready-to-train entry point

Use the existing Python 3.9 environment from the repository root. The prepared
training/validation directories can be reused for the three declared training
seeds (7, 17, 29), without recollecting or changing the validation cohort:

```bash
conda activate highd-ppo-clean39
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_conditional_baseline train --data outputs/v46_conditional_baseline_v1/data --output outputs/v46_conditional_seed17 --seed 17 --iterations 100
```

Outputs include per-iteration metrics, RLlib checkpoints, the validation-selected
portable policy in `ppo/`, the last policy in `ppo/last_iteration/`, the selected
constant-epsilon baseline in `selected_fixed/`, and `baseline_run.json`.
Existing destinations are never overwritten. An interrupted stage is retained;
use a new destination for that stage. `--interface-only` explicitly permits a
training-only software check, without checkpoint selection or a tuned fixed
baseline; it is not the standard experiment.

## Protocol and objective

The policy receives 22 causal geometry/context features and produces two
epsilon values in [0.05,1]. The actor/critic have two 64-unit tanh layers; the
bounded Beta head starts at epsilon=(0.1,0.1). Default PPO runs 100 iterations,
batch 1024, minibatch 128, five SGD passes, learning rate 0.0003, and gamma=1.
Training configuration: [highd_v46_conditional_ppo_v1.json](../configs/highd_v46_conditional_ppo_v1.json).

Each episode has **one intervention decision**, followed by four seconds of
native dynamics with natural later actions. This is a conditional contextual
PPO baseline and warm start, not the full 60-second multi-decision policy.
The two outputs control the existing cross-unit proposal's two likelihood
factors. NDE tables, natural partners, risk matrix, interaction H, strict gates,
physics and CAV-first-collision event remain the existing implementations.
BV-only first collisions remain negative competing endpoints.

The conditional law chooses a selected eligible source episode uniformly,
then one selected state within that source uniformly. Each continuation has
target mass `alpha_case/n_case`; cases from the same source share its mass.
For behavior Qb and candidate Qtheta, the empirical loss estimates
`sum_case alpha_case E_Qb[I * (P/Qb) * (P/Qtheta)]`, using a fixed positive
numerical scale. This is the event second moment, not raw collision count.
Positive replay fraction 0.5 has an exact replay-sampling correction. No event
weight or reward is clipped; all zero-event states and continuations remain.
Conditional validation standard errors use fixed-state stratification.

The new observation contract explicitly sets unavailable **past acceleration**
slots to zero and sets the initial remaining-time context to four seconds.
It never fabricates past accelerations from future outcomes. The remaining
features describe the current CAV, ordered pair, pre-draw interaction mass,
factor masks and signed relative geometry. A distinct data/observation/target
contract prevents this model from silently entering the population collector.

Training imports all 21 selected eligible states from 16 source episodes in
the checksum-locked 200000-200063 training calibration: 1,344 dual-Q draws,
31 CAV events from two sources. The teacher-ranker three-source fitting guard
from the previous task remains unchanged: that ranker is not fitted here.
This PPO baseline can be trained with sparse real labels, but two event-bearing
training sources are a material generalization limit.

Validation uses new source seeds 210000-210031 and continuation seeds from
2600000, 128 per selected state. It retains mixed5 discovery, rank-1/rank-2/audit,
all strict gates and the fixed two-state/source cap. A zero-event validation
collection is retained and blocks standard checkpoint selection. It is never
extended conditionally on outcomes. Checkpoint selection includes iteration 0;
the fixed comparator searches the same validation objective over the Cartesian
grid {0.0501,0.1,0.3,0.6,0.99}. Validation needs at least one event to define
selection, which is not a claim of sufficient statistical precision.

Test sources are 220000-220031. Both portable policies are checksum-locked
**before** these sources are opened. The evaluator rejects source overlap or
policy changes after that lock. It compares natural, first-factor-only,
second-factor-only, fixed (0.1,0.1), validation-selected fixed, and learned PPO
with matched continuation seeds, keeping every endpoint. Factor-only policies
are ablations, not the original paper's single-BV D2RL agent. Test reports show
IS means, second moments, source counts, event ESS, 90% RHW, paired comparisons
and CPU time. Existing 20-event/three-source comparison guards remain active.
Population acceleration stays undefined because this law conditions on a
selected state bank; original-initialization evaluation is still required.

## Reproduce each stage

The original training calibration outputs are required, with the summary SHA
in the configuration. They can be regenerated using the preceding calibration
guide, but timing changes its summary SHA: verify immutable artifacts and use
an explicitly versioned baseline config rather than silently bypassing the lock.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_conditional_baseline prepare --split train --output outputs/v46_conditional_baseline_v1/data/train
python -m scenario_reconstruction.highd_v46_conditional_baseline prepare --split validation --output outputs/v46_conditional_baseline_v1/data/validation
python -m scenario_reconstruction.highd_v46_conditional_baseline train --data outputs/v46_conditional_baseline_v1/data --output outputs/v46_conditional_baseline_v1/seed7 --seed 7 --iterations 100
python -m scenario_reconstruction.highd_v46_conditional_baseline prepare --split test --policy outputs/v46_conditional_baseline_v1/seed7/ppo --fixed outputs/v46_conditional_baseline_v1/seed7/selected_fixed --output outputs/v46_conditional_baseline_v1/data/test
python -m scenario_reconstruction.highd_v46_conditional_baseline evaluate --bank outputs/v46_conditional_baseline_v1/data/test --policy outputs/v46_conditional_baseline_v1/seed7/ppo --fixed outputs/v46_conditional_baseline_v1/seed7/selected_fixed --smoke --output outputs/v46_conditional_baseline_v1/native_smoke
python -m scenario_reconstruction.highd_v46_conditional_report --run outputs/v46_conditional_baseline_v1/seed7 --output outputs/v46_conditional_baseline_v1/curves
```

`--smoke` evaluates the predeclared first 16 continuations per case/arm. Omitting
it runs the full declared 128. Those two runs share the prefix and must never
be pooled as independent data; keep the policies fixed when completing that
budget. A later changed policy needs a newly versioned independent test cohort.
For multiple training seeds, collect/select all policies before defining their
new common test protocol. The current test lock deliberately binds one learned
policy and its fixed comparator.

```bash
python -m pytest tests/test_highd_v46_conditional_baseline.py tests/test_highd_v46_validation.py tests/test_highd_v46_native_calibration.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Raw banks, labels, checkpoints, policy weights and curves stay in ignored
`outputs/`. Only code, protocol, documentation and compact evidence are tracked.
