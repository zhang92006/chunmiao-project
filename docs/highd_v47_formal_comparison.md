# Prospective matched Single-BV / Dual-BV action D2RL

Current execution status: v1 and the separate recovery v2 each completed
both 8,192-draw collections and 2,518 scheduled native replays, but **both
modes still have zero CAV events in both attempts**. Formal PPO has not
started; Validation/FinalTest remain unopened.
The frozen v1 protocol/data are unchanged. The separate
[training-behavior recovery record](highd_v47_formal_recovery.md) retains this
failed action-exploration attempt. The subsequent 7,578 supported temporal
probes also recorded zero CAV events; this does not establish action persistence
as the cause. Do not repeat completed recovery commands or remove the guard.
The subsequent [teacher restoration and source expansion](highd_v47_teacher_expansion.md)
completed the historical reproduction and the full 4,096-source budget: 2,009
CAV events from 209 sources in 66,752 shared behavior trajectories. Collection
size passed, but does not automatically establish target-specific PPO readiness.
The [shared-teacher PPO handoff](highd_v47_shared_teacher_ppo.md) now reconstructs
both independent target likelihoods on this immutable data. Its broader teacher
bank is a different conditional target from the formal v1 bank; this document's
old collection/validation commands are historical, not commands for the new
checkpoints. Production and held-out validation remain unchanged.

This protocol implements the user's formal comparison request on 2026-10-07.
The prior six 100-iteration runs are retained as pipeline smoke evidence in
[the compact record](../results/highd_v47_fixed_alpha_ppo_smoke_v1.json).
They used a strict-B state bank, risk-bin exploration and fixed-last-iteration
selection. They are not the original-critical-state main experiment and their
training moments do not select a formal policy.

## Frozen comparison and scope

The [formal protocol](../configs/highd_v47_formal_comparison_v1.json) fixes
Q_S=alpha, state LR=1 and no state model. Both modes use the SAME source/state
bank, IID state sequence, source/state masses, continuation seeds, four-second
conditional horizon and continuation count. Alpha is uniform over sources
with original Single-critical states, then uniform over their selected states.
Abstaining sources remain in discovery records and coverage denominators.
Every continuation endpoint and every zero-event case stays in the data.

Natural source episodes use the original initialization and fixed 60-second
CAV-first-collision endpoint. The bank checks the original
`NativeExperiment.choose(..., 'single')` at every natural decision. Up to five
eligible times per source are sampled uniformly, independently of native
continuation labels. No pair score or pair eligibility selects the bank.
This bounded bank is a conditional training target, not a natural reset density.

Single uses the existing supported H=P*challenge/sum(P*challenge), one BV and
one epsilon. Dual uses the frozen existing following-pair rule and two-factor
joint proposal from `NativeExperiment.choose(..., 'dual')`. Both apply the
same original Single critical-state gate at each decision. A missing actionable
pair uses that SAME Single action proposal with the second factor inactive.
This rule is the repository's native following-pair D2RL extension; the later
cross-unit interaction-gated proposal and risk-bin teacher are separate designs.
The primary comparison estimates the effect of this frozen Dual action system,
including its pair selection and joint action law; it does not isolate every
internal mechanism or claim exact reproduction of the upstream SUMO experiment.

NDE, probability tables, initialization, lane changes, CAV dynamics, original
criticality thresholds, stopped event, simulator and correlated conditional
partners are inherited without editing their implementation. The freeze pins
all these components, the sampler, PPO, observation and replay reward code,
dependencies and assets. Changing a component invalidates the lock. Critical
sequences omit only natural decisions whose P/Q factor is exactly one. The
full action log likelihood is reconstructed from ALL native decisions.

The 22-feature observation uses actual source episode age / 60 s in training,
validation and original-initialization deployment. Its prefix is the causal
policy action-weight prefix. Snapshot accelerations are retained. Actor
selection updates at each critical decision; actors are not locked for the
entire four-second continuation.

## Source splits and matched budgets

| Split | Natural source episodes | Purpose |
| --- | --- | --- |
| Train | 1000000000–1000000255 | Common original-critical bank and PPO trajectories |
| Validation | 1000100000–1000100255 | Native checkpoint selection only |
| FinalTest | 1000200000–1000201023 | Original-initialization matched comparison |

All source IDs below one billion are conservatively excluded, including old
development, validation and reserved final-test namespaces. Freeze also checks
persisted configuration, result, source-bank, sequence, manifest, protocol and
summary metadata for overlapping used/reserved IDs and saves their hashes.
These checks concern available recorded history; deleted or unrecorded private
experiments cannot be recovered by an inventory. Whole natural episodes define
split membership; no timestamp within a source can move to another split.
Freeze creates all three unopened manifests without simulating a fresh source.

Training has 8,192 real-Q continuations per mode, with shared state RNG seed
1700017 and continuation seeds 1100000000–1100008191. Both initial and subsequent
decisions use fixed behavior epsilon 0.1 during collection. Full source/state
and action likelihoods are audited; scheduled roots are replayed before PPO.
The same empirical second-moment reward formula and exact positive/negative
replay correction are used. Both modes share one training-derived positive
numerical reward scale. No IS weight or event reward is clipped.

PPO uses paired seeds 7/17/29/41/53, 200 iterations, two 64-unit tanh layers,
batch 1,024, minibatch 128, five SGD passes and learning rate 0.0003. The existing
bounded Beta epsilon head, gradient clipping and PPO ratio clipping are retained
as optimizer hyperparameters. They are distinct from clipping IS weights.
Iteration 0 and every tenth iteration are archived as RLlib checkpoints and
portable policies with hashes, giving 21 candidates per run / 210 candidates.

Each iteration records replay reward, empirical scaled/unscaled M2, fixed
dataset event/source counts, candidate likelihood log quantiles, epsilon
quantiles, active second-factor influence, last-minibatch gradient norm,
entropy/KL, policy loss and value loss. Replay TV is labelled conditional TV
at logged action rows; native evaluation computes distribution-level Q/P TV.
Training diagnostics do not count as additional native observations.

## Native Validation and model freeze

All checkpoints receive 1,024 native continuations on a common validation bank,
with the same initial-state schedule and seeds 1200000000–1200001023. The fixed
native budget is 215,040 validation continuations. All plans are written before
each job's outcomes; policy weights and the candidate set are frozen by the
completed training catalog. There is no validation model fitting or extension.
Native M2 is unscaled mean(Y*exp(2*action_log_weight)). Equal M2 breaks toward
the earlier iteration. Iteration 0 wins unless a supported candidate improves
its native M2. An unsupported zero-event candidate cannot win by having M2=0.

Reports retain CAV events, event sources, IS mean/M2, estimator variance, 90%
RHW, source/event concentration, effective M2 sources, weight p95/p99/max and
log tails, exact full likelihood and both actual factor Q/P TVs. Sparse or
degenerate observed evidence leaves precision unsupported. No epoch or seed
is discarded after selection. The seal reconstructs every checkpoint report
from retained hashed native endpoints and verifies the selection and weights.

FinalTest opens only after the predeclared stop checks pass: at least three
improved seeds per mode; 20 events across five sources for each selected policy;
maximum source and single-event M2 shares <=0.8; effective M2 sources >=3; and
effective two-factor coverage for Dual. Otherwise seal fails with explicit
reasons, leaves FinalTest unopened and retains all failures. Stop checks are
engineering evidence requirements, not a guarantee of population superiority.

## Original-initialization FinalTest

Each of the ten selected policies and natural / fixed-Single / fixed-Dual
controls receives all 1,024 ORIGINAL resets, not bank continuations. Every arm
has the same seed, initial fleet, CAV and 60-second horizon. Exact initial-state
identities are verified after sampling; common random numbers provide partial
coupling despite different factor draw counts. State LR is one. Full action
P/Q corrects every intervention through the stopped path.

The fixed final budget is 13,312 full episodes. There is no model update,
checkpoint replacement, threshold adjustment or positive-only restart.
Results report every paired training seed, source contributions, M2/variance/
RHW curves, likelihood tails and episode/critical-decision CPU. Native paths
sharing a source across policies are paired observations, not extra independent
traffic sources. Each per-seed comparison resamples whole sources in common.
Five percentile-bootstrap intervals use nominal Bonferroni family adjustment;
more than 10% censored replicates leaves an interval undefined. These remain
finite-sample bootstrap diagnostics with event/source guards, not distribution-
free variance certificates for heavy tails.

AF is N_Single/N_Dual at the same RHW threshold (0.3/0.2/0.1), requiring an
observed suffix to stay below the threshold and sufficient source evidence.
If either arm never reaches precision, observed AF is undefined. Asymptotic
plug-in required-count projections remain separate from observed efficiency.
The final conclusion supports consistent Dual superiority only when all five
paired M2 ratio intervals lie below one AND all final policies satisfy the
same event/source, concentration, tail and two-factor evidence guards; otherwise the report explicitly states
that consistent superiority is not established. No best-seed promotion occurs.

## Commands for the researcher

### Frozen-v1 JSON identity compatibility, 2026-10-07

The first completed bank exposed a representation bug: `last_acceleration`
uses integer actor-ID keys in memory, but JSON reloads them as strings.
`identity(..., sort_keys=True)` sorts integers numerically before saving and
strings lexicographically after loading. The outer bank checksum remains
correct while 1,254 of the 1,263 inner state hashes fail the naive read-back
check. Restoring ONLY that mapping's original integer keys reproduces every
stored state hash; case IDs, sources and alpha masses match without changes.
This is not changed physical state content or a likelihood error.

Do NOT edit bank hashes, re-freeze the parent, delete sources or repeat discovery.
The separately versioned
[representation-only adapter](../configs/highd_v47_formal_identity_compat_v1.json)
pins the original formal configuration/runner and preserves every parent code,
lock and data byte. It retains outer-file checksums and exact inner identity
checks, allowing only the provably reproducible pre-JSON acceleration-key hash.
It neither changes returned states nor modifies Q, alpha, seeds, PPO or budgets.

For an existing v1 frozen lock and training bank, declare the adapter once:

```bash
python -m scenario_reconstruction.highd_v47_formal_identity_compat prepare \
  --frozen outputs/v47_formal_v1_frozen \
  --bank outputs/v47_formal_v1_train_bank \
  --output outputs/v47_formal_v1_identity_compat.json
```

Subsequent commands below already use the compatibility entry point and
`--identity-compat outputs/v47_formal_v1_identity_compat.json`. All sampling
arguments stay unchanged. The renderer needs no change.
Every delegated command writes an exclusive-create provenance sidecar, verifies
the adapter/config hashes and original parent lock, and uses the unchanged
original entry point except for its bank identity reader. The same compatibility
applies to the later validation bank; it is not a new sampling protocol.

The local failure occurred before training collection created its destination:
there was no `outputs/v47_formal_v1_data/summary.json` or audit. Start with
`collect-train`, not `audit-train`, after declaring compatibility. If a complete
collection already exists on another machine, retain it and proceed to audit;
never overwrite or append to partial collection outputs.

The compatibility regression suite plus the six original relevant modules
passed 103 tests. A read-only reconstruction checked all 1,263 states and all
21 parent frozen components; the original bank/summary/state-law hashes and
state LR=1 remain intact. No source was regenerated and no native trajectory
was run. Compact diagnostic evidence:
[identity compatibility v1](../results/highd_v47_formal_identity_compat_v1.json).

The following commands document the original v1 workflow. Its completed
zero-event collection must not be overwritten or retried until favorable;
use the [versioned recovery commands](highd_v47_formal_recovery.md) now, not
the v1 `train` command below.

The assistant prepares and tests the implementation. Run long jobs yourself.
Use NEW output directories; preserve failed runs. Changes to code/configuration
after freezing require a versioned protocol and lock, not a silent re-freeze.
Do not pool a labelled restart with an earlier attempt using repeated seeds.

```bash
# Run from the repository root.
conda activate highd-ppo-clean39
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m scenario_reconstruction.highd_v47_formal_comparison freeze --output outputs/v47_formal_v1_frozen
python -m scenario_reconstruction.highd_v47_formal_comparison bank --frozen outputs/v47_formal_v1_frozen --split train --output outputs/v47_formal_v1_train_bank
python -m scenario_reconstruction.highd_v47_formal_identity_compat prepare --frozen outputs/v47_formal_v1_frozen --bank outputs/v47_formal_v1_train_bank --output outputs/v47_formal_v1_identity_compat.json
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json collect-train --frozen outputs/v47_formal_v1_frozen --bank outputs/v47_formal_v1_train_bank --output outputs/v47_formal_v1_data
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json audit-train --frozen outputs/v47_formal_v1_frozen --bank outputs/v47_formal_v1_train_bank --data outputs/v47_formal_v1_data --replay --output outputs/v47_formal_v1_data/audit.json
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json train --frozen outputs/v47_formal_v1_frozen --data outputs/v47_formal_v1_data --output outputs/v47_formal_v1_training
```

After all ten training runs have completed, the next commands collect the
independent validation bank and measure every frozen checkpoint:

```bash
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json bank --frozen outputs/v47_formal_v1_frozen --split validation --training outputs/v47_formal_v1_training --output outputs/v47_formal_v1_validation_bank
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json validate --frozen outputs/v47_formal_v1_frozen --bank outputs/v47_formal_v1_validation_bank --training outputs/v47_formal_v1_training --output outputs/v47_formal_v1_validation
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json seal --frozen outputs/v47_formal_v1_frozen --validation outputs/v47_formal_v1_validation --output outputs/v47_formal_v1_selected_lock.json
```

Read the selection and stop-condition report before running FinalTest. If seal
fails, stop there. If it creates the selected lock:

```bash
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json final-test --frozen outputs/v47_formal_v1_frozen --policy-lock outputs/v47_formal_v1_selected_lock.json --output outputs/v47_formal_v1_final_test
python -m scenario_reconstruction.highd_v47_formal_identity_compat --identity-compat outputs/v47_formal_v1_identity_compat.json audit-final --frozen outputs/v47_formal_v1_frozen --result outputs/v47_formal_v1_final_test --replay --output outputs/v47_formal_v1_final_audit.json
python -m scenario_reconstruction.highd_v47_formal_report --result outputs/v47_formal_v1_final_test --audit outputs/v47_formal_v1_final_audit.json --output outputs/v47_formal_v1_report
```

Each training run prints its log path. Follow it with `tail -f`; terminating
the log viewer leaves the separately running training process intact. Raw
trajectories, matrices, checkpoints and logs stay in ignored outputs. This
document and protocol currently declare the experiment; no formal native
result or FinalTest conclusion has been generated.

The final audit reconstructs all paired statistics from hashed paths and
replays one predetermined original-reset path per arm (13 total). Replays are
verification, not extra data. The report renders M2/variance/RHW curves for
all five seeds and retains censored AF/intervals as null. Matplotlib's version
and the renderer hash accompany the figures; rendering never changes policies,
statistics or the sampling protocol.

## Implementation checks, 2026-10-07

All 84 tests in the formal comparison, fixed-alpha PPO, native D2RL, validation,
cross-unit likelihood and conditional-baseline modules passed. Frozen NDE and
runtime asset verification also passed. Two-iteration synthetic PPO checks for
both modes exercised portable checkpoint export (including iteration 0),
per-iteration diagnostics, masked factors and state LR=1. Their portable-policy
maximum discrepancy was 1.49e-8. They were interface checks, not native fitting
or formal performance evidence, and opened none of the new source splits.

```bash
python -m pytest tests/test_highd_v47_formal_comparison.py tests/test_highd_v47_fixed_alpha_ppo.py tests/test_highd_v46_validation.py tests/test_highd_v46_d2rl.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_conditional_baseline.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```
