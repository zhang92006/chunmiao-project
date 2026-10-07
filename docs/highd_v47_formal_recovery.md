# Formal training status and versioned behavior recovery

## Where the experiment actually stopped

The completed formal v1 training bank contains 1,263 states from 253 selected
sources (256 natural source episodes were retained). Single and Dual each
completed 8,192 continuations under the same alpha law. The native audit
verified 2,518 predetermined state/mode replays; maximum complete/factor log
error was 4.04121180963557e-14. The bank identity compatibility is working.

Both collections contain **zero actual CAV first collisions**. Single has 77
BV-only competing endpoints and Dual has 78; these stay negative. Formal PPO
has not started, and Validation/FinalTest have not been opened. The earlier
six 100-iteration jobs are still smoke runs, not these formal runs. The
complete zero-event data and audit remain immutable, with collection summary
SHA `584cf11b70d9f0e83b54f6d2ab45e49de8de4ca036586adbc897ecc5b56b7e9f`.

The current failure is **missing training reward signal**, not an identity or
likelihood exception. Removing the replay guard cannot repair it: every
event-weighted objective term would still be zero. BV-only collisions cannot
be relabelled as the target event.

## Demonstrated mismatch, and what remains uncertain

The inherited `NativeExperiment.challenge` holds selected initial BV
accelerations throughout its four-second surrogate calculation; native
rollouts resample BV decisions every second. Its exponential minimum-clearance
score is not a collision probability. It also does not implement the native
first-collision competing-endpoint stopping rule. These are directly observed
code-level differences, not evidence of a corrupted simulator or likelihood.

The original critical gate admitted a very broad bank: the median natural
source had 56 eligible decision times. Single sampled an initial proxy score
>=0.5 in 169 continuations, yet none produced a CAV endpoint. Thus high proxy
score alone was not enough to produce training labels. The finite cohort does
not prove that all selected states have zero CAV risk, nor isolate one mismatch
as the sole cause of every negative continuation.

Do not relax the original state gate, change the proxy/target H, enlarge the
horizon, change the CAV reward, or open held-out sources to manufacture events.
Instead the separately versioned recovery changes **training behavior only**.
It is a development repair after a failed full collection, not an unnoticed
extension of formal v1 or a successful efficiency result.

## Separate behavior exploration from the evaluated proposal

The [recovery v2 protocol](../configs/highd_v47_formal_recovery_v2.json) keeps
the exact common original-Single bank, alpha, original actor/pair choice,
Single fallback, target H, original native physics, stopped event and PPO.
Q_S=alpha and state LR=1. No transferable state model is fitted.

At each original critical decision, behavior H is uniform over naturally
supported selected action cells, zero elsewhere. Single uses its selected
action marginal; Dual factorizes this supported joint behavior. The existing
epsilon=0.1 factor sampler and exact natural conditional partners are reused.
Natural decisions outside the original critical gate remain natural. The
target proposal optimized by PPO and used by Validation/FinalTest is still
the original `epsilon*P+(1-epsilon)*H_target`, not this exploratory behavior.

Every step records target H and behavior H separately. The full behavior LR
is reconstructed from every native decision. Replay recomputes the target LR
using target H and the candidate policy's causal target prefix. The objective
remains exactly

`E_Q_behavior[Y * (P/Q_behavior) * (P/Q_target_epsilon)] = M2(Q_target_epsilon)`.

This is a change to **training data acquisition**, not to the event or evaluated
proposal. Substituting behavior H for target H, or using twice the behavior log
weight in the numerical reward scale, would optimize the wrong objective.
The new replay contract explicitly rejects both mistakes. Original replay
episode corrections and the PPO optimizer/head are inherited unchanged.
Runtime sidecars pin the actual off-policy replay subclass as well as its
unchanged base environment; the old trainer's base-environment hash alone
must not be mistaken for complete runtime provenance.

## Fixed budget and clean stop conditions

Declare one complete collection of 8,192 continuations/mode, with the same
state RNG 1700017 and new common continuation seeds 1300000000–1300008191.
All states, negatives and competing endpoints remain. Existing v1 outcomes
are retained as failure evidence, not pooled into new estimates or used to
select positive states. The complete schedule, original input hashes, new
implementation, environment and target identities are frozen before outcomes.

Collection computes training readiness before any Ray/PPO process. Each mode
requires at least 20 nonzero, finite CAV reward observations with an active
target P/H contrast, across at least five training sources.
Dual must additionally contain event-weighted active second-factor contrast;
an entirely padded-Single reward dataset cannot launch it as a Dual agent.
The gate also reports
training-objective event concentration and effective event count; the count
gate is not a claim of balanced natural-risk mass or good estimator variance.
Only a complete native replay audit and passing readiness permit PPO. A
scientific stop returns `training_blocked`, exits 2 and creates no PPO output
directory. It does not suggest collecting indefinitely or removing the guard.

This design **does not guarantee enough events**. If it fails its fixed budget,
retain the negative result; no extra batch, positive-only restart, model change
on Validation or automatic retry is authorized. Behavior event frequencies
are not target-Q evaluation probabilities or efficiency evidence. Actual
recovery outcomes and formal PPO results are pending researcher execution.

## Commands for the researcher

Run these after reviewing/committing the new files. Do not overwrite any v1
output. The new exclusive-create destinations below must not already exist.

```bash
conda activate highd-ppo-clean39
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m scenario_reconstruction.highd_v47_formal_recovery freeze \
  --parent-frozen outputs/v47_formal_v1_frozen \
  --bank outputs/v47_formal_v1_train_bank \
  --data outputs/v47_formal_v1_data \
  --output outputs/v47_formal_v2_recovery_frozen

python -m scenario_reconstruction.highd_v47_formal_recovery collect-train \
  --frozen outputs/v47_formal_v2_recovery_frozen \
  --output outputs/v47_formal_v2_recovery_data

python -m scenario_reconstruction.highd_v47_formal_recovery audit-train \
  --frozen outputs/v47_formal_v2_recovery_frozen \
  --data outputs/v47_formal_v2_recovery_data \
  --replay \
  --output outputs/v47_formal_v2_recovery_data/audit.json

python -m scenario_reconstruction.highd_v47_formal_recovery train \
  --frozen outputs/v47_formal_v2_recovery_frozen \
  --data outputs/v47_formal_v2_recovery_data \
  --output outputs/v47_formal_v2_recovery_training
```

The ten formal jobs retain seeds 7/17/29/41/53, 200 iterations, checkpoint 0
and every tenth iteration, and a common training-derived numerical scale.
Progress is printed at collection/replay boundaries. PPO logs go to each
`single_seed*.log` / `dual_seed*.log` in the new training directory.

Do not run the old identity wrapper's `train` on recovery data: it intentionally
does not accept the new off-policy role. The recovery entry point also provides
`bank --split validation`, `validate`, `seal`, `final-test` and `audit-final`,
delegating to the original native target implementation after recovery checks.
Its `seal` and `final-test` additionally require `--training`; keep these stages
for after complete audited training. Original source-disjoint validation,
checkpoint selection and FinalTest stop criteria are not relaxed. No validation
or final run is part of the four training commands above.

Read-only status remains available:

```bash
python -m scenario_reconstruction.highd_v47_formal_recovery status \
  --parent-frozen outputs/v47_formal_v1_frozen \
  --bank outputs/v47_formal_v1_train_bank \
  --data outputs/v47_formal_v1_data
```

Implementation tests cover distinct behavior/target factors, correlated
partners, zero-target rows, inactive factors, exact multidecision off-policy
M2 and derivatives, replay correction, causal prefixes, label validation,
fixed state law and stopping before PPO with zero/sparse reward. Short synthetic
PPO checks are interface tests only, never native evidence or a formal launch.

Implementation verification: 118 relevant regression tests passed; two
iterations of synthetic PPO completed for each action dimension with state
LR=1 and portable checkpoints. Frozen code/assets verified. No native recovery
collection, formal PPO or held-out source evaluation was run by the assistant.
