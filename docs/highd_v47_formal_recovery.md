# Formal training status and versioned behavior recovery

**Current status:** recovery v2 also completed with zero CAV events in both
modes. The historical reproduction commands below are not instructions to
repeat it. See the final section for the new, bounded reachability preflight.
Formal PPO and held-out evaluation have not started.

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
recovery outcomes are recorded below; formal PPO remains blocked.

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

## Completed recovery v2: exploration alone did not supply reward

The researcher completed the full 8,192 draws per mode and native audit.
Single has 8,111 horizon endpoints, 81 BV-only first collisions and zero CAV
events; Dual has 8,112, 80 and zero respectively. All 2,518 prescribed native
replays matched, with maximum full/factor log error 5.684341886080802e-14.
The guard correctly returns `training_blocked` before Ray/PPO. This is a
failed training-data design, not another Python exception to suppress.

The competing BV endpoints account for approximately 1% of each collection.
They do not explain why the other 99% also lack CAV reward. Recovery expands
the action law only on the actors chosen by the original selector; it neither
explores actor identity nor enforces temporal persistence of an adverse action.

Read-only reconstruction of all 1,263 bank states found Single's initial
selected actor to be the direct CAV leader in 1,071 states, a same-lane rear
actor in 156, another-lane actor in 35 and a more distant same-lane front
actor in one. Only 114 states have selected-actor maximum supported original
proxy risk >=0.5. This threshold is descriptive, not a new selection rule.
The proxy's four-second held-action calculation is not the native one-second
reselection/resampling process. An exponential clearance score also assigns
positive criticality to close but successfully controlled following traffic.

As a bounded read-only physics check, the first 32 **existing** recovery paths
per mode were replayed using their stored complete actions. All endpoints and
elapsed times matched. All 32 Single initial choices were direct leaders;
Dual included those 32 leaders and 15 further same-lane front actors. Minimum
CAV clearances over these old paths were at least 6.68766 m (Single) and
6.41220 m (Dual). Native CAV acceleration responded to the executed actions.
This subset is not proof of safety for the full bank, and does not establish
which mechanism caused every zero. No new continuation was sampled for this
check. The negative collections do not establish zero true CAV probability.

## Next command: supported-path reachability, not another PPO retry

The new [protocol](../configs/highd_v47_formal_reachability_v1.json) and
`highd_v47_formal_reachability` entry point test the missing temporal-control
hypothesis on **every** one of the 1,263 locked training states. Each mode
receives one four-second probe per state under each of three declared rules:

| Probe | Initial decision | Later decisions |
| --- | --- | --- |
| `initial_peak` | Original selected actor(s), supported maximum-risk action | Natural actions |
| `reselected_peak` | Same | Original actor selector, supported maximum-risk action at each eligible decision |
| `persistent_peak` | Same | Keep initial actor identities, recompute their supported maximum-risk actions each second |

Maximum-risk ties use largest natural action mass, then flat action index.
These are **forced supported-action diagnostics**, not probabilistic teachers.
All unforced partners follow the exact natural conditional distribution.
Persistent pairs use their current joint marginal even if natural pairing
changes; the original evaluated target selector is not overwritten. Persistence
continues independently of later original-gate eligibility, explicitly as a
diagnostic intervention. Neither persistent identities nor persistent peak
actions are silently adopted by Single/Dual evaluation.

The complete plan has 7,578 probes; Single and Dual have identical state
coverage and budget. Seeds 1600000000–1600001262 are disjoint by case and
shared across arms/modes as common random numbers, not independent source
replications. All negatives, sources and BV-only stopped endpoints remain.
The bank target alpha and all frozen training/evaluation laws stay unchanged.
The exhaustive diagnostic schedule is not an IID draw from alpha; it produces
no IS mean, M2, state-LR estimate or PPO manifest. Every path records forced
action natural mass, original target actors, actual actors/actions, conditional
partner log probability, native clearance and first-collision identities.

A witness under `reselected_peak` would establish a supported path accessible
without changing actor choice and motivate a separately declared risk-directed
behavior proposal. A witness only under `persistent_peak` would motivate
investigating actor continuity. Neither finite contrast proves a unique cause;
unforced futures differ after actions diverge. A negative probe is not a
certificate of unreachability. There is no automatic further batch or teacher
promotion, and these forced labels cannot be inserted into ordinary PPO replay.

Run the following once after code/protocol review and commit. No long native
job is launched by the assistant. The two commands stop on errors and use new
exclusive-create destinations; do not delete old evidence or rerun old training.

```bash
conda activate highd-ppo-clean39
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m scenario_reconstruction.highd_v47_formal_reachability freeze \
  --recovery-frozen outputs/v47_formal_v2_recovery_frozen \
  --data outputs/v47_formal_v2_recovery_data \
  --output outputs/v47_formal_reachability_v1_frozen \
&& python -u -m scenario_reconstruction.highd_v47_formal_reachability run \
  --frozen outputs/v47_formal_reachability_v1_frozen \
  --output outputs/v47_formal_reachability_v1
```

Progress prints every 32 probes, including the current CAV witness count.
The final summary is `outputs/v47_formal_reachability_v1/summary.json`.
After completion, a separate native replay audit is available:

```bash
python -u -m scenario_reconstruction.highd_v47_formal_reachability audit \
  --frozen outputs/v47_formal_reachability_v1_frozen \
  --data outputs/v47_formal_reachability_v1 --replay \
  --output outputs/v47_formal_reachability_v1/audit.json
```

Implementation tests check support/ties, current conditional partners,
actor-persistence boundaries, first-collision stopping, complete schedules,
exclusive output creation and freeze-before-outcomes. They are not evidence
that this prospective diagnostic has found CAV events. The zero-CAV training
problem remains unresolved until a supported probabilistic collection actually
passes the training-signal gate.

Verification for this addition: 121 relevant regression tests passed and frozen
code/assets verified. A read-only load of the actual completed recovery passed
its locks and reproduced the full 7,578-task plan without sampling outcomes.
Compact evidence: [zero-signal v2](../results/highd_v47_formal_zero_signal_v2.json).
