# Source-disjoint native validation of shared-teacher PPO

All ten Single/Dual PPO runs completed 200 iterations, retaining 21 checkpoints
per run (0,10,...,200). The completed recovery summary SHA is
`55772f97174ef69f8e590ceaa8868fbc0e93e7233f07fd2190c50df56e337564`.
Single seed 7 remains at its original location; the remaining nine runs are
linked from the recovery summary. Training used the same 66,752 fixed behavior
paths; those paths and the restored teacher remain unchanged.

This is a prospective independent **validation/checkpoint-selection** stage,
not a second training stage and not a post-selection FinalTest. The
[protocol](../configs/highd_v47_shared_teacher_validation_v1.json) must be
reviewed, committed and frozen before opening any reserved validation episode.

## Common held-out source and state law

Open exactly the previously reserved 1,024 sources 1800010000–1800011023.
They are disjoint from all 4,096 teacher training sources and earlier consumed
research cohorts. The new entry point accepts this exact range, never a
training/consumed-validation replacement. No FinalTest source is opened.
Freeze validates training dependencies, provenance and all 210 portable
checkpoint identities before creating the validation lock; it opens no source.
The bank stage creates an exclusive source-claim record before its first rollout,
preventing another invocation from silently reopening the same reserved cohort.

Discovery is identical to teacher acquisition: natural episodes, mixed5 bank,
bounded rank-1/rank-2/audit pairs, two original calibration-style strict cases
plus maximum supported risk >=0.5, with unchanged ties and deduplication.
At most three cases per source are retained, based on pre-continuation features,
never native continuation labels. All source rows, abstentions, natural CAV
endpoints and BV-only competing endpoints remain recorded. The inherited
low-level discovery helper's historical `training_discovery` progress label is
not fitting: the new bank explicitly declares validation-only usage.

Alpha is uniform over the selected sources, then uniform over their selected
cases. Both agents and every checkpoint receive the identical outcome-blind
4,096-draw IID state schedule from alpha, with state RNG seed 2000017 and
continuation seeds 2000000000–2000004095. Thus Q_S=alpha and state LR=1.
Abstentions are part of discovery coverage, not imputed zero-risk continuations.
Training was fixed-stratum collection; validation is IID alpha, so its raw
sample means already target the same uniform-source/within-source case law.
No learned Q_S, per-source risk rebalancing or original-reset density correction
is introduced. This is a new conditional bank drawn by the same discovery rule,
not the old training bank or the failed formal v1 original-Single-critical bank.

## Native policies and fixed budget

Each of all 210 locked checkpoints receives 4,096 real target-policy draws.
An additional unchanged-natural reference gets the same schedule. Total:
**864,256 logical native continuations**, plus 1,024 natural discovery episodes.
Discovery has at most 5,120 bank states and 15,360 bounded pair audits.
Every continuation stops at the first any-actor collision or four seconds;
CAV involvement defines the positive event, and BV-only first collisions stay
negative competing endpoints. Every endpoint is saved, including zero events.

The online target choices are the same independent original Single selector
and common-Single-gated following-pair Dual with Single fallback used during
training. PPO epsilon applies at each active decision, with causal target
likelihood-prefix observation and elapsed-time/four-second feature. Snapshot
restoration matches the teacher replay exactly; the older formal native entry
point's additional acceleration-history restoration/global-time observation
is not silently substituted. Pair correlations and conditional natural partners
remain intact. Single timing does not include an unnecessary Dual selector call.

Likelihood is the full stopped-path product P/Q_target, with state factor one.
Sampling is now under each target, so M2 is mean(I*(P/Q_target)^2), NOT the
training off-policy product involving Q_behavior. Logged factor/full densities,
observations, masks, actions and all labels are retained. Reused seeds supply
common random numbers, not independent cross-checkpoint evidence. Iteration-0
replications across training seeds must not be pooled as extra independent data.

Four workers alter execution order only. Each arm is stored in 128-draw shards;
all schedules and model identities are saved before continuation outcomes.
There is no stop-at-positive rule, favorable-seed selection, extra continuation,
retuned epsilon or changed checkpoint grid after seeing results. Even a run
with few/no events is a complete retained validation result, not a reason to
recollect until favorable. An empty eligible bank stops explicitly.

## Selection, uncertainty and audit

Use the existing native M2 selection rule, retaining iteration 0 unless a
strictly lower M2 candidate meets the minimum observed evidence guard of 20
weighted events across five sources. Ascending iteration breaks ties. An
unsupported baseline or all-zero empirical moment cannot promote a trained
policy. All five training seeds remain paired and reported; no best-seed pick.

Report event counts/rates, conditional IS mean/M2/variance, weight tails,
source/event M2 contributions, effective M2 sources and second-factor TV.
The existing complete screening checks remain (share <=0.8, effective sources
>=3, effective Dual factor coverage, at least three improved seeds per mode).
Passing a numerical selection rule or a training-signal gate alone does not
establish a statistically significant policy improvement.

Reported Monte Carlo precision is conditional on this fixed validation bank.
Repeated continuations from one source are NOT independent traffic sources.
There is no source-population bootstrap claim here. The checkpoint search also
uses these outcomes; winner metrics are selection-biased, not an independently
confirmed Single-versus-Dual result or population acceleration estimate. Even
if every screening check passes, this entry point does not open FinalTest or
automatically announce Dual superiority. A later untouched test protocol is
required after selected policies are reviewed and sealed.

The audit hashes all shards, reconstructs the entire outcome-blind bank and
IID schedule, all policy/factor/full likelihoods and every checkpoint metric,
then replays the predetermined first draw of each arm (211 checks). Timing
fields alone are excluded from native replay equality. Replays add no data.
It reconstructs checkpoint selection/readiness and writes a separate audit.

## Researcher-run commands

Use the inherited highd-ppo-clean39 environment. The complete script freezes,
discovers, runs and audits in order; it stops on any error. On macOS,
`caffeinate -i` prevents idle system sleep while the script runs (keep the
laptop plugged in and open). This is a substantial multi-hour job; no native
validation was launched during implementation.

```bash
conda activate highd-ppo-clean39
caffeinate -i bash scripts/run_v47_shared_teacher_validation.sh
```

Outputs:

- `outputs/v47_shared_teacher_validation_v1_lock.json`
- `outputs/v47_shared_teacher_validation_v1_bank/`
- `outputs/v47_shared_teacher_validation_v1/selection.json`
- `outputs/v47_shared_teacher_validation_v1/summary.json`
- `outputs/v47_shared_teacher_validation_v1_audit.json`

All destinations are exclusive-create. Do not delete a source claim, overwrite
an opened protocol, rerun the whole script after an interruption or point old
formal-validation commands at these models. Retain the exact partial attempt
and log for explicit recovery review. No automatic interruption restart exists.

Implementation preflight passed 95 relevant regression tests, including a
synthetic bank/run/audit pipeline, exclusive source claims, sparse-event
selection guards and Single/Dual factor/partner-law checks. All 446 protected
training/checkpoint input files verified. Five bounded diagnostics on an OLD
training state (natural; Single and Dual at iterations 0 and 200), each repeated
once with the same seed, matched deterministic native replay. Maximum full /
factor likelihood error was 1.33e-14. They are implementation checks, not fresh
validation or additional fitting data. Frozen NDE/assets and shell syntax
verified; the reserved validation cohort remains unopened at this milestone.

```bash
python -m pytest tests/test_highd_v47_shared_teacher_validation.py tests/test_highd_v47_shared_teacher.py tests/test_highd_v47_shared_teacher_recovery.py tests/test_highd_v47_formal_comparison.py tests/test_highd_v47_teacher_expansion.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```
