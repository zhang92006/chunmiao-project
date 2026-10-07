# Restore the collision-producing teacher, then expand training sources

Execution update: the researcher completed the full historical reproduction,
expansion and native audit described below. See the completed evidence at the
end; do not rerun completed collection commands. The next job is the
[shared fixed-behavior Single/Dual PPO handoff](highd_v47_shared_teacher_ppo.md).

This supersedes further zero-CAV recovery reruns, not their retained evidence.
The original V47 teacher is the positive control: 67 sampled CAV first
collisions / 2,112 continuations (3.1723%), across seven event sources from
25 selected sources / 33 cases. Those selected sources came from 128 natural
source episodes. The fixed behavior, not a trained PPO target, generated them.
The failed formal v1/recovery v2 PPO runs never started; their zero events
cannot be attributed to a learned target suppressing collisions.

## Restoration evidence and gate

On 2026-10-07, the current teacher implementation and all inherited dependencies
matched the **original** lock, with no repinning or destructive Git rollback.
All 68 original artifact hashes verified. A short native replay reproduced
41 historical trajectories exactly, ignoring execution timing only: the first
continuation of every case and the first positive continuation in each positive
case, deduplicated. These diagnostic replays include all seven collision sources;
they are not new samples or an independent collision-rate estimate.

The full 2,112-draw reproduction is intentionally left for the researcher to
run. It uses the original lock and seeds, writes to a new exclusive directory,
and must reproduce every design, action likelihood, endpoint, label and moment,
not just the aggregate count 67. Only timing and timing-induced artifact hashes
may differ. The expansion freeze refuses an absent/failed full reproduction.
No historical data or old freeze is overwritten.

## Fixed behavior and fresh source budget

[Protocol](../configs/highd_v47_teacher_expansion_v1.json):

- Exactly 4,096 new TRAINING sources, 1800000000–1800004095.
- 1800010000–1800011023 are reserved for later source-disjoint validation;
  this entry point cannot open them. All earlier cohorts remain separate.
- Inherited natural initialization, mixed5 bank and rank-1/rank-2/audit pairs
  in the existing nearest-eight/115 m universe, including cross-unit pairs.
- Per source, take the calibration's highest-interaction-mass strict case;
  take a second distinct state, preferring another mechanism, when available.
  Add the maximum-supported-risk pair if risk >=0.5, irrespective of the
  cooperative strict gate. Deduplicate state/ordered-pair identities. Ties use
  ascending time and actor IDs. At most three cases/source; all absent and
  below-threshold sources remain in the discovery record.
- This rule, applied read-only to the old two calibration banks, selects
  **exactly the original 33 teacher cases**, including all negative cases and
  all eight event-bearing cases. New sources need not reproduce old yield.
- The identical teacher H chooses a nonempty supported risk bin uniformly,
  then a supported action cell uniformly. Both factor epsilons remain 0.1.
  Conditional natural partners remain correlated correctly. Only the first
  decision is tilted; the remaining four-second continuation is natural.
- Exactly 64 independent continuation seeds per case; source/case/action laws
  and the full schedule are saved before any teacher outcomes. Continuation
  seeds start at 1900000000. Maximum 12,288 cases / 786,432 continuations;
  actual counts depend only on outcome-blind discovery, not collision results.

The maximum discovery budget is 20,480 states / 61,440 pair audits, plus one
proxy re-audit for each selected case. Four workers change execution order
only. Discovery runs in fixed 64-source chunks and reports progress; every
chunk must complete before collection begins. Interrupted artifacts remain
intact. Outputs are exclusive-create; there is no automatic resume, extension,
positive-only restart or stop-at-50/100 rule. Do not rerun the whole script over
existing directories or delete partial outputs to make a failure disappear.

The target is uniform selected source, then uniform cases within source.
There is **no learned Q_S and no state likelihood tilt**. As in the original
teacher, collection is a fixed 64-draw-per-case stratified design, not IID
uniform-source sampling. Thus summaries use alpha-weighted stratum means and
within-stratum variances. A later uniform-case replay requires the separately
logged `number_of_cases * alpha` design correction; calling state LR=1 does
not permit pooling the cases unweighted. Single and Dual must use the same
data and target; this collector does not create separate mode-specific samples.

## Data-size acceptance versus PPO readiness

The requested **50–100 sources means independent collision-bearing source
episodes**, not total opened sources, trajectory seeds or repeated collisions
at one state. Predeclare a minimum of 50 collision sources and 500 real sampled
CAV events, with 100 collision sources desired. These are engineering data-size
checks, not statistical sufficiency guarantees. Scaling the old seven sources
cannot guarantee reaching them. All 4,096 sources are processed regardless of
intermediate counts; the final report says whether the fixed attempt passed.

Report raw event yield, all-source yield, alpha-weighted mean/M2, per-source
event counts, largest event/source M2 shares and effective M2 sources. Zero-event
concentration is undefined, not zero true risk. High source/event counts alone
can consist of extremely low-natural-probability actions; they do not prove
useful reward diversity or lower testing variance.

Consequently `collision_count_gate_passed` and `ppo_ready` are separate;
**`ppo_ready` remains false here**, even when the count gate passes. The next
handoff must audit both independent Single-BV and Dual-BV target likelihoods
on this SAME fixed-behavior dataset, useful event-weighted reward/gradient
contrast (including Dual's second factor), source concentration, and replay
design corrections. Do not relabel a first-factor ablation as the paper's
Single agent. Reward is based on `-I*(P/Q_behavior)*(P/Q_target)`; target PPO
updates must not delete or recollect behavior collision labels. This stage
restores acquisition and scale, not automatic authorization of formal PPO.

## Researcher-run commands

Use Python 3.9.19, NumPy 1.23.1, SciPy 1.13.1, one BLAS/OMP thread. From the
repository root, run the checked-in, fail-fast sequence:

```bash
conda activate highd-ppo-clean39
bash scripts/run_v47_teacher_expansion.sh
```

The script runs the following stages, stopping on the first failure. Expect a
long, potentially multi-hour job for the expansion; this was not started by
the coding agent. It does not train PPO or open validation.

```bash
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_action_teacher run \
  --lock outputs/v47_action_teacher_v1_lock.json \
  --output outputs/v47_action_teacher_v1_reproduction
python -m scenario_reconstruction.highd_v47_teacher_restore \
  --reproduction outputs/v47_action_teacher_v1_reproduction \
  --output outputs/v47_teacher_full_reproduction_v1_audit.json
python -m scenario_reconstruction.highd_v47_teacher_expansion freeze \
  --reproduction-audit outputs/v47_teacher_full_reproduction_v1_audit.json \
  --output outputs/v47_teacher_expansion_v1_lock.json
python -m scenario_reconstruction.highd_v47_teacher_expansion discover \
  --lock outputs/v47_teacher_expansion_v1_lock.json \
  --output outputs/v47_teacher_expansion_v1_bank
python -m scenario_reconstruction.highd_v47_teacher_expansion collect \
  --lock outputs/v47_teacher_expansion_v1_lock.json \
  --bank outputs/v47_teacher_expansion_v1_bank \
  --output outputs/v47_teacher_expansion_v1_data
python -m scenario_reconstruction.highd_v47_teacher_expansion audit \
  --lock outputs/v47_teacher_expansion_v1_lock.json \
  --bank outputs/v47_teacher_expansion_v1_bank \
  --data outputs/v47_teacher_expansion_v1_data --replay \
  --output outputs/v47_teacher_expansion_v1_audit.json
```

The audit reconstructs discovery selections, serialized state identities,
source/case masses, every sample's initial action and full conditional-partner
likelihood as well as P/H/Q factors, schedules, labels and moments,
then replays the predetermined first continuation of every selected case.
Replays never add training observations. Synthetic stage-wiring tests open no
native source; old-positive-control replays are explicitly distinguished from
fresh-cohort evidence.

Pre-run checks passed 99 relevant regression tests, including a synthetic
complete discovery/collection/audit pipeline, exclusive-output and tamper
checks. Frozen NDE code/assets verified. Compact restoration evidence is in
[the preflight record](../results/highd_v47_teacher_restore_preflight_v1.json).
No full reproduction or new-source collection was run by the coding agent.

```bash
python -m pytest tests/test_highd_v47_teacher_expansion.py \
  tests/test_highd_v47_action_teacher.py tests/test_highd_v47_event_source_diagnosis.py \
  tests/test_highd_v46_cross_unit.py tests/test_highd_v46_multistate_q.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

## Completed expansion and next handoff

All 4,096 declared natural sources were processed. Outcome-blind discovery
selected 790 source episodes and 1,043 cases: 551 sources had one case, 225 had
two and 14 had three. Each received exactly 64 draws, for 66,752 total; source
maximum was 192, not an event-dependent extension. All 805 zero-event cases
remain in the record.

There were **2,009 CAV first collisions across 209 independent event sources**
and 637 BV-only competing endpoints. Raw continuation event yield was 3.00965%;
event-source yield over all 4,096 sources was 5.10254%. The count gate (50
sources/500 events, 100 sources desired) passed. This is fixed-behavior yield,
not a natural collision probability or a trained PPO result.

Under the declared source/case law, the teacher IS mean is 0.00193285524536
and event M2 is 0.0124173643600. Largest source M2 share is 56.7499%, and
effective M2 sources are 2.49193. Almost all the largest source's contribution
comes from one event; the three largest sources contribute about 56.75%,
24.75% and 12.99%. The larger event-source count does not establish balanced
natural-risk contributions or low target-policy gradient variance.

The completed audit verified all collection likelihoods and 1,043 predetermined
native replays, with maximum full/factor log error 2.84217e-14. A subsequent
read-only handoff check rehashed all 2,153 bank/collection artifacts and verified
the parent's recorded dependencies/configuration/assets unchanged. The original
67-event reproduction also passed. Validation sources remain unopened.

Bank summary SHA:
`161a8c49e76937e126e17aed04b54844b03eabee1ba9e28967d42307b411c9ec`.
Collection summary SHA:
`e852abfd3f4822319ccc25a01a6c97c1d1fe35279cab4b6174b595931e3e0918`.

The next handoff reuses these paths, not another collection. Its own complete
target replay/audit must establish useful Single and Dual reward signals before
the guarded PPO command starts. No PPO outcome is claimed by this collection.
