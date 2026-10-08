# Shared fixed teacher data for independent Single/Dual PPO

Execution update: all ten runs have now completed 200 iterations and portable
policy export. Proceed to the separately versioned
[source-disjoint native validation](highd_v47_shared_teacher_validation.md),
not another training run or the incompatible historical formal validator.

This handoff consumes the completed teacher expansion at `52c1929`, without
collecting another sample, changing labels or modifying historical protocols.
The expansion retained 66,752 continuations, including 2,009 CAV first collisions
from 209 source episodes. Its complete 1,043-case replay audit passed. These
counts pass the collection-size gate, not the target-specific training gate.

## Same behavior paths, independent target policies

[Protocol](../configs/highd_v47_shared_teacher_ppo_v1.json) pins the complete
teacher bank/data summaries and audit. The behavior remains the risk-bin joint
teacher at the first decision followed by natural continuation. Replaying its
exact RNG schedule reconstructs both target views on the SAME physical paths:

- Single uses the existing formal independent Single-BV selector and its H.
  Its selected actor can differ from either teacher actor.
- Dual uses the existing formal following-pair selector when the common Single
  gate is active; otherwise it uses Single fallback or no intervention. A
  padded second factor has P=H=1 and a masked policy gradient.
- Both targets are evaluated at each decision in the four-second stopped
  continuation, not only at the teacher's initial decision. Natural partner
  correlations and all unselected actors retain their conditional natural law.
- No target chooses or samples a replacement physical action. The sampled
  actions, CAV labels and complete behavior likelihood remain immutable.

This preserves the existing **target** families; it does not relabel the first
teacher factor as Single. The teacher-selected state bank is broader than the
failed original-Single-critical formal bank. Their conditional targets differ;
this handoff is not a same-target improvement claim over the zero-event run.
Within this new comparison both targets have exactly the same state law.

State target alpha is uniform over 790 selected sources, then their 1,043 cases.
State LR is exactly one; there is no learned Q_S. Collection had 64 draws/case,
so the per-trajectory target mass is alpha_case/64. Uniform trajectory replay
must include N*alpha_case/64; positive/nonpositive replay adds its exact sampling
correction. Neither raw collision counts nor unweighted case pooling defines
the target. All zero-event cases and target-inactive event paths remain stored.
An event with no target-active decision contributes a constant to the complete
objective, not a trainable reward signal.

For each mode the minimized objective is

`sum_case alpha_case * mean_teacher[I * exp(log(P/Q_behavior) + log(P/Q_target))]`.

PPO receives its negative with exact replay corrections and one common positive
reward scale frozen for both modes. No weight clipping, event deletion or
source rebalancing is used. Target prefix likelihood is recomputed causally
under the current policy; the behavior's terminal weight is never an observation.
The 22-dimensional observation uses elapsed continuation time / four seconds.
This is an explicit new dataset/proposal contract; old checkpoints cannot be
silently substituted. Initial snapshots retain the original teacher's native
restoration semantics, including its initialization of history-dependent state.

## Guards, audit and training

Before PPO, require at least 500 target-dependent event trajectories from 50
sources in EACH mode. Dual additionally requires 20 second-factor signal events
from five sources. At reference epsilon=0.1, useful sampled-factor contrast is
`abs((P-H)/(0.1P+0.9H)) >= 1e-6`. Counts, objective/gradient concentration,
target-inactive constants and exact source masses are all reported. These are
engineering signal checks, not independent variance or generalization evidence.
Large event counts can still conceal concentrated rewards; the report does not
declare concentration solved or equate these gates with validation readiness.

Export recomputes every existing path and checks original initial likelihood,
endpoint and clearance score. It stores full natural/behavior densities, all
actions, each target's factor PMFs, masks and compact training views. The audit
checks every label, target action, conditional PMF, support, TV and likelihood,
then repeats the predetermined first root of each case (1,043 replays). These
are implementation verifications, not extra observations. Missing/failed signal
returns `training_blocked` BEFORE Ray or PPO starts; nothing is recollected.

The parent freeze used a directory-wide dependency inventory. This new consumer
checks its RECORDED dependency set and all old config/asset/artifact hashes,
rather than injecting new consumer files into the historical inventory. No
parent freeze is silently regenerated or repinned. The new lock separately
records the consumer, environment, dataset adapter and unchanged PPO runtime.

If the audited signal passes, train five paired seeds [7,17,29,41,53], 200
iterations per mode, under the existing formal PPO optimizer/head settings.
Keep iteration 0 and every ten-iteration checkpoint. All data remain fixed
through training; training objective values are resubstitution diagnostics,
not held-out testing performance. Portable policy metadata and a separate
shared-runtime provenance record pin the actual new dataset adapter.

No validation source or final test is opened. Source-disjoint native checkpoint
selection and evaluation need their own reviewed protocol for this new state
bank and target contract; the old formal validation command must not be pointed
at these checkpoints without that compatibility work.

## Researcher-run command

After review/commit, use the inherited Python 3.9.19 environment:

```bash
conda activate highd-ppo-clean39
bash scripts/run_v47_shared_teacher_ppo.sh
```

This runs freeze, full shared export, replay audit and guarded training. Export
is a substantial deterministic replay job; it prints progress every 16 cases.
Each PPO subprocess prints its own log path. For the first run, monitor:

```bash
tail -f outputs/v47_shared_teacher_ppo_v1_training/single_seed7.log
```

All outputs are exclusive-create. Do not rerun the entire script over existing
destinations, delete failed artifacts or change frozen dependencies mid-run.
If a stage fails, retain its log and resume discussion at that stage. This
handoff supplies commands; the coding agent does not launch the long job.

Preflight uses synthetic likelihood/replay tests and bounded deterministic
replays of old paths, including a positive and a negative teacher trajectory.
Full target-signal counts remain unknown until the researcher completes export.

Implementation checks passed 97 relevant regression tests, including a complete
synthetic export/audit handoff and finite-enumeration off-policy likelihood and
derivative checks. One-iteration synthetic PPO runs for BOTH modes completed
initialization, reward updates, diagnostics and portable checkpoint export.
They are interface checks only, not teacher-data training or new native evidence.
Frozen NDE/assets, the complete teacher evidence and shell syntax also verified.

```bash
python -m pytest tests/test_highd_v47_shared_teacher.py tests/test_highd_v47_teacher_expansion.py tests/test_highd_v47_fixed_alpha_ppo.py tests/test_highd_v47_formal_comparison.py tests/test_highd_v46_d2rl.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

## Ray startup interruption and continuation

The researcher completed export and all 1,043 audit replays. Both target-signal
gates passed: 1,992 usable event trajectories from 209 sources per mode; Dual's
second factor has 394 usable events from 36 sources. Single seed 7 completed
all 200 iterations and all scheduled policy exports. Dual seed 7 failed BEFORE
PPO initialization: Ray's registry tried sending 108,261,462 bytes against its
104,857,600-byte gRPC limit. There are no Dual iterations or checkpoints to
restore. This is a software startup failure, not a failed learning result.

The frozen model's `forward` closure references `check.observation_contract`,
but captures the complete sequence environment and therefore the whole Dual
dataset. The separate recovery adapter replaces ONLY that method's `check`
closure cell with a small object containing the identical observation-contract
string. Its bytecode, other closure cells, model parameters, active-factor mask,
optimizer, reward and all training data remain unchanged. The live trainer
environment is not altered. No gRPC limit is raised, old frozen implementation
is edited, or old lock is repinned. The runtime adapter receives its own hash
in the new recovery protocol.

The continuation validates all 200 Single iteration records, its 21 scheduled
checkpoint exports and original provenance, then retains that completed run.
It records checksums of all original attempt files, preserves the failed Dual
log and empty directory, and runs the remaining NINE mode/seed combinations in
a new exclusive directory, starting with Dual seed 7. Any nonempty unfinished
run is rejected: this command must never silently restart a partially trained
model instead of restoring its optimizer/checkpoint. It is specifically the
documented pre-training registration-failure recovery, not generic resumption.

Run this INSTEAD of the original all-stage script:

```bash
conda activate highd-ppo-clean39
bash scripts/resume_v47_shared_teacher_ppo.sh
```

Monitor the first remaining run with:

```bash
tail -f outputs/v47_shared_teacher_ppo_v1_training_recovery/dual_seed7.log
```

No discovery, collection, target export, native audit or completed Single run
is repeated. Validation stays unopened. The final recovery summary links the
original completed run and all nine remaining results; it does not pool them
as extra seeds. A second interruption retains the recovery output for explicit
review rather than overwriting it or silently restarting partial training.

Recovery checks passed 29 focused tests. With the actual complete Dual dataset,
serialized model size fell from 108,261,403 to 3,274 bytes (the gRPC request has
additional framing). This preflight stopped before Ray startup. Separate
one-iteration synthetic PPO checks for both modes completed under the adapter;
their metrics matched the earlier unmodified synthetic checks and portable
policy errors were 0 / 7.45e-9. These are software checks, not resumed native
training. The original 318 attempt files and all frozen inputs were retained.

```bash
python -m pytest tests/test_highd_v47_shared_teacher_recovery.py tests/test_highd_v47_shared_teacher.py -q
```
