# Training-only native discovery calibration

This task builds on `9f85b7c`. The previous fresh real-Q validation confirmed
CAV-event exercise at two lateral states, but failed the precision/source
diversity guards. It is now consumed evidence, not a source of fitting labels.
This task does not change NDE, physical P, production Q, proxy, strict gates,
PPO, or reserved final-test episodes.

## Fixed training protocol

[highd_v46_native_calibration_v1.json](../configs/highd_v46_native_calibration_v1.json)
opens 64 new **training** natural source episodes, seeds 200000–200063.
Consumed validation cohorts including 190000–190031 are explicitly rejected.
The existing mixed5 bank retains two geometry peaks, two uniform times and
one lateral/phase-aware state, with four seconds remaining. The pair budget
remains rank 1 + rank 2 + one audit slot within nearest-eight BVs/115 m:
at most 320 states and 960 expensive matrices, never all 28 pairs per state.
The mixed bank is used for training diversity, not declared a validated winner.

Cases are fixed before native outcomes. Uniformly sample at most 16 source
episodes with strict-positive pairs. Within each, retain the maximum
interaction-mass pair/state, plus at most one different state, preferring a
different mechanism and then interaction mass. Retain four seeded abstention
controls from episodes with no eligible pair. This is a teacher/calibration
budget, not a deployed online rule. Missing cases are not backfilled with
known collision witnesses; no native-negative case is replaced.

Each case receives 64 continuations in each of the existing natural `(1,1)`,
first-only `(0.1,1)`, second-only `(1,0.1)` and dual `(0.1,0.1)` arms. Q draws
the first actions; four-second subsequent dynamics and correlated natural
partners remain unchanged. There are at most 9,216 continuations. Within a
case, arms share seeds; different cases have disjoint seed ranges beginning
2500000. Every attempt, zero event, and competing BV-only endpoint is retained.
No PPO dataset, policy checkpoint or population manifest is exported.

The comparison targets the fixed conditional law: uniform selected eligible
source episode, then uniform selected cases within that source. The case plan
and masses are saved before native samples. Fixed-stratum variance uses the
actual masses, and event-source guards count distinct episodes rather than
states. Bootstrap resamples matched continuations within fixed states and
does not establish source-population generalization. First-only/second-only
are factor ablations, not the original paper's single-BV agent.

## Native-yield ranker and limits

All audited pairs additionally receive features from the already computed
proxy matrix: dual-Q expected clearance risk, maximum supported proxy risk,
and a lateral indicator. There is no extra proxy query or change to H/Q.
At nonqualifying pairs, the hypothetical Q feature is only diagnostic: those
pairs are not tilted by the native collector.

A fixed ridge-regularized binomial model predicts **CAV first-event frequency
under fixed dual Q**, not natural risk. It uses four coefficients including
the intercept, ridge 4, and at most 100 Newton steps. Zero-event cases enter
the likelihood as observed zeros, never as proof of zero true risk. Each
source episode has equal fitting weight; multiple cases share that weight.
Model fitting requires at least 20 dual events, at least three event-bearing
source episodes and at least three zero-event source episodes. These are
engineering training-quality guards, not significance or efficiency criteria.
No candidate model is exported when labels are insufficient or fit fails.

Leave-one-source-episode-out predictions exclude **all** states and labels of
the checked episode. Prespecified interaction-mass, dual expected proxy risk,
maximum proxy risk, and (if available) cross-fitted native scores are replayed
on at most two calibrated cases per source. This is training-side diagnosis,
not fresh validation, global state recall, or online performance. In-sample
native labels are never used as an oracle selector. Unqueried states/pairs
remain outside this replay comparison.

Real-Q endpoint yield is only the first discovery objective. The report also
retains IS means, second moments, event/weight ESS, RHW, BV-only counts and
descriptive first-minus-dual second-moment gain per measured CPU. A predictor
of collision exercise alone is **not** a learned optimal-variance selector.
Zero baseline events or failed 20-event/three-source guards cannot support
efficiency conclusions; high raw event density is not a natural risk estimate.

## Reproduction

Freeze the configuration, module and dependency hashes before opening sources.
The output path is exclusive-create; an interrupted run is preserved and any
restart uses the same protocol with a new output path, never only positive cases.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_native_calibration --output outputs/v46_native_calibration_v1_train
python -m pytest tests/test_highd_v46_native_calibration.py tests/test_highd_v46_multistate_q.py tests/test_highd_v46_discovery_v2.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Raw banks, case plans, likelihood-audited continuations, calibration labels and
candidate model live only in ignored outputs. Commit protocol, code, tests
and compact evidence. Any subsequent state-law/selector update needs an
explicitly locked, source-episode-disjoint fresh validation cohort; a trained
policy still needs independent original-initialization evaluation. No formal
fresh validation or PPO is launched by this training-only stage.

## Evidence, 2026-10-04

Protocol and implementation were frozen at
`fc773612197bab8c7c288580ec8dbd49b946930d` before opening the new sources.
The complete fixed-budget run took 737.60 s locally. All 104 relevant regression
tests passed and frozen code/assets verified. No NDE, production Q, proxy/gate,
PPO, consumed validation or reserved final-test change was made.

The 64 natural episodes supplied 3,662 pre-draw states: 58 episodes reached
the horizon and six ended at a BV-only collision; none had a CAV first collision.
The bank retained five states per episode: 128 geometry, 128 uniform, 55
lateral/phase and nine phase-fallback states. It evaluated 939 pair matrices,
at most three per state, finding 23 strict-positive pairs, 22 positive states
and 16 positive source episodes. The main first failures were leverage (850)
and interaction (64); two failed risk-H factor TV. No gate was relaxed.

The 22 positive states comprised 13 geometry, two uniform and seven
lateral/phase states. Phase/uniform slots added six proxy-positive source
episodes beyond those found by the two geometry slots in this bank. This
within-training-bank comparison is not an equal-budget fresh-validation result
or a population recall estimate. In particular, source 200051's native-event
state came from the lateral/phase slot and was absent from the geometry slots.

The pre-outcome plan selected 21 eligible states across all 16 eligible source
episodes, plus four abstention controls. The two-state-per-source cap was
retained; no missing or native-negative case was backfilled. All 6,400 planned
continuations completed. Each primary arm has 1,344 continuations (21 x 64);
controls are retained separately rather than mixed into the eligible law.

| Arm | CAV first events | Distinct event source episodes | Conditional IS mean | Event second moment |
| --- | ---: | ---: | ---: | ---: |
| Natural | 0 | 0 | 0 | 0 |
| First factor only | 12 | 1 | 0.00013755 | 0.00000161445 |
| Second factor only | 2 | 2 | 0.00035582 | 0.000116303 |
| Dual | 31 | 2 | 0.00033968 | 0.0000276788 |

Means and second moments use the predeclared source-then-state masses, not
equal state masses or self-normalized weights. Zero observed natural events
are not evidence of zero true risk. These estimates have no natural-initial
density correction and must not be relabelled as population risk.

Only two selected states produced CAV events; the other 19 eligible states
produced none in these continuations. Every outcome was retained:

| Source / time / pair | Bank slot / pair mechanism | Natural / first / second / dual CAV events, 64 each |
| --- | --- | --- |
| 200017 / 20 s / [12,14] | Geometry / bracket | 0 / 0 / 1 / 3 |
| 200051 / 20 s / [26,28] | Lateral-phase / lateral | 0 / 12 / 1 / 28 |

This adds a training-side bracketing event mechanism alongside the lateral
case; it does not establish generalization or a significant dual advantage.
Four abstention controls reproduced identical endpoint, clearance and weight
records across all four arms (apart from timing), with no CAV events and
weight one. Native BV-only counts across all cases were 1/0/4/2 respectively.
The actual Q-versus-P second-factor TV ranged from 0.41408 to 0.67202 at
eligible cases, but nonzero factor influence alone did not predict collision.

### Event density is not variance efficiency

Dual has 31 events but only two event-bearing sources; its source guard fails,
90% RHW is 0.7763 and event-contribution ESS is 4.2847. Its descriptive
second-moment ratio to first-only is **17.1444**, not a variance improvement.
Both-arm comparison guards fail, so bootstrap intervals and efficiency claims
remain unavailable. The different finite-run IS means are also too uncertain
to infer different natural conditional targets.

At source 200051, two of the 28 dual events have zero H probability in the
second conditional factor. Q still samples these actions through its natural
epsilon floor, so likelihood support is correct. Their weights are each
approximately 0.117374 and they account for 97.94% of that case's observed
event second moment. The case's event ESS is only 3.8756 despite its high raw
event count. The first-only case has second moment 0.0000258311 versus dual's
0.000439563. This is concrete training evidence of interaction-H/native-event
misalignment, not a likelihood bug or proof about the true efficiency ratio.
No weights were clipped, and these events were not removed.

All 105 output artifact hashes, predeclared case/seed schedules, unique
within-arm continuation seeds and source-based masses were independently
checked. Every per-case summary and the full conditional comparison reproduced
exactly from raw sequences. Full-versus-two-factor log likelihood error was
at most 1.42e-14; protocol/configuration content still matched the frozen
commit. No policy checkpoint or population sequence manifest exists.

### Decision and next stage

The native-yield ranker was **not fitted or exported**: 31 events satisfy its
count gate but two event sources do not satisfy its three-source gate; 14
eligible source episodes have zero observed events. No threshold was relaxed.
The three prespecified proxy-only replay scores each retain the same 31 events
when selecting one calibrated case per source. They do not demonstrate a
ranking improvement; their comparison covers only the queried cases.

Phase coverage has now found a useful native-event source in training, while
most strict-positive states still yield no event and a few event weights
dominate the second moment. The next discovery version should therefore
target **diverse native CAV-event sources and stable weighted utility together**,
not merely larger interaction mass or raw collision yield. Use the negative
cases and H-zero-event diagnostics to design training-side discovery features
or a teacher budget; do not deploy post-hoc hard thresholds, change production
Q/NDE, or add PPO epochs. Define any expanded training protocol before opening
new episodes. Only after label/source guards and a frozen candidate screen
permit fitting should a separately locked, episode-disjoint fresh validation
check event-source coverage, weighted second moments/ESS/RHW, both factor TVs,
intervention yield and CPU. Consumed 190000-190031 validation remains excluded
from fitting, and original-initialization evaluation remains required later.

Discovery scanning took 45.92 s and proxy CPU 366.52 s (413.21 s total discovery)
in this local run; the four native arms each used approximately 78-79 CPU s.
These are development timings, not a testing-acceleration benchmark.
The compact evidence is
[highd_v46_native_calibration_pilot_v1.json](../results/highd_v46_native_calibration_pilot_v1.json).

## Fixed-budget training-source supplement

The [expansion protocol](../configs/highd_v46_training_expansion_v1.json) was
committed as `e1c1189` before opening 64 fresh natural training episodes,
230000–230063. Continuation seeds start at 2800000. This supplements independent
sources rather than repeatedly extending the two known event states. Discovery
still uses mixed5 and rank 1 + rank 2 + audit; the proxy, strict gates, native
NDE, Q and PPO are unchanged. All eligible sources may enter (cap 64), at most
two states per source, plus four separately retained abstention controls.
Every case receives the fixed 64 continuations per arm. Neither the budget nor
case selection changes in response to native events. No validation/test source
is opened, reused for training, or replaced in this task.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_native_calibration --config configs/highd_v46_training_expansion_v1.json --output outputs/v46_training_expansion_v1
```

After completion, the training-only exporter checksum-locks both calibrations,
checks unchanged code/dependencies, replays outcome-blind case selection, and
checks every planned continuation and full/factor likelihood. It combines all
eligible dual sequences, including zeros, with the original locked calibration.
Source-then-state masses are recomputed across the union, not concatenated from
separately normalized batches. Raw controls and other arms remain in their
original archives; the four-arm union comparison is a training diagnostic.

```bash
python -m scenario_reconstruction.highd_v46_training_expansion --expansion outputs/v46_training_expansion_v1 --expansion-sha256 <completed-summary-sha256> --output outputs/v46_training_expansion_v1_dataset
python -m pytest tests/test_highd_v46_training_expansion.py tests/test_highd_v46_conditional_baseline.py tests/test_highd_v46_native_calibration.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

`conditional_sequence_manifest.json` is a verified training replay artifact;
`training_quality.json` reports all-source event coverage, empirical unscaled
second-moment concentration, H2-zero event contributions, and the four arms.
The descriptive effective number of second-moment sources is
`1 / sum(source_fraction**2)`, not an independent sample size or precision test.
All-zero observations yield undefined concentration, not evidence of zero risk.

This new data law has its own identity. It cannot silently replace the old
baseline's train directory or reuse its selected checkpoints. No fresh
validation is prepared and `checkpoint_selection_ready` remains false. A
separately frozen fresh validation protocol and matched data identity are
required before standard training/selection. No additional PPO run is launched
by collection or export. More event sources would improve training coverage,
but do not by themselves establish generalization or population acceleration.

### Supplement evidence, 2026-10-04

The complete fixed-budget run took 638.86 s locally. All 64 natural sources
reached the horizon, supplying 3,840 states. Mixed5 retained 320 states and
audited 944 pair matrices, finding 11 strict-positive pairs at ten states
from eight source episodes. All ten states and four abstention controls were
retained. All 3,584 planned native continuations completed: **zero CAV events
in every arm**, including controls. BV-only competing endpoints totaled
3/3/3/4 across natural/first-only/second-only/dual. Zero observed CAV events
do not establish zero true risk. No ranker was fitted or exported.

The completed collection summary SHA is
`6cd41910ab556c0aa5ad4da262808f0bff1e9d65296b08484fff0e266eccdef3`.
All 61 collection artifact locks passed, including the full continuation
schedules and likelihood checks. The exporter also verified all original
training artifacts before creating
`outputs/v46_training_expansion_v1_dataset/conditional_sequence_manifest.json`.

| Training union | Before | After |
| --- | ---: | ---: |
| Natural source episodes scanned | 64 | 128 |
| Selected eligible source episodes | 16 | 24 |
| Selected states | 21 | 31 |
| Dual continuation sequences | 1,344 | 1,984 |
| CAV events | 31 | 31 |
| Event-bearing source episodes | 2 | 2 |
| Largest source's empirical second-moment share | 99.26% | 99.26% |

The event-source diversity problem is **not solved** by this supplement.
Under the recomputed uniform-source law, the empirical second moment falls
from 2.76788e-5 to 1.84525e-5 only because zero-observed-event strata now receive
one third of target mass. It is not a policy improvement or a same-target
efficiency comparison. The effective second-moment source count remains 1.015.
No weights, endpoints or negative cases were removed or clipped.

Training-side diagnosis: all ten newly selected states have maximum supported
proxy risk between 0.1304 and 0.3424. The two original event states both had
proxy maxima of 1.0, though other original maximum-1.0 states had no observed
events. These are clearance-risk scores, **not collision probabilities**.
Positive joint interaction can occur without a collision-proximate absolute
risk level. This motivates a separately declared training-only comparison of
endpoint-oriented risk features alongside interaction; it does not justify
deploying a post-hoc risk threshold or relaxing strict gates. State selection
and the temporal coverage budget also remain possible limitations.

All 129 relevant tests passed and frozen NDE/assets verified. Old validation
artifacts are unchanged, no fresh validation/test source was opened, and no
PPO run was launched. The merged replay is ready for data inspection, but is
not a newly validated training/selection baseline. The next experiment should
address native-event discovery utility and require fresh independent validation
after freezing any candidate change, rather than spending more PPO epochs on
this unchanged two-source signal.

Compact evidence:
[highd_v46_training_expansion_v1.json](../results/highd_v46_training_expansion_v1.json).
