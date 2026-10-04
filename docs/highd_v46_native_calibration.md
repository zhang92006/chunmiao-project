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
