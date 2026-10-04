# Discovery v2: mixed temporal bank, bounded pair screening

This experiment builds on `f56f2e4`. Frozen NDE, natural initialization, CAV,
event definition, production Q, risk surrogate, strict gates and PPO remain
unchanged. Only offline state/pair discovery and its diagnostics are new.

## Fixed selector and controls

The protocol is [`highd_v46_discovery_v2.json`](../configs/highd_v46_discovery_v2.json).
Every bank considers one-second natural pre-draw states with four seconds left
in the 60-second source episode. Selection never reads the collision outcome.

Three arms share each natural source episode:

| Arm | State budget | Pair budget per state |
| --- | --- | --- |
| `geometry2_diverse` | 2 geometric peaks, >=5 s apart | Highest geometry pair, different-mechanism preference, random audit |
| `geometry5_top2` | 5 geometric peaks, >=5 s apart | Rank 1 + rank 2 + random remaining pair |
| `mixed5_top2` | 2 geometric + 2 uniform-time + 1 lateral/phase | Rank 1 + rank 2 + random remaining pair |

The five-geometry arm is an equal-budget control: a larger candidate count alone
is not evidence for mixed sampling. The two-geometry control applies the common
remaining-horizon guard before selection, so it is a horizon-safe analogue,
not a byte-identical replay of all v1 bank choices.

Uniform times are drawn without replacement from remaining eligible times,
without geometric scoring or a spacing restriction. The last slot first
considers remaining states with nonzero lateral-pair geometry score (which
requires natural lane-change support). It prefers a still-unrepresented third
of the episode, then the largest lateral-pair score, then earliest time. If no
lateral-active candidate exists, it uses phase coverage and geometry score.
Slots do not duplicate states; short episodes may yield fewer states. Budgets
are fixed at 2+2+1 for this pilot, without threshold fitting on validation.

The pair universe and cheap scores stay at nearest eight BVs / 115 m. Computing
these cheap scores does not run 28 risk matrices. Each arm runs at most three
33x33 risk matrices per selected state. Shared state/pair results are reused
between comparison arms; no exhaustive-pair fallback is allowed. Audit RNGs are
keyed by source seed and time so arm execution order does not change samples.

This remains an **offline episode-wide bank selector**, not an online causal
trigger. Its high-risk state distribution is not the natural reset law.

## Freeze, then fresh validation

Development uses the existing 16 training source episodes 160000-160015.
Fresh validation is fixed at **32 source episodes 180000-180031**. These do not
overlap training, earlier validation, prior discovery examples, or final-test
episodes 100000-101023. The completed training record locks configuration,
implementation and dependencies before validation is opened. Changes cause
validation to refuse execution. Do not tune on the fresh cohort after reading it.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_discovery_v2 --split train --output outputs/v46_discovery_v2_train
python -m scenario_reconstruction.highd_v46_discovery_v2 --split validation --training-lock outputs/v46_discovery_v2_train/summary.json --output outputs/v46_discovery_v2_validation
```

## Eight diagnostics and their denominators

1. Candidate states per source episode, including role counts and shortages.
2. Strict-positive states / all selected states (at least one passing pair).
3. Strict-positive pairs / all audited pairs.
4. Distinct positive source episodes / all source episodes.
5. Second-factor TV for all audited pairs and the best strict pair per state.
   H-versus-P uses the existing H-first-marginal-weighted conditional TV.
   Actual fixed-Q TV uses `sum_a Q1(a) TV(Q2(.|a), P2(.|a))` with eps=(0.1,0.1).
6. **Shadow conditional activation**, both / selected states and / all scanned
   natural decisions. At a passing state, draw once from the unchanged Q with
   its natural conditional partners and audit complete versus two-factor
   likelihood. These draws are never fed back into the source natural rollout.
   Actual online closed-loop intervention rate is therefore `null`, not the
   shadow rate. Measuring it requires a separately specified causal trigger.
7. Selector CPU seconds: cheap pair scores, bank/pair selection, and charged
   proxy evaluation cost for each arm. Shared proxy costs are charged to every
   arm using them; total unique-run wall time is reported separately. Native
   simulation, snapshot serialization and calibration are not selector CPU.
   Timings are local measurements, not tests-to-precision acceleration.
8. Proxy-to-native interaction calibration, using the fixed procedure below.

All negative states and episodes stay in the results. Empty rates with zero
denominators/TV distributions remain undefined; states within an episode are
not independent observations. No significance or global recall claim follows
from this pilot's point estimates.

## Predeclared native calibration

For each arm, select the largest interaction-mass strict pair per positive
source episode, then a seeded uniform sample of up to eight such episodes.
Deduplicate identical cases across arms. Add up to four seeded, episode-distinct
nonqualifying controls with a positive supported four-corner proxy contrast.
Zero-contrast rows are excluded from these controls because their arbitrary
argmax may lie outside natural support. This is not an audit of all negatives
or a population proxy-calibration curve.

Each selected pair receives 16 stochastic native continuations of each of the
four forced baseline/first/second/joint action corners, with common random
numbers. Training seeds are 2100000-2100015, validation seeds 2200000-2200015.
Record mean risk contrast, its Monte Carlo SE, native-minus-proxy difference,
and CAV event counts, including negative results. A native contrast >=0.05 is
a predeclared descriptive confirmation threshold, not a statistical test or
guarantee of collision uplift. These forced corners are not draws from Q and
not population event probabilities.

## Validation

```bash
python -m pytest tests/test_highd_v46_discovery_v2.py tests/test_highd_v46_discovery_coverage.py tests/test_highd_v46_state_discovery.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_pair_synergy.py tests/test_highd_v46_d2rl.py tests/test_highd_v46_validation.py tests/test_highd_paper_discrete_pairs.py tests/test_highd_mean_precision_policy.py tests/test_highd_dynamic_pairs.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Raw banks, pair rows and continuation outcomes stay in ignored `outputs/`.
Only protocol, implementation, tests and compact evidence enter Git. This
experiment exports no PPO dataset, checkpoint, or population acceleration.

## Development evidence and pre-validation freeze

The implementation and protocol were committed as `2205ce8` after the training
run and before opening validation. All 89 relevant regression tests passed.
The training lock identifies implementation SHA-256
`50b86b3b3ed018c589e96e9031bd184347064ff33e8d485a68d272182e54ab32`.
Training source episodes, including the BV-only early termination, were retained.

| Training arm | Selected states | Audited pairs | Strict-positive states | Strict-positive pairs | Positive source episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometry 2, diverse pairs | 32 | 96 | 2 | 2 | 2/16 |
| Geometry 5, top two + audit | 80 | 240 | 5 | 6 | 2/16 |
| Mixed 2+2+1, top two + audit | 80 | 240 | 3 | 4 | 2/16 |

Mixed discovery did not outperform equal-budget geometry on training. No rules
were changed to reverse that outcome. Both selected positive source episodes
per arm met the native interaction >=0.05 descriptive threshold; these were
the same two known training episodes across arms, not new independent evidence.
One joint-corner continuation collided at the known seed 160009 / 19 s; it is
not a population collision rate. Four nonqualifying controls were also retained.

Training used 121 unique states and 388 unique pair matrices. Discovery took
172.23 s and the full run including native calibration took 192.58 s locally.
The next result must come from the locked fresh cohort, not from further
training-side tuning after viewing validation.

## Fresh validation evidence, 2026-10-04

The locked 32 episodes 180000-180031 completed with 1,901 natural decision
states. There were 31 horizon completions, one BV-only termination and no CAV
events in the natural source trajectories. All outcomes were retained. The
three comparison arms shared 246 unique states / 743 unique pair matrices.
No arm evaluated more than three pairs per state, and no source/config change
was made between training and validation.

| Diagnostic | Geometry 2 + diverse | Geometry 5 + top two | Mixed 2+2+1 + top two |
| --- | ---: | ---: | ---: |
| Candidate states per episode | 2 | 5 | 5 |
| Total states / audited pairs | 64 / 184 | 160 / 460 | 160 / 461 |
| Strict-positive states / rate | 4 / 6.25% | 9 / 5.625% | 7 / 4.375% |
| Strict-positive pairs / rate | 4 / 2.174% | 10 / 2.174% | 7 / 1.518% |
| Positive source episodes / coverage | 4 / 12.5% | 5 / 15.625% | 5 / 15.625% |
| Chosen-pair second H-factor TV, mean | 0.67083 | 0.60412 | 0.70771 |
| Chosen-pair second Q-factor TV, mean | 0.58213 | 0.52311 | 0.60441 |
| Shadow activation / selected states | 4/64 | 9/160 | 7/160 |
| Shadow activation / natural decisions | 4/1901 (0.2104%) | 9/1901 (0.4734%) | 7/1901 (0.3682%) |
| Selector CPU seconds | 73.26 | 183.49 | 183.92 |
| Native-positive cases checked / contrast >=0.05 | 4 / 4 | 5 / 5 | 5 / 5 |

Shadow draws preserved full versus two-factor likelihood to maximum absolute
error 4.44e-15. These rates are not online closed-loop intervention rates.
The mixed bank had exactly 64 geometry, 64 uniform and 32 lateral/phase states.
Its seven strict-positive states comprised four geometry, two uniform and one
lateral/phase state. The last category supplied a source episode (180003) not
found by the geometry arms; the five-geometry arm instead found episode 180007.
The other four positive source episodes were shared. This is complementarity,
not an improvement in total episode coverage over the equal-budget control.

Discovery wall time was 322.83 s for all arms together, including 24.59 s of
natural scanning; total time including calibration was 358.18 s. Per-arm CPU
charges include reused pair computations, so they must not be added to infer
the unique-run wall time. These are not collision testing acceleration factors.

### Native calibration and collision witnesses

There were seven distinct selected-positive cases across arms and four
nonqualifying controls. All selected-positive cases had mean native interaction
>=0.05; only sampled/selected cases were checked, not every passing state.
For compactness, G2/G5/M below denote the three table columns above.

| Source seed / time | Selected for | Proxy interaction | Native interaction (SE) | Base / first / second / joint CAV events, 16 each |
| --- | --- | ---: | ---: | --- |
| 180003 / 55 s | M | 0.06536 | 0.05631 (0.00491) | 0 / 0 / 0 / 0 |
| 180007 / 47 s | G5 | 0.13419 | 0.22356 (0.00972) | 0 / 0 / 0 / 0 |
| 180010 / 23 s | G2, G5, M | 0.30549 | 0.41128 (0.01120) | 0 / 0 / 0 / 0 |
| 180011 / 48 s | G2, G5, M | 0.05654 | 0.08380 (0.00409) | 0 / 0 / 0 / 0 |
| 180029 / 55 s | G2, G5, M | 0.06124 | 0.09446 (0.00555) | 0 / 0 / 0 / 0 |
| 180030 / 22 s | G2, M | 0.63701 | 0.62942 (0.00885) | 0 / 0 / 0 / 14 |
| 180030 / 29 s | G5 | 0.62699 | 0.53962 (0.00767) | 0 / 0 / 0 / 9 |

The new collision mechanism at source 180030 is also found by baseline arms.
The two times share a source episode and continuation seed schedule: do not
pool them as independent collision evidence or attribute them exclusively to
mixed sampling. These are forced-corner conditional events, not Q event rates.

Three controls had native contrasts close to zero. One control, source 180026
at 30 s / pair [24,26], had proxy contrast 0.00554 versus native 0.41331
(SE 0.04279), with no CAV collision in any of its corners. This is a concrete
conditional-corner underprediction. However, its natural-probability-weighted
proxy interaction mass was only 2.18e-18 and several gates failed. It is **not**
proof that rejecting it loses a useful importance-sampling opportunity, nor
grounds for relaxing gates. The controls are a selected small audit, not a
global false-negative-rate estimate.

### Research decision

Fresh discovery and conditional two-factor influence are demonstrated: mixed
sampling found more positive states than the two-state control and all five
selected positive source episodes reproduced the native risk interaction.
But **mixed sampling did not beat the equal-budget five-geometry control**:
7 versus 9 positive states, equal 5/32 source coverage, lower pair yield, and
essentially equal CPU cost. The 2+2+1 allocation therefore is not established
as a superior selector. Retain the equal-budget control; do not promote mixed
sampling as the winning default or launch PPO on the strength of this pilot.

Any later training-side study of temporal allocation or proxy calibration must
use another predeclared fresh validation cohort. Seeds 180000-180031 are now
consumed validation, not a tuning pool. NDE, Q, PPO and final-test episodes
remain unchanged. See the complete compact provenance record in
[`highd_v46_discovery_v2_pilot.json`](../results/highd_v46_discovery_v2_pilot.json).
