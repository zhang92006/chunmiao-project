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
