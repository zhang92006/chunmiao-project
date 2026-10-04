# Discovering states and pairs before more PPO

This stage builds on `cea7e3a` and changes no frozen NDE, production proposal,
PPO objective, trainer, or precision guard. It builds an offline discovery bank
and measures where the existing gate rejects candidate pairs.

## Two levels of screening

Every natural decision state (one-second cadence) receives a cheap geometric
screen. Consider pairs among the nearest eight BVs within 115 m. Each actor's
pressure decays with bumper gap after subtracting positive closing speed times
four seconds. Adjacent-lane actors contribute direct pressure only when a lane
change toward the CAV has natural support. A pair's score combines the two
pressures and a declared bracket/same-side/lateral factor. These scores are
ranking heuristics, not crash probabilities or a learned model.

For each source episode, compare three equal state-budget banks:

- `geometry_1s`: the two highest-ranked states, at least five seconds apart;
- `geometry_5s`: the same ranking restricted to the old five-second grid;
- `random_1s`: a seeded random-priority control with the same budget/spacing.

The bank selector does not read the episode collision outcome. It does rank
states across a complete episode, so it is an **offline** pool builder, not a
causal online trigger. A future online screen needs its own threshold/budget
policy. The bank retains source seed, time, exact natural units, conditional
tables, histories and pre-draw RNG state. It cannot be treated as a natural
reset distribution or passed directly to population-risk PPO.

For each bank state, expensive evaluation checks the highest-ranked pair,
prefers a different mechanism for the second ranked pair, and audits one
random remaining pair. All use the unchanged 33x33 reactive proxy. Duplicate
states shared by methods are evaluated once. The random competitor provides
a limited missed-candidate audit; it is not an exhaustive/global recall test.
Ranked-candidate and random-audit discoveries are reported separately.

## Rejection funnel

The audit records each gate independently and the first failed gate:
two-actor risk leverage, positive four-corner interaction, excess over the sum
of global single-action maxima, risk-weighted-H factor TV, interaction mass,
and the actual interaction-weighted-H factor TV.

Three reporting views were declared before collection:

1. The unchanged production strict rule.
2. An ablation omitting global single-maximum excess.
3. A local-interaction view additionally omitting the risk-weighted-H TV check,
   while retaining both factor-influence checks for the actual H used by Q.

These are diagnostics, **not deployed threshold relaxations**. A local
interaction can be real even when another isolated action already has high
risk; conversely, dropping a gate may admit surrogate artifacts. Native
continuation checks are needed before treating such candidates as useful.

## Episode-disjoint validation

The fixed protocol is
[`highd_v46_state_discovery_v1.json`](../configs/highd_v46_state_discovery_v1.json):
16 training-source episodes (160000-160015), eight validation-source episodes
(170000-170007), both using natural initialization and up to 60 s. Old handpicked
examples and reserved final-test seeds are excluded. Implementation, proxy,
interaction helper, physical config and all screen/gate settings are locked
after the training run. The validation command refuses changed identities or
overlapping source episodes. Do not tune this rule after inspecting validation.

```bash
python -m scenario_reconstruction.highd_v46_state_discovery --split train --output outputs/v46_state_discovery_v1_train
python -m scenario_reconstruction.highd_v46_state_discovery --split validation --training-lock outputs/v46_state_discovery_v1_train/summary.json --output outputs/v46_state_discovery_v1_validation
```

Bank states within one source trajectory remain correlated. Discovery yield
is reported by pairs, states **and distinct source episodes**, with off-grid
times recorded separately. These counts are not collision probabilities,
independent sample counts, or tests-to-precision/acceleration estimates.

## Native confirmation

A separately predeclared witness protocol selects the two largest interaction-
mass ranked candidates from the one-second bank/local-interaction view, with
at most one case per source episode. Selection occurs before native outcomes.
Only states with at least four seconds left in the source 60 s window qualify;
raw bank proxy scores near the horizon are otherwise four-second scores and
must not be confused with remaining-horizon event probability.

Each case has 16 stochastic native continuations of the four baseline/single/
single/joint action corners, preserving natural conditional partners. Continuation
seeds are 1800000-1800015 for training cases and 1900000-1900015 for validation
cases; corners share random numbers. Report risk contrast and actual CAV events,
including failures to reproduce a proxy effect. This is limited conditional
confirmation, not a population-risk or significance study.

```bash
python -m scenario_reconstruction.highd_v46_discovery_witness --discovery outputs/v46_state_discovery_v1_train --output outputs/v46_discovery_witness_v1_train
python -m scenario_reconstruction.highd_v46_discovery_witness --discovery outputs/v46_state_discovery_v1_validation --output outputs/v46_discovery_witness_v1_validation
python -m pytest tests/test_highd_v46_state_discovery.py -q
```

Raw banks, action matrices and trajectories stay in ignored `outputs/`. Only
configuration, tests and concise evidence belong in Git. No high-risk reset
dataset is silently relabelled as a natural-initialization training collection.

## Pilot evidence, 2026-10-04

The locked run completed 16 training-source and eight independent validation-
source episodes. Training scanned 925 states / 16,479 cheap pairs and evaluated
276 expensive matrices at 92 unique bank states. Validation scanned 480 states /
6,236 cheap pairs and evaluated 128 matrices at 45 unique bank states. No source
episode had a CAV event; one training episode terminated on a BV-only collision.

Each method had 32 training and 16 validation bank states. Counts below are
strict-proxy-qualified states (distinct source episodes in parentheses), not
native collisions or independent Bernoulli observations:

| Bank method | Training | Validation |
| --- | ---: | ---: |
| Every-second geometry | 2 (2) | 0 (0) |
| Five-second geometry | 1 (1) | 0 (0) |
| Random state control | 0 (0) | 0 (0) |

The two every-second strict cases occurred at 19 s and 53 s, off the five-second
grid. This shows that exact snapshots can be missed by the cadence; it does not
show that all nearby useful states are missed. The local-interaction reporting
view found three training states for each geometric method, but zero validation
states for every method. Removing only global excess did not improve the
every-second bank's yield. Its leading failure was two-actor leverage: 88/96
audited training pairs and 42/46 validation pairs failed that first gate.
All four remaining validation pairs failed interaction. This pilot therefore
does not support blaming global-excess strictness alone or relaxing thresholds
to obtain a desired result. It also cannot distinguish absent mechanisms from
mechanisms missed by this limited bank/pair budget.

Predeclared native confirmation selected two new training-source episodes:

| Source seed, time, pair | Base / first / second / joint CAV events (16 each) | Native risk interaction mean (SE) |
| --- | --- | ---: |
| 160009, 19 s, [7, 9] | 0 / 0 / 0 / 3 | 0.25042 (0.00923) |
| 160013, 53 s, [21, 5] | 0 / 0 / 0 / 0 | 0.07906 (0.00330) |

Both are bracketing pairs and pass the unchanged strict gate. These are **forced
four-corner conditional continuations**, not samples from the probabilistic Q
proposal and not population collision rates. The first case adds a new local
collision witness beyond the old handpicked seed; the second reproduces a risk
interaction but no collision. The locked validation witness selector returned
an empty list rather than substituting a training example.

Natural scanning took 11.90 s (training) / 5.91 s (validation); total discovery
including proxy audits took 121.88 s / 51.73 s on this local run. These are
implementation timings, not a controlled speedup benchmark. All 77 relevant
regression tests passed and frozen-code/asset verification succeeded.

```bash
python -m pytest tests/test_highd_v46_state_discovery.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_pair_synergy.py tests/test_highd_v46_d2rl.py tests/test_highd_v46_validation.py tests/test_highd_paper_discrete_pairs.py tests/test_highd_mean_precision_policy.py tests/test_highd_dynamic_pairs.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

The concise provenance and evidence record is
[`highd_v46_state_discovery_pilot_v1.json`](../results/highd_v46_state_discovery_pilot_v1.json).
The discovery infrastructure works, but **reliable held-out discovery and a
multi-state training pool are not established**. Next work should measure
missed states/pairs on training data (including broader candidate audits and
mechanism coverage), then predeclare an improved screen with a fresh validation
cohort. This cohort is now consumed; do not retune on it or launch more PPO
epochs to compensate for the missing state diversity.
