# Cross-natural-unit dual intervention

This opt-in experiment builds on `b439dd6`. Frozen NDE tables, natural pairing,
initialization, CAV dynamics, event definition, the existing collector and PPO
trainer remain unchanged. The new entry point is `highd_v46_cross_unit`.

## Complete likelihood, including natural partners

Let P(A) be the product of the current disjoint natural units, and let i,j be
the selected actors. Their marginal Pij is the existing joint table when both
belong to one unit, and the product of exact unit marginals otherwise.
After drawing the controlled pair, draw every other actor under its original
natural conditional law:

`Q(A) = Qij(ai,aj) P(A_rest | ai,aj)`

`P(A)/Q(A) = Pij(ai,aj)/Qij(ai,aj)`.

A partner is not resampled from an independent marginal. Selecting one actor
from each of two correlated natural units can involve four actors in the
likelihood decomposition, while only two factors are directly tilted. The
partner marginals may change indirectly through their natural dependence.

For `H = Pij * positive_interaction / sum(Pij * positive_interaction)`, use

`Qij = [eps1 P_i + (1-eps1) H_i] [eps2 P_j|i + (1-eps2) H_j|i]`.

Both epsilon values lie in [0.05,1]. Zero H-mass rows use P's conditional row.
Consequently Q preserves P's support and Qij >= eps1*eps2*Pij. Exact enumeration
tests cover same-unit, reversed-order and cross-unit control, zero-H rows,
normalization, E_Q[P/Q]=1 and recovery of natural event probabilities. A second
test compares actual sample frequencies to the independently enumerated full
four-actor law. The per-decision log records both two-factor P/Q and full
log P/log Q with the cancelled partner conditional term.

## Selection, context and deliberate limitations

Candidates use the diagnostic's nearest-four pairs plus incident natural
pairs, ordered rear-to-front by x (actor ID breaks ties). The reactive
four-second surrogate supplies the four-corner interaction. Candidates must
pass the earlier leverage/interaction/excess gates and both factor-influence
gates for the *actual interaction-based H*. Rank by natural-probability-weighted
positive interaction mass; abstain to P when no candidate passes.

This pilot evaluates candidates every five seconds to bound CPU cost; other
seconds use P exactly. This schedule is explicit Q configuration. It can miss
short-lived opportunities. The proxy remains a deterministic, coarse-step
approximation and the strict gates may produce few interventions. Neither
the schedule nor thresholds have been tuned using final-test episodes.

The 22-D causal context remains compatible with the sequence replay interface.
The two BV slots now describe the ordered controlled pair, which need not be
natural followers or even occupy the same lane. The pair gap/speed entries are
signed relative geometry. Both factor masks are active; an epsilon fixed at 1
in a baseline disables tilting that factor, but the factor still exists in the
policy contract. Counts of directly biased factors use epsilon<1.

New proposal-structure hashes include the selector/proxy/conditional-sampler
source, the pair-order convention and selection cadence. Old following-pair
datasets and models cannot silently be used with this proposal. The physical
P identity remains the unchanged inherited native experiment. Changing epsilon
changes Q only. A loaded policy must match P, observation, action dimension and
proposal structure; evaluation seeds cannot overlap training/validation seeds.

Interaction-weighted H deliberately emphasizes cooperative risk. It is a
research proposal, not a guarantee of lower variance than risk-weighted H.
Comparing those proposal scores and auditing proxy calibration remain future
ablations; the original PPO variance objective has not been changed.

## Reproducible natural-initialization pilot

The predeclared protocol is
[`highd_v46_cross_unit_pilot_v1.json`](../configs/highd_v46_cross_unit_pilot_v1.json):
eight 60-second episodes per arm, seeds 140000-140007, with all outcomes retained.

```bash
python -m scenario_reconstruction.highd_v46_cross_unit --mode natural --episodes 8 --seed 140000 --output outputs/v46_cross_unit_v1_natural8
python -m scenario_reconstruction.highd_v46_cross_unit --mode dual --epsilon 0.1 1.0 --episodes 8 --seed 140000 --output outputs/v46_cross_unit_v1_first_factor8
python -m scenario_reconstruction.highd_v46_cross_unit --mode dual --epsilon 0.1 0.1 --episodes 8 --seed 140000 --output outputs/v46_cross_unit_v1_dual8
python -m scenario_reconstruction.highd_v46_statistics --collections natural=outputs/v46_cross_unit_v1_natural8 first_factor_only=outputs/v46_cross_unit_v1_first_factor8 dual=outputs/v46_cross_unit_v1_dual8 --bootstrap 30 --output outputs/v46_cross_unit_v1_comparison
```

The first-factor-only arm uses the same pair selector and two-factor contract
with eps2=1. It isolates the second factor's effect; it is **not** the paper's
single-BV agent or a validation-tuned baseline. These are development runs,
not an efficiency benchmark. Reserved test seeds 100000-101023 remain unused.

The existing trainer and fixed-baseline selector accept the resulting
likelihood sequences when event counts and independent validation permit.
Use this new collector with `--policy <export_directory>` for online evaluation;
the old workflow's evaluator correctly rejects the changed proposal structure.
No PPO training is launched by this pilot.

## Separate conditional probe

```bash
python -m scenario_reconstruction.highd_v46_cross_unit_probe --state-pool outputs/v46_pair_synergy_diagnostic_v1/state_pool.json --output outputs/v46_cross_unit_conditional_probe_v1
```

This predeclared probe uses the known training state (source seed 120013 at
25 s), 256 independent continuation seeds 1500000-1500255 per arm, and a four-
second horizon. The *pair selector* chooses the pair, then the new Q samples
its first actions; unlike the earlier witness test, no particular adverse
action is forced. All subsequent actions are natural. All three arms share
the same fixed source state and a common-random-number seed schedule.

Ordinary IS estimates here target only the conditional event probability
given that state. They have no initial-state-density correction and cannot be
mixed into a natural-initialization estimate. The probe exports no population
sequence manifest, no checkpoint and no acceleration estimate. It tests whether
the probabilistic proposal can exercise the known mechanism while preserving
likelihood accounting, not whether the method generalizes.

Run regression tests with:

```bash
python -m pytest tests/test_highd_v46_cross_unit.py tests/test_highd_v46_pair_synergy.py tests/test_highd_v46_d2rl.py tests/test_highd_v46_validation.py tests/test_highd_paper_discrete_pairs.py tests/test_highd_mean_precision_policy.py tests/test_highd_dynamic_pairs.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

## Pilot evidence, 2026-10-04

All 69 regression tests passed and frozen-code/asset verification succeeded.
The three fresh natural-initialization arms each completed eight episodes
without a CAV collision. Each biased arm evaluated 538 pair candidates, but
none passed the strict gate, so no intervention occurred and all three arms
followed the same trajectories. The comparison correctly reports undefined
RHW/acceleration and censored bootstrap replicates. These data do not measure
an online dual advantage. Candidate evaluation also cost roughly 264 s per
biased arm versus 6.4 s for natural sampling in this concurrent local run;
this is implementation overhead, not a controlled timing benchmark.

A separately labelled mechanism replay from the previously selected training
seed 120013 did trigger a cross-unit intervention at 25 s. The complete 60 s
episode had no collision; its full likelihood and two-factor likelihood agreed
within 1.56e-15. The exported real sequence passed the offline replay audits.
It is excluded from the fresh-seed comparison.

The conditional one-decision probe produced:

| Arm | CAV events / 256 | Mean likelihood ratio |
| --- | ---: | ---: |
| Natural | 0 | 1.0000 |
| First factor only | 0 | 1.0739 |
| Dual | 7 | 1.2538 |

Here Q sampled the initial actions rather than forcing the earlier witness's
worst corner. The dual weighted conditional event estimate was 0.000168989,
SE 0.0000642805, and 90% RHW 0.6257; seven events are below the 20-event guard
and RHW remains above 0.3. Its raw event fraction 7/256 is not a natural risk
estimate. The empirical mean weight's Monte Carlo SE was 0.4263 (first factor:
0.1184), so finite-run averages need not equal the exact expectation 1.
The two H-versus-P factor TVs at this source state were 0.5268 and 0.6289.
Both output factors now materially affect the proposal locally.

The concise record is
[`highd_v46_cross_unit_pilot_v1.json`](../results/highd_v46_cross_unit_pilot_v1.json).
No PPO was trained: the fresh cohort has no event sequences, and the known-
state conditional probe cannot be relabelled as population-initialized data.
The next research step is a separately defined multi-state conditional training
and validation distribution (split by source episode), along with improved
state discovery/candidate-screening cost. Any eventual trained policy still
requires independent original-initialization evaluation. Do not enlarge the
current dataset by appending only collision episodes or relax precision guards.

The next-stage [state-discovery pilot](highd_v46_state_discovery.md) and
[training-only coverage audit](highd_v46_discovery_coverage.md) investigate
temporal-bank and pair-budget misses without changing this sampler or NDE.
