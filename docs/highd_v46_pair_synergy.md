# V46 pair-synergy diagnostic before PPO

This experiment depends on the dual-validation implementation at `729ed95`.
It does not modify the frozen NDE or `highd_v46_d2rl.py`, and leaves previous
sequence/physical-target identities valid. No PPO training is performed.

## What is being tested

The previous collector already evaluates a 33x33 pair action matrix; it does
not simply choose the two highest single-car criticalities. Its candidates
are restricted to supported natural following units, and its legacy proxy
holds candidate acceleration for the full horizon while other BVs use fixed
mean accelerations. These restrictions can conceal interactive effects.

The diagnostic considers every pair among the nearest four BVs within 115 m,
plus any old natural pair incident to those actors. A same-unit pair uses its
exact natural joint distribution (transposed when needed). A cross-unit pair
uses the product of its exact unit marginals, not independent replacements of
the natural units. These are valid pair marginals for diagnostics only.

For supported actions, compute the second-actor leverage

`Delta_j(a,b) = R(a,b) - sum_b' P(b'|a) R(a,b')`,

and the symmetric first-actor quantity. The natural action is averaged using
its conditional law, not an arbitrary sampled or independent action.
Leverage by itself does not establish synergy.

Choose the joint natural mode `(a0,b0)` as a reproducible anchor and compute

`I(a,b) = R(a,b) - R(a,b0) - R(a0,b) + R(a0,b0)`.

Only actions whose four corners all have positive natural support enter I.
The report records that support's mass, the anchor and witness actions. The
ranking score is `sum_ab P(a,b) max(I(a,b),0)`; this avoids ranking purely by
one extremely unlikely action's maximum. Additional gates require leverage
from both actors, nontrivial H-versus-P influence in both sequential factors,
and positive joint excess above the sum of the two single-action increments.
The selector explicitly abstains when no pair passes; it never fills a second
slot merely to call an intervention dual.

`C_i`, `C_j` and `C_ij` in this implementation are maximum risk **increments
relative to the same modal anchor**, not calibrated crash probabilities or
the paper's natural criticality definition. Positive interaction depends on
the anchor and the nonlinear risk scale. An AND-risk fixture passes, whereas
single-actor and additive-risk fixtures fail. Correlation alone and omitted
unsupported corners must not be interpreted as physical cooperation.

## Three controlled diagnostic variants

| Variant | Selected action duration | Subsequent BV response |
| --- | --- | --- |
| `legacy_hold` | Entire four-second proxy horizon | Existing fixed mean acceleration |
| `mean_1s` | One second | Initial mean acceleration |
| `reactive_1s` | One second | State-dependent fitted longitudinal IDM |

The last two share exact speed-bound kinematics, dynamic CAV IDM response,
one-second linear lane changes and first-any-collision truncation. Their
difference isolates longitudinal BV response. Both are deterministic
surrogates: later stochastic actions/lane changes and natural partner
conditioning in counterfactual continuations are not integrated out. Therefore
a larger interaction in `reactive_1s` is a sensitivity finding, not evidence
that it predicts real collisions better. One-second execution is regression
checked against the native executor at a 1/15 s physics step. The pilot proxy
uses a coarser 0.2 s step; fine-step/horizon sensitivity and stochastic
continuation validation remain necessary before deployment.

## Natural state pool, separate from population evaluation

The predeclared config is
[`highd_v46_pair_synergy_diagnostic_v1.json`](../configs/highd_v46_pair_synergy_diagnostic_v1.json):
32 natural rollouts, seeds 120000-120031, fixed 60 s horizon, observations every
5 s before the action draw. High-risk flags use current TTC, bumper gap,
closing speed and supported adjacent-lane cut-in opportunity. They never use
a future collision or selected candidate action. Reservoir sampling uses its
own RNG (seed 41), retaining at most 16 states in each of the disjoint high-risk
and control strata. It cannot perturb the rollout RNG. Source stratum sizes,
episode/time identifiers, actor states, PMFs, natural units, histories and
pre-draw RNG state are saved for audit. Snapshots are **after prepare**, so
blindly calling prepare again would advance pairing history twice.

The pool is a risk-stratified diagnostic/training distribution, not a new natural
initialization. It is deliberately incompatible with the existing PPO
sequence-manifest contract. Before training from these states, define the
conditional task and sampling/replay correction, preserve episode-level split
boundaries, and implement a tested snapshot-restoration interface. Action-only
P/Q weights cannot correct a changed initial-state distribution. Formal risk
and acceleration testing must still begin from the original natural
initialization, or include an explicitly derived initial-state density ratio
with support coverage. The reserved final-test seeds remain untouched.

The author's [training workflow](https://github.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning/blob/main/README_D2RL_Train.md)
collects behavior-policy trajectories before offline D2RL training and performs
separate online validation. The state-pool diagnostic here is an extension,
not a claim that arbitrary high-risk resets reproduce that procedure.

## Run and inspect

```bash
python -m scenario_reconstruction.highd_v46_pair_synergy --output outputs/v46_pair_synergy_diagnostic_v1
python -m pytest tests/test_highd_v46_pair_synergy.py -q
```

Runtime outputs contain the state pool, every supported pair's three risk
matrices, pair metrics, abstaining selector decisions and a compact summary.
They stay in ignored `outputs/`; only configuration and concise evidence are
committed. Observations/pairs within a rollout are correlated, so pair counts
are not independent event counts or a basis for a significance claim.

Cross-unit choices cannot be passed to the current online sampler, which
tilts one natural unit only. Deployment must preserve partners' conditional
law across all affected units, account for the complete P/Q factorization,
and verify support and unbiasedness before collecting fresh training data.

## Native conditional witness check

```bash
python -m scenario_reconstruction.highd_v46_pair_witness --diagnostic outputs/v46_pair_synergy_diagnostic_v1 --output outputs/v46_pair_witness_v1
```

The separately declared witness config selects at most four distinct states
by the reactive proxy's interaction mass, before observing continuation
outcomes. Each uses 16 continuations, seeds 1300000-1300015. Its four corners
share random numbers. Only the initial pair actions are fixed; all unforced
actors, including partners from affected natural units, are drawn from the
exact conditional natural law. Later actions use unchanged native V46 dynamics
and stochastic NDE for four seconds at 15 Hz. Saved pairing histories are
restored without an extra prepare call. This is a conditional experiment, not
importance sampling of the original initial-state population.

## First pilot findings

The 32 natural source episodes had no collisions. Of 384 five-second
pre-draw observations, 244 (63.5%) passed the broad geometry flags; the bank
retained 16 high-risk and 16 control states. This balancing actually reduces
the flagged fraction to 50%, so no increase in rare-event density has been
established. The flags need calibration rather than being labelled a
successful densification scheme.

Across 189 state-pair combinations (46 same-unit, 143 cross-unit), the legacy,
one-second mean and reactive proxies passed 1, 2 and 2 pairs respectively.
Both reactive selections are cross-unit pairs bracketing the CAV. An important
example is seed 120013 at 25 s, CAV 37 between rear BV 36 and front BV 38.
The selected first-second actions change rear acceleration from -2.4 to
+0.8 m/s2 and front acceleration from 0 to -3.0 m/s2.
The joint action's natural conditional probability is approximately 0.000175;
the forced-corner event fraction below is not its natural event probability.

| Initial corner | Mean native clearance score | CAV events / 16 |
| --- | ---: | ---: |
| Modal baseline | 0.0349 | 0 |
| Rear change only | 0.1055 | 0 |
| Front change only | 0.0526 | 0 |
| Both changes | 0.8837 | 5 |

The paired clearance interaction mean was 0.7605 (Monte Carlo SE 0.0518).
This supports a local front-braking/rear-approach interaction in the native
simulator. The second selected state had a weaker clearance interaction
(0.0941, SE 0.00454) and no collisions in any corner. Neither is a population
risk estimate or independent method-level significance claim. The old proxy
overestimated isolated action danger in the strong example by holding actions
or mean accelerations too long, and the old selector excluded this cross-unit
pair entirely. The stochastic witness supports revising those assumptions;
it does not establish global surrogate accuracy.

The compact record is
[`highd_v46_pair_synergy_pilot_v1.json`](../results/highd_v46_pair_synergy_pilot_v1.json).
The next implementation step is an audited cross-unit two-factor intervention
operator, followed by separately seeded conditional training-data collection.
This branch deliberately leaves the production collector and PPO unchanged.
