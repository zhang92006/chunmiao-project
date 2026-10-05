# V47: collision-probability allocation

This opt-in experiment builds on `c92950a`. It addresses probability allocation
under the same frozen NDE, state/pair bank, physical dynamics and four-second
CAV-first-collision endpoint. The first milestone is a training-only calibration
and allocation audit. Its protocol is
[highd_v47_probability_allocation_v1.json](../configs/highd_v47_probability_allocation_v1.json).

## The objective for the current intervention horizon

Let `f_s(a)=Pr(CAV first collision within four seconds | s, controlled pair a)`.
This integrates correlated natural partners and the subsequent natural actions.
Only the initial controlled action changes distribution. For the realized
Bernoulli event Y and weight `w=P(a|s)/Q(a|s)`, the conditional second moment is

`J_s(Q) = E_Q[Y*w^2] = sum_a P(a|s)^2 * f_s(a) / Q(a|s)`.

The event probability `sum P*f_s` is independent of Q. For fixed states and
fixed replication counts, minimizing J therefore minimizes estimator variance.
Without support-floor constraints the solution is `Q_star ∝ P*sqrt(f_s)`.
The square root arises because `E[Y^2|s,a]=E[Y|s,a]=f_s(a)`. Substituting
`P*f_s` would optimize a different experiment that evaluates the exact
conditional probability instead of one stochastic continuation per draw.
This specialization follows the stochastic-simulator result in Lemma 1 of
[Chen and Choe, Importance Sampling and its Optimality for Stochastic Simulation Models](https://arxiv.org/html/1710.00473#S2).
The constrained two-factor derivation below is our adaptation to this code.

For `P=P1*P2`, retain the factor bounds `q1>=0.1P1`, `q2>=0.1P2`. For each
supported first action i, solve

`L_i = min_q2 sum_j P2(j|i)^2 * f_s(i,j) / q2(j|i)`.

The conditional solution is `q2(j|i)=max(0.1P2(j|i), c_i P2(j|i)*sqrt(f_s(i,j)))`.
Then solve `q1(i)=max(0.1P1(i), c P1(i)*sqrt(L_i))`, normalizing each distribution.
An active-set algorithm implements these equations without a gradient optimizer.
All-zero objectives return the natural distribution. Unsupported actions keep
zero mass. The first-factor-only comparator fixes q2=P2 and uses its unoptimized
row cost in the q1 calculation. Dual allocation cannot have a larger **predicted**
objective than any feasible comparator using the same probability estimates.
That mathematical property does not establish an improvement under true native f.

## Explicit Q factors and likelihood contract

V47 samples Q factors directly using a new contract. A row can reach q1's
natural-mixture floor while still tilting q2. V46's shared joint H cannot always
represent this: H1=0 makes its conditional H2 default to P2. Thus this is a
larger proposal family under the same factor bounds, not merely a new H score.
The new record contains P/Q factors and the allocation contract, without
inventing a legacy joint H. Old PPO sequence/checkpoint interfaces are not used.

The complete law remains `Q(full)=q1*q2*P(rest|controlled)` and
`P(full)/Q(full)=P1*P2/(q1*q2)`. Natural partners are drawn from the exact
original conditional unit law. The bounds imply `Qij>=0.01Pij`; both individual
LRs are at most 10 and their product at most 100 for the declared epsilons.
Tests independently enumerate four-actor laws, reproduce legacy draws for
identical Q/RNG, check actual sampling frequencies, and compare the allocation
objective against an independent constrained optimizer in joint coordinates.
Production V46 code and frozen assets remain unchanged.

## Training data and source-held-out calibration

The checksum-locked V46 training comparison and its two original calibrations
supply all 31 eligible cases from 24 training sources. Every zero-event case
is retained. On each case, use 64 draws from each of the four original arms
(natural, first-factor, second-factor, dual) and three later proposal arms
(interaction, absolute risk, mixture): 13,888 logged observations total.
This includes the old interaction tail realizations missed by the later
interaction run. Consumed validation cohorts 240000–240255 and 250000–250255
and reserved final-test sources never enter this loader. Training sources are
restricted to the locked 200000–200063 and 230000–230063 cohorts.

The target remains uniform source, then uniform eligible states within source.
Each behavior component has equal weight within case. A logged observation
has design mass `case_mass/(7*64)`; calibration multiplies it by its own `P/g`,
where g is its actual generating proposal. Mixing uncorrected collision rates
from different behavior distributions would estimate the wrong action law.
Common random numbers are retained as paired blocks, not additional independent
sources or IID uncertainty replicates.

The first calibrator is deliberately small: five predeclared absolute-risk bins
crossed with whether positive interaction is at least 0.05. Each bin uses the
natural-action weighted event fraction, shrunk toward the fitted global event
fraction with total regularization mass 0.01 divided equally among ten bins.
Predictions are clipped to [1e-6, 1-1e-6]. This is model regularization, not
clipping of IS weights, a Beta posterior, or evidence of a positive true risk
in a zero-event bin. Bin averages approximate f only if the features transfer
across states; individual state/action calibration is an empirical question.

Every source receives predictions from a fit that excludes **all** its states,
arms and continuation outcomes. The full-training fit is saved as a candidate
artifact for later development, not used in the source-held-out estimates.
Report held-out weighted Brier/log loss against raw clearance score and the
held-out global-probability baseline, bin reliability, and event-source coverage.
The five-event-source review guard is retained. No hyperparameter or bin is
selected using the consumed validation outcomes.

## Probability-allocation audit and its interpretation

Each held-out case receives the same six candidate distributions: V46
interaction, absolute risk, 50/50 mixture, calibrated-linear joint H, optimized
first factor, and optimized two factors. The latter three share the same held-out
collision-probability model, distinguishing calibration from allocation effects.

For a candidate Q and a sample generated under g, the correct off-policy
second-moment contribution is `Y*P^2/(g*Q)`. The script averages each of the
seven separately corrected component estimates, then applies the fixed
source/state masses. Simply recomputing `Y*(P/Q)^2` on g samples is incorrect.
Exact-enumeration tests cover this distinction. Predicted J and this corrected
observed J are reported separately, along with factor TV, source contributions,
paired-block concentration and action LR at the logged events. The original
event observations remain intact; new-Q collision counts/rates are undefined.

This first milestone runs **zero new native continuations**. The fitted folds
overlap in their other training sources and the behavior arms share random
numbers, so no naive IID/whole-source bootstrap significance interval is
asserted. No proposal is selected automatically, no production replacement or
PPO export occurs, and no population acceleration is estimated. These diagnostics
identify allocation/calibration problems before committing another native budget.

## Reproduction

Use the pinned Python 3.9.19, NumPy 1.23.1 and SciPy 1.13.1 environment. Freeze
the reviewed implementation/configuration before running the fixed audit:

```bash
conda activate highd-ppo-clean39
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_probability_allocation freeze --output outputs/v47_probability_allocation_v1_lock.json
python -m scenario_reconstruction.highd_v47_probability_allocation run --lock outputs/v47_probability_allocation_v1_lock.json --output outputs/v47_probability_allocation_v1_train
python -m pytest tests/test_highd_v47_probability_allocation.py tests/test_highd_v46_proposal_comparison.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_collision_coverage.py tests/test_highd_v46_multistate_q.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Outputs are exclusive-create. Input artifacts, original seed schedules, full
factor likelihoods, protocol/environment/dependencies and source exclusions are
checked. A changed implementation or protocol needs an explicit new version and
lock; do not silently overwrite evidence. Raw observations, fitted bin tables
and per-case Q matrices stay in ignored outputs. Only compact evidence is
committed. Later native comparison must freeze the model and sampler identity
before its outcomes, use a fixed budget and held-out sources, and keep the
original-initialization evaluation separate from this conditional target.
