# Measuring state second moments without treating pilot zeros as certainty

This opt-in training experiment follows `ad22e41`. The target remains the
25-source/33-state teacher bank. NDE, state selection, native dynamics, endpoint,
production proposals and PPO are unchanged. In particular, the teacher action
proposal Q_A is the **estimand**, not the new measurement distribution.

## Failure being addressed

The direct observation `Z=Y*(P_A/Q_A)^2` estimates a state second moment J.
Estimating its sampling variance involves fourth powers of the action LR.
Sixty-four zero-event draws give both zero empirical J and zero empirical
variance, without establishing either zero risk or precision. The previous
allocation used these noisy point estimates as though the other states were
negligible. A larger state floor bounded damage but did not fix measurement.

Risk-only stratification is insufficient: the observed high-weight collision
actions at state 19/source 200053 have proxy scores about 0.00053 and 0.00057.
This is development evidence, not a newly discovered validation result.

## Exact coefficient measurement

For each joint action a with natural support, set `c(a)=P_A(a)^2/Q_A(a)`.
If f(a) is its native conditional collision probability, including natural
partners and subsequent natural actions, then `J=sum_a c(a)*f(a)`.

First isolate **every distinct historical training event action** as a singleton
stratum, not just the largest contributor. Teacher, first allocation and the
completed update supply these action IDs; repeated state/seed observations are
deduplicated and checked. Their labels determine the measurement design only
and are not reused as new measurements. This ensures previously discovered
tail actions are actually probed instead of waiting for another rare draw.

Partition all remaining supported actions using the Cartesian product of frozen risk
boundaries `[.2,.4,.6,.8]` and log10 action-LR boundaries `[-6,-3,-1,0]`.
Low-risk and zero-event regions are never discarded. For each nonempty h:

`C_h = sum_{a in h} c(a)`

`G_h(a) = c(a)/C_h`

`J_hat = sum_h C_h * mean(Y_h)`.

The initial pair action is drawn from G_h and every other actor follows its
exact natural conditional law. Subsequent actions remain natural. Although
the native conditional helper accepts an initial action as a forced argument,
these actions are randomly drawn from the declared G, not cherry-picked
collision witnesses selected using this run's outcomes. In a historical
singleton stratum G is one and the native endpoint is still newly sampled.
They are **moment probes**, not Q_A draws or population
risk samples. Their stratum correction is exactly `c/G_h=C_h`; do not apply
the old Q_A likelihood again. Within-stratum random weight magnitudes disappear;
native Bernoulli noise and between-action risk variation do not.

The unbiased conditional variance estimate is
`sum_h C_h^2 * p_hat_h*(1-p_hat_h)/(n_h-1)`.
It can still be zero with no events. Therefore it never independently certifies
precision. This design is not guaranteed to beat direct sampling at a fixed
budget; C-heavy low-event strata can remain expensive to measure.

## Zero-event uncertainty and a safe allocation decision

Use exact Clopper–Pearson binomial intervals, implemented by SciPy 1.13.1
[`binomtest(...).proportion_ci(method="exact")`](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
Bonferroni over each state's
nonempty strata gives a 95% state J interval. A second Bonferroni over **all**
state/stratum cells gives simultaneous 95% bank intervals `[L_s,U_s]`.
These probabilities integrate random action draws and native continuations;
they are not confidence claims conditional on a particular realized action
schedule or generalization intervals over new traffic sources.

Zero-event strata retain positive upper bounds. Precision is reported only
when the state lower bound is positive and interval half-width <=0.5 J_hat.
Neither a zero MCSE nor many repeated draws from one source passes that test.

Compare every frozen Q_S on the **same** J measurements, rather than separate
tail realizations. For target masses alpha and candidate q, let
`d_s=alpha_s^2/q_s-alpha_s`. The true IID M2 difference is `sum_s d_s*J_s`.
Its simultaneous upper bound uses U where d>=0 and L where d<0. Only a strictly
negative upper bound certifies improvement over target-IID. Otherwise the
diagnostic decision is **retain alpha / insufficient evidence**, not acceptance
of a noisy square-root allocation. This is a same-target M2 statement; it is
not a raw concentration threshold or a population acceleration estimate.

Simultaneous coverage also protects a data-dependent q fitted on these J
estimates. Its plug-in fixed-contrast MCSE is deliberately undefined, because
that contrast was fitted on the same measurements. No candidate is deployed.

## Frozen matched-budget run

The [protocol](../configs/highd_v47_state_moments_v1.json) retains all 33 states.
Each gets 128 fresh direct Q_A continuations and 128 stratified moment probes,
8,448 total. Divide the latter evenly among nonempty strata, with deterministic
remainder allocation and at least two draws per stratum. The stratum law uses
existing matrices and completed training event-action IDs, never new native
outcomes. All actions and counts are
saved before simulation. Independent direct/probe native seed blocks start at
5100000/5200000; action scheduling uses 5000017. There is no extra run, adaptive
budget, selective restart, validation fitting, source expansion or PPO export.

The same measurements evaluate the earlier two frozen tabular Q_S proposals
and a diagnostic square-root fit with floor 0.5 and regularization 0.01. No
direct/probe samples are pooled into the old experiment estimates. Empirical
variance comparisons remain descriptive; neither arm supplies ground truth J.

Use the inherited Python 3.9.19/NumPy 1.23.1/SciPy 1.13.1 environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_state_moments freeze --output outputs/v47_state_moments_v1_lock.json
python -m scenario_reconstruction.highd_v47_state_moments run --lock outputs/v47_state_moments_v1_lock.json --output outputs/v47_state_moments_v1
python -m pytest tests/test_highd_v47_state_moments.py tests/test_highd_v47_state_allocation.py tests/test_highd_v47_action_teacher.py tests/test_highd_v47_probability_allocation.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Source locks, implementation, dependencies, complete measurement design and
candidate proposals are verified before exclusive output creation. Retain all
negative endpoints and incomplete attempts. This estimator can prevent an
unsupported allocation; it cannot manufacture information about rare failures
or certify precision when its intervals remain wide.

## Completed evidence: local improvement, unresolved unknown-action tail

Implementation `b5ae6a1` was committed and frozen before outcomes. The full
8,448-draw run completed in 397.98 s. All 33 states and 359 nonempty strata
were retained, including 124 historical-event-action singleton strata. The
historical design read 14,784 old rows, representing 14,194 unique state/seed
draws; none became a new evaluation observation. Direct sampling observed 127
events and probes 266, each spanning eight source episodes. These are different
measurement designs, not improved natural collision rates or new sources.

The following relative SEs are **empirical point diagnostics**, not verified
relative-error guarantees:

| State / source | Direct J | Stratified J | Direct relative MCSE | Stratified relative MCSE |
| --- | ---: | ---: | ---: | ---: |
| 18 / 200051 | 1.27063e-4 | 3.84143e-5 | 41.78% | 63.24% |
| 19 / 200053 | 1.85267e-5 | 7.15003e-5 | 100.00% | 49.95% |
| 32 / 230053 | 6.77080e-2 | 5.62542e-5 | 99.60% | 57.23% |

The known 200053 tail was deliberately remeasured and its empirical relative
noise fell. Improvement is not uniform: 200051's relative MCSE increased, and
the huge 230053 discrepancy prevents a claim of overall stabilization.

### The new unanticipated event region

At source 230053, direct continuation seed **5104146**, initial actions `[2,16]`
produced a CAV event. Their natural probability is 0.00948884, clearance proxy
score 0.312239 and full action LR about 2.93803. This observation contributes
99.6010% of that state's direct J estimate. It is not an astronomically small-P
witness and was not among the historical event-action singletons.

Its remaining-support stratum has coefficient mass **8.14354**, but equal
stratum allocation gave it only **three probes**. The action's conditional G
probability is 0.00342339; none of those probes selected it and the stratum had
zero observed events. Thus isolating known tail actions helps those actions,
but equal counts across strata still undersample a high-coefficient region
whose native collision probability is unknown. The new estimator did not
demonstrate comprehensive tail coverage. The direct event is retained and is
not dismissed because the stratified point estimate is smaller.

Across the declared target, direct measurement estimates M2=9.08596e-4 with
empirical MCSE=8.99150e-4; stratification estimates 5.19805e-6 with MCSE=1.78096e-6.
These measure the **same** true quantity, not two improved testing proposals.
Their different tail realizations rule out interpreting that gap as variance
reduction or a natural-risk change.

### Uncertainty guard outcome

None of the 33 states passes the predeclared relative-precision check. For
example, the 95% state interval at 230053 is approximately
`[5.04315e-7, 7.56121]`. The large upper bound reflects unmeasured event
probability in coefficient-heavy zero-event strata. It is conservative, but
it correctly prevents treating the small empirical MCSE as a precision proof.

On the common stratified measurements, first-allocation and updated-allocation
M2 ratios to target-IID are 1.78381 and 0.44601, respectively. The diagnostic
measurement-fitted candidate gives 0.17827. **None is certified**: every
nontrivial simultaneous M2-difference interval includes zero. The fitted
candidate is not deployed and its in-sample contrast MCSE remains undefined.
The self-comparison has only floating-point roundoff (difference below 1e-21),
not evidence of a target-law change.

An additional, explicitly **post-hoc** same-probe diagnostic estimates what
the direct estimator's variance would be using those same measurements:

`T_hat = sum_h C_h * mean_Gh(Y*w_A^2)`

`V_direct_hat = (T_hat - J_hat^2 + V_hat(J_hat))/128`.

The `+V_hat` term removes the bias from squaring J_hat. This is not another
native experiment and not a new primary estimate. Its stratified/direct
variance point ratio is 0.04763 at 200053, but 0.70588 at 200051. It shares the
probe design's missed-region limitation, particularly at 230053, and provides
no variance-ratio confidence claim. Negative finite variance reconstructions
are retained, with ratios suppressed rather than clipped into success.

### Verification and decision

The independent read-only auditor reconstructs all 35 artifact hashes, all
8,448 measurement records, coefficient corrections, direct likelihoods, J and
variance estimates, beta-quantile confidence bounds and common-probe contrasts.
Maximum direct log-weight discrepancy is 1.42e-14. Eight protected training,
validation and prior-experiment artifacts are unchanged. All 110 relevant
regression tests and frozen-code/assets verification passed.
Sixty-six existing-seed native replays (one direct draw and one probe per
state, prioritizing the largest observed contributions) also reproduced their
stored endpoints. They add no new observations and are not pooled as data.

```bash
python -m scenario_reconstruction.highd_v47_state_moments_audit
```

**Decision:** known-action weight noise and false zero-event certainty are
addressed by the implementation, but reliable state-level second moments are
**not yet established**. Do not promote the fitted state proposal, use the
smaller stratified M2 as a variance win, or append favorable repeats. The next
measurement design must handle uncertainty in the *remaining action region*,
not only replay known tails: coefficient/uncertainty-aware measurement budgets
or explicit action coverage are candidates, with a declared cost cap and
independent measurements. They are not implemented or validated results here.
NDE, Q_A, production Q_S and PPO remain unchanged.

Compact evidence: [highd_v47_state_moments_v1.json](../results/highd_v47_state_moments_v1.json).
The immutable summary SHA is
`931b50da9a07f9a06490a034bae3384fb5f48a76e4b6d2afbd66d06ebe2e0ee9`.
