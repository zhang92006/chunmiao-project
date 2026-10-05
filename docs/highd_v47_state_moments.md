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
`binomtest(...).proportion_ci(method="exact")`. Bonferroni over each state's
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
