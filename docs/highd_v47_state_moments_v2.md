# Coefficient action coverage and uncertainty-aware measurement budgets

This training experiment follows `3a554c4`. The v1 measurement revealed two
causes of unstable J estimates: equal allocation gave a coefficient-heavy
stratum only three draws, and that stratum still randomly omitted an important
action. V2 changes both the partition and measurement counts. The estimand is
the same fixed teacher Q_A conditional second moment on all 33 states from 25
sources. NDE, the physical endpoint, production proposals and PPO are inherited.

## Action coverage before outcomes

For c(a)=P_A(a)^2/Q_A(a), make every supported action with
`c(a) >= 0.001 * sum(c)` a singleton, regardless of its risk score or event
history. Also retain historical event-action singletons above
`0.000001 * sum(c)`. Smaller historical cells rejoin the supported remaining
risk-by-log-LR strata; their mass and possible events are retained.
The thresholds are development choices, not tuned on validation.

This directly covers large coefficient cells that previously shared a broad
stratum. In particular, the v1 witness at source 230053/actions [2,16] becomes
a singleton because its coefficient ranks 25th, independently of its observed
collision. The remaining support continues to use G_h=c/C_h. Therefore
`J_hat=sum_h C_h*mean(Y_h)` remains unbiased conditional on the frozen design.
Initial-action randomness disappears in singleton cells; natural partner and
future-action randomness remain.

## Budget uses coefficient mass and planning uncertainty

All completed old teacher/state-allocation/v1-moment training observations,
including negatives, fit per-action planning counts. Repeated physical QA
state/seed observations are deduplicated. The conditional probe RNG kernel is
tracked separately. The 240000/250000 validation cohorts and final-test bank
are never fitting inputs.

A Beta(.5,.5) working prior gives per-action planning probability ranges.
Project their endpoints through G_h, and take the largest Bernoulli variance
`v_upper_h=max_{p in planning_range} p*(1-p)`. These are **planning model
ranges**, not frequentist evaluation intervals or proof that zeros are safe.
CRN dependence and prior adaptive development do not supply an independent
planning calibration claim. Every stratum receives at least two new draws.
Any historically event-bearing stratum retains its equal-arm count; every
other stratum retains at least half its equal-arm count. The planner can move
only the remaining budget. These floors imply **true measurement variance
<=2 times equal allocation**, for any native collision probabilities on the
same partition. This is a within-stratum variance bound, not an empirical
promise of improvement, and needs no accurate planning model.

The allocated arm minimizes `sum_h C_h^2*v_upper_h/n_h` under the fixed integer
budget by repeatedly assigning the next draw to the largest marginal gain
`C_h^2*v_upper_h/[n_h*(n_h+1)]`. Separability and diminishing gains make this
the exact integer minimum for those planning scores and vector floors. Equal
allocation remains a feasible control. It may still misallocate
when the planning model is wrong; its fitted objective is not native evidence.

## Matched comparison and independent repeat

The [v2 protocol](../configs/highd_v47_state_moments_v2.json) fixes two arms:
equal allocation and uncertainty allocation. They share exactly the same v2
partition and 256 logical draws per state. Two independent batches yield
33,792 logical draws. This matches cost **within v2**; direct numerical
comparisons to v1 also change partition and budget and are not causal effects
of allocation alone.

Within each batch, arms share the same action/native prefixes within each
stratum. A physical draw used by both arms is simulated and stored once.
Unique simulations are at most 33,792; duplicate arm use is never treated as
independent evidence. Batch seed blocks begin at 5300000 and 5400000, with
separate action scheduling seed 570017. All counts, actions, native seeds,
sharing flags and negative cases are locked before any new outcome. Four
worker processes execute independent cases; scheduling order changes no draw.

Report each arm's J, unbiased empirical variance, independent-batch agreement
and zero-event uncertainty. Exact binomial Bonferroni intervals cover the J
family across states, arms and batches. Their mean bounds cover the two-batch
estimate without calling shared arm draws independent.

For each stratum's union observations, sample variance estimates the same
Bernoulli variance for both allocation designs. The paired estimate difference
has covariance `C_h^2*p_h*(1-p_h)/max(n_equal,n_allocated)`. Thus the variance
contrast uses the same outcomes rather than different tail realizations.
A separate 95% simultaneous binomial family bounds the signed variance
contrast `sum C_h^2*(1/n_allocated-1/n_equal)*p_h*(1-p_h)`. Zero sample variance
cannot certify a win. Point ratios remain descriptive unless that upper bound
is negative. J and variance bounds are separately defined 95% families.

Compare the old frozen Q_S candidates using these common J measurements. The
stable contrast `alpha*(alpha/q-1)` gives exactly zero at q=alpha; a numerical
margin prevents roundoff from certifying an improvement. No new Q_S fit or
PPO launch is selected automatically. A repeatable conditional measurement
would still require a later fresh-source check for a transferable model.

## Reproduction

Use Python 3.9.19, NumPy 1.23.1, SciPy 1.13.1 and one BLAS/OMP thread:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_state_moments_v2 freeze --output outputs/v47_state_moments_v2_lock.json
python -m scenario_reconstruction.highd_v47_state_moments_v2 run --lock outputs/v47_state_moments_v2_lock.json --output outputs/v47_state_moments_v2
python -m pytest tests/test_highd_v47_state_moments_v2.py tests/test_highd_v47_state_moments.py tests/test_highd_v47_state_allocation.py tests/test_highd_v46_pair_synergy.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Source/config/code locks and all parent artifacts are checked. Outputs are
exclusive-create; partial attempts remain distinct. The complete fixed budget
is run once without outcome-dependent extensions, extra batches, dropping
dominant observations or claiming success from a small empirical MCSE alone.
