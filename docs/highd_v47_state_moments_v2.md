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

## Completed evidence, 2026-10-05

Implementation `f9b23f6` was committed and frozen before the new outcomes.
All 66 batch/state cases completed in 375.96 s: 33,792 arm-use observations,
22,234 unique native simulations. Both arms retain all 33 states/25 sources.
The matrix rule created 1,549 strata, including 1,297 coefficient singleton
actions. Planning used 22,642 unique prior physical draws; 590 duplicate
arm uses were excluded. Neither the action teacher nor production Q_S changed.

| Metric | Equal budget | Uncertainty budget |
| --- | ---: | ---: |
| Observed CAV events, both batches | 100 | 100 |
| Observed event sources | 6 | 6 |
| Target second moment | 0.00106080 | 0.00157980 |
| Empirical target-M2 MCSE | 0.00104451 | 0.00117447 |
| States passing bank-interval precision check | 0 / 33 | 0 / 33 |

Shared events are not independent replications. All 66 same-probe variance
contrasts have intervals spanning zero; none certifies an allocation win.
The native result does **not** establish that the new planner reduces noise
overall. Its deterministic <=2x equal-variance protection remains valid for
the same partition; that bound is not a guarantee of absolute precision.

Selected state estimates, averaged over the two independent batches:

| Source / state | Equal J | Equal relative MCSE | Allocated J | Allocated relative MCSE |
| --- | ---: | ---: | ---: | ---: |
| 200020 / 6 | 0.0522223 | 100% | 0.0261112 | 100% |
| 200051 / 18 | 0.000038747 | 38.5% | 0.0263383 | 99.9% |
| 200053 / 19 | 0.000063557 | 70.7% | 0.000063557 | 70.7% |
| 230053 / 32 | 0.000919832 | 91.0% | 0.000112250 | 45.2% |

These MCSEs are empirical conditional diagnostics, not evidence of adequate
coverage. State 19 has identical batch estimates, but each batch contains only
one event; identical point values do not imply a precise state moment. Every
simultaneous state interval remains far too wide for point allocation fitting.

### The old missed cell is covered; future Bernoulli uncertainty remains

The previous 230053/[2,16] witness received four direct conditional draws in
each batch, shared by both arms: eight unique probes, all negative. Its
inclusion did not depend on its old event label. These negatives do not prove
that the action is safe or that its conditional event probability is zero.

Two new coefficient-heavy singleton actions expose the remaining bottleneck:

| Source / initial action | Prior action labels / events | c=P^2/Q_A | New equal / allocated draws per batch | New event seed |
| --- | ---: | ---: | ---: | ---: |
| 200020 / [18,21] | 4 / 0 | 0.522223 | 5 / 10 | 5402119 |
| 200051 / [20,19] | 8 / 0 | 0.788987 | 4 / 15 | 5406192 |

Both are **singleton** layers: this is no longer random omission of the
initial action inside a broad layer. Partners and future natural actions still
vary. The first event is shared by both budgets; the second lies beyond the
equal arm's prefix and is seen only by the allocated arm. Their natural joint
probabilities are 0.04105 and 0.05404, versus teacher Q_A probabilities 0.003227
and 0.003702. The teacher underallocates those naturally common cells, making
their moment coefficients large. A handful of conditional binary outcomes
cannot resolve their native continuation risk. The new planner had explicitly
nonzero planning uncertainty for both; it did not declare them safe, but the
fixed total budget allowed only 10/15 probes, not precise tail measurement.

This separates three issues: initial-action coverage is improved, allocation
is protected against arbitrary overconcentration, but **rare native future
outcomes multiplied by large coefficients remain unresolved**. Merely adding
previously observed event actions to a list moves the blind spot to another
action. Empirical zero-event J values are not suitable fitting targets.

The read-only consumer `checked_moments` therefore rejects a full-bank point
fit unless every state passes the inherited half-width <=50% J check, using
the simultaneous bank interval. It never drops failed/zero states to make the
bank appear precise. This is an additive fail-closed development check, not
an outcome-tuned replacement of the frozen experiment's results or a wired-in
production/PPO change. It rejects both completed arms.

No old Q_S candidate has a certified same-target M2 improvement. For the updated
candidate, the point ratio to target is 1.9929 under equal measurements and
0.8483 under allocated measurements; both uncertainty intervals span zero
improvement. The different point orderings are not a basis for selecting a
winner or refitting Q_S.

**Decision:** keep the entire completed run, but do not promote these point
moments into Q_S or restart PPO. The next estimator design must address native
continuation uncertainty itself (e.g. separately weighted continuation
stratification/importance sampling or conditional averaging), rather than
another point-J rebalance. Any such design needs explicit future-action
likelihoods and an independently locked evaluation; no such extra run was
launched here. Production, consumed validation and reserved final tests remain
unchanged. The 95% intervals are conservative; failed certification is not
proof that every allocation is intrinsically inferior.

Independent reconstruction verified all 68 artifact hashes, every frozen
schedule and arm prefix, coefficient correction, integer optimum and floors,
endpoint labels, stratum estimates, covariance, exact intervals and common-J
Q_S comparisons. Maximum coefficient-correction error was 5.55e-17. The summary
SHA is `38fd9eeb002da1cc4ce4d503d5d23f7ec0446d00e222fde355c7aa102ac1818d`.
Three new event continuations and one negative singleton continuation were
replayed at their original seeds and matched exactly; they are verification,
not added observations. All 133 relevant regression tests and frozen-code/asset
verification passed. Protected prior summaries/manifests remain unchanged.
Raw outputs remain ignored. Compact evidence:
[highd_v47_state_moments_v2.json](../results/highd_v47_state_moments_v2.json).

```bash
python -m scenario_reconstruction.highd_v47_state_moments_v2_audit
python -m pytest tests/test_highd_v47_state_moments_v2_audit.py tests/test_highd_v47_state_moments_v2.py tests/test_highd_v47_state_moments.py -q
```
