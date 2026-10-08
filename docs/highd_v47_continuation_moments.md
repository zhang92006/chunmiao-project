# Bounded continuation importance sampling for state-moment measurement

This experiment follows `98b72e5`. It addresses the v2 failure in which fixed
initial-action singleton probes still encountered sparse, coefficient-heavy
events under later natural actions. It changes the **measurement** law of
those later actions, not NDE, native physics, the event, the 33-state/25-source
target, teacher Q_A, production Q_S or PPO. Consumed validation episodes and
reserved final-test data are not fitting inputs or new collection sources.

## Target and correction

For initial controlled action a, c(a)=P_A(a)^2/Q_A(a) is unchanged. Let tau
include unforced correlated partners at the initial decision and the stopped
subsequent native continuation. The desired moment is

`J_s = sum_a c(a) E_Pfuture[Y | s,a]`.

At each later pre-draw state, a deterministic rule selects the nearest two BVs
within 115 m, ordered rear-to-front then actor ID. The inherited held-action
clearance proxy enumerates their 33x33 actions over the remaining horizon with
step 1/3 s, scale 10 m. H is proportional to `P * R^4`, with P fallback at zero
mass. The usual separate-factor epsilon 0.1 mixture and exact natural-partner
conditional sampler define a sequential proposal T_future. This score is an
uncalibrated development heuristic, not a native collision probability. No
cooperative gate or outcome label selects later actions. With fewer than two
nearby BVs, T uses P exactly.

Choose a single latent component at the start of each measured path:

`G_future(tau) = 0.5 P_future(tau) + 0.5 T_future(tau)`.

The initial chosen actions are forced equally in P and T. All initial natural
partners are drawn under their exact conditional law, unchanged. T begins only
at the next decision. Both path densities are evaluated along **every** path,
including natural-component paths. A first-step endpoint has equal P/T
densities and is simulated once. Stop on the same first collision/horizon as
the original experiment; BV-only first collisions remain negative endpoints.

The complete future correction is computed in the log domain:

`W_future = 1 / [0.5 + 0.5 exp(log T_future - log P_future)] <= 2`.

This is a path mixture, not the product of per-decision defensive mixtures.
The initial partner law cancels, as do all unmodified later unit conditionals.
Selected actors may belong to different natural units. Full unit likelihoods
and tilted factors must agree; partners are never replaced with independent
marginals. Pre-draw state-adaptive selection is included in T's definition.

For an initial action stratum h sampled from G_h=c/C_h, the measured root is
`Z=Y*W_future` and

`J_hat_s = sum_h C_h * mean_h(Z)`.

The future LR occurs **once**, because it corrects a conditional event
expectation. Squaring it would change the estimand. Exact finite stopped-path
enumeration tests verify normalization, E_G[W]=1, event recovery, and the bound.
The complete-pair initial coefficient already contains the original square.
The protective bound implies E_G[Z^2]<=2 Pr_P(E), not a guaranteed centered
variance reduction, guaranteed precise J, or testing acceleration.

## Frozen matched budget and independent batches

The [protocol](../configs/highd_v47_continuation_moments_v1.json) reuses the
checksum-locked v2 **equal** action partition and counts without changing any
source, stratum, initial action law or 256-root/state budget. Natural and
defensive controls use common initial actions and native seeds. Natural latent
components share their physical path with the control; T components use a
separate path with the same seed for common random numbers. Identical shared
paths are not independent observations. These are paired estimators, not two
independent traffic populations.

Two independent batches use seed blocks 5800000/5900000 and scheduling seed
580017. All initial actions, latent components, seeds and negative states are
written before native outcomes. There are 33,792 logical arm uses and at most
25,414 planned native path simulations; initial-decision endpoints can reduce
the latter further. Four workers only change execution order. No case
substitution, outcome-dependent extension, third batch or selective restart
is allowed. Timing includes proxy evaluation and is an implementation profile,
not a controlled efficiency benchmark. Any partial attempt remains distinct.

Report both batches' state J, empirical MCSE, paired differences and estimated
variance, actual future weights, source/event contributions and common-J
comparisons of the old frozen Q_S candidates. No new Q_S is fitted or exported.
Variance ratios are descriptive point estimates, not automatic success claims.
Differences from v2 also involve new native seeds; do not attribute every
numerical change to the proposal.

Z in the defensive arm is bounded in [0,2], not Bernoulli. Intervals use the
scaled, two-sided [Maurer/Pontil empirical Bernstein bound, Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf),
with sample variance (n-1 denominator) and log(4/error) after unioning the two
tails. Bonferroni covers every action stratum/state/arm/batch; interval sums and
batch averages cover J. The known expectation is a probability <=1, although
individual weighted observations can exceed one. Shared arm paths need no
independence for the union bound; independent roots within each fixed stratum
do. Zero empirical variance retains positive interval width. The intervals may
be conservative at this budget, and cannot turn zero events into zero true risk.
The inherited half-width <=50% J precision check remains descriptive and
does not waive the previous full-bank fitting guard. There is no PPO launch,
population risk estimate or source-disjoint generalization claim.

## Reproduction

Use inherited Python 3.9.19, NumPy 1.23.1, SciPy 1.13.1, and one BLAS/OMP thread.
Review/commit before freezing the full design. Config, all prior artifacts,
current dependencies, environment and design hashes are checked before output.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_continuation_moments freeze --output outputs/v47_continuation_moments_v1_lock.json
python -m scenario_reconstruction.highd_v47_continuation_moments run --lock outputs/v47_continuation_moments_v1_lock.json --output outputs/v47_continuation_moments_v1
python -m pytest tests/test_highd_v47_continuation_moments.py tests/test_highd_v47_state_moments_v2.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Outputs are exclusive-create. Raw paths and action/likelihood logs stay ignored;
only the protocol, implementation, tests and compact evidence are committed.
The fixed conditional bank cannot by itself demonstrate transfer to new source
episodes or the paper's original-initialization acceleration.

## Completed evidence, 2026-10-06

Implementation `5c42104` was committed/frozen before the 5800000/5900000-series
outcomes. The entire two-batch cohort completed in 582.51 s: all 33 states/25
training sources, 33,792 logical uses and 25,414 native paths. Every endpoint
and zero-event case was retained. No initial-decision early endpoint reduced
the predeclared upper path budget. The following are matched conditional
estimators, not new independent traffic populations:

| Metric | Natural continuation | Defensive path mixture |
| --- | ---: | ---: |
| Observed CAV events / 16,896 roots | 93 | 135 |
| Observed event sources | 8 | 10 |
| Target teacher-action M2 estimate | 0.0000381555 | 0.00183522 |
| Empirical measurement MCSE | 0.0000298826 | 0.000460351 |
| Empirical relative MCSE | 78.32% | 25.08% |
| First-batch target M2 | 0.00000998830 | 0.000683755 |
| Second-batch target M2 | 0.0000663228 | 0.00298668 |
| States passing simultaneous precision check | 0 / 33 | 0 / 33 |

Defensive events add sources 200024/200034 to the natural control's observed
set. These are previously initialized training episodes, not ten fresh traffic
sources. Shared paths are not independent arm replications. Raw defensive event
counts/fractions do not estimate natural risk. The two means differ by about
48x despite targeting the same J; small empirical baseline M2/variance is not
a reliable reference when it omits coefficient-heavy events. In particular,
the primary empirical measurement-variance ratio is **237.32**, not a supported
claim that the true defensive variance is 237 times worse. Do not hide this
primary comparison or replace it with a favorable secondary calculation.

The bounded likelihood mechanism works: all 16,896 actual defensive root
weights lie between 0.000270424 and 1.999935, with empirical mean 1.000382.
The expectation-one property is exact for the law, not for every finite run.
All complete-versus-factor log differences agree to 1.16e-13, including
correlated partners. Coverage repair here does not certify accurate moments.

| State / source | Natural J (relative MCSE) | Defensive J (relative MCSE) |
| --- | ---: | ---: |
| 6 / 200020 | 1.1372e-11 (100%) | 0.0109436 (81.69%) |
| 18 / 200051 | 0.0000597875 (32.85%) | 0.0352598 (28.67%) |
| 19 / 200053 | 0.000113540 (45.66%) | 0.00307895 (97.28%) |
| 32 / 230053 | 0.00233881 (95.56%) | 0.00462619 (69.15%) |

State 32's defensive batch estimates are 0.00476347 and 0.00448890; state 18's
are instead 0.000754610 and 0.0697650. Local repeatability improved at one
known state, but not across the bank. Relative MCSE can fall merely because
the estimated mean increased; it is not by itself proof of noise reduction.

### Separate same-draw variance diagnosis

After seeing the completed primary comparison, a **post-hoc explanatory**
calculation uses the defensive paths to estimate both measurement variances
on the identical initial-action strata. For a stratum, f=E_G[Z]=E_P[Y] and

`Var_G(Z) - Var_P(Y) = E_G[Z^2-Z]`.

Thus `C_h^2*mean(Z^2-Z)/n_h` estimates the signed variance difference without
depending on another arm's finite event realization. A distinct-root
U-statistic estimates f^2 when reconstructing the natural variance; a finite
negative estimate is retained, not clipped to manufacture a ratio. Separate
empirical-Bernstein intervals use the range [-1/4,2] of Z^2-Z, Bonferroni over
all strata/states/batches. They are a separate 95% analysis family and are not
an independently confirmed result or an outcome-blind training criterion.
No proposal parameter or native budget changed in this analysis.

| Same-draw quantity, averaged over two batches | Point estimate |
| --- | ---: |
| Natural measurement variance inferred from G | 4.56470e-6 |
| Defensive measurement variance from G | 2.11923e-7 |
| Defensive / natural variance, point only | 0.04643 |
| Signed variance difference | -4.35278e-6 |
| Separate simultaneous difference interval | [-0.00722538, 0.05780305] |

Of the 135 defensive events, 123 have future W<=1. The largest favorable
contrast contributions include source 200051/[20,19] continuations with
weights near 0.079-0.081. This is useful evidence that the new future proposal
can reach event-producing continuations at reduced likelihood weight. The
point ratio is favorable, but its interval spans zero widely. It cannot be
reported as a validated 95.36% variance reduction. The primary and secondary
contrasts answer different finite-sample diagnostic questions and both remain
in the evidence. A toy exact enumeration test checks the secondary identity.

### Decision and remaining noise

The future weight tail is bounded and additional event regions are measured,
but stable full-bank J is **not established**. The 3-7 probes in many action
strata make the conservative simultaneous bounded intervals effectively
[0,sum C_h]; no full-bank precise fitting certificate was possible at this
budget. Do not promote these point moments into Q_S, retune on consumed
validation, claim population acceleration or restart PPO. No extra native
continuation, favorable restart or case removal was performed.

The next estimator design should fix natural/T component counts within each
action stratum rather than rely on an IID latent coin with a handful of draws.
For even n and equal component counts, the same balance-heuristic weights give
an unbiased estimate and remove the nonnegative between-component variance
term `C_h^2*(mu_P-mu_T)^2/(4*n)`. This is a derivable structural improvement,
not a promise that remaining within-component future noise is small. It needs
its own fixed protocol; it was not silently substituted into this completed
run. More future-label/model calibration may still be necessary.

Independent arithmetic reconstruction checked all 68 artifact hashes, every
scheduled root/component, sharing flags, future densities, weights, stratum
variances, batch estimates and common-J Q_S comparisons. One predetermined
root per batch/state was replayed (66 total); all native trajectories, actions,
endpoints and logs matched exactly. These replays are not extra independent
observations. The prior v2 and allocation summaries remain unchanged; frozen
assets/code verification and all 155 relevant regression tests passed. Compact evidence:
[highd_v47_continuation_moments_v1.json](../results/highd_v47_continuation_moments_v1.json).

```bash
python -m scenario_reconstruction.highd_v47_continuation_moments_audit --replay
python -m pytest tests/test_highd_v47_continuation_moments.py tests/test_highd_v47_continuation_moments_audit.py tests/test_highd_v47_state_moments_v2.py -q
```
