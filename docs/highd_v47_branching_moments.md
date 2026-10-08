# Branching continuation and uncertainty-aware allocation

This experiment follows `4965ca9`. It addresses sparse future-event labels in
coefficient-heavy action strata, not random fixed-count dispersion. Keep NDE,
native dynamics, teacher Q_A, the 33-state/25-source conditional target, event,
future P/T score and production Q_S/PPO unchanged. No consumed validation or
final-test episodes are opened.

## Shared prefix and exact correction

Each independent root draws its initial action under inherited c/C_h and its
initial natural partners conditionally. A fair P/T latent component produces
the first two native seconds. The first controlled action remains forced in
both laws; T begins at the next decision. At two seconds, surviving roots fork
into two natural and two tilted independent suffixes, each lasting at most
two more seconds. Clone every dynamic state, histories and physics metric;
prepare the next decision exactly once. Preserve terminal prefix endpoints.

For prefix x, the mixture posterior is
`eta=P_prefix(x)/(P_prefix(x)+T_prefix(x))`.
Full-path G remains `(P_future+T_future)/2`. Each suffix has
`Z=Y*P_full/G_full<=2`, evaluated using both complete stopped-path densities.
Use `B=eta*mean_P(Z)+(1-eta)*mean_T(Z)`, NOT equal suffix weights after an
informative prefix and NOT another multiplication/square of the prefix LR.
The initial partner term cancels. E[B] is the original natural-future event
probability. With early terminal prefixes, B=Y*P_prefix/G_prefix.

The single-suffix control chooses one measured P or T suffix with posterior
eta, using an independent predeclared uniform. It shares a physical path with
the branching estimator, so it is not independent evidence. Conditional
variance is reduced from
`eta*v_P+(1-eta)*v_T+eta*(1-eta)*(mu_P-mu_T)^2`
to `eta^2*v_P/2+(1-eta)^2*v_T/2`. Shared-prefix variance is **not** removed.
Total-variance conditioning is the basis of this comparison; see
[Owen, chapter 8, conditioning](https://artowen.su.domains/mc/Ch-var-basic.pdf).
The implemented finite branching is not exact analytic Rao-Blackwellization.
Independent roots, not leaves, determine sample size, sample variance and
confidence intervals. Improvement per root is not improvement per CPU cost.

## Measurement and allocation uncertainty

The [protocol](../configs/highd_v47_branching_moments_v1.json) retains all
1,549 initial-action strata. Instead of equal counts, exactly minimize
`sum_h C_h^2/n_h` subject to two roots per stratum and 256 roots per state.
Since B lies in [0,2], this protects an unknown root-variance upper envelope;
zero old events do not remove a layer. All cases and negatives remain.

Two independent batches use seeds 6200000/6300000 and schedule seed 620017,
with disjoint prefix/suffix seeds. All root actions, latent components, control
uniforms and seeds are saved before outcomes. Forks are unconditional on event
labels, except that already stopped paths cannot be continued. There is no
extension, substitution or favorable restart. This is 16,896 independent roots
and at most 67,584 suffix paths, not 67,584 independent states/traffic sources.
Record actual native decisions and wall cost as well as uncertainty.

Batch 0 builds simultaneous root-level J intervals. Fit an upper-envelope
state proposal, not a plug-in zero-event table: `q>=alpha/1.1` and minimize
`sum alpha^2*U/q`. The floor guarantees **true** `M2(q)<=1.1*M2(alpha)` for
any nonnegative J, even if fitted uncertainty is wrong. It does not bound
centered variance or realized finite-sample estimates. Strict promotion uses
the simultaneous upper bound on `sum alpha*(alpha/q-1)*J`; if it is not
negative, recommend unchanged alpha. This objective test does not require
every zero/near-zero state to pass relative precision, and does not waive the
need for evidence of actual improvement.

Save the batch-0 proposal/decision before opening batch 1. Batch 1 compares
that fixed candidate on common J measurements; it does not retune it. A safe
10% loss cap alone is not an efficiency success. No state-sampling rollout,
production replacement, PPO sequence export or source-transfer claim is made.

## Run

Use inherited Python 3.9.19/NumPy 1.23.1/SciPy 1.13.1, one BLAS/OMP thread.
Review/commit before freezing. All old artifacts and code dependencies are
checksum checked; outputs are exclusive-create and raw trees stay ignored.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_branching_moments freeze --output outputs/v47_branching_moments_v1_lock.json
python -m scenario_reconstruction.highd_v47_branching_moments run --lock outputs/v47_branching_moments_v1_lock.json --output outputs/v47_branching_moments_v1
python -m pytest tests/test_highd_v47_branching_moments.py tests/test_highd_v47_fixed_mixture_moments.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

## Completed measurement, 2026-10-06

Implementation/protocol `2fc5c13` was committed and frozen before new native
outcomes. Both full batches completed: 16,896 independent roots, 67,508 suffix
paths and 151,783 future native decisions, across every original state/source
and action stratum. Nineteen stopped prefixes were not forked. Local wall time
was 947.99 s, not a controlled efficiency benchmark.

| Quantity | Single suffix | Weighted branching |
| --- | ---: | ---: |
| Target teacher-action M2 measurement | 0.000823905 | 0.000927583 |
| Measurement MCSE | 0.000217292 | 0.000216390 |
| Relative empirical MCSE | 26.37% | 23.33% |
| First-batch target M2 | 0.000845894 | 0.001085325 |
| Second-batch target M2 | 0.000801915 | 0.000769841 |

The actual branching/single measurement-variance point ratio is 0.99172.
The same-root unbiased removed-conditional-variance estimate is 1.89342e-8,
giving inferred single variance 6.57590e-8 versus branching 4.68248e-8 (ratio
0.71207). These are two different finite-sample diagnostics, not conflicting
identities or supported variance confidence claims. Cost must also be considered.
Neither comparison establishes precise state moments or a superior testing Q_S.

### The fork is too late for important prefix uncertainty

| State / source | Branching J, batch 0 / batch 1 | Combined relative empirical MCSE |
| --- | --- | ---: |
| 6 / 200020 | 0.0106707 / 0.0234004 | 40.10% |
| 18 / 200051 | 0.0138845 / 0.0050830 | 36.47% |
| 19 / 200053 | 0.00288293 / 0.0000979904 | 94.46% |
| 32 / 230053 | 0.0145574 / 0.00545561 | 57.04% |

For example, state 19's root 6240880/action [32,18] has four identical positive
suffix values 0.955127. Its rare dangerous prefix is already shared by all
suffixes. Several largest state-6/18 roots likewise have all four suffixes
positive with almost identical weights. More suffixes from those same prefixes
cannot remove the randomness that selected the prefix. This is an observed
mechanism of the remaining noise, not proof that all uncertainty has that cause.

Batch 0's objective upper bound remained positive (0.449865), so the protected
allocator recommended unchanged alpha. Its exploratory upper-envelope candidate
gave a batch-1 common-measurement M2 ratio 1.10, not improvement; the objective
interval was [-0.833719, 0.449865]. No state meets simultaneous relative precision,
and no Q_S fit or PPO dataset is promoted. The 10% true uncentered-M2 safety
bound is implemented, but is not an empirical efficiency result. Importantly,
the uncertain point estimates no longer force an unsafe allocation decision.

Independent reconstruction verified all 69 artifact hashes, all densities,
posterior weights, schedules, root-level variances, allocation decisions and
66 predetermined complete native trees. Maximum full/factor log error was
1.37e-13. Replays are checks, not additional observations. Earlier evidence
and production remain unchanged. The completed summary SHA is
`300458b995b4a9ad7d2a049540ee9e06c6a25d1b5ca937b942692649a90b4156`.
All 197 relevant measurement/allocation/sampler regression tests and frozen
code/assets verification passed at the subsequent decision-layer handoff.

**Decision:** prefix/suffix correction and uncertainty protection are verified;
the large measurement problem is not solved. Continue prospective development
with weighted branching before future decisions, and conditional integration
of a terminal action instead of only more suffixes at two seconds. Such a
measurement must retain the same target and native endpoint, independent-root
statistics, exact correction and all negative states. Do not retune or append
outcomes to this completed experiment.

The separately versioned [decision-layer measurement](highd_v47_decision_branching.md)
implements that next design; it does not append to this completed run.

Compact evidence:
[highd_v47_branching_moments_v1.json](../results/highd_v47_branching_moments_v1.json).

```bash
python -m scenario_reconstruction.highd_v47_branching_moments_audit --replay
python -m pytest tests/test_highd_v47_branching_moments.py tests/test_highd_v47_branching_moments_audit.py -q
```
