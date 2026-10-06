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
