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
