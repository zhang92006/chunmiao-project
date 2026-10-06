# Fixed-component continuation measurement

This training-only experiment follows `8c94edd`. It removes one identifiable
source of state-moment measurement noise: random P/T component counts in small
initial-action strata. NDE, native dynamics, stopped event, teacher Q_A, future
P and T, the 33-state/25-source target, production Q_S and PPO are unchanged.
Consumed validation and final-test episodes are not opened.

## Estimator and structural variance reduction

Use the inherited stopped-path mixture `G=(P_future+T_future)/2` and balance
weight `W=P_future/G<=2`. For each inherited initial-action stratum h, the
initial actions still have law `c/C_h`, where `c=P_A^2/Q_A`. Each component
draws correlated partners naturally. T begins at the next decision.

The fixed estimator draws n/2 independent roots from P and n/2 from T, then
uses `Z=Y*W` **once** and `J_h=C_h*(mean_P Z+mean_T Z)/2`. Both path densities
are evaluated on every path; the initial conditional partner law cancels.
The roots are not identically distributed across components. Its variance is

`V_fixed,h=C_h^2*(Var_P Z+Var_T Z)/(2*n)`.

For the same n and G, random-component IID variance exceeds this by

`V_IID,h-V_fixed,h=C_h^2*(mu_P-mu_T)^2/(4*n)>=0`.

This removes only between-component noise, not within-component native future
uncertainty. It does not guarantee small relative error. Component means can
exceed one; only their equal mixture is the desired native probability.

Report fixed variance with within-component sample variances. Pooled s^2/n
would wrongly add component dispersion. A predeclared same-fixed-draw
comparison estimates `(mu_P-mu_T)^2` by `(mean_P-mean_T)^2-s_P^2/n_P-s_T^2/n_T`.
Finite negative estimates are retained. It infers the IID variance from the
same observations, separately from the actual IID-control variance estimate.
Bounded component-mean intervals give a simultaneous interval for the removed
nonnegative term; a favorable point ratio is not a supported precision claim.

## Fixed matched design

The [protocol](../configs/highd_v47_fixed_mixture_moments_v1.json) locks the
complete previous measurement and inherits every action stratum. Counts are
even and at least four, permitting two observations per component. State 8
has 65 strata and therefore uses 260 roots; all other states use 256. This is
four extra roots at one state per batch, explicitly declared before outcomes.
All three arms share these counts, so this change cannot explain their contrast.

Compare natural control, IID defensive mixture and fixed defensive mixture.
The latter two have identical marginal G and weights. Fixed assignments are
uniformly permuted within strata, independent of initial actions/outcomes;
IID assignments use independent fair coins. Common initial actions/seeds are
shared, and identical physical paths are simulated once. CRN arms are not
independent replications. First-decision endpoints share their common law.

Two independent batches use seeds 6000000/6100000 and schedule seed 600017.
There are 50,712 logical arm uses and at most 29,566 native paths in the frozen
plan; shared paths and initial-decision endpoints are not duplicated.
The whole plan is saved before native outcomes. Four workers affect only
execution order. Keep all states, strata, negatives and competing BV endpoints;
no extension, extra batch, replacement or repeat-until-favorable comparison.

Simultaneous intervals use bounded empirical Bernstein, separately within
each component; union over natural, IID, two fixed components, strata and
batches. Component intervals stay in [0,2], with known probability restriction
applied only after mixing. At 2-5 roots/component these bounds can remain
uninformative. The full-bank precision check is not removed or silently waived.
Old Q_S comparisons use a common measured J; no new allocation is fitted.
This is not a new source-disjoint validation or population efficiency test.

## Reproduction

Use Python 3.9.19 / NumPy 1.23.1 / SciPy 1.13.1 and one BLAS/OMP thread.
Review and commit the implementation/protocol before freezing new outcomes.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_fixed_mixture_moments freeze --output outputs/v47_fixed_mixture_moments_v1_lock.json
python -m scenario_reconstruction.highd_v47_fixed_mixture_moments run --lock outputs/v47_fixed_mixture_moments_v1_lock.json --output outputs/v47_fixed_mixture_moments_v1
python -m pytest tests/test_highd_v47_fixed_mixture_moments.py tests/test_highd_v47_continuation_moments.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

All outputs are exclusive-create. Interrupted attempts remain separate and
cannot be selectively restarted/pooled. Raw likelihoods/paths stay ignored;
commit only protocol, tests and concise evidence. No PPO/training manifest.
