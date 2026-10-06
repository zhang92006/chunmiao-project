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

## Completed evidence, 2026-10-06

Implementation/protocol `5227aec` was committed and frozen before any new
6000000/6100000-series outcome. All 33 states/25 training sources and 1,549
strata completed both batches: 16,904 roots per arm, 50,712 logical arm uses
and 29,566 physical paths. All endpoints and negative states were retained.
Local wall time was 550.36 s, not a controlled efficiency benchmark.

| Metric | Natural | Random-component IID | Fixed components |
| --- | ---: | ---: | ---: |
| Observed CAV events | 98 | 129 | 123 |
| Observed event sources | 6 | 9 | 9 |
| Target teacher-action M2 estimate | 0.0000527144 | 0.000960429 | 0.000914590 |
| Empirical measurement MCSE | 0.0000318871 | 0.000394591 | 0.000350906 |
| Relative empirical MCSE | 60.49% | 41.08% | 38.37% |
| First-batch target M2 | 0.0000640670 | 0.000604537 | 0.000188840 |
| Second-batch target M2 | 0.0000413617 | 0.00131632 | 0.00164034 |
| Simultaneously precise states | 0 / 33 | 0 / 33 | 0 / 33 |

Arms share paths and have different event-source sets; do not sum their counts
as independent replications. These are old training traffic episodes, not
fresh-source validation. Raw weighted-measurement event rates are not natural
collision probabilities. All three estimates target the same J but still
differ substantially. Fixed variance is 0.79084 times actual IID variance
(20.92% lower as a point estimate), and 121.10 times the natural control's
empirical variance. Neither ratio is a validated true-variance comparison:
the natural control misses coefficient-heavy event observations, and rare
events still differ between the matched defensive estimators.

### Count fixing is correct; its effect size is unresolved

The outcome-blind same-fixed-draw comparison gives:

| Quantity | Point estimate |
| --- | ---: |
| Fixed measurement variance | 1.23135e-7 |
| Removed between-component variance, unbiased estimate | 1.32714e-10 |
| Inferred IID variance from the same fixed draws | 1.23268e-7 |
| Fixed / inferred IID ratio | 0.998923 |
| Simultaneous interval for removed nonnegative term | [0, 0.0303663] |

The removed fraction point is only 0.1077%, **not** a supported upper bound
on the real benefit of count fixing. Sparse component observations leave an
extremely wide interval. In particular, with two T observations containing
one event and one zero and two zero P observations, the unbiased squared-mean
contrast estimate cancels to zero despite a nonzero raw mean difference.
There is too little information to identify the true component means well.
Do not attribute the actual 20.92% point contrast entirely to fixed counts,
or conclude from the tiny same-draw point that count noise is proven absent.

The exact finite-path enumeration test confirms unbiasedness, correct
within-component variance and the nonnegative true removed term. The design
therefore eliminates this avoidable noise structurally, but does not establish
precise J or overall variance superiority over natural continuation.

### The remaining high-contribution observations

| State / source | Fixed J | Two independent batch estimates | Relative empirical MCSE |
| --- | ---: | --- | ---: |
| 6 / 200020 | 0.00935159 | 0 / 0.0187032 | 100% |
| 18 / 200051 | 0.0157656 | 0.00167033 / 0.0298608 | 46.59% |
| 19 / 200053 | 0.0000923443 | 0.000102030 / 0.0000826591 | 12.89% |
| 32 / 230053 | 0.00523701 | 0.00805730 / 0.00241672 | 57.45% |

States 6 and 18 account for about 98.5% of the fixed estimator's empirical
measurement variance. Source 200020's only fixed event occurred at seed
6101664, initial action [18,21], in the T branch. Its future LR is **0.143258**,
not a large or invalid weight; the initial stratum coefficient is 0.522223.
Its layer has only two T roots per batch, and the other batch is negative.
Source 200051's largest target-J contribution is another T event, seed
6104744/action [20,20], future LR 0.0778874 and coefficient 0.605696.
Even weights below one leave substantial uncertainty when native outcomes in
coefficient-heavy action layers are represented by very few observations.
No tail observation or source was removed to improve the estimates.

The main unresolved measurement is now **within-component conditional future
event probability in coefficient-heavy action layers**, not an uncorrected
state LR or a random-count implementation defect. This describes the observed
variance bottleneck; the sparse run does not prove the true unseen event
probabilities or rule out additional regions. State 19 is locally repeatable,
but that is not a full-bank or fresh-source precision certificate.

At the declared 2-5 roots per component, simultaneous distribution-free
intervals are effectively [0,sum C_h]. This is partly a budget/bound limitation,
not proof that every empirical J is equally inaccurate. Moreover, exact
relative precision at a truly zero J cannot be certified by a finite negative
sample. Do not turn the full-bank relative guard into repeat-until-favorable
simulation, nor silently waive it and fit Q_S to these sparse point values.
All existing Q_S comparisons remain uncertified.

### Verification and decision

Independent reconstruction checked all 68 artifact hashes, exact component
counts and schedules, density/factor logs, shared paths, weighted events,
component variances, batch/target moments and same-draw contrasts. One
predetermined root per state/batch was replayed (66 total), matching every
native path/action/endpoint/log exactly; these replays are not extra data.
Maximum full/factor log error is 1.22e-13 and fixed future weights range from
0.00973139 to 1.999955. Earlier continuation/v2/allocation summaries stayed
unchanged. All 180 relevant tests and frozen code/assets verification passed.

**Decision:** retain the correct fixed-component estimator, but do not fit a
new Q_S, export PPO data or claim that state-moment noise is solved. No extra
batch or favorable restart is launched. The next design should address
conditional future uncertainty directly (for example, explicitly weighted
branching of shared native prefixes), rather than another count-only rerun.
Such a design must retain exact likelihood and account for dependence between
branches; leaves sharing a prefix are not independent roots. Allocation
decisions also need uncertainty-aware objective checks, rather than requiring
every near-zero state to pass the same relative-error rule. These are separate
prospective design tasks, not changed criteria for this completed experiment.

The [branching and protected-allocation implementation](highd_v47_branching_moments.md)
now targets within-prefix future uncertainty directly, with parent-root
statistics and an explicit true-M2 loss cap, rather than another count-only run.

Compact evidence:
[highd_v47_fixed_mixture_moments_v1.json](../results/highd_v47_fixed_mixture_moments_v1.json).

```bash
python -m scenario_reconstruction.highd_v47_fixed_mixture_moments_audit --replay
python -m pytest tests/test_highd_v47_fixed_mixture_moments.py tests/test_highd_v47_fixed_mixture_moments_audit.py tests/test_highd_v47_continuation_moments.py tests/test_highd_v47_continuation_moments_audit.py -q
```
