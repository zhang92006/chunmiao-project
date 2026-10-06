# State-level probability allocation on the fixed training bank

This experiment follows `7b050d4`. It changes Q_S only; the teacher action Q_A,
NDE, native physical dynamics and four-second first-CAV-collision endpoint are
unchanged. The old 25-source/33-state conditional target is retained exactly.
P_S below means that declared bank target, **not** the natural reset density.

## Objective and full likelihood

For a locked state/pair k, let alpha_k be its target mass (uniform source,
then uniform locked cases within source). Hold action Q_A fixed and define
`J_k = E_QA[Y*(P_A/Q_A)^2 | k]`, `mu_k = E_QA[Y*P_A/Q_A | k]`.
An IID state proposal q_k followed by Q_A has full weight

`w = (alpha_k/q_k) * (P_A/Q_A)`.

Its event second moment is `J(q) = sum_k alpha_k^2 J_k/q_k` and its estimator
variance at N draws is `(J(q) - (sum alpha_k mu_k)^2)/N`. Minimizing J(q) gives
`q_k proportional to alpha_k*sqrt(J_k)` without floors. The implementation
reuses the tested water-filling solver for `q_k >= 0.2 alpha_k`. States with
zero observed pilot events retain support. These are zeros in a finite pilot,
not a claim that their risk is zero.

The tabular model uses **only** the checksum-locked completed teacher pilot:
`J_hat_k = mean(Y*(P_A/Q_A)^2)` from 64 draws/state, regularized by adding
`0.01 * sum(alpha*J_hat)` to every entry. Both constants are fixed before new
outcomes, with no parameter search. Pilot observations fit Q_S, but are not
pooled into evaluation. Q_S is fixed throughout the new run. Its source marginal
is the sum of q over that source's cases; q within a source is its normalized
case mass. Repeated draws from one source are not independent traffic sources.

For this pilot the target assigns source 200051 mass 4%; the fitted proposal
assigns it about 66.05%. Its contribution to the pilot weighted mean is about
98.07%, so eliminating its *target* influence would change the problem. The
predicted IID M2 ratio is 0.0626; this is a fitted prediction, not native evidence.
The state LR is bounded by 5 and the combined state/two-action-factor LR by 500.
No clipping or renormalization of observed IS weights is used.

## Fixed-budget comparison

The [protocol](../configs/highd_v47_state_allocation_v1.json) predeclares three
arms with 2,112 new continuations each, 6,336 total:

1. `target_iid`: sample state k from alpha, then the frozen teacher Q_A.
2. `allocated_iid`: sample from fitted q, then **the same** Q_A.
3. `equal_state_stratified`: 64 draws in every state, matching the previous
   collection design; estimate with the original alpha-weighted stratum means.

IID arms share state uniforms from a separate RNG (470017). All three arms use
continuation seeds 4800000-4802111 indexed by draw, providing paired random
numbers when states coincide. Arms remain separate estimators, never pooled
independent replications. All state schedules are recorded before outcomes.
The stratified estimator's variance is `sum alpha_k^2 s_k^2/64`, **not** pooled
IID variance divided by N. Its logged state factor alpha/(1/33) is a design
weight: the fixed schedule is not misrepresented as an IID state draw.

Every trajectory logs state target/proposal masses, state log ratio, action log
ratio and their sum. Original action likelihood audits and natural partner
conditional sampling are unchanged. Report same-target means, conditional MCSE,
estimator variance, events, source/state draws and second-moment concentration.
Variance ratios are point estimates unless separately supported by uncertainty
analysis. A falling maximum-source share is not the optimization objective;
even optimal allocation may leave genuine target-risk heterogeneity. The old
diversity/concentration checks are not silently declared passed or removed.

This is a tabular fixed-bank allocation test with independent continuation
draws, **not** source-disjoint validation of a transferable Q_S model. It does
not estimate population testing acceleration or export a PPO-ready manifest.
A useful variance result here would justify a later state-feature model and
fresh-source test; it cannot discover sources absent from the bank.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_state_allocation freeze --output outputs/v47_state_allocation_v1_lock.json
python -m scenario_reconstruction.highd_v47_state_allocation run --lock outputs/v47_state_allocation_v1_lock.json --output outputs/v47_state_allocation_v1
python -m pytest tests/test_highd_v47_state_allocation.py tests/test_highd_v47_action_teacher.py tests/test_highd_v47_probability_allocation.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Use inherited Python 3.9.19, NumPy 1.23.1 and SciPy 1.13.1. Source artifacts,
implementation, allocation, complete schedules and target identities are
locked before new samples. Outputs are exclusive-create; interruptions retain
their artifacts and cannot be selectively restarted or pooled. No validation
cohort is opened and no production selector, action Q or PPO code is modified.

## First complete run: zero-pilot states were underestimated

The `2f660eb` implementation completed all 6,336 draws in 305.64 s. No sample
was dropped or substituted. The fitted 0.0626 M2-ratio prediction did not hold:

| Arm | Events | Conditional mean | Estimator variance |
| --- | ---: | ---: | ---: |
| Target IID | 64 | 5.02730e-5 | 6.71362e-10 |
| Allocated IID | 516 | 6.60644e-4 | 1.70921e-7 |
| Equal-state stratified | 66 | 1.90700e-4 | 1.38647e-8 |

Allocated variance is 254.59 times target-IID and 12.33 times stratified, as
point estimates in this run. Source 200053, state 19, had zero pilot events.
Its Q_S was only 0.013238 versus target mass 0.04. Two new native events at
seeds 4800284 and 4800566 had action LRs 0.180826 and 0.225464; multiplying
by state LR 3.02161 gives full weights 0.546387 and 0.681264. This source now
supplies 99.96% of allocated M2. The nominal state floor preserved unbiased
support, but did not prevent a materially mistaken allocation. Baseline IID
missed this source's events in its finite run; the empirical ratio is not
evidence violating the theoretical M2 bound from the floor. Stratified
sampling also observed an event there. The old 99.54% concentration at 200051
was not enough evidence to treat all other sources as negligible.

The first run remains immutable at summary SHA
`a816d1183722f8df0d39be0522d259d0172de1e39026b624eb13673e20414ffd`.
It is failed development evidence, not a promoted proposal or hidden restart.

## Explicit update and second fixed comparison

The separate [update protocol](../configs/highd_v47_state_allocation_update_v1.json)
fits tabular J on **all** original teacher and all completed first-run samples,
including every negative. Identical state/seed samples shared between arms
are deduplicated and checked for matching labels; repeats are not extra data.
Use action P_A/Q_A to estimate conditional J_k, not the total state/action
weight (which targets the global distribution). Fitting from this accumulated
adaptive development design is not a new unbiased primary risk estimate.

The state floor becomes 0.5 alpha. This explicitly chosen conservative limit
caps the state LR at 2 and true IID M2 at twice target-IID M2 for **any** native
conditional risk, holding Q_A fixed. It is not a centered variance-ratio bound
or a promise of improvement. All constants, updated masses and schedules are
frozen before this update's new outcomes. No model is changed during the run.

The same three arms receive 2,112 new draws each, using continuation seeds
4900000-4902111 and state RNG seed 490017. The previous experiment is not
pooled into these evaluation estimates. This remains development on the same
fixed training bank, not a fresh source-generalization evaluation. There is no
repeat-until-positive rule or automatic subsequent run.

```bash
python -m scenario_reconstruction.highd_v47_state_allocation_update freeze --output outputs/v47_state_allocation_update_v1_lock.json
python -m scenario_reconstruction.highd_v47_state_allocation_update run --lock outputs/v47_state_allocation_update_v1_lock.json --output outputs/v47_state_allocation_update_v1
```

## Completed update: mixed evidence, not a validated winner

Implementation `2dd0fa7` froze the update before opening any 4900000-series
continuation. The fitting input contained 8,448 rows, of which 313 repeated
the same physical state/seed across CRN arms; only 8,135 unique rows entered
the tabular fit. All negative observations and states were retained. The
resulting source probabilities are 0.395891 for 200053, 0.144109 for 200051,
and 0.02 for each of the other 23 sources. Target mass remains 0.04/source.
The largest state LR is exactly two. These are fitted bank lookup values,
not predictions for new episodes.

The complete second comparison retained all 6,336 draws in 302.74 s:

| Arm (2,112 draws each) | CAV events | Conditional mean | Conditional MCSE | Estimator variance | Largest source M2 share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target IID | 63 | 7.90470e-5 | 2.47886e-5 | 6.14476e-10 | 65.30% |
| Updated allocated IID | 116 | 1.15912e-4 | 3.68605e-5 | 1.35869e-9 | 62.78% |
| Equal-state stratified | 60 | 1.13527e-4 | 4.94052e-5 | 2.44087e-9 | 90.74% |

Updated allocation's variance point estimate is 0.55664 times the old fixed-
stratum design (44.34% lower), but **2.21114 times target-IID**, which had no
observed event from 200053 in this schedule. Allocated IID observed that source;
the stratified arm's seven event sources instead include 200034 and not 200053.
Thus source-tail realizations still differ materially even with matched target
and budget. The point ordering does not establish a universally better state
proposal, and no variance-ratio confidence or acceleration claim is made.

In particular, comparing the old pilot's 99.54% directly to this allocated
arm's 62.78% would overstate the intervention: target-IID also had 65.30% in
this batch. State concentration alone is not a stable objective or a reliable
estimate of which sources dominate true natural conditional risk. All reported
means retain exact target and action correction; changing the target to make
sources look balanced was never performed. The stratified pooled M2 uses its
fixed-design weights, while its variance uses within-stratum variances, as
specified above; these two quantities must not be interchanged.

Independent reconstruction verified all five artifacts in each experiment,
all 12,672 scheduled native draws, initial action/partner likelihood replay,
endpoint labels, full state/action log weights, means and estimator variances.
Maximum full log-weight reconstruction error was 2.84e-14 in each run. The
teacher pilot, failed first run and consumed validation summaries are unchanged.
All 51 allocation/teacher regression tests and frozen-code/asset verification
passed. The completed update summary SHA is
`0a50edb8530818ba30d45626b56aed74caecd95d953f4fa22a5694df8c0cb0d3`.

**Decision:** Q_S and its full likelihood are now implemented and exercised;
the earlier omission of state allocation is addressed. Stable variance
superiority and source generalization are **not** established. Retain both
experiments and all baselines, keep production/PPO unchanged, and do not repeat
this comparison until a favorable ordering appears. A later design should
predeclare a common-draw, correctly reweighted moment comparison to reduce
tail-realization mismatch, and assess state-moment uncertainty on independent
sources before proposing a transferable state model. This is a next research
design, not an extra run or an analysis substituted for the results above.

Compact evidence:
[highd_v47_state_allocation_v1.json](../results/highd_v47_state_allocation_v1.json).

The next [state-moment measurement experiment](highd_v47_state_moments.md)
implements action-stratified probes, zero-event uncertainty bounds and common-
measurement Q_S comparisons. It improves some known-tail measurements but
discovers another unmeasured action region; no precise state model or new
production allocation is promoted. The original allocation results above
remain unchanged.

The subsequent [v2 measurement](highd_v47_state_moments_v2.md) explicitly covers
coefficient-heavy actions and compares protected uncertainty budgets on two
independent batches. It retains the complete bank, but still finds rare native
future outcomes in previously negative singleton actions; no state passes the
simultaneous precision gate. Neither completed measurement promotes a Q_S fit.

The [branching measurement](highd_v47_branching_moments.md) adds exact posterior
prefix/suffix correction and an uncertainty-aware objective guard. Its completed
two-second fork retains substantial shared-prefix uncertainty and recommends
unchanged alpha. The new [decision-layer design](highd_v47_decision_branching.md)
therefore branches before the early future decisions and integrates a terminal
natural action. Its completed matched measurement reduced relative empirical
MCSE from 22.68% to 11.50%, but retained uninformative simultaneous J bounds.
The next [objective-level certificate](highd_v47_allocation_certificate.md)
measures allocation benefit and loss directly with separately frozen training
roots. Neither design reinterprets the unsuccessful earlier allocations or
fits on consumed validation. A per-state precision certificate is not claimed.

The [continuation measurement experiment](highd_v47_continuation_moments.md)
then changes only the later-action measurement law with a complete path mixture
and future likelihood bounded by two. It improves event-region access but still
does not provide stable full-bank moments; primary and separate same-draw
variance diagnostics are both retained, without selecting a production winner.

The [fixed-component continuation design](highd_v47_fixed_mixture_moments.md)
then removes random P/T count dispersion within each initial-action stratum,
with correctly separated component variances and unchanged target correction.
Its complete matched run retains every state and source; sparse outcomes
inside coefficient-heavy layers still dominate empirical measurement noise.
It does not promote a precise state model or new production allocation.
