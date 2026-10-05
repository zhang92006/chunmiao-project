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
