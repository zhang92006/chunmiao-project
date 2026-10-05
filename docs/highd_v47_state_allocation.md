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
