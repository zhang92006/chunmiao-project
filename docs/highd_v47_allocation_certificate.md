# Objective-level uncertainty certificate for fixed-bank state allocation

This prospective training design follows the complete
[early-decision branching measurement](highd_v47_decision_branching.md). It
addresses a different remaining bottleneck: simultaneous per-stratum bounds
with only a few roots can be essentially uninformative even when conditional
integration reduces empirical noise. Requiring every near-zero J to have a
precise relative estimate cannot certify a useful allocation at this budget.
Those completed measurements and their unsuccessful decisions remain intact.

NDE, native dynamics, event definition, teacher Q_A, production and PPO stay
unchanged. The target alpha remains uniform over 25 locked sources, then their
33 states. It is not the natural reset law. No validation/test source opens.

## Fit a protected candidate; measure its decision directly

Fit the water-filling candidate on both completed TRAINING branching batches,
with q>=alpha/1.1. Finite pilot zeros do not justify eliminating support. This
universal floor gives true M2(q)<=1.1*M2(alpha) for any nonnegative J, but is
not a centered variance-ratio guarantee. The point J model is not promoted.

For fixed Q_A and d_k=alpha_k*(alpha_k/q_k-1),

`M2(q)-M2(alpha) = sum(d_k*J_k) = L-B`,

where L sums positive d and B sums negative d in absolute value. Thus one can
certify the allocation's objective without claiming all J_k are precise.
The same conditional target mean cancels in the IID variance difference.

At every inherited initial-action stratum h, C_h=sum(P_A^2/Q_A), and initial
actions have law (P_A^2/Q_A)/C_h. Independent root measurements z have mean
equal to the native natural-future conditional event probability:

- Benefit: early P/T branching and terminal conditional integration, z in [0,2].
- Loss: plain natural future, z=Y in {0,1}, without proxy/T calculation.
- Exact-zero d: retain at least two natural roots per stratum as controls.

Each independent root contributes X=abs(d_k)*C_h*z/n_h to its side. Different
roots need not be identically distributed. Descendants/cells share one root;
they never multiply the statistical sample size.

Loss counts are `max(2,ceil(d_k*C_h/1e-5))`, set before outcomes. Thus an unseen
event's single objective contribution is at most 1e-5. Benefit keeps inherited
coefficient-minimax counts (256/state). This directs budget toward uncertainty
in possible allocation harm rather than every state's relative error. A
100,000-natural-root/batch cap is checked at freeze, not extended after events.
The unstarted v1 planning cap was 60,000; the completed training candidate
requires 99,155 at the same 1e-5 bound. The explicitly versioned v2 declares
the larger cap before any freeze or new outcome. No v1 evaluation was opened,
extended or pooled. Actual benefit budget is 512 branching roots/batch.

## Non-IID fixed-bet bounds and proof

For independent nonnegative X_i, m=sum E X_i and fixed lambda>0,

`E[exp(-lambda*m)*prod(1+lambda*X_i)] <= 1`.

For deterministic 0<=X_i<=b_i and lambda*max(b_i)<1,

`E[exp(+lambda*m)*prod(1-lambda*X_i)] <= 1`.

Proof: independence factors expectations; apply 1+u<=exp(u) or
1-u<=exp(-u) to their known nonnegative means. Markov inversion yields

`B_lower=max(0,(sum log(1+lambda_B*X_i)-log(1/delta))/lambda_B)`,

`L_upper=min(sum b_i,(log(1/delta)-sum log(1-lambda_L*X_i))/lambda_L)`.

All-zero loss observations still have positive uncertainty unless the loss
coefficient is identically zero. Bounds are deterministic design quantities,
not observed maxima. The derivation here specializes the nonnegative betting
and Markov principle; see [Shafer's primary paper](https://www.glennshafer.com/assets/downloads/articles/article104_jrss_shafer-testingbybetting-with-discussion-response.pdf).
Finite heterogeneous-law enumeration tests check the e-value and coverage.

Choose ONE benefit bet from a fixed grid using training data only, maximizing
the worse training-batch lower expression. Those fitted pilot expressions are
not valid evaluation bounds. Lock this bet, q, support bounds, both complete
schedules and every count before new outcomes. Lambda_L=0.8/max(b_i).

Two independent new batches receive error delta=0.0125 per side/batch; a union
bound gives 95% family coverage for the four bounds. Require L_upper-B_lower<0
in BOTH batches. Otherwise retain alpha. No optional stopping, outcome-driven
bet search, extra batch, favourable rerun or old/new pooling. The confidence
statement concerns this prospective protocol conditional on its frozen pilot,
not all adaptive research experiments or source-population generalization.

## Exact Q_S/Q_A verification and reproduction

The [protocol](../configs/highd_v47_allocation_certificate_v2.json) additionally
predeclares 2,112 real draws each from alpha and q with unchanged teacher Q_A
and natural continuation. Both run regardless of the certificate outcome.
Full likelihood is `(alpha_k/q_k)*(P_A/Q_A)`. Shared state/seed physical paths
are executed once and not counted as independent A/B evidence. The finite
IID variance ordering is descriptive; it cannot substitute for the primary
common-target objective certificate. No training/PPO export is made.

The protocol now pins the complete parent summary; its independent audit must
also complete before freezing or opening new outcomes. Commit the reviewed
code/configuration, then freeze the complete schedule and bets before new data.
Use Python 3.9.19, NumPy 1.23.1 and SciPy 1.13.1; one BLAS/OMP thread.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_allocation_certificate freeze --output outputs/v47_allocation_certificate_v2_lock.json
python -m scenario_reconstruction.highd_v47_allocation_certificate run --lock outputs/v47_allocation_certificate_v2_lock.json --output outputs/v47_allocation_certificate_v2
python -m scenario_reconstruction.highd_v47_allocation_certificate_audit --replay
python -m pytest tests/test_highd_v47_objective_bounds.py tests/test_highd_v47_allocation_certificate.py tests/test_highd_v47_allocation_certificate_audit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Exclusive-create artifacts, source/dependency/target locks and all negative
records remain required. Deterministic scheduled root replays are verification,
not extra observations. A successful fixed-bank certificate would solve the
bank allocation decision despite imperfect point J estimates; it would not
produce a transferable state model, discover unseen sources, establish
original-initialization acceleration or authorize PPO automatically.

The new plain-natural collector was checked against 66 predetermined OLD
training paths (one per state/batch). Every initial/future action, stopped
endpoint and natural action log matched exactly; the maximum natural-log
difference was zero. These are implementation replays, not new fitting or
evaluation observations. All 233 relevant measurement/allocation/sampler tests
and frozen-code/assets verification passed before the prospective freeze.
