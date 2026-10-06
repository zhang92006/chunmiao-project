# Decision-layer branching and conditional action integration

This prospective training measurement follows the completed
[two-second branching experiment](highd_v47_branching_moments.md). Important
prefix regimes remained shared by every suffix. The new design conditions
earlier instead of appending more suffixes to those completed paths. It retains
all 33 states/25 sources, teacher Q_A, natural future target, first-collision
endpoint, physical equations and production Q_S/PPO. Consumed validation and
test sources are not opened. New outcomes are not yet evidence of improvement.

## Two early branch layers

After the initial forced teacher action, surviving roots branch before BOTH
the first and second future decisions: two P and two T children at each node.
The posterior eta=P_prefix/(P_prefix+T_prefix), evaluated before the action,
weights the two component means. Clone complete dynamics and histories before
preparing a child exactly once. Already stopped CAV/BV endpoints are retained.

The measurement law mixes whole P/T prefixes only through the third second;
the last decision is natural. The target remains natural over all four seconds.
At each terminal prefix the ratio W=P_prefix/G_prefix<=2 is used **once**.
Intermediate branch aggregation does not multiply or square W. A control follows
one posterior-chosen child using saved uniforms. It shares paths with the tree.

## Integrate a terminal action, not another Bernoulli draw

At a surviving three-second prefix choose the nearest BV by distance/ID before
any action. Integrate every action in its natural marginal support. For each
action draw other actors under their exact natural conditional law, using the
same partner uniforms in all cells. Execute the entire traffic assignment with
the native 15-substep IDM/bounded-acceleration/lateral/competing-endpoint rules.
The batched kernel is an execution adapter, not the clearance-risk proxy.

Numerically near-boundary collision assignments are deterministically replayed
by the original scalar native method; their endpoints replace the batched
ones. This verification fallback does not redraw actions or add observations.

For partner randomness U and supported action a, terminal value is
`F(U)=sum_a P(a)*Y(a, partners(U|a))`; its prefix contribution is W*F.
Conditional expectations recover the original joint natural law, including
correlated partners. The natural-control terminal value draws one a with an
independent uniform, then uses the identical recorded partner assignment.
Thus integration removes its conditional action variance W^2*F*(1-F) exactly;
the partner randomness remains. Intermediate tree averaging removes additional
future-decision variance per root. These statements do not prove lower cost-
adjusted variance than the previous measurement law or testing Q.

The control law now differs from the older all-four-second defensive mixture;
do not attribute a cross-experiment point contrast solely to branching.
All terminal action cells are dependent conditional measurements, not new
collision trajectories from independent traffic episodes.

## Frozen budget and uncertainty decision

The [protocol](../configs/highd_v47_decision_branching_v1.json) uses two complete
independent batches, 256 parent roots/state/batch and inherited coefficient-
minimax counts with at least two roots in every stratum. Initial actions are
new independent c/C_h draws, not reused selected actions. Continuation blocks
start at 8000000/9000000 with schedule seed 800017. At most 16,896 independent
roots, 337,920 future prefix decisions and 8,921,088 terminal native action-cell
decisions are planned. Stopped prefixes reduce actual counts without backfill.
Four workers only affect execution order. Save all schedules before outcomes.

Variance uses parent-root observations, never a pooled leaf sample size.
The within-tree variance-removal recursion is an unbiased descriptive estimate;
finite negative estimates remain. All full-bank precision checks stay visible.
Batch 0 builds the protected allocation and saves it before batch 1. The floor
q>=alpha/1.1 bounds true uncentered M2 loss for arbitrary J, but improvement
still requires the uncertainty-aware objective certificate and an independent
batch check. Otherwise recommend alpha. No new allocation, model, PPO dataset
or acceleration claim is promoted merely because branching is implemented.

## Reproduction

Use Python 3.9.19/NumPy 1.23.1/SciPy 1.13.1 and one BLAS/OMP thread. Review and
commit implementation/protocol, then freeze once. The terminal adapter is locked
along with dependencies, complete parent evidence and target identities. All
raw trees remain ignored; exclusive output creation prevents overwrite/restart.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_decision_branching freeze --output outputs/v47_decision_branching_v1_lock.json
python -m scenario_reconstruction.highd_v47_decision_branching run --lock outputs/v47_decision_branching_v1_lock.json --output outputs/v47_decision_branching_v1
python -m pytest tests/test_highd_v47_decision_branching.py tests/test_highd_v47_conditional_native.py tests/test_highd_v47_branching_moments.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

The execution adapter has matched 960 randomized complete scalar native paths,
including CAV, BV-only and negative endpoints, speed bounds and existing/new
lateral maneuvers. This is implementation verification, not fresh-source data.
Independent complete-root replay and density/allocation reconstruction remain
required before accepting the new experiment's results.
The full 197-test measurement/allocation/sampler suite and frozen-code/assets
verification passed before freezing this prospective design.

```bash
python -m scenario_reconstruction.highd_v47_decision_branching_audit --replay
```
