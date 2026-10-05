# Fixed-bank proposal-score comparison

This training-only experiment follows the collision-action coverage audit at
`3995790`. It adds an opt-in proposal family; it does not replace production Q.
Frozen NDE, native initialization, physical dynamics, CAV event definition,
discovery, pair selection, original eligibility gates and PPO remain unchanged.
Consumed validation sources 240000–240255 are not opened by this entry point.

## Fixed law and normalized H

The protocol is
[highd_v46_proposal_comparison_v1.json](../configs/highd_v46_proposal_comparison_v1.json).
Its two checksum-locked **training** calibrations supply all 31 eligible cases
from 24 source episodes, including zero-event cases. Abstention controls remain
in their original artifacts but outside this eligible-case target. No case is
selected using the new outcomes. The target is uniform source, then uniform
locked eligible states within source; it is not uniform over the 31 cases.

For natural joint action distribution P and unchanged clearance-risk proxy R:

`H_R = P*R / sum(P*R)`

`H_I = P*positive_interaction / sum(P*positive_interaction)`

`H_lambda = (1-lambda)*H_R + lambda*H_I`.

The three predeclared candidates have lambda 1, 0 and 0.5. Each component with
zero mass uses P **before** mixing. Thus the component balance is interpretable
and does not depend on the two unnormalized score scales. R is a deterministic
clearance score, not a calibrated collision probability. Absolute risk can
restore high-risk actions on the four-corner reference row/column that pure
interaction excludes. This does not guarantee improved native event variance
or balanced source contributions.

Every case reproduces its old proxy and complete strict-gate audit before any
new native sample. These original eligibility decisions are held fixed even
when a new H changes factor TV. Candidate Q uses the unchanged factor sampler:

`Qij = [0.1 P1 + 0.9 H1] [0.1 P2|1 + 0.9 H2|1]`.

Natural partners retain their original conditional distributions. Full P/Q
agrees with the product of the two factors, and Qij >= 0.01 Pij. A zero H1 row
uses P2 for H2, exactly as before. The score implementation/configuration and
dependencies define a new proposal identity; old datasets/checkpoints must not
silently adopt it. Natural/physical identities are unchanged.

## Fixed native comparison, not a training launch

Each case receives 64 draws per candidate, 5,952 continuations total. The
case-indexed schedule starts at 4100000; candidates share common random numbers
and different cases have disjoint blocks. Only the initial decision is tilted;
the remaining four-second continuation is natural. All endpoints are retained,
including BV-only competing collisions and zero-event cases. There is no
outcome-dependent extension or substitution.

Before new outcomes, a separate descriptive replay checks **all old training
arms' event actions** under each candidate H. Its Q probabilities and hypothetical
action ratios are coverage diagnostics, not new-Q event estimates. Old arms
are kept separate: their events cannot be pooled into independent replications.

The native report includes both-factor Q/P TV, actual event H-support categories,
source-weighted conditional IS mean/M2, fixed-state stratified variance/MCSE,
maximum single-event/source M2 shares and descriptive effective M2 sources.
Matched-state ratios compare the same conditional target. Whole-source paired
bootstrap intervals retain every state and continuation in the source; they
require at least 20 events across five distinct observed event sources in each
candidate. Replicates without sufficient distinct sources are censored; more
than 10% censored replicates leaves the interval undefined. Finite fixed-state
MCSE does not repair poor source generalization. Zero-event empirical M2 is not
evidence of a zero-risk target or an optimal proposal.

Engineering checks also require five effective-dual event sources, maximum
source M2 share <= 0.8 and effective M2 source count >= 3. No winner or PPO
launch is selected automatically, and no sequence manifest is exported. This
conditional experiment has no initial-state-density correction and cannot
support original-initialization acceleration claims.

## Run and validation boundary

Review and commit the protocol/code, then freeze before the fixed-budget run:

```bash
conda activate highd-ppo-clean39
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_proposal_comparison freeze --output outputs/v46_proposal_comparison_v1_lock.json
python -m scenario_reconstruction.highd_v46_proposal_comparison run --lock outputs/v46_proposal_comparison_v1_lock.json --output outputs/v46_proposal_comparison_v1_train
python -m pytest tests/test_highd_v46_proposal_comparison.py tests/test_highd_v46_collision_coverage.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_source_selector_validation.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

All outputs are exclusive-create. An interrupted run retains its opened protocol
and partial artifacts; any labelled restart uses the entire same locked budget
and cannot be pooled with the earlier attempt. Locks verify complete source
artifacts, implementation/dependencies, environment, case masses and target
identities. Raw states, matrices, trajectories and logs stay in ignored outputs;
only configuration, tests and compact evidence are committed.

Next, choose candidates on **training evidence only**, freeze a separately
versioned validation design and use previously unopened, episode-disjoint source
episodes. Keep selector/state-bank/pair budget matched across proposals. Native
confirmation must sample within the eligible source set before continuation
outcomes, rather than a large union that can miss an entire candidate. Review
event-action coverage, matched-state variance, independent event-source yield
and source/event concentration together. Improving one H support witness alone
does not establish lower population variance, source diversity or a PPO-ready
dataset. Consumed validation and reserved final-test data remain held out.
