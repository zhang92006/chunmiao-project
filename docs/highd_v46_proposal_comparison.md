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

## Predeclared fresh matched-source validation

The separate protocol is
[highd_v46_proposal_validation_v1.json](../configs/highd_v46_proposal_validation_v1.json).
Its entry point first requires the checksum-locked complete training comparison.
All three candidates are retained for validation; this avoids declaring a
training winner from a few event-bearing sources. Sources 250000–250255 are
reserved and have not been opened by implementation tests or training.

The mixed5 bank and rank-1/rank-2/audit budget are unchanged. The old strict B
selector selects at most one state/pair per source by maximum supported proxy
clearance risk, with ascending time/actor-ID ties. Every H candidate shares
that exact selection. New H does not silently relax eligibility or replace the
state when its factor TV is weak. The target is uniform **selected** source and
its fixed state; abstentions remain in all-source coverage/event-yield
denominators, without being labelled zero risk.

Each selected case receives 64 continuations per candidate, with source-indexed
common random numbers starting at 4200000. The upper bound is 49,152 Q draws;
the actual budget shrinks with selector abstention. Before Q outcomes, sample
up to 16 sources uniformly **within the common eligible source set** for native
four-corner confirmation, 16 draws/corner and at most 1,024 total, with a separate
source-indexed schedule starting at 4300000. Each physical confirmation case
is measured once, not once per H; it checks the pair/proxy mechanism rather
than comparing proposal sampling distributions. No missing or negative case
is substituted. There is no adaptive budget, PPO or training export.

Coverage, independent event sources, actual second-factor Q/P TV, per-event and
per-source moment tails, paired same-target M2/stratified-variance ratios,
source-cluster intervals and proxy/native interaction calibration are reported
separately. Original-initialization efficiency still requires a later independent
population experiment. The prior consumed validation cohort is excluded.

After freezing the reviewed code/configuration and the complete training hash:

```bash
python -m scenario_reconstruction.highd_v46_proposal_validation freeze --output outputs/v46_proposal_validation_v1_lock.json
python -m scenario_reconstruction.highd_v46_proposal_validation run --lock outputs/v46_proposal_validation_v1_lock.json --output outputs/v46_proposal_validation_v1
python -m pytest tests/test_highd_v46_proposal_validation.py tests/test_highd_v46_proposal_comparison.py tests/test_highd_v46_collision_coverage.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_source_selector_validation.py -q
```

Freeze validates training only and does not open the reserved fresh sources.
If regenerating the training comparison, timing fields change its summary
checksum. Verify the same protocol, implementation, source locks, environment
and all retained artifacts, then explicitly version the validation protocol
for that regenerated evidence; do not bypass or silently repin the checksum.

## Completed training evidence, 2026-10-05

The committed implementation/protocol at `aa95f86` completed all 31 cases and
all 5,952 probabilistic continuations. Each candidate has 1,984 draws and the
same 24-source conditional target. No extra continuation or selected positive
case was appended. The local wall time was 301.42 s, not a controlled efficiency
benchmark. An independent read-only reconstruction verified all 127 artifacts,
all seed schedules, normalized components and sampled P/H/Q factors, full
likelihoods and the source-weighted mean/M2/stratified variance. Maximum full
versus factor log-weight discrepancy was 1.42e-14.

| Metric | Interaction | Absolute risk | 50/50 mixture |
| --- | ---: | ---: | ---: |
| Native CAV events | 30 | 1 | 18 |
| Independent event sources | 2 | 1 | 2 |
| Conditional IS event mean | 4.2293e-5 | 3.3178e-5 | 6.4515e-5 |
| Conditional event M2 | 2.2297e-7 | 1.6908e-6 | 8.7519e-7 |
| M2 ratio to interaction, point only | 1 | 7.5831 | 3.9251 |
| Stratified variance ratio, point only | 1 | 11.0585 | 4.0390 |
| Source-weighted second-factor Q/P TV | 0.5635 | 0.0244 | 0.2887 |
| Largest source M2 share | 62.5098% | 100% | 51.9615% |
| Effective M2 sources, descriptive | 1.8822 | 1 | 1.9969 |

Every current-draw event has both implemented H factors positive, even under
the interaction baseline. The new candidates' observed events also lie in
cells with positive **original** joint H_I. Thus these draws do not directly
exercise the old H-zero collision witness under a new Q. With only 1–2 event
sources, all 2,000 whole-source bootstrap replicates fail the five-source guard;
generalization intervals are undefined. No candidate passes the engineering
readiness checks and no PPO/data export is launched.

The deterministic old-training action replay *does* reproduce the structural
coverage repair. Across 31 original dual events, interaction has two H2-zero
events; both absolute risk and mixture assign positive H1/H2 to all 31. Across
12 original first-factor events, six H2-zero actions likewise become covered.
The two original second-factor events already had positive H factors. These
arms are separate descriptive records, not a pooled new-Q estimate. At the
two old dual-tail actions from source 200051, the initial action ratios change
from 0.11737 to 0.05543/0.05890 (risk) or 0.07545/0.07859 (mixture). Restored
positive mass is not the same as optimal allocation to the actual event region.

Importantly, replaying the **original frozen draws on this identical 31-case,
24-source target** gives M2 1.84525e-5 and maximum source share 99.2555%, including
two H2-zero events. The new interaction baseline keeps exactly the same Q but
samples no such event in its 64 draws/state, giving M2 2.22970e-7 and share
62.5098%. This large seed-schedule sensitivity is not a proposal improvement;
absence of a known tail observation does not eliminate the tail. Neither the
mixed arm's smaller maximum share nor its larger current M2 establishes a
generalized benefit or definitive rejection from this sparse sample. All
observations remain intact; the two schedules are not pooled or selectively
restarted.

**Decision:** retain the normalized family as an opt-in experiment, not a new
production baseline. Coverage repair is verified, but native variance and
independent source diversity have not demonstrably improved. Absolute-risk H
barely tilts the second factor in this bank; mixture trades some cooperative
concentration for broader coverage. Keep all three predeclared scores in the
separate fresh validation protocol, without claiming a trained/validated winner.
Sources 250000–250255 remain unopened at this training milestone. NDE,
production selector/Q, PPO, old training manifests and consumed validation are
unchanged. All 162 relevant regression tests and frozen code/assets passed.

Compact evidence:
[highd_v46_proposal_comparison_v1.json](../results/highd_v46_proposal_comparison_v1.json).

The fresh protocol was frozen after review at `5536040`, verifying all training
artifacts without opening source 250000 or any other reserved fresh source.
The local lock is `outputs/v46_proposal_validation_v1_lock.json`. Its compact
[freeze record](../results/highd_v46_proposal_validation_v1_rule_lock.json)
records status **frozen, not started**; it contains no validation result. With
that existing lock, the next native step is the `run` command above, not another
`freeze` into the already existing output. Any code/config/environment change
requires an explicitly versioned protocol and lock before opening sources.
