# Fixed-bank proposal comparison and fresh validation

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
training winner from a few event-bearing sources. At the pre-run declaration,
sources 250000–250255 were reserved and had not been opened by implementation
tests or training. Their completed validation evidence is recorded below.

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
records the historical status **frozen, not started**; it contains no validation
result. The completed run below used that existing lock, without another
`freeze` into the already existing output. Any code/config/environment change
requires an explicitly versioned protocol and lock before opening sources.

## Completed fresh validation evidence, 2026-10-05

The frozen protocol completed on code `94b49d2`, without changing any score,
epsilon, selector, gate, discovery budget, native dynamics or PPO. All 256
natural source episodes (250000–250255) were retained. There were 1,276 bank
states and 3,720 bounded pair audits; early natural endpoints reduced the bank
without backfill. Discovery found 77 strict-positive pairs, 75 positive states
and 57 eligible sources (22.2656% of all sources). This is offline selected-state
coverage, not online time-step intervention rate or global collision recall.
Natural source rollouts had zero CAV events and 11 BV-only competing endpoints.

Every candidate used the same 57 selected states/pairs and 64 continuations per
case: 3,648 draws per candidate, 10,944 total. All endpoints were retained. The
conditional target is identical across these three proposals, unlike the old
A/B selector comparison. No sample was added after seeing events.

| Metric | Interaction | Absolute risk | 50/50 mixture |
| --- | ---: | ---: | ---: |
| Native CAV events / 3,648 | 20 | 2 | 9 |
| Independent effective-dual event sources | 5 | 2 | 4 |
| Raw conditional event rate | 0.5482% | 0.0548% | 0.2467% |
| Conditional IS event mean | 8.3395e-5 | 3.3108e-5 | 5.0462e-5 |
| Conditional event M2 | 1.5861e-5 | 2.5791e-6 | 5.1614e-6 |
| M2 ratio to interaction, point only | 1 | 0.1626 | 0.3254 |
| Stratified variance ratio, point only | 1 | 0.1635 | 0.3252 |
| Mean second-factor Q/P TV | 0.5615 | 0.0275 | 0.2865 |
| Largest source M2 share | 99.8642% | 91.7443% | 99.3938% |
| Largest single-event M2 share | 98.4272% | 91.7443% | 96.2104% |
| Effective M2 sources, descriptive | 1.0027 | 1.1785 | 1.0122 |

Mixture's M2 point estimate is about 67.46% below interaction on this matched
conditional target. This is not a supported generalization or acceleration
claim: mixture has only four observed event sources, and risk-only only two.
Interaction meets the point minimum of 20 events/five sources, but 1,808 of
2,000 whole-source bootstrap replicates fail the distinct-source/event guard;
risk and mixture each censor all 2,000. Every M2/variance-ratio interval remains
undefined. Fixed-state MCSE cannot repair this source-level evidence gap.

Source 250200 dominates all proposals. Interaction's event sources were
250089, 250096, 250162, 250188 and 250200; risk retained two, and mixture four,
of that observed set. Their union is five independent sources, not eleven
independent replications. Zero events at other cases do not establish zero
true risk. No proposal meets the maximum-share <=80% and effective-M2-sources
>=3 checks, and no proposal passes the complete readiness decision. The smaller
risk-only M2 point is especially sparse and must not be treated as an optimal Q.
Its weak second-factor tilt also persists from training.

### New-source collision-action coverage witness

At source 250200, 51 s, actors [7,25], interaction sampled actions [0,21] at
continuation seed 4212853 and observed a CAV event. The raw clearance-risk score
was 0.569548; its four-corner contrast was negative (about -0.002457), so positive
interaction and H2 were zero. The natural modal reference was [19,20]; this
extends the earlier reference-column witness to an excluded negative-contrast
action. The first factor LR was 0.0238647 and the second factor LR 10, yielding
a full weight 0.238647. That one event contributes 98.4272% of interaction's M2.

Mixture independently sampled the same initial actions at seed 4212852 and
observed a CAV event with H2=0.0455786, Q2=0.0635551 and full weight 0.134592.
Risk-only sampled [0,20] at seed 4212839, another original-H_I-zero cell, with
positive implemented H factors and weight 0.0929076. Thus **both new scores
actually exercised the repaired action region on fresh data**; this is no
longer only an old-training action replay. These are different continuations,
not a paired counterfactual native outcome comparison.

All nine mixture events and both risk events have positive implemented H1/H2.
Nevertheless their respective largest events still account for 96.2104% and
91.7443% of their M2. Positive H mass repairs a structural omission but does not
ensure adequate allocation to the native collision region, eliminate source
risk heterogeneity, or prove a lower true estimator variance. The training and
fresh-validation point orderings differ; their sparse tail realizations are
not evidence for choosing or retuning lambda on this consumed cohort.

### Native confirmation, cost and independent verification

The outcome-blind eligible-source confirmation subset measured all 16 planned
cases with 16 draws at each supported action corner (1,024 separate forced
continuations). Fourteen passed the predeclared native-interaction mean >=0.05
threshold: 87.5% descriptive confirmation. The two remaining means were
0.020424 (source 250087, MCSE 0.017746) and -0.012115 (250146, MCSE 0.060780).
No case was replaced. This fixes the previous design's missing eligible-rule
confirmation coverage, but confirms minimum-clearance interaction, not collision
probability calibration, real-Q event rates or population testing efficiency.

Local total wall time was 2,208.78 s, with 1,578.71 s common discovery wall time,
1,390.01 s recorded proxy CPU and 20.81 s selection/re-audit CPU. Native Q CPU
was 183.02/182.50/181.66 s for interaction/risk/mixture. These are local profiles,
not controlled timing benchmarks or acceleration factors.

Independent read-only reconstruction verified all 248 artifact hashes, bounded
screening, outcome-blind case/probe schedules, all 10,944 initial action draws
including correlated natural partners, complete/factor likelihoods, event
counts, IS means/M2/stratified variance, source contributions, confirmations
and paired whole-source bootstrap censoring. Maximum replayed full-likelihood
log error was 6.39e-14. Eight protected old-manifest/summary/protocol/code hashes
remained unchanged. All 180 regression tests in the 14 relevant modules and
frozen code/assets verified. Raw outputs remain ignored; no PPO or training
manifest was exported.

**Decision:** fresh native evidence supports structural action-coverage repair,
but does not establish a successful variance/source-diversity baseline. Keep
production unchanged; do not launch PPO, add continuations from these cases,
select a winner automatically or retune parameters on sources 250000–250255.
These sources are now consumed validation. The next design work belongs on
declared training data: calibrate absolute risk against native collision actions
and audit proposal mass/tails across independent sources. Any source-balanced
training law must be explicit; do not clip weights, discard the dominant source
or rescale primary IS contributions to manufacture balance. A revised proposal
needs a separately versioned fresh validation and later independent
original-initialization evaluation.

Compact reproducible evidence:
[highd_v46_proposal_validation_v1.json](../results/highd_v46_proposal_validation_v1.json).
