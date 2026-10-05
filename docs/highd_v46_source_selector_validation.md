# Fresh source-level selector validation

This experiment compares absolute-risk ranking with and without the existing
dual research gate. It follows the fixed-training ranking audit at `822ace9`.
NDE, native initialization, CAV dynamics, risk proxy, H/Q algebra, likelihood
sampler, event definition and PPO implementation remain unchanged. A new
validation adapter admits non-strict pairs for the A ablation; the production
collector's strict checks remain intact.

The protocol is
[highd_v46_source_selector_validation_v1.json](../configs/highd_v46_source_selector_validation_v1.json).
Sources 240000–240255 are reserved for this experiment. Previously consumed
training/validation/test ranges and reserved final-test sources are rejected.
Freezing the protocol verifies both checksum-locked training calibrations,
current code/dependencies, environment and frozen natural/physical target.
The run checks this lock before opening a fresh episode. An implementation,
configuration or dependency change requires a new explicitly versioned protocol.

## Candidate rules and bounded discovery

Both rules use exactly the same frozen mixed5 bank: two geometry peaks, two
uniform times and one lateral/phase-aware state per natural source episode,
with four seconds remaining. At each bank state only rank 1, rank 2 and one
audit pair are evaluated in the existing nearest-eight/115 m universe.

| Rule | Rank score | Research eligibility |
| --- | --- | --- |
| A | Maximum supported proxy clearance risk | Valid supported P/H/Q only |
| B | Same score | Existing complete `production_strict` gate |

B retains the unchanged leverage, interaction, global excess, risk-H TV,
interaction-mass and actual-H TV checks. This is an ablation of the whole
existing research gate, not evidence isolating each individual threshold.
Neither rule introduces an absolute-risk cutoff or ranks by an outcome label.
Ties break by time and ordered actor IDs. Each rule selects at most one fixed
state/pair **per source**, without a global top-k competition between sources.
The bank is selected across an offline natural episode; this is not a causal
online selector or a deployed 60-second policy.

For A's supported non-strict pairs, `H = Pij * positive_interaction / mass`
and the existing two-factor Q and correlated natural-partner law are unchanged.
Zero interaction mass records natural fallback; no alternative state/pair is
substituted. Each candidate records actual Q/P TV for both factors in every
arm. Effective dual-pair coverage requires both dual-Q TVs >= 0.001. Native
events from natural fallback remain observed events, but cannot satisfy the
effective-dual event-source diversity check.

## Fixed native budget and source pairing

Every selected state receives the existing 64 continuations per arm: natural,
first-factor-only, second-factor-only and dual. Only the initial decision is
tilted; the next four seconds use unchanged natural actions. CAV involvement
in the first collision is the event; BV-only first collisions remain competing
negative endpoints. These factor ablations are not the paper's single-BV agent.

The source-indexed continuation schedule begins at 2900000. All cases from
the same source use the same 64-seed block, providing A/B common random numbers;
different sources use disjoint blocks. A/B selecting the identical state/pair
shares one physical case, without duplicate simulation or duplicated data.
The entire case plan, fallback flags and probe selection are saved before
native outcomes. Every source and every endpoint remains in the record.
Abstentions count in coverage denominators but have no imputed native outcome.

The upper bounds for this fixed cohort are 1,280 bank states, 3,840 proxy pair
audits, 512 distinct selected cases and 131,072 probabilistic continuations.
Selected cases also recompute their proxy once (at most 512 re-audits) to verify
the locked audit. Discovery time and selection/re-audit CPU are reported separately.
Actual case counts shrink with abstention and shared A/B choices. This budget
increases independent sources; the per-state 64-draw budget is unchanged.
There is no event-dependent extension, restart with only positives or backfill.
The budget does not guarantee five or ten event-bearing sources.

## Separate native interaction confirmation

Before outcomes, a seeded uniform sample of at most 16 sources from the A/B
candidate-source union is selected. Each selected method's case, when it has
a positive supported contrast, receives 16 continuations at each of four
supported action corners. At most 2,048 confirmation continuations are possible.
The source-indexed confirmation schedule starts at 3900000 and is disjoint
from probabilistic Q samples. Corners and A/B cases from the same source share
random numbers. Missing candidates or unsupported/no-positive contrasts remain
explicit statuses and are never substituted after an outcome.

Native interaction is `R11 - R10 - R01 + R00` in actual native minimum-clearance
risk. The predeclared descriptive confirmation threshold is 0.05, inherited
from the existing discovery calibration. Its conditional MCSE is retained.
The report gives both confirmations among measured cases and confirmed-source
yield within the planned source subset. Neither is global recall. Forced corner
outcomes are separate from real-Q event rates and IS moments, and do not enter
their likelihood estimates or a training dataset.

## Source-level statistics

Each rule reports all-source candidate/effective-pair coverage, event sources,
continuation event rates, actual second-factor Q/P TV, conditional IS mean,
second moment and each source's contribution. Within a rule, the conditional
target selects one chosen source uniformly and its one chosen state. It has
no original-initialization density correction. A/B generally select different
state laws, so their moments cannot establish same-target variance improvement
or population testing acceleration.

For source second moment `m2_k = mean(I * (P/Q)^2)` and K selected sources:

`M2_k = m2_k / K`,
`c_k = m2_k / sum_j m2_j`,
`effective_second_moment_sources = 1 / sum_k c_k^2`.

The effective number is descriptive, not an independent sample size. Zero-event
cases retain their source mass. All-zero observations yield undefined shares,
not zero true risk. Observed event-source yield divides discovered event sources
by **all** natural source episodes; it does not label abstentions as zero-risk.

Uncertainty has two distinct targets:

- Fixed-state stratified MCSE uses independent continuation draws conditional
  on each selected state. More draws can improve this conditional precision.
- Generalization intervals use 2,000 paired whole-source bootstrap replicates,
  retaining all within-source observations. A/B use the same source draws.
  Continuations are never resampled as independent traffic sources.

Event/moment intervals require at least 20 events across five observed event
sources. Unsupported intervals remain null and retain censored replicate counts.
All-zero/all-one coverage bootstrap distributions are marked as degenerate
observed boundaries rather than exact population bounds. Neither extra bootstrap
repeats nor small fixed-state MCSE repairs insufficient independent event sources.
Maximum-source share is a point diagnostic; no artificial bootstrap confidence
claim is made for it. Coverage and observed event-source yield have paired
B-minus-A comparisons. Conditional dual event rate also has a paired source
interval; IS moments remain selector-specific conditional estimates.

The predeclared engineering screening checks require five effective-dual event
sources (ten desired), maximum source share <= 0.8 and effective second-moment
source count >= 3. These are candidate data-quality checks, not significance or
paper-efficiency guarantees. The historical 99.26% training share comes from a
different state law and serves as context, not a matched reference estimate.
Source diversity, native confirmation and contributions must be reviewed together.
No winner or PPO launch is selected automatically. Validation remains held out
from fitting; later PPO requires its own training data and independent evaluation.

## Run and artifacts

Use the pinned Python 3.9 environment. Freeze once after code/configuration
review, then run the complete fixed cohort:

```bash
conda activate highd-ppo-clean39
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_source_selector_validation freeze --output outputs/v46_source_selector_validation_v1_rule_lock.json
python -m scenario_reconstruction.highd_v46_source_selector_validation run --lock outputs/v46_source_selector_validation_v1_rule_lock.json --output outputs/v46_source_selector_validation_v1
```

Freeze opens only the two locked old training artifacts, not validation sources.
The run destination is exclusive-create. An interruption retains its protocol
and incomplete artifacts; an explicitly labelled restart uses the same frozen
budget in a new destination, never only the event cases. Repeated seeds across
restarts or A/B are not independent data and must not be pooled.

Outputs include `opened_protocol.json`, natural state/pair audits, the outcome-
blind case plan, all four native arms, separate confirmations, per-source
records and the complete summary. Artifact hashes and lock identity accompany
the report. No PPO checkpoint, training sequence manifest or acceleration
estimate is exported. Raw outputs remain ignored by Git.

```bash
python -m pytest tests/test_highd_v46_source_selector_validation.py tests/test_highd_v46_selector_ranking.py tests/test_highd_v46_multistate_q.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Implementation checks on 2026-10-05 passed 162 relevant regression tests and
frozen-code/asset verification. Implementation tests did not open fresh sources.

## Completed evidence, 2026-10-05

The locked full cohort completed: all 256 natural sources, 1,280 bank states,
3,736 bounded pair audits and 257 distinct selected cases. A/B shared 46 cases;
only one of B's 47 cases differed from A's choice in the same source. All
65,792 probabilistic continuations were retained. Natural source episodes had
zero CAV events and 11 BV-only competing endpoints; these are separate from
the selected-state continuations below. No PPO or training export was produced.

| Metric | A: absolute risk only | B: absolute risk + frozen strict gate |
| --- | ---: | ---: |
| Selected sources / 256 | 256 | 47 |
| Effective dual-pair sources / 256 | 177 | 47 |
| Natural fallback sources | 78 | 0 |
| Dual continuations | 16,384 | 3,008 |
| Dual CAV events | 63 | 63 |
| Raw conditional dual event rate | 0.3845% | 2.0944% |
| Independent dual event sources | 3 | 3 |
| Observed event-source yield / all sources | 1.1719% | 1.1719% |
| Second-factor dual Q/P TV, mean | 0.4097 | 0.5460 |
| Second-factor TV, min–max | 0–0.8941 | 0.4253–0.6947 |
| Largest source second-moment share | 99.9247% | 99.9247% |
| Effective second-moment sources, descriptive | 1.0015 | 1.0015 |

Strict discovery found 65 positive pairs, 61 positive states and 47 distinct
sources. B's effective-pair source coverage was 18.3594%, with a predeclared
90% source-cluster bootstrap interval of 14.4531–22.2656%. This is offline
selected-state coverage, not a time-step online intervention rate or global
collision-source recall. No exhaustive native reference establishes recall.

All observed dual events came from the same three physical cases in both rules:

| Source | Dual CAV events / 64 | Share of dual second moment |
| --- | ---: | ---: |
| 240022 | 47 | 99.9247% |
| 240099 | 14 | 0.07214% |
| 240151 | 2 | 0.003181% |

B retained all observed event sources while requiring 81.64% fewer selected
cases than A. That improves *conditional event density / screening economy*,
not source diversity or a same-target variance comparison between A and B.
The all-source event yield was identical. Adding A's non-strict candidates
introduced no observed additional event source in this fixed cohort; zero
observed events do not establish zero true risk in those candidates.

The historical two-event-source / 99.2555% training result and this three-source
/ 99.9247% validation result use different cohorts, budgets and state laws.
They are not a matched improvement estimate. The larger fresh source budget
still did not reach the predeclared five-effective-event-source guard, maximum
share <=80% or effective second-moment source count >=3. Both diversity and
concentration checks failed. With only three event sources, all event/moment
generalization intervals and paired event-rate comparisons remain censored,
despite 63 events and 2,000 bootstrap replicates.

Within each rule's fixed state law, the four native arms observed respectively
4 natural, 35 first-factor-only, 4 second-factor-only and 63 dual CAV events,
each from three sources, but **not the same three across arms** (see the
post-hoc support audit below). The empirical dual/first-factor second-moment
ratio was 0.3385, a favorable point diagnostic but not a supported generalization
or precision/acceleration claim. These factor arms are not the paper's agent.

### The remaining likelihood-weight tail

At source 240022, continuation seed 2901420 produced a real CAV event with
second-factor `P2=0.23889344`, `H2=0`, `Q2=0.02388934`. Its full weight was
0.19425933, whereas the other dual event weights at this state were approximately
0.00880–0.00954. This one observation accounted for **91.0754% of the entire
dual event second moment**. It is not a likelihood failure: the unchanged
epsilon mixture preserves support and the full/factor logs agree. It shows
that substantial factor TV alone does not ensure coverage of actual collision
actions. More trajectories from the same dominant state would not create
independent source diversity.

### Native confirmation is incomplete, not negative evidence for B

The outcome-blind uniform confirmation sample selected 16 sources, none in B's
47-source eligible set. B therefore has no measured native interaction cases;
its confirmation rate is **undefined**, not a measured zero. A measured nine
supported contrasts and confirmed none at the frozen 0.05 threshold; seven
cases had no positive supported contrast. These non-strict A probes cannot
be transferred to B. No case was substituted or added after seeing outcomes.
A future separately versioned confirmation design should predeclare coverage
of each candidate rule before native outcomes, on independently reserved data.

### Cost, verification and decision

Local total wall time was 4,987.61 s; common discovery wall time was 1,600.73 s,
including 1,405.55 s of recorded proxy CPU. Selection plus selected-pair
re-audit CPU was 98.78 s. Dual continuation CPU for the rule-specific case sets
was 802.42 s (A) and 148.90 s (B). This is an offline implementation profile,
not a controlled online timing benchmark or population testing acceleration.

An independent read-only reconstruction verified all 1,049 artifact hashes,
the complete source and continuation schedules, outcome-blind case/probe
selection, every arm's event counts and moments, source summaries, confirmations
and the paired source bootstrap. Maximum full-versus-factor log-weight error
was 1.47e-14. The old training/validation manifests remain unchanged. The 58
entry-point/ranking/sampler/likelihood regression tests passed; frozen code and
assets verified.

**Decision: screening density improved, but the source-diversity and
concentration bottlenecks did not. Do not restart PPO or append more
continuations from these validation cases.** Keep NDE/Q/PPO frozen. The next
training-side diagnostic should audit actual collision-action coverage and
native risk calibration, then test a separately declared source-discovery rule
and source-balanced confirmation design on fresh episodes. This consumed
validation cohort must not become fitting data.

Compact reproducible evidence is recorded in
[highd_v46_source_selector_validation_pilot_v1.json](../results/highd_v46_source_selector_validation_pilot_v1.json).

## Post-hoc collision-action coverage audit, 2026-10-05

The checksum-locked diagnostic in
[highd_v46_collision_coverage_v1.json](../configs/highd_v46_collision_coverage_v1.json)
rechecks all 65,792 existing probabilistic sequences, including zero-event
cases, and reports each selector and sampling arm separately. It verifies
the logged factor mixtures, the product likelihood and the full likelihood.
The zero test is exact equality in the logged effective H factors; tiny
positive values are not reclassified using a tuned threshold. Natural
fallback is a separate category. No native continuation or parameter search
is performed. This is an explanatory audit of consumed validation data.

The dual-arm event counts and unnormalized sums are identical for A and B
because their observed events come from shared physical cases. Those cases
are read once, and are not counted as independent A/B replications.

| Effective H category | CAV events | Sum of I w^2 | Share of dual second moment |
| --- | ---: | ---: | ---: |
| H1 > 0, H2 > 0 | 62 | 0.003697867905 | 8.9246% |
| H1 = 0, H2 > 0 | 0 | 0 | 0% |
| H1 > 0, H2 = 0 | 1 | 0.037736689194 | 91.0754% |
| H1 = 0, H2 = 0 | 0 | 0 | 0% |

There were no natural-fallback CAV events. Dividing each sum by K*64 gives
its conditional M2 contribution; K is 256 for A and 47 for B. Arms are not
pooled because they were sampled under different Q. These are contributions
to the uncentered second moment, not an exact decomposition of centered
stratified estimator variance.

The implemented proposal mixes each factor separately:
`Qij = q1 * q2`, `ql = eps_l * Pl + (1-eps_l) * Hl`.
For the dominant event, `P1/Q1 = 0.0194259335` and `P2/Q2 = 10`, giving a full
weight of 0.1942593349. The factor's multiplier is ten; the trajectory weight
is their product and remains below one in this case. When H1 is zero, the
sampler defines its otherwise undefined H2 row as P2. Thus the logged
`H1=0,H2>0` category denotes natural conditional fallback, not supported mass
in the original joint H; a naturally supported draw cannot have both logged
factors zero in this implementation.

### The risk score was high; the interaction contrast removed it

The audit reconstructs the frozen proxy at source 240022, 34 seconds, actors
[15,2], without a new native rollout. The collision actions were [32,21]; the
natural modal reference actions were [19,21]. All four corners have natural
support. Recomputed P/H/Q factors match the generation record.

`R11 = R10 = 0.8884343432`,
`R01 = R00 = 0.0380021667`,
`I = R11 - R10 - R01 + R00 = 0`.

R is a clearance-risk proxy score, not collision probability. The proxy already
assigns high absolute risk to this action. The second action equals its natural
reference, so the four-corner interaction cancels. Algebraically, this contrast
is zero on the reference row and column, even if those actions have high
absolute risk (floating-point residuals can be tiny rather than exact zero).
Pure interaction-weighted H therefore omits some dangerous actions by design.
The demonstrated defect lies in using cooperative gain alone as the proposal
score for a target event that also includes collisions without positive gain;
this witness does not demonstrate a failure of the raw absolute-risk proxy.
The epsilon mixture still preserves Q support, so the ordinary IS likelihood
remains valid.

### Both factor coverage and source concentration need attention

| Native arm | Observed event sources |
| --- | --- |
| Natural | 240012, 240022, 240111 |
| First factor only | 240022, 240099, 240141 |
| Second factor only | 240012, 240022, 240111 |
| Dual | 240022, 240099, 240151 |

The union contains six independent source episodes, not three. This corrects
the earlier prose that all arms shared three sources; the stored per-arm
counts and likelihood estimates were correct. The two natural events at
240012 and 240111 had H1=0 and H2=P2. They demonstrate first-factor adversarial
coverage gaps as well. A dual zero in 64 draws does not establish a zero event
probability, and the six-source union cannot be substituted for the dual arm's
three-event-source evidence guard. Source 240141's first-factor event had both
H factors positive, so zero-support categories do not capture every coverage
or finite-sampling issue.

As a descriptive sensitivity check only, excluding the single largest event
from the moment calculation leaves 0.003697867905 in summed I w^2; source
240022 still contributes **99.1561%**. Every original observation remains in
the dataset and in all primary estimates. This is not a corrected estimate
or a new Q evaluation. It rules out attributing all source concentration to
that one observed tail point. Conditional risk heterogeneity and source
coverage can still matter alongside proposal quality.

### Concrete direction for the next proposal experiment

Hold the training state/pair bank fixed when comparing proposal scores. The
first candidate family should include normalized absolute-risk and interaction
components: `H_R = P*R/sum(P*R)`, `H_I = P*I_positive/sum(P*I_positive)`, and
`H_lambda = (1-lambda)*H_R + lambda*H_I`. A zero component mass uses an explicitly
declared fallback. Normalization makes the component balance interpretable;
parameters, fallback behavior and acceptance criteria must be chosen on
training data and locked before a new source-disjoint validation cohort.
No lambda has been chosen or tested on this consumed cohort.

A per-factor floor `H_l_new=(1-delta_l)*H_l+delta_l*P_l` is algebraically an
epsilon ablation with `eps_l_new=eps_l+(1-eps_l)*delta_l`. It should not be
reported as an independent new mechanism. Smoothing the joint H before
factorization is a different operation and needs its own likelihood checks.
Any adaptive epsilon must use information available before the sampled action.
Success should include event-action coverage, per-source and per-event M2
concentration, and matched-state variance evidence as well as event yield.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_collision_coverage --output outputs/v46_collision_coverage_v1_complete.json
python -m pytest tests/test_highd_v46_collision_coverage.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_source_selector_validation.py -q
```

All 46 relevant tests passed. The original validation summary, all 1,049
artifacts, NDE, selector, Q and PPO code remain unchanged. Compact evidence:
[highd_v46_collision_coverage_v1.json](../results/highd_v46_collision_coverage_v1.json).
