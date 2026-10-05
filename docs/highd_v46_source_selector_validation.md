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
frozen-code/asset verification. The full fresh validation cohort has not been
opened by implementation tests; source-generality results remain pending.
