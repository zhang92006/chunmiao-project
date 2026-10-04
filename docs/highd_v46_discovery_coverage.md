# Training-only state/pair coverage audit

This task builds on `da04d33`. It does not change the NDE, natural reset law,
production Q, risk proxy, strict gates, PPO, or the consumed validation cohort.
The question is whether the v1 discovery budget missed useful states or pairs.

## Fixed audit design

The protocol is [`highd_v46_discovery_coverage_v1.json`](../configs/highd_v46_discovery_coverage_v1.json).
It accepts only the checksum-locked **training** summary from discovery v1:
16 natural source episodes, seeds 160000-160015. The existing discovery module,
proxy, cross-unit helper, physical configuration, natural target and source
artifacts must match their recorded hashes. Validation input is rejected.

The pinned summary identifies an existing local run. When regenerating v1 from
scratch, its timing fields change the summary checksum: first verify its config,
implementation, dependencies, state-bank and pair-audit hashes against the
committed v1 evidence, then make an explicitly versioned coverage configuration
pointing to that regenerated summary. Do not silently bypass the source lock.

The natural episodes are replayed without changing their action RNG. Every
original geometry-bank snapshot must match its complete pre-draw state,
including natural tables, histories and RNG state. The new bank comprises:

1. The original `geometry_1s` bank states with at least four seconds remaining.
   Near-horizon states are excluded without backfilling.
2. Two uniformly sampled, previously unselected times per episode, also with
   four seconds remaining. Sampling is without replacement, using a separate
   RNG (seed 61), with no score or collision-label input. These times need not
   be separated by five seconds and are not independent traffic episodes.

At every retained state, audit **all** pairs in the unchanged nearest-eight-
BVs/115 m universe, including pairs whose cheap geometry score is zero. This
is at most 28 pairs per state, not exhaustive coverage of all road actors.
The original 33x33, four-second reactive proxy and all gates remain unchanged.
Previously audited pairs are recomputed and must match their old metrics.

## What the comparisons mean

Within original selected states, compare the former two ranked plus one random
pair audit with all candidate pairs. Report positive pairs, positive states,
distinct source episodes, pair ranks and mechanisms. The fraction of proxy-
positive pairs covered by the old audit is defined only within this evaluated
universe. It is `null` when there are no positives, not a perfect recall score.
A new positive pair in an already-positive state is separated from a state
that previously had no discovered positive pair.

Within the uniformly sampled omitted times, positives show that the v1 temporal
bank can skip an opportunity. Zero positives would not prove that all omitted
times are safe. We do not estimate global temporal recall from this small audit.
Both groups retain all results, including all-negative states and episodes.

For diagnosis only, summarize strict and the two previously declared ablated
gate views. Report top-k geometric pair state hits at k=1,2,3,5,10,28. This is
not permission to tune on validation or claim an online selector is deployed.

## Conditional native checks

Before seeing the audit outcomes, the
[`highd_v46_coverage_witness_v1.json`](../configs/highd_v46_coverage_witness_v1.json)
protocol specifies up to two new strict-positive pairs, ranked by natural-
probability-weighted interaction mass, one per source episode. Pairs already
audited by v1's geometry bank are ineligible. Each receives 16 natural stochastic
continuations per forced action corner, with common random numbers and seeds
2000000-2000015. No case is substituted because its native outcome is negative.

The four-corner experiment is a mechanism check, not a draw from Q, an ordinary
IS population estimate, or a test of statistical significance. A proxy-positive
candidate is not itself proof of a real joint-action advantage.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_discovery_coverage --source outputs/v46_state_discovery_v1_train --output outputs/v46_discovery_coverage_v1_train
python -m scenario_reconstruction.highd_v46_discovery_witness --discovery outputs/v46_discovery_coverage_v1_train --config configs/highd_v46_coverage_witness_v1.json --output outputs/v46_coverage_witness_v1_train
python -m pytest tests/test_highd_v46_discovery_coverage.py tests/test_highd_v46_state_discovery.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Python/dependency versions are recorded in the audit configuration. Raw state
banks and per-pair rows remain in ignored `outputs/`; only compact evidence is
committed. Any improved selector requires a separately declared fresh validation
cohort. Neither prior validation seeds 170000-170007 nor reserved final-test
seeds 100000-101023 are used in this audit.

## Evidence, 2026-10-04

All source snapshots and re-audited v1 metrics matched. Four of the original
32 geometry-bank states lacked four remaining seconds, leaving 28 selected
states plus 32 uniformly sampled omitted states. The completed run evaluated
1,132 pair matrices in 475.82 s, including 11.78 s of natural scanning. This is
a local diagnostic timing, not an acceleration benchmark. One initial attempt
stopped on an exclusive-create artifact-writing error after two states; its
incomplete files were retained separately. The corrected run restarted the
same protocol from scratch, retaining every outcome. A two-state integration
test now covers that output path.

| Group | States audited | Pairs audited | Strict-positive pairs | Strict-positive states / source episodes |
| --- | ---: | ---: | ---: | ---: |
| Existing geometry bank | 28 | 540 | 3 | 2 / 2 |
| Uniform omitted-time audit | 32 | 592 | 1 | 1 / 1 |

At existing bank states, the old three-pair budget found two of the three
strict-positive pairs, but already found both positive states. The missed pair
was [7,10], source 160009 at 19 s, geometry rank 2, another bracketing pair.
The old audit checked rank-1 bracket [7,9], a different-mechanism same-side pair
[6,7], and a random lateral pair [32,10]. Thus forced mechanism diversity
displaced a promising second-ranked pair, **without losing this positive state**.

The omitted-time audit found [12,29], source 160008 at 34 s, a lateral pair at
rank 2. This adds one source episode beyond the original geometry bank's strict
positives. All four strict-positive pairs in the full audit had geometry rank
1 or 2. This is training-side evidence for testing top-two-plus-random checks,
not proof that top two suffice in new traffic or outside the nearest-eight
universe. There was no observed strict-positive pair below rank 2 in this audit.

Two-actor leverage still rejected 524/540 selected-bank and 588/592 omitted-time
pairs. Removing global excess alone did not add strict positives in either
group. The existing local-interaction view found four pairs / three states in
the selected bank and one pair / one state among omitted times. No production
gates were relaxed.

The predeclared native confirmations of the two new candidates were:

| Source / time / pair | Proxy interaction | Native interaction mean (SE) | Base / first / second / joint CAV events, 16 each |
| --- | ---: | ---: | --- |
| 160008 / 34 s / [12,29] | 0.37935 | 0.40229 (0.00965) | 0 / 0 / 0 / 0 |
| 160009 / 19 s / [7,10] | 0.05182 | 0.13492 (0.02720) | 0 / 0 / 0 / 0 |

Both reproduce positive interaction in minimum-clearance risk under the frozen
native dynamics. Neither adds a collision witness in these small continuations.
Risk interaction is not collision probability, and these selected conditional
cases do not establish significance or population testing advantage.

### Decision from this audit

The temporal bank is a demonstrated source of lost **state diversity**. Increasing
pair budget at the old bank found another useful proxy pair but no new strict-
positive state. Uniform temporal exploration found a new source episode and a
lateral mechanism, so an improved bank should reserve temporal exploration and
mechanism/phase coverage instead of retaining only two global geometry peaks.
For pair budget, test retaining both leading candidates plus an audit slot,
rather than forcing a distinct mechanism into the second slot. Broader pair
audits should remain a sampled monitoring control, not the default expensive
online computation. These are hypotheses for a new version, not deployed rules.

Do not change NDE, relabel the bank as natural-initialized data, or start more
PPO epochs based on these results. A new screen still needs fresh episode-
disjoint validation, then a separately specified conditional training law and
independent original-initialization evaluation. The consumed v1 validation
cohort remains untouched during this task.

The compact record is
[`highd_v46_discovery_coverage_pilot_v1.json`](../results/highd_v46_discovery_coverage_pilot_v1.json).
All 82 relevant regression tests passed; frozen code and assets verified:

```bash
python -m pytest tests/test_highd_v46_discovery_coverage.py tests/test_highd_v46_state_discovery.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_pair_synergy.py tests/test_highd_v46_d2rl.py tests/test_highd_v46_validation.py tests/test_highd_paper_discrete_pairs.py tests/test_highd_mean_precision_policy.py tests/test_highd_dynamic_pairs.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```
