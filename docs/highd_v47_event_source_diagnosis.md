# Training event-source bottleneck diagnosis

This training-only diagnostic follows `cd296fc`. It does not advance PPO or
probability-allocation changes. Both old training calibrations and all their
artifacts are verified through the existing locked loader. Validation is not
opened, and original artifacts and production code remain unchanged.

The audit covers all 128 sources and 1,883 previously computed pair matrices.
For each source, choose the maximum supported absolute-risk pair in its cached
mixed5 bank, before strict filtering; ties use time and ordered actor IDs.
Retain all source rows, including absent/below-threshold candidates. Probe all
11 source candidates with score >=0.5, a training diagnostic triage threshold,
not a calibrated probability or a proposed production gate. This design is
informed by previously observed training evidence, not fresh validation.

Before any new native outcomes, freeze the protocol, source hashes, complete
case plan and supported action probes. Each candidate gets 64 draws under
unchanged interaction-H and 64 under the existing 50/50 normalized H mixture.
Those two arms share continuation seeds beginning 4400000, disjoint by case.
An additional 16 continuations force the supported maximum-risk action and
16 force the natural modal action; these share separate seeds beginning
4500000. Total budget: 1,408 Q draws and 352 forced probes. Partners follow
their exact natural conditional law and subsequent four seconds remain natural.
All endpoints, including negatives and BV-only first collisions, are retained.

Forced probes diagnose whether specific actions can exercise a collision;
they are not Q draws, ordinary IS estimates, training sequences, or proof of
global source recall. Real-Q evidence is reported separately. Existing source
episodes reused here are not new independent sources. Negative finite probes
do not prove safety. The cached bank does not cover every time or road actor.
No substitute or event-conditioned extension is allowed. A full prospective
training collection, if needed, needs its own declared protocol.

Initial read-only funnel evidence: the original cohort has nine source maxima
>=0.5, the expansion only two. The expansion's source 230053 at 33 s has
proxy risk 1.0 for pair [5,7], but failed `global_excess` and was never queried.
This is a candidate omission, not yet a demonstrated missed native collision.
The audit separates this eligibility hypothesis from proxy/native mismatch
and finite-Q action coverage. It does not assume a cause before native results.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_event_source_diagnosis freeze --output outputs/v47_event_source_diagnosis_v1_lock.json
python -m scenario_reconstruction.highd_v47_event_source_diagnosis run --lock outputs/v47_event_source_diagnosis_v1_lock.json --output outputs/v47_event_source_diagnosis_v1_train
python -m pytest tests/test_highd_v47_event_source_diagnosis.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_multistate_q.py -q
```

Use the inherited Python 3.9.19/NumPy 1.23.1/SciPy 1.13.1 environment. Outputs
are exclusive-create; failed runs are retained, never pooled with a restart.
No new training readiness or population efficiency claim is produced.

## Completed diagnostic and training-collector repair

The protocol frozen on `4236dad` completed its entire budget in 89.91 s.
Interaction Q observed event sources 200017, 200051 and **230053** (1/31/1
events); mixture observed only 200051 (15 events). The maximum-risk forced
probe found seven event-bearing sources, while every modal-action probe was
negative. These paired arms/probes are not independent source replications.

| Source | Maximum-risk forced events /16 | Natural probability of that action |
| --- | ---: | ---: |
| 200002 | 15 | 4.46e-18 |
| 200017 | 12 | 2.25e-8 |
| 200034 | 1 | 1.54e-52 |
| 200041 | 14 | 1.79e-14 |
| 200047 | 13 | 3.90e-17 |
| 200051 | 16 | 1.62e-5 |
| 230053 | 14 | 1.28e-4 |

This separates two demonstrated problems. First, source 230053 was excluded
solely by `global_excess`; its other five gates passed. Its two actual-H TVs
were 0.6256 and 0.5804, and positive interaction reached 0.4682. Because the
bounded risk already permits a single-actor increment of 0.7542 (reaching
risk 1), a joint increment cannot exceed that ceiling; adding the other
single increment makes the global-excess contrast -0.5456. Demanding that
contrast as a condition for *collision-probability training access* rejects
an actual collision-capable state with useful local interaction. This does
not by itself invalidate a research gate for a narrower cooperative claim.

Second, multiple unobserved event sources are physically exercisable but their
forced actions have extremely small P. Multiplying a bounded risk/interaction
score by P can make them too rare to label in a testing-oriented Q budget.
This is not necessarily poor natural-risk allocation: those individual actions
may have negligible natural risk contribution. A raw event-source count can
therefore be increased without solving IS concentration. Finite negative probes
also leave native stochasticity and untested actions unresolved.

The opt-in [action teacher protocol](../configs/highd_v47_action_teacher_v1.json)
repairs **training access and action exploration**, not production sampling.
Its case law is the deduplicated union of all 31 original eligible cases and
all 11 diagnostic candidates, including every negative. It does not select
only the seven forced-positive sources. Source mass is uniform and states
within source split that mass; this is a new conditional training law.

Teacher H first chooses uniformly among nonempty supported risk bins
`[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]`, then uniformly among supported joint
action cells in that bin. It no longer multiplies its exploration score by P.
The unchanged epsilon=0.1 factor sampler supplies exact P/Q and correlated
partners, retaining Q>=0.01P. Each case gets 64 fresh draws, with case-indexed
seeds starting 4600000. All matrices, states and action/outcome labels are
saved, so later calibration can explicitly choose its fitting measure; pooling
uncorrected teacher event frequencies as natural probabilities is forbidden.
For pointwise f(s,a), labels are conditional native Bernoulli observations;
finite-feature calibration under natural actions additionally requires the
appropriate P/Q and source/state design weighting. Forced probes are excluded.

This exploratory collector is not an efficiency baseline, does not replace
the failed V47 model, and does not export PPO sequences. It intentionally
measures event-action P alongside source counts to prevent counting negligible
natural-risk witnesses as a variance success. New model fitting and source-held-
out checking are still necessary after label collection.

```bash
python -m scenario_reconstruction.highd_v47_action_teacher freeze --output outputs/v47_action_teacher_v1_lock.json
python -m scenario_reconstruction.highd_v47_action_teacher run --lock outputs/v47_action_teacher_v1_lock.json --output outputs/v47_action_teacher_v1_train
python -m pytest tests/test_highd_v47_action_teacher.py tests/test_highd_v47_event_source_diagnosis.py tests/test_highd_v46_cross_unit.py tests/test_highd_v46_multistate_q.py -q
```

## Completed teacher evidence

The teacher design was committed/frozen on `839e162` before its new outcomes.
All 33 cases from 25 training sources completed 64 draws each (2,112 total)
in 118.79 s. Original negative cases were not dropped; two extra cases came
from the declared whole-bank risk selection, not native event labels. This
adds only one source to the old 24-source selected-case union: 230053 already
had other, non-event states in that union, whereas 230013 was absent.

| Source | Teacher CAV events | Share of teacher conditional M2 |
| --- | ---: | ---: |
| 200002 | 9 | 1.29e-21 |
| 200017 | 10 | 2.00e-11 |
| 200020 | 1 | 4.36e-7 |
| 200041 | 5 | 3.44e-26 |
| 200047 | 5 | 2.40e-28 |
| 200051 | 21 | 0.995442 |
| 230053 | 16 | 0.00455718 |

Shares above are fractions, not percentages. All other source rows, with zero
observed events, remain in the conditional target and artifacts. There are 67
sampled events across **seven** source episodes, without forcing an adverse
action. These are labels from existing independent traffic episodes, not seven
newly generated source episodes or an independent validation cohort.

The improvement is in action-conditioned label coverage, not natural-risk
variance. Source 200051 still supplies **99.5442%** of empirical M2. At sources
200002/200041/200047, even the largest natural probability of a sampled event
action is only 3.48e-14/1.80e-16/3.90e-17. Giving their labels equal raw counting
weight cannot be presented as balanced natural-risk contributions. Source
200020's event action instead has P=5.55e-6; source 230053 has event actions up
to P=2.93e-5. The teacher also samples a different, much lower-P event region
at source 200017 than the old Q. No point-count source guard automatically
promotes this teacher to a testing proposal or repairs the failed bin model.

Teacher conditional IS mean is 9.59139e-5 and M2 2.71005e-6 under its explicit
25-source/33-state law. They are not a same-target improvement comparison with
the old 24-source/31-state bank. The new data can support action-conditional
model development, but cross-source calibration, meaningful natural-risk
coverage and an independent evaluation remain unresolved.

Independent read-only reconstruction checked all 46 diagnostic and 68 teacher
artifact hashes, action distributions, seeds, labels, case masses and moment
calculations. Maximum reconstructed likelihood-log errors were 8.88e-16 and
1.42e-14. Seven protected earlier artifacts remained unchanged; all 36 relevant
tests and frozen-code/assets verification passed. No forced diagnostic label
was included in the teacher data. Compact evidence:
[highd_v47_event_source_repair_v1.json](../results/highd_v47_event_source_repair_v1.json).
