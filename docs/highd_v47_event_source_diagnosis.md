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
