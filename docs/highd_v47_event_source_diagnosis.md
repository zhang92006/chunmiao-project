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
