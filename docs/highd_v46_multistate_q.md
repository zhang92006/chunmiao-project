# Multi-state real-Q conditional pilot

This experiment builds on `7836c05`. It leaves NDE tables, natural reset,
native dynamics/event definition, production Q, strict gates and PPO unchanged.
It asks whether sampled dual interventions improve the **weighted** conditional
event estimator, rather than merely supplying forced-action collision witnesses.

## Predeclared distribution and arms

The protocol is [highd_v46_multistate_q_v1.json](../configs/highd_v46_multistate_q_v1.json).
The training discovery summary is checksum-locked. Fresh validation uses source
episodes 190000–190031, disjoint from training, consumed validation 170000–170007
and 180000–180031, and reserved final-test episodes 100000–101023.

The previous mixed bank did not beat geometry5 at equal state/pair budget.
This pilot fixes geometry5: five geometry-ranked states per episode, five-second
minimum separation and four seconds remaining; rank 1 + rank 2 + one seeded
audit pair. Only this arm is recomputed on fresh validation; no full pair audit.
Episode-wide state ranking is offline discovery, **not an online causal policy**.

For each source episode with a strict-positive pair, select its largest
natural-probability-weighted interaction-mass pair/state. Cap at six source
episodes by seeded uniform selection. Add two seeded controls from episodes
with no eligible pair anywhere in the bank. At most one case per episode.
All discovery rows and all selected continuations are retained, including
negatives. Selection has no native-event input and there is no outcome-based
replacement or additional sampling. The conditional law is written before
the first continuation: uniform state mass among eligible cases (primary),
and uniform mass among all selected cases including controls (secondary).

The four fixed arms are natural `(1,1)`, first factor only `(0.1,1)`, second
factor only `(1,0.1)`, and dual `(0.1,0.1)`. All use the same pair and actual
interaction-based H. Q samples the initial actions; correlated natural
partners keep their exact conditional P law. All subsequent actions are NDE.
Controls abstain to P in every arm. These ablations are **not** the original
paper's single-BV selector/agent.

There are 128 continuations per case/arm in development and 256 in validation,
with a four-second horizon. Different cases have disjoint continuation seeds;
the four arms share a common-random-number schedule within each case. The
snapshot's initial preparation is not repeated. The event remains CAV
involvement in the first any-actor collision; BV-only first collisions are
negative competing endpoints.

## Estimation and decision safeguards

For `Y=I(CAV event)*Pij/Qij`, report ordinary IS means, second moments,
weight-mean MCSE, maximum weight, weight/event ESS, largest event contribution,
RHW and per-arm CPU/wall time. Never truncate or self-normalize weights.
For uniform state mass `alpha_s`, the primary estimator is
`sum_s alpha_s mean(Y_s)` and `J=sum_s alpha_s mean(Y_s^2)`.
Its fixed-stratum variance estimate is
`sum_s alpha_s^2 sampleVar(Y_s)/n_s`, not a pooled iid standard error that
includes between-state risk differences. Paired bootstrap resamples matched
continuations within each fixed state; it does not estimate source-population
generalization uncertainty.

Predeclared guards require at least 20 observed events and at least three
event-bearing source episodes **in both compared arms** before interval-based
second-moment superiority is reported. Descriptive nonzero-denominator ratios
remain labelled descriptive. Zero-event sample moments are not evidence of
zero risk/variance or a winning baseline. Heavy-weight normal/bootstrap
intervals remain diagnostic even after guards pass. RHW target is 0.3 at 90%
confidence. The fixed budget is not increased based on events.

These estimates target only the declared conditional state law. No initial
state density correction, population sequence manifest, checkpoint or
original-initialization acceleration is exported. Controls assess abstention
behavior, not recall over all unselected road states. CPU timings are local,
not a controlled end-to-end efficiency benchmark.

## Reproduction

The frozen discovery source must first be present with its pinned checksum.
A regenerated discovery summary has different timing bytes; verify all source
artifacts and make a separately versioned protocol rather than bypassing locks.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v46_multistate_q --split train --output outputs/v46_multistate_q_v1_train
# Freeze source/config/dependencies before opening fresh validation.
python -m scenario_reconstruction.highd_v46_multistate_q --split validation --training-lock outputs/v46_multistate_q_v1_train/summary.json --output outputs/v46_multistate_q_v1_validation
python -m pytest tests/test_highd_v46_multistate_q.py tests/test_highd_v46_discovery_v2.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Raw banks, likelihood-audited continuations and locks live in ignored outputs.
Only protocol, implementation, tests, documentation and compact evidence are
committed. If the validation remains sparse or dual does not improve weighted
moments, this pilot does not authorize more PPO epochs, changing NDE/Q/gates,
or tuning on that now-consumed cohort.
