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

## Evidence, 2026-10-04

The protocol/source were committed as `1958ea1` after development and before
opening validation. Development retained two eligible cases (160009 at 19 s,
160013 at 53 s) and two abstention controls. All 2,048 continuations had zero
CAV events. There was no native-event-driven baseline choice or rule change.

Fresh natural discovery scanned 1,823 decisions from 32 source episodes:
29 reached the horizon and three stopped at BV-only first collisions; none
had a CAV first collision. There were five candidate states per episode,
160 states and 472 evaluated pairs (at most three per state). Twelve pairs
in eleven states from ten source episodes were strict-positive: state rate
6.875%, pair rate 2.542%, and source coverage 31.25%. These rates are not a
paired improvement estimate against the previous, different validation cohort.

The fixed cap selected six eligible source episodes and two controls. All
8,192 real-Q continuations were retained. Eligible-only counts below have
1,536 continuations per arm; each case has 256 per arm.

| Source / time / pair | Mechanism | Natural / first / second / dual CAV events |
| --- | --- | --- |
| 190003 / 18 s / [27,29] | bracket | 0 / 0 / 0 / 0 |
| 190015 / 32 s / [4,24] | bracket | 0 / 0 / 0 / 0 |
| 190016 / 27 s / [18,4] | lateral | 0 / 1 / 0 / 20 |
| 190021 / 27 s / [17,19] | bracket | 0 / 0 / 0 / 0 |
| 190022 / 36 s / [5,21] | lateral | 0 / 2 / 0 / 4 |
| 190027 / 39 s / [8,25] | bracket | 0 / 0 / 0 / 0 |

Controls 190004 at 54 s and 190008 at 29 s produced no events; all four arms
had identical seed-matched actions' likelihood audits, endpoint events and
clearance-risk outcomes, with weight exactly one. The six eligible cases'
second-factor Q-versus-P TV ranged from 0.4727 to 0.5854. The second factor
therefore changes the action law, even at cases with no observed CAV event.

| Eligible-only arm | CAV events | Weighted mean | Event second moment J | 90% RHW |
| --- | ---: | ---: | ---: | ---: |
| Natural | 0 | 0 observed | 0 observed | undefined |
| First factor only | 3 | 5.66635e-5 | 1.77611e-6 | 0.9854 |
| Second factor only | 0 | 0 observed | 0 observed | undefined |
| Dual | 24 | 2.53875e-5 | 1.51097e-7 | 0.6397 |

The dual/first second-moment point ratio is 0.08507. It is descriptive, **not
an 11.75x acceleration or a significant superiority result**. First-only has
three events; dual events occur in only two source episodes, not the required
three; RHW exceeds 0.3. Nine of 200 primary paired-bootstrap replicates had
zero first-only second moment. All comparison guards failed and superiority
intervals remain null. Zero-event arms cannot be selected as variance winners.
All three IS arms target the same conditional risk; differences in their mean
point estimates are not causal risk differences. The all-case secondary
distribution retains controls and also fails the guards.

Dual event-contribution ESS is 6.55, with the largest event contributing 36.7%
of the estimate. At 190022, one of four dual events contributes 78.5% within
that state (event ESS 1.58). At 190016, the all-draw weight ESS is only 1.07,
while event-contribution ESS is 19.73: all-draw weight ESS alone is not the
variance of the collision estimator. Its mean weight is 0.4033 with MCSE
0.3906 and maximum weight 100. Exact likelihood accounting remains intact;
finite-run normalization deviations are not corrected by self-normalization.

Source discovery took 215.89 s (proxy CPU 192.70 s); continuation wall totals
were 90.02 / 92.54 / 89.05 / 91.61 s in the four-arm order above. The complete
validation took 582.89 s. These are local diagnostics, not controlled timing
or original-initialization efficiency estimates. Four eligible cases had no
observed CAV event despite proxy positivity; at 190015, natural continuations
ended in BV-only collisions 241/256 times (dual 129/256). Competing endpoints
remain negative rather than being relabelled as CAV failures.

All artifact checksums and seed schedules matched. Recalculation from the raw
files reproduced the stratified estimates exactly. Full-versus-two-factor
log-likelihood error was at most 1.49e-14. All 97 relevant regression tests
passed and frozen code/assets verified. Compact evidence is
[highd_v46_multistate_q_pilot_v1.json](../results/highd_v46_multistate_q_pilot_v1.json).

The experiment strengthens **local, fresh-state** dual-mechanism evidence,
but does not establish efficient population testing or a trainable general
PPO advantage. The next useful step is a separately declared training-only
state pool and bounded native real-Q calibration: rank states/pairs for CAV
first-event exercise and estimated event-moment gain per CPU, preserving
negative and BV-only cases. Clearance interaction positivity alone is not
that objective. The two observed lateral successes motivate a training-side
mechanism-coverage hypothesis, not a validation-tuned lateral-only rule.
Validation 190000–190031 is now consumed and must not supply fitting labels,
threshold tuning or repeat validation. A subsequent screen needs another
source-episode-disjoint validation, and any policy still needs independent
original-initialization evaluation. No PPO or production selector was changed.
