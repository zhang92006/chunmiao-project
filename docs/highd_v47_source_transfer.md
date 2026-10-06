# Fresh-source transfer of state probability allocation

This prospective experiment follows the completed [fixed-bank certificate](highd_v47_allocation_certificate.md),
but does not run, refine or fit that bank again. The old certificate demonstrates
a decision principle, not a source-generalizing model. No new-source result is
claimed here. NDE, initialization, native dynamics, stopped CAV-first-collision
event, teacher action proposal, production and PPO remain unchanged.

## Separate new sources and an explicit conditional target

The [versioned protocol](../configs/highd_v47_source_transfer_v1.json) reserves
**260000–260255 for training** and **270000–270255 for validation**. Earlier
development/validation sources and reserved final-test sources are rejected in
code. Tests use synthetic fixtures and do not open either cohort. Training
collection never opens validation; validation freeze verifies completed
training artifacts and an independent read-only audit with predetermined root
replays, without generating a source episode.

Both cohorts use unchanged mixed5 discovery (two geometry, two uniform, one
phase-aware state) and rank-1/rank-2/audit pair screening, nearest-eight/115 m.
The frozen strict-B rule chooses at most one state/pair per source by maximum
supported proxy risk, with time/actor-ID ties. All sources, abstentions, negative
states and competing BV-only endpoints remain recorded. No source is chosen
using continuation labels. This changes neither production gates nor selector.

Within each new bank, alpha is uniform over selected sources and their one
selected state. It differs from the old 25-source/33-state development law and
is **not the natural reset density**. Comparisons use the same new states,
alpha, teacher Q_A and natural four-second future in both proposals. Old/new
M2 or concentration values are not matched improvement estimates.

## Transferable features instead of a state/source lookup

The model predicts J(s)=E_QA[Y(P_A/Q_A)^2|s]. Its eleven inputs are available
before sampling: log coefficient mass C=sum(P_A^2/Q_A), c-weighted clearance
risk/second risk/high-risk mass, natural-weighted risk/high-risk mass, maximum
supported risk, pair nearness/closing speed/lateral activity/bracketing. Actor
IDs only locate vehicles; IDs, seeds, absolute positions, timestamps and native
outcomes are absent from the feature vector. The clearance score is not treated
as a calibrated collision probability.

Each selected training state receives 128 independent early-branching roots in
each of two independent batches. Top twelve coefficient-heavy cells are
isolated **without outcomes**, and every remaining supported action stays in a
risk-by-action-LR stratum. Roots draw initial actions c/C_h; count allocation
minimizes the coefficient-only worst-case sum(C_h^2/n_h), with two roots/stratum.
P/T prefix branching and terminal conditional integration use the tested exact
density recursion. One tree is one independent root; leaves/cells never enlarge
the sample size. All negatives are retained. At most 65,536 training roots are
possible, shrinking only with selector abstention.

J/C estimates receive deterministic design-effective-root shrinkage toward a
training-only global fraction before equal-source log-fraction ridge fitting.
This regularizes finite pilot zeros; it is not a posterior or certified risk
floor. Four whole-source folds select ridge from 1/10/100 by held-out,
common-measurement M2 ratio. Each checking source is excluded from fitting,
scaling and prior calculation. Full-training fitting follows this training-only
selection. The OOF score is development evidence, not a validation interval.
At least five measured-positive training sources and OOF ratio below one are
required before validation opens. Failure leaves validation unopened rather
than substituting cases or tuning on validation.

## Freeze the mapping; validate its allocation objective

Validation freeze locks the complete training summary/model/artifacts, all
dependencies, feature contract, selected ridge and one training-selected benefit
bet. New validation features determine predictions, q and normalization before
continuation outcomes. There is no source-ID memorization or validation refit.

Water-filling uses predicted J with q>=alpha/1.1. Thus full IS likelihood remains
(alpha/q)(P_A/Q_A), state LR<=1.1, and true uncentered M2(q)<=1.1 M2(alpha)
for any nonnegative J. This is a worst-case M2 protection, **not** a centered
variance-ratio bound, nor a guarantee that this candidate improves.

For d=alpha(alpha/q-1), directly measure Delta=sum d J=loss-benefit. Recipient
states use 128 branching roots/state/batch; donor states use natural futures
with n_h=max(2,ceil(d C_h/2e-5)). Neutral states keep natural controls. A cap of
150,000 natural roots/batch is checked before new continuation outcomes. A cap
failure retains discovery artifacts and stops: do not silently increase it.

Independent-root fixed-bet bounds retain the inherited [betting/Markov principle](https://www.glennshafer.com/assets/downloads/articles/article104_jrss_shafer-testingbybetting-with-discussion-response.pdf)
and tested algebra, not simultaneous relative precision in every tiny J.
Two new batches receive delta=.0125 per side/batch (95% joint conditional-bank
coverage). Both upper bounds on Delta must be negative. Model uncertainty is
handled by prospective objective measurement, not by trusting ridge error bars.
This tests a transferable rule on a new bank; that certificate alone is still
conditional on the realized validation bank, not all unseen source populations.

Separately, 2,000 paired **whole-source** bootstrap replicates resample all
natural sources, retaining each source's complete measured roots and abstention.
Each replicate recomputes q normalization from frozen predictions, never refits
the model. At least five distinct observed positive sources per replicate are
required; more than 10% censored replicates leaves the interval undefined.
Root measurement MCSE is reported separately. These are approximate source
generalization diagnostics, not distribution-free population bounds; duplicating
a source in a bootstrap draw does not create another traffic episode.

Validation review requires both objective certificates and a supported
source-bootstrap M2-ratio upper endpoint below one. Source contributions,
effective M2 sources, real-IID event sources and factor TVs must also be reviewed;
no model is automatically promoted and PPO readiness remains false. The two
2,112-draw real Q_S/Q_A arms run regardless of certificate outcome, with complete
state/action likelihoods and outcome-blind CRN reuse. Their empirical variance
ordering is secondary, not a substitute for the common-measurement objective.

## Reproduction and handoff

Use Python 3.9.19 / NumPy 1.23.1 / SciPy 1.13.1 and one BLAS/OMP thread. Review
and commit implementation/configuration before freezing any new-source run.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_source_transfer freeze-train --output outputs/v47_source_transfer_v1_train_lock.json
python -m scenario_reconstruction.highd_v47_source_transfer train --lock outputs/v47_source_transfer_v1_train_lock.json --output outputs/v47_source_transfer_v1_train
python -m scenario_reconstruction.highd_v47_source_transfer_audit --run outputs/v47_source_transfer_v1_train --replay --output outputs/v47_source_transfer_v1_train_audit.json
python -m scenario_reconstruction.highd_v47_source_transfer freeze-validation --training outputs/v47_source_transfer_v1_train --training-audit outputs/v47_source_transfer_v1_train_audit.json --output outputs/v47_source_transfer_v1_validation_lock.json
python -m scenario_reconstruction.highd_v47_source_transfer validate --lock outputs/v47_source_transfer_v1_validation_lock.json --output outputs/v47_source_transfer_v1_validation
python -m scenario_reconstruction.highd_v47_source_transfer_audit --run outputs/v47_source_transfer_v1_validation --replay --output outputs/v47_source_transfer_v1_validation_audit.json
python -m pytest tests/test_highd_v47_state_features.py tests/test_highd_v47_source_transfer.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Destinations are exclusive-create. Interrupted artifacts remain separate, with
no event-conditioned extension, selective restart, old/new pooling or favorable
repeat. Raw state banks, matrices, fitted models and trees stay ignored. A changed
feature/model/dependency/protocol requires explicit versioning and a new lock;
completed validation cannot become fitting data for a claimed fresh evaluation.
No PPO checkpoint or sequence manifest is exported. Passing this transfer test
would justify a separately specified training law and independent original-
initialization evaluation, not an automatic formal D2RL training launch.

Implementation checks on 2026-10-06 passed 284 relevant regression tests and
frozen-code/asset verification. These checks used synthetic source fixtures;
they did not open either new natural source cohort or establish transfer benefit.

## Launch record, 2026-10-06

Implementation/protocol `d431b24` was committed before freezing and opening new
training sources. The complete 256-source training collection has started at
`outputs/v47_source_transfer_v1_train`, using the new 50/60-million-series root
seeds. It is not completed evidence: no fitted model, OOF benefit, transfer
interval or native improvement result is yet reported. Validation 270000–270255
remains unopened; its model lock cannot be created until training and the
independent audit pass the declared checks. No PPO run is launched.

The compact [historical lock/launch record](../results/highd_v47_source_transfer_v1_rule_lock.json)
records this status. The local freeze lock is
`outputs/v47_source_transfer_v1_train_lock.json`. Old fixed-bank/measurement and
consumed validation summaries remain unchanged. The task is on
`experiment/v47-source-transfer-allocation`; local main was refreshed read-only.
Publication is still blocked: local HTTPS Git lacks authentication and the
connected GitHub integration reports repository `push=false`. No remote branch
or PR has been created, and no merge is claimed.
