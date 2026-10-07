# Training-only revision of the transferable state model

The completed [new-source training experiment](highd_v47_source_transfer.md)
failed its held-source allocation check. Its log-fraction model underallocated
source 260081 in both measurement batches. This revision investigates two
separate hypotheses: a surrogate objective that does not reflect M2 cost, and
insufficient traffic context. It does not reopen the old fixed-bank certificate.
Training labels have already been inspected; this is explicitly adaptive
development, not a prospective independent validation claim.

The [protocol](../configs/highd_v47_state_model_revision_v1.json) pins the complete
256-source training summary, its independent 112-root replay audit and the
56-selected-source conditional target. All states, zeros and tiny-risk positives
remain included. Native measurement, NDE, dynamics, selector, teacher Q_A,
likelihood, production and PPO are unchanged. **No new native outcome is run.**
Validation sources 270000–270255 and final-test sources stay unopened.

## Four declared ablations, one fixed primary candidate

| Features | Log-J ridge surrogate | Direct M2 allocation loss |
| --- | --- | --- |
| Original eleven features | Control | Objective-only ablation |
| Extended pre-outcome context | Context-only ablation | Fixed primary candidate |

The extra 29 features distinguish front/rear bumper clearances, closing speeds,
vehicle chains, local lane-entry/density context and action coefficient mass on
reference rows/columns. They use cached states before continuation outcomes.
They exclude source/actor identities, timestamps, absolute road position,
seeds and outcome labels. Geometry uses front-bumper coordinates and is
translation, actor-renaming and vehicle-order invariant. Original features and
production selection are not silently replaced.

The direct model outputs positive **relative allocation scores**, not calibrated
J or collision probability. For standardized features z and score exp(z beta),
the unchanged water-filling solver gives q>=alpha/1.1. Fit beta by minimizing

`sum(alpha^2 * J_hat / q(beta)) / sum(alpha * J_hat) + lambda*||beta||^2/2`.

This retains every measured J without clipping IS weights or labels. On the
active recipient set, `d log(q_k)/d beta=(z_k-sum_rec(q*z)/sum_rec(q))/2`;
floor donors have zero derivative. Finite-difference tests verify the gradient.
A single deterministic, bounded L-BFGS-B fit starts at beta=0 (q=alpha), with
regularization grid 0.01/0.1/1. A failed fit is explicit and falls back to neutral
scores; it cannot pass the review. Coefficient bounds constrain a working
predictor, not observed likelihood weights. Log-ridge retains its original
1/10/100 grid, prior-root shrinkage and positive prediction floor.

Direct variance-oriented proposal objectives are related to the chi-squared
formulation in [Neural Importance Sampling](https://arxiv.org/abs/1808.03856).
The discrete protected water-fill derivative and traffic design here are our
own implementation, not a claim that that paper validates this experiment.

## Nested whole-source checking

Four outer source folds use the original seed 260017, common to all ablations.
Inside each outer training subset, three source folds select regularization.
Outer labels never enter scaling, prior, fitting or hyperparameter selection.
Selection uses aggregate common-measurement M2, weighted by checking cohort
size, rather than an unweighted mean of fold ratios. The two old branching
batches give separate outer objective diagnostics under the same fitted scores.
Neither tree leaves nor repeated batch observations become new sources.

The primary candidate is fixed to extended context + direct M2 before fitting
this revision. Its development screen requires successful nested fits, at least
five measured-positive sources and aggregate outer M2 ratio <1 in **both**
batches. A better-looking ablation cannot replace it after seeing outcomes.
Full-data hyperparameters are then selected by source CV for a potential future
frozen model; full-data fitting is not the outer evaluation. Scores, source
folds, all grid fits and failures remain recorded. Objective MCSE is conditional
measurement noise, not a source-generalization confidence certificate.

A passing development screen still requires a separately reviewed/versioned
validation adapter, model hash and outcome-blind budget before fresh sources
open. This entry point cannot freeze or run validation, export sequences, change
production or launch PPO. The failed v1 model and its rejection stay intact.

## Reproduction

Review/commit code and protocol before the training-only fit. Use the inherited
Python 3.9.19 / NumPy 1.23.1 / SciPy 1.13.1 environment, one BLAS/OMP thread.
The original 174 native artifacts and independent audit are verified without
new native replays. All destinations are exclusive-create.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_state_model_revision freeze --training outputs/v47_source_transfer_v1_train --training-audit outputs/v47_source_transfer_v1_train_audit.json --output outputs/v47_state_model_revision_v1_lock.json
python -m scenario_reconstruction.highd_v47_state_model_revision run --lock outputs/v47_state_model_revision_v1_lock.json --output outputs/v47_state_model_revision_v1
python -m scenario_reconstruction.highd_v47_state_model_revision audit --directory outputs/v47_state_model_revision_v1 --output outputs/v47_state_model_revision_v1_audit.json
python -m pytest tests/test_highd_v47_state_context.py tests/test_highd_v47_state_objective_model.py tests/test_highd_v47_state_model_revision.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

Raw feature matrices, models and full fitting reports stay in ignored outputs.
Commit only reproducible code/protocol, tests and compact evidence. Old source,
measurement, model and validation artifacts must remain unchanged. There is
no optional native extension or repeat-until-favorable fit.
