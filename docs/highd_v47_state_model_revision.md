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

## Completed training development, 2026-10-06

Implementation/protocol `a58cb64` was committed before freezing this fitting
design. All four declared families completed on the same 56 source/state rows,
with zero new native outcomes. The original 14,336-root measurements and all
negative states were retained. The primary candidate did **not** pass:

| Family | Aggregate nested outer M2 ratio | Batch 0 ratio | Batch 1 ratio |
| --- | ---: | ---: | ---: |
| Original features + log J | 1.09747 | 1.09444 | 1.09998 |
| Extended context + log J | 1.06343 | 1.03887 | 1.08387 |
| Original features + direct M2 | 1.01163 | 1.01790 | 1.00640 |
| Extended context + direct M2, primary | 1.09922 | 1.09829 | 1.09998 |

These are development point diagnostics from already consumed training data,
not confidence statements about true source-population variance. Nested
selection and aggregate M2 differ from the original v1 mean-fold-ratio design;
the original 1.05183 is not a matched control for this new table.

The primary full-training fitted M2 ratio is 0.31268, yet every outer checking
fold has ratio >1. Source 260081 still receives q/alpha=0.909091, while the same
checking fold gives 260214 q/alpha=1.85015 (their measured J are 0.0971331 and
0.000159295). No source identifier enters inference. Optimizing training M2
and adding traffic context therefore did not repair this held-source ordering;
a large in-fit improvement is not a transferable allocation result. The two
batch diagnostics both reject this candidate, without proving a unique causal
explanation for every state's measurement uncertainty.

### Numerical failure is recorded separately from source misranking

Protected water-fill has active-set boundaries; a correct within-active-set
gradient does not make the objective globally smooth. Original-feature direct
M2 had two outer line-search failures and correctly used neutral q=alpha there.
Its near-one aggregate ratio must not be advertised as learning progress: the
largest-J checking fold is one of those neutral fallbacks. Extended-context
direct M2 had no outer optimizer failure, but one outer fold's inner selection
had no fully successful regularization candidate. That fallback and every
failed grid fit remain in the artifacts and also prevent promotion. A solver's
relative-objective convergence flag is not a global optimum certificate.

Thus these data do not establish a clean superiority/rejection theorem for
all direct-M2 learning methods. They demonstrate that **this implemented
linear-score family** fails the declared training review. Both numerical
robustness and transferable high-J ranking need attention before an independent
test. The next development design should use a differentiable protected
allocation parameterization, with explicit out-of-source uncertainty handling,
rather than dropping failed folds or selecting a favorable ablation. Any such
change needs its own version; none was substituted into this completed run.

### Verification and decision

Read-only reconstruction reproduced feature extraction, all nested scalers,
priors, grid fits, failure/fallback statuses and final models exactly. A separate
held-fold calculation reproduced all objective contributions, with maximum
absolute discrepancy 1.73e-18. Three revision artifacts and the original 174
native artifacts were checksum-verified; the inherited independent native audit
remains the evidence for the old 112 root replays. No new native replay or
continuation became fitting data. All 275 V47 regression tests and frozen
code/assets verification passed. The original training summary and seven
protected earlier summaries stayed unchanged.

**Decision:** retain this failed development experiment and alpha as the
operational fallback. Do not open validation 270000–270255, reinterpret the
full-training ratio as generalization, automatically choose another family,
retune on validation or launch PPO. The fixed-bank certificate is not refined;
production, natural target, Q_A and all old manifests remain unchanged.

Compact evidence:
[highd_v47_state_model_revision_v1.json](../results/highd_v47_state_model_revision_v1.json).
Complete summary SHA:
`cfd3effbe02b53fd21f45411fdeae35a7230d22688b167221b52e51aadf04f97`.

## Subsequent user decision, 2026-10-06

Transferable Q_S training is stopped. The new
[fixed-alpha PPO protocol](highd_v47_fixed_alpha_ppo.md) sets Q_S=alpha and
state LR=1, then trains actual Single-BV and Dual-BV policies from new real-Q
critical sequences. Passing this failed state-model screen is no longer a
prerequisite for that separate training experiment. The completed results
above are preserved; no transferable state model is promoted.
