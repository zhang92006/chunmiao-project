# Fixed-alpha Single-BV and Dual-BV PPO

The 2026-10-07 formal comparison declaration classifies these completed six
runs as pipeline smoke evidence. Their original protocol and artifacts remain
intact. The new [matched original-critical-state experiment](highd_v47_formal_comparison.md)
uses independent native validation for checkpoint selection, including iteration
0, before a sealed original-initialization FinalTest. No smoke seed is promoted.
The new trainer instrumentation has a new hash; old frozen run locks remain
historical records and must not be silently repinned to rerun them.

User direction on 2026-10-06 ends transferable Q_S development and starts actual
PPO with `Q_S=alpha`, `state LR=1`, and `state log-LR=0`. This is a new training
protocol; the preceding failed state models and their results remain recorded.

Alpha retains all 56 selected sources/states from training cohort 260000–260255,
uniform over sources. These are the same initial states used by the preceding
state-model work. Old branching measurements are not training trajectories.
The [protocol](../configs/highd_v47_fixed_alpha_ppo_v1.json) collects 4,096 fresh
real-Q native episodes per mode using common IID state draws from alpha and
continuation seeds 100000000–100004095. Every negative and competing BV endpoint
is retained. All state identities, target/proposal masses and the complete
schedule are locked before outcomes. No learned state score is loaded or fitted.

## Two actual agents on a common initial-state law

Single-BV independently selects the highest natural absolute-risk-mass actor
among the nearest eight BVs within 115 m. Dual-BV uses the original strict-B
pair assigned to that initial state. Selected actor identities remain fixed
for the four-second episode. Every second, each agent recomputes supported
action risk and H for its one or two actors, then supplies one or two epsilon
values. This trains multi-decision critical sequences, with up to four PPO
decisions per episode. It is a conditional-state experiment, with a four-second
CAV-first-collision target; the initial state bank is not a 60-second natural
reset population. Its strict-B initial eligibility also limits the population
scope of a Single/Dual comparison.

Both modes use the inherited uniform-risk-bin exploration teacher: uniformly
select a nonempty supported risk bin, then a supported action cell within it.
Single H is one-dimensional; Dual H is joint and factorized by the existing
sampler. The reactive clearance proxy uses the remaining episode horizon.
The 33-action single proxy is checked against the existing pair proxy with one
actor varied. Single is independently selected and scored, not the first-factor
ablation of a preselected pair. Uncontrolled natural partners keep their exact
conditional action law, including when natural unit membership changes later.

The natural physics, CAV dynamics and first-collision endpoint remain the
frozen implementation. A BV-only first collision terminates as a competing
negative endpoint. Only the proposal changes. Full path action likelihood is
the product of every logged decision factor. The state factor is identically
one, so `w=product_t(P_A/Q_A)`. State probability fields, the IID draw schedule,
full/factor likelihood agreement and the new dataset contract are checked
before the data can enter PPO. An old learned-Q_S dataset cannot be relabelled
as fixed-alpha training.

## Training and checkpoint rule

The [PPO configuration](../configs/highd_v47_fixed_alpha_ppo_train_v1.json)
uses 100 iterations, seeds 7/17/29 for each mode, two 64-unit tanh layers,
batch 1,024, minibatch 128, five SGD passes and learning rate 0.0003. The one/two
dimensional bounded Beta epsilon head starts at 0.1 and stays in [0.05,1].
The new 22-feature observation includes the policy's causal action-weight
prefix and remaining time. Missing historical acceleration at the initial
snapshot is explicitly zero; subsequent decisions use available past dynamics.

Training minimizes the complete empirical trajectory second moment
`mean_Qb[Y * (P_A/Qb_A) * (P_A/Qtheta_A)]` with the existing fixed positive
numerical reward scaling. A 50/50 positive/negative replay sampler retains its
exact replay correction. Replay correction is separate from state LR and does
not introduce a learned state proposal. Likelihoods, event contributions and
terminal rewards are not clipped.

Iteration 100 is selected in advance for every run. All training seeds and
curves are retained; there is no validation-based checkpoint/seed search.
This authorizes actual optimization and checkpoint export despite earlier
Q_S failures. Training loss reductions remain training evidence. The reserved
270000–270255 sources remain available for a separately locked native evaluation
of the completed policies; consumed validation and final test are excluded
from this run. Formal training does not imply demonstrated testing acceleration.

## Reproduce

Use Python 3.9.19, NumPy 1.23.1, SciPy 1.13.1, Ray 1.11.0, Torch 1.11.0 and
Gym 0.21.0. Code/configuration and source hashes are frozen before collection.
The complete source artifact audit is checked without refitting a Q_S model.
Outputs are exclusive-create; partial failures retain their logs and records.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scenario_reconstruction.highd_v47_fixed_alpha_ppo freeze --output outputs/v47_fixed_alpha_ppo_v1_frozen
python -m scenario_reconstruction.highd_v47_fixed_alpha_ppo collect --frozen outputs/v47_fixed_alpha_ppo_v1_frozen --output outputs/v47_fixed_alpha_ppo_v1_data
python -m scenario_reconstruction.highd_v47_fixed_alpha_ppo audit --frozen outputs/v47_fixed_alpha_ppo_v1_frozen --data outputs/v47_fixed_alpha_ppo_v1_data --replay --output outputs/v47_fixed_alpha_ppo_v1_audit.json
python -m scenario_reconstruction.highd_v47_fixed_alpha_ppo train --frozen outputs/v47_fixed_alpha_ppo_v1_frozen --data outputs/v47_fixed_alpha_ppo_v1_data --audit outputs/v47_fixed_alpha_ppo_v1_audit.json --output outputs/v47_fixed_alpha_ppo_v1_training
python -m pytest tests/test_highd_v47_fixed_alpha_ppo.py tests/test_highd_v46_conditional_baseline.py tests/test_highd_v46_validation.py tests/test_highd_v46_cross_unit.py -q
python -m scenario_reconstruction.highd_v46_workflow verify
```

The training folder contains six run directories, per-iteration metrics,
RLlib checkpoints, portable policies, exact data/config identities and process
logs. Real native traces and model weights stay in ignored outputs. Commit
code, configuration, tests and compact run evidence only.
