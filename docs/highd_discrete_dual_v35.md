# V35: dual following on the accepted V34 baseline

The user accepted V34 as the single-vehicle development baseline and requested
this dual extension. Each second, adjacent follower/leader candidates form a
deterministic disjoint matching; multiple pairs can act at once. Each actor is
sampled exactly once. A 33 x 33 joint preserves both V34 single-action marginals;
only its both-stay 31 x 31 acceleration block has fitted dependence. Lane-change
probabilities, one-second duration, zero LC acceleration, pre-draw feasibility
and 15 Hz execution stay the same. Lane-change residual dependence is not fitted.

The six rank-kernel weights were freshly fitted at 1 Hz on the existing ten train
recordings, with equal recording weight and shrinkage selected on ten calibration
recordings. Only directly supported states with three seconds of causal stable
leader/lane history receive dependence. Other pairs use independence. No old
10 Hz weights, validation/test data, new safety mask or sampled-action override
was used. The explicit V34 support restrictions remain part of this baseline.

## Collision-first result

Same initial fleets, seeds 7/19/29, 60 seconds each:

| Quantity | Single V34 | Dual V35 |
| --- | --- | --- |
| Completed without collision | 3/3 | 3/3 |
| Lane changes per run | 10 / 10 / 8 | 5 / 6 / 7 |
| Speed Hellinger | 0.16305 | 0.16852 |
| Gap Hellinger | 0.21005 | 0.23607 |
| Relative-speed Hellinger | 0.12486 | 0.10974 |

Dual runs contain 660/602/669 correlated pair decisions, with at most 17/17/16
simultaneous pairs. Pair membership changes as adjacency changes. Maximum
marginal discrepancy is 3.34e-16. Disabling dependence exactly reproduces all
V34 per-second action and acceleration records and execution diagnostics.
Thus the dual mechanism works and passes this collision screen; it does not
improve every distribution. These fixed development seeds do not establish
general safety, stationarity, long-run acceptance or D2RL readiness.

Distribution scoring was performed only after the collision screen passed.
The initial V35 output has acceleration-log IDs incorrectly zipped to pair
insertion order. Physical execution was correct, but that output's replay
scores are invalid. The fix keys held accelerations by vehicle ID. Preserve
the original directory for diagnosis; use only the verified directory below.
Three focused checks passed: the log-order regression, full joint marginals
and LC blocks, and independence/zero-support handling. No broad suite was run.

## Reproduction

See `configs/highd_paper_discrete_dual_v35.json` for seeds, hashes, environment
versions and exact results. Runtime artifacts live under the ignored
`data_analysis/raw_data` directory; licensed highD data and fitted model files
are external and are not committed. The empirical and lateral artifacts must
match the configuration's hashes. Commands below assume those V34 artifacts
and fixed initial fleets are already available. Set `$highdSource` to the
licensed highD root containing `data/`; use fresh output directories on rerun.

```powershell
$raw = 'data_analysis/raw_data'
$model = "$raw/highd_paper_native_train_v1_20261001"
$lateral = "$raw/highd_paper_discrete_lateral_v1_20261003"
$bank = "$raw/highd_paper_native_matched_v2_20261001/initial_fleets"
$pair = "$raw/highd_paper_discrete_pair_fit_v1_20261003"
$output = "$raw/highd_paper_discrete_dual_v35_verified_20261003"
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --source-root $highdSource --output $pair
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --pair-model $pair --initial-bank $bank --mode fitted --output "$output/fitted"
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --pair-model $pair --initial-bank $bank --mode independent_control --output "$output/independent_control"
# Only after all declared scenes complete without collisions:
python -m scenario_reconstruction.highd_paper_discrete_score --model $model --reference "$raw/highd_paper_native_reference_v2_20261001" --initial-bank $bank --rollout "$output/fitted" --output "$output/fitted/fidelity_summary.json"
```
