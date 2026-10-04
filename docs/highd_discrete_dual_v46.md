# V46: align dual labels to one-second execution

The executor holds acceleration for one second. Previous dual fits sampled
instantaneous highD accelerations at 1 Hz. V44-V46 instead fit pair dependence
using `(abs(v(t+1s))-abs(v(t)))/1s` on synchronized source rows. The individual
V34 action distributions are retained; this changes only their rank coupling.

Both source actors must pass the existing current-state and three-second past
history filters, have an observed endpoint at frame+25, and remain quiet in
lane/leader history over that interval. Tracks without endpoints and labels
outside [-4,2] are excluded. Future endpoints are offline labels, never runtime
features. Fully observed stable-interval selection is a coverage limitation,
not proof of an unbiased population law. The original ten train and ten
calibration recordings are used; validation/test recordings are untouched here.

V45 fits speed bins below 27, 27-33 and at least 33 m/s within the original
twelve contexts. Calibration chooses a convex mixture of each fine law and
its coarse parent; sparse cells use the parent. V46 chooses the smallest fine
fraction within one recording-level standard error of the best calibration
gain. Five of the 36 cells retain a nonzero speed refinement; all others use
the interval-aligned coarse parent. This limits unstable fine-cell fitting.

Runtime keeps V38 supported, continuity-preserving disjoint pairing. A cell
whose calibration selected pure independence no longer reserves actors either.
Every vehicle samples once. Single marginals, the explicit V34 feasibility
rule, one-second LC at zero longitudinal acceleration and 15 Hz physics are
unchanged. No sampled action is rewritten and no model is selected per seed.

## Results and selection

All rows use fixed seeds 7/19/29, the original fleets and reference bins. Score
only when all three collision screens pass. All failures remain on disk.

| Version | Change | Completed | H(speed) | H(gap) | H(rr) |
| --- | --- | --- | --- | --- | --- |
| V34 | Accepted single | 3/3 | 0.16305 | 0.21005 | 0.12486 |
| V38 | Previous selected dual | 3/3 | 0.16400 | 0.22306 | 0.11069 |
| V40 | Recompute nearest matching without continuity | 2/3 | Not scored | Not scored | Not scored |
| V41 | Exact matching maximizing pair acceleration covariance | 2/3 | Not scored | Not scored | Not scored |
| V42 | Instantaneous labels, separate speed contexts | 2/3 | Not scored | Not scored | Not scored |
| V43 | Instantaneous labels, hierarchical speed refinement | 3/3 | 0.16461 | 0.22835 | 0.11381 |
| V44 | One-second labels, coarse contexts | 3/3 | 0.15864 | 0.21892 | 0.11654 |
| **V45** | **One-second labels, hierarchical speed refinement** | **3/3** | **0.16633** | **0.20696** | **0.11576** |
| **V46** | **V45 with one-standard-error calibration** | **3/3** | **0.15716** | **0.21796** | **0.11598** |

V46 is the current balanced development candidate. V45 is retained as the
gap-priority alternative. V46 reduces speed/gap H by about 4.2%/2.3% versus
V38, while rr H worsens. V45 has lower gap H than even V34 but worse speed H.
Neither candidate dominates all metrics. The earlier 0.147 speed / 0.197 gap
targets remain unmet. Reusing the fixed development seeds for seven candidates
does not establish independent improvement or statistical significance.

V46 completes every run for 60 seconds without collision, makes 7/7/6 lane
changes and records 749/653/700 correlated pair decisions, at most 16/15/15
pairs simultaneously. Maximum marginal error is below 3.9e-16. No extra
collisions appear in scoring replay. The independent control exactly matches
every V34 action, acceleration and execution diagnostic. Four focused tests
pass, covering log actor IDs, supported matching, weighted-path matching and
forward-interval labels with truncated-track exclusion. No broad suite is run.

## Reproduction

Use `$raw`, `$model`, `$lateral`, `$bank`, and the licensed `$highdSource` from
[V35](highd_discrete_dual_v35.md). Output directories must be new. Fitted models
and licensed source files are external; configuration hashes identify them.

```powershell
$parent = "$raw/highd_paper_discrete_pair_fit_interval_v4_20261003"
$child = "$raw/highd_paper_discrete_pair_fit_interval_speed_v5_20261003"
$pair = "$raw/highd_paper_discrete_pair_fit_interval_1se_v7_20261003"
$output = "$raw/highd_paper_discrete_dual_v46_interval_1se_20261003"
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --source-root $highdSource --one-second-actions --output $parent
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --source-root $highdSource --one-second-actions --speed-context --output $child
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --refine-speed-model $child --parent-pair-model $parent --one-standard-error --output $pair
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --pair-model $pair --initial-bank $bank --supported-pairs-only --output $output
# Only after all three scenes complete without collisions:
python -m scenario_reconstruction.highd_paper_discrete_score --model $model --reference "$raw/highd_paper_native_reference_v2_20261001" --initial-bank $bank --rollout $output --output "$output/fidelity_summary.json"
```

Omit `--one-standard-error` when creating the refined artifact to reproduce
V45 in fresh directories. Optional `--pair-selection nearest` and `covariance`
preserve the failed V40/V41 experiments; neither is selected. CLI defaults
retain V35 compatibility. Exact selected flags, hashes, package versions and
the complete development table are in `configs/highd_paper_discrete_dual_v46.json`
and the retained alternative `configs/highd_paper_discrete_dual_v45.json`.
