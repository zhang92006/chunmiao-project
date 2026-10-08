# V38: check support before reserving a pair

V35 forms disjoint following pairs before checking whether both actors have
direct empirical state support and three seconds of stable leader/lane history.
An old pair that must fall back to independence can therefore reserve a vehicle
and prevent its supported neighboring relation from being paired.

V38 applies those existing eligibility checks before matching. For example,
when (0, 1) has no direct support but (1, 2) does, the old pair releases vehicle 1
so (1, 2) can receive the fitted joint law. Every actor still draws exactly one
action; all V34 single marginals, LC rules, action support and fitted V35 weights
are unchanged. This changes which eligible relation receives dependence; it
does not insert emergency actions or change individual action probabilities.

## Collision-first development

All variants use the same initial fleets, seeds 7/19/29 and 60 s duration. Only
variants completing all three collision screens were distribution-scored.
Full-exposure Hellinger distances use the unchanged reference and 10 Hz bins.

| Version | Change | Completed | H(speed) | H(gap) | H(rr) |
| --- | --- | --- | --- | --- | --- |
| V35 | Original dual | 3/3 | 0.16852 | 0.23607 | 0.10974 |
| V36 | Global dependence scale 0.5 | 2/3 | Not scored | Not scored | Not scored |
| V37 | Dependence tapers beyond 30 m to zero at 115 m | 3/3 | 0.17018 | 0.23621 | 0.11273 |
| **V38** | **Support checked before matching** | **3/3** | **0.16400** | **0.22306** | **0.11069** |
| V39 | V38 with dependence scale 0.5 | 3/3 | 0.16127 | 0.23861 | 0.12563 |

V36 seed 19 collides at 33.667 s after a lane change, while both actors use
independent backoff; its output is preserved. Uniform shrinkage therefore
cannot be treated as a safety improvement. V38 is selected for the better
speed/gap tradeoff, with a small rr regression. The selected version has
724/632/743 correlated pair decisions, at most 16/15/16 simultaneous pairs,
and 7/8/7 lane changes. Maximum marginal error is 3.34e-16. No extra collisions
were observed in the scoring replay. Turning dependence off exactly matches
every V34 action/acceleration record and all execution diagnostics in all seeds.

The two focused regression tests pass: reordered pair IDs cannot corrupt
acceleration logs, and an unsupported retained pair cannot block a supported
neighbor. No broad test or probability-audit batch was run. These reused
development seeds establish an observed improvement, not independent validation
or statistical significance. V38 still has worse gap H than V34 (0.21005) and
does not establish general safety, unrestricted highD fidelity or stationarity.

## Run the selected configuration

Use the artifact variables in [the V35 instructions](highd_discrete_dual_v35.md).
No refit or highD read is needed. Output directories must be new.

```powershell
$output = "$raw/highd_paper_discrete_dual_v38_supported_20261003"
python -m scenario_reconstruction.highd_paper_discrete_pairs --model $model --lateral-model $lateral --pair-model $pair --initial-bank $bank --supported-pairs-only --output $output
# Only after all three collision screens pass:
python -m scenario_reconstruction.highd_paper_discrete_score --model $model --reference "$raw/highd_paper_native_reference_v2_20261001" --initial-bank $bank --rollout $output --output "$output/fidelity_summary.json"
```

`configs/highd_paper_discrete_dual_v38.json` records the selected flags, hashes,
environment and all four development results. CLI defaults still reproduce
V35; `--supported-pairs-only` selects V38. Diagnostic options
`--dependence-scale 0.5` and `--gap-taper` reproduce V36/V39 and V37 respectively;
neither is enabled in the selected V38 configuration. Full old outputs remain
under the ignored raw-data directory. Remote publication remains pending the
earlier explicit authorization request; only local changes are saved here.
