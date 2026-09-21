"""Compare repeated equal-budget uniform and stratified closed-loop validations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


def _outcome(root: Path, episode_id) -> dict:
    paths = [root / folder / f"{episode_id}.json"
             for folder in ("crash", "tested_and_safe", "rejected")]
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Episode {episode_id} in {root} has {len(paths)} outcomes")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def analyze_batch(root: str | Path) -> dict:
    root = Path(root)
    audit = json.loads((root / "closed_loop_audit.json").read_text(encoding="utf-8"))
    run = json.loads((root / "manifest_run_summary.json").read_text(encoding="utf-8"))
    if not audit.get("audit_passed"):
        raise ValueError(f"Audit did not pass: {root}")
    stratified = audit.get("stratified_estimation")
    allocation = run.get("stratified_allocation")
    if not stratified or not allocation:
        raise ValueError(f"Experiment lacks stratified estimation metadata: {root}")
    planned = {str(Path(path)): int(count)
               for path, count in allocation["rollouts_by_template"].items()}
    template_count = len(planned)
    contributions = []
    collision_sources = set()
    source_contributions = defaultdict(float)
    for result in run["results"]:
        if result["status"] != "ok":
            raise ValueError(f"Failed rollout in audited experiment: {root}")
        episode = _outcome(root, result["episode"])
        if not (episode.get("collision_result")
                and "CAV" in (episode.get("collision_id") or [])):
            continue
        template = str(Path(result["template"]))
        coefficient = 1.0 / (template_count * planned[template])
        contribution = coefficient * math.exp(float(episode["log_importance_weight"]))
        contributions.append(contribution)
        source = str(
            episode.get("scenario_metadata", {}).get("source_event_id", "unknown")
        )
        collision_sources.add(source)
        source_contributions[source] += contribution
    total = math.fsum(contributions)
    ordered = sorted(contributions, reverse=True)
    normalized_source_contributions = {
        source: value / total for source, value in sorted(source_contributions.items())
    } if total > 0.0 else {}
    effective_sources = [
        source for source, value in normalized_source_contributions.items()
        if value >= 0.01
    ]
    return {
        "experiment_path": str(root),
        "attempted": int(audit["attempted"]),
        "raw_cav_crashes": int(audit["raw_cav_crashes"]),
        "weighted_mean": float(stratified["weighted_cav_collision_mean"]),
        "estimated_95pct_relative_half_width": (
            None if stratified["estimated_95pct_relative_half_width"] is None
            else float(stratified["estimated_95pct_relative_half_width"])
        ),
        "crash_contribution_ess": float(stratified["crash_contribution_ess"]),
        "largest_normalized_contribution": (
            ordered[0] / total if total > 0.0 else None
        ),
        "top_three_normalized_contribution": (
            math.fsum(ordered[:3]) / total if total > 0.0 else None
        ),
        "collision_source_count": len(collision_sources),
        "collision_sources": sorted(collision_sources),
        "normalized_source_contributions": normalized_source_contributions,
        "effective_contributing_source_count_at_1pct": len(effective_sources),
        "effective_contributing_sources_at_1pct": effective_sources,
    }


def _optional_median(values):
    values = [value for value in values if value is not None]
    return median(values) if values else None


def summarize(arms: dict[str, list[str | Path]]) -> dict:
    if len(arms) < 2 or any(not roots for roots in arms.values()):
        raise ValueError("At least two non-empty arms are required")
    results = {}
    for name, roots in arms.items():
        batches = [analyze_batch(root) for root in roots]
        means = [row["weighted_mean"] for row in batches]
        results[name] = {
            "batch_count": len(batches),
            "total_rollouts": sum(row["attempted"] for row in batches),
            "weighted_mean_across_batches": mean(means),
            "weighted_mean_between_batch_std": stdev(means) if len(means) > 1 else None,
            "weighted_mean_between_batch_relative_std": (
                stdev(means) / mean(means)
                if len(means) > 1 and mean(means) > 0.0 else None
            ),
            "median_within_batch_relative_half_width": _optional_median(
                row["estimated_95pct_relative_half_width"] for row in batches
            ),
            "median_crash_contribution_ess": median(
                row["crash_contribution_ess"] for row in batches
            ),
            "median_largest_normalized_contribution": _optional_median(
                row["largest_normalized_contribution"] for row in batches
            ),
            "minimum_effective_contributing_source_count_at_1pct": min(
                row["effective_contributing_source_count_at_1pct"] for row in batches
            ),
            "batches": batches,
        }
    decision = None
    if "uniform" in results and "stratified" in results:
        uniform = results["uniform"]
        stratified = results["stratified"]

        def ratio(numerator, denominator):
            return (
                numerator / denominator
                if numerator is not None and denominator not in (None, 0.0) else None
            )

        ratios = {
            "weighted_mean": ratio(
                stratified["weighted_mean_across_batches"],
                uniform["weighted_mean_across_batches"],
            ),
            "median_relative_half_width": ratio(
                stratified["median_within_batch_relative_half_width"],
                uniform["median_within_batch_relative_half_width"],
            ),
            "median_crash_contribution_ess": ratio(
                stratified["median_crash_contribution_ess"],
                uniform["median_crash_contribution_ess"],
            ),
            "median_largest_contribution": ratio(
                stratified["median_largest_normalized_contribution"],
                uniform["median_largest_normalized_contribution"],
            ),
            "between_batch_relative_std": ratio(
                stratified["weighted_mean_between_batch_relative_std"],
                uniform["weighted_mean_between_batch_relative_std"],
            ),
        }
        checks = {
            "at_least_three_batches_per_arm": (
                uniform["batch_count"] >= 3 and stratified["batch_count"] >= 3
            ),
            "median_relative_half_width_reduced_by_20pct": (
                ratios["median_relative_half_width"] is not None
                and ratios["median_relative_half_width"] <= 0.8
            ),
            "median_ess_increased_by_25pct": (
                ratios["median_crash_contribution_ess"] is not None
                and ratios["median_crash_contribution_ess"] >= 1.25
            ),
            "median_largest_contribution_reduced_by_20pct": (
                ratios["median_largest_contribution"] is not None
                and ratios["median_largest_contribution"] <= 0.8
            ),
            "between_batch_relative_std_not_increased": (
                ratios["between_batch_relative_std"] is not None
                and ratios["between_batch_relative_std"] <= 1.0
            ),
            "at_least_two_contributing_sources_in_every_stratified_batch": (
                stratified["minimum_effective_contributing_source_count_at_1pct"] >= 2
            ),
        }
        decision = {
            "status": "supports_stratification" if all(checks.values()) else "inconclusive_or_reject",
            "preregistered_ratios_stratified_over_uniform": ratios,
            "checks": checks,
            "interpretation": (
                "A pass supports the frozen allocation on this 14-template validation "
                "mixture only; it does not unlock test or establish real-world risk."
            ),
        }
    return {
        "schema_version": 1,
        "scope": (
            "Descriptive repeated-validation comparison on the frozen 14-template "
            "mixture; not a real-world SHRP2 risk estimate or a test-split result."
        ),
        "arms": results,
        "decision": decision,
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Arm must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Arm must use non-empty NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, type=_parse_arm)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    arms = defaultdict(list)
    for name, path in args.arm:
        arms[name].append(path)
    result = summarize(dict(arms))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
