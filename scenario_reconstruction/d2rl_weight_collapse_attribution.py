"""Attribute closed-loop importance-weight collapse across audited experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median


def _normalized_weights(log_weights: list[float]) -> list[float]:
    if not log_weights:
        return []
    peak = max(log_weights)
    scaled = [math.exp(value - peak) for value in log_weights]
    total = sum(scaled)
    return [value / total for value in scaled]


def _ess(normalized_weights: list[float]) -> float:
    return 1.0 / sum(value * value for value in normalized_weights) if normalized_weights else 0.0


def _episode_id(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def _episode_sort_key(value):
    return (0, value) if isinstance(value, int) else (1, str(value))


def _load_audited_crashes(root: Path) -> list[tuple[Path, dict]]:
    audit_path = root / "closed_loop_audit.json"
    if not audit_path.is_file():
        raise ValueError(f"Missing closed-loop audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("audit_passed"):
        raise ValueError(f"Closed-loop audit did not pass: {audit_path}")
    crashes = []
    for path in sorted((root / "crash").glob("*.json"), key=lambda value: str(value)):
        episode = json.loads(path.read_text(encoding="utf-8"))
        if episode.get("collision_result") and "CAV" in (episode.get("collision_id") or []):
            crashes.append((path, episode))
    if len(crashes) != int(audit.get("raw_cav_crashes", -1)):
        raise ValueError(
            f"Crash count disagrees with audit in {root}: "
            f"{len(crashes)} != {audit.get('raw_cav_crashes')}"
        )
    return crashes


def analyze_experiment(name: str, root: str | Path, top_step_count: int = 10) -> dict:
    root = Path(root)
    crashes = _load_audited_crashes(root)
    log_weights = [float(episode["log_importance_weight"]) for _, episode in crashes]
    normalized = _normalized_weights(log_weights)
    source_rows = defaultdict(list)
    actor_log_totals = defaultdict(float)
    actor_log_penalties = defaultdict(float)
    episode_rows = []
    step_rows = []

    for (path, episode), log_weight, contribution in zip(crashes, log_weights, normalized):
        episode_id = _episode_id(path)
        source = str(episode.get("scenario_metadata", {}).get("source_event_id", "unknown"))
        source_rows[source].append((log_weight, contribution))
        per_episode_steps = []
        online_steps = episode.get("online_policy_step_info", {})
        for time, record in episode.get("log_probability_step_info", {}).items():
            step_log_weight = float(record["log_importance_weight"])
            actor_ratios = {}
            sampled_terms = online_steps.get(time, {}).get("sampled_terms", {})
            for actor, terms in sampled_terms.items():
                p = float(terms["p"])
                q = float(terms["q"])
                if p > 0 and q > 0:
                    actor_log_ratio = math.log(p) - math.log(q)
                    actor_ratios[actor] = actor_log_ratio
                    actor_log_totals[actor] += actor_log_ratio
                    actor_log_penalties[actor] += max(0.0, -actor_log_ratio)
            row = {
                "episode_id": episode_id,
                "source_event_id": source,
                "time": time,
                "log_importance_weight": step_log_weight,
                "actor_log_importance_weights": actor_ratios,
                "online_status": online_steps.get(time, {}).get("status"),
            }
            per_episode_steps.append(row)
            step_rows.append(row)
        worst = min(per_episode_steps, key=lambda row: row["log_importance_weight"], default=None)
        inferred_steps = sum(
            record.get("status") == "inferred" for record in online_steps.values()
        )
        episode_rows.append({
            "episode_id": episode_id,
            "source_event_id": source,
            "log_importance_weight": log_weight,
            "normalized_crash_contribution": contribution,
            "probability_step_count": len(per_episode_steps),
            "inferred_step_count": inferred_steps,
            "worst_step_time": None if worst is None else worst["time"],
            "worst_step_log_importance_weight": (
                None if worst is None else worst["log_importance_weight"]
            ),
        })

    episode_rows.sort(key=lambda row: row["normalized_crash_contribution"], reverse=True)
    step_rows.sort(key=lambda row: row["log_importance_weight"])
    source_summary = {}
    for source, rows in sorted(source_rows.items()):
        source_logs = [row[0] for row in rows]
        within_source = _normalized_weights(source_logs)
        source_summary[source] = {
            "crash_count": len(rows),
            "global_normalized_contribution": sum(row[1] for row in rows),
            "within_source_ess": _ess(within_source),
            "largest_within_source_contribution": max(within_source),
            "maximum_log_importance_weight": max(source_logs),
        }

    sorted_logs = sorted(log_weights, reverse=True)
    total_actor_penalty = sum(actor_log_penalties.values())
    return {
        "name": name,
        "experiment_path": str(root),
        "crash_count": len(crashes),
        "crash_contribution_ess": _ess(normalized),
        "largest_normalized_crash_contribution": max(normalized, default=None),
        "top_two_normalized_crash_contribution": sum(
            row["normalized_crash_contribution"] for row in episode_rows[:2]
        ),
        "log_weight_distribution": {
            "maximum": max(log_weights, default=None),
            "second_maximum": sorted_logs[1] if len(sorted_logs) > 1 else None,
            "median": median(log_weights) if log_weights else None,
            "minimum": min(log_weights, default=None),
            "maximum_minus_second": (
                sorted_logs[0] - sorted_logs[1] if len(sorted_logs) > 1 else None
            ),
            "maximum_minus_median": (
                sorted_logs[0] - median(log_weights) if log_weights else None
            ),
        },
        "source_contributions": source_summary,
        "actor_log_importance_weight_totals_across_crashes": dict(sorted(actor_log_totals.items())),
        "actor_negative_log_penalty_share": {
            actor: value / total_actor_penalty if total_actor_penalty else 0.0
            for actor, value in sorted(actor_log_penalties.items())
        },
        "episodes_by_contribution": episode_rows,
        "most_negative_probability_steps": step_rows[:top_step_count],
    }


def summarize_experiments(experiments: dict[str, str | Path], top_step_count: int = 10) -> dict:
    if not experiments:
        raise ValueError("At least one experiment is required")
    results = {
        name: analyze_experiment(name, path, top_step_count=top_step_count)
        for name, path in experiments.items()
    }
    episode_sets = {
        name: {row["episode_id"] for row in result["episodes_by_contribution"]}
        for name, result in results.items()
    }
    common = set.intersection(*episode_sets.values()) if episode_sets else set()
    union = set.union(*episode_sets.values()) if episode_sets else set()
    return {
        "schema_version": 1,
        "scope": (
            "Attribution of audited conditional closed-loop runs; not a real-world "
            "SHRP2 crash-rate estimate."
        ),
        "experiment_count": len(results),
        "common_crash_episode_ids": sorted(common, key=_episode_sort_key),
        "union_crash_episode_ids": sorted(union, key=_episode_sort_key),
        "experiments": results,
    }


def _parse_experiment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Experiment must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Experiment must use non-empty NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", action="append", required=True, type=_parse_experiment,
        help="Audited closed-loop experiment as NAME=PATH; repeat for multiple seeds.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_step_count", type=int, default=10)
    args = parser.parse_args()
    if args.top_step_count < 1:
        parser.error("--top_step_count must be positive")
    experiments = {}
    for name, path in args.experiment:
        if name in experiments:
            parser.error(f"Duplicate experiment name: {name}")
        experiments[name] = path
    result = summarize_experiments(experiments, top_step_count=args.top_step_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
