"""Build a conservative template-stratified rollout allocation from audited pilots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev


def _outcome_path(root: Path, episode_id) -> Path:
    paths = [
        root / folder / f"{episode_id}.json"
        for folder in ("crash", "tested_and_safe", "rejected")
    ]
    existing = [path for path in paths if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"Episode {episode_id} in {root} has {len(existing)} outcome files"
        )
    return existing[0]


def _load_rows(name: str, root: str | Path) -> list[dict]:
    root = Path(root)
    audit = json.loads((root / "closed_loop_audit.json").read_text(encoding="utf-8"))
    if not audit.get("audit_passed"):
        raise ValueError(f"Closed-loop audit did not pass: {root}")
    run = json.loads((root / "manifest_run_summary.json").read_text(encoding="utf-8"))
    rows = []
    for result in run["results"]:
        if result["status"] != "ok":
            raise ValueError(f"Experiment {name} contains a failed rollout")
        episode = json.loads(
            _outcome_path(root, result["episode"]).read_text(encoding="utf-8")
        )
        cav_crash = bool(
            episode.get("collision_result")
            and "CAV" in (episode.get("collision_id") or [])
        )
        log_contribution = (
            float(episode["log_importance_weight"]) if cav_crash else -math.inf
        )
        rows.append({
            "experiment": name,
            "episode_id": result["episode"],
            "template": str(result["template"]),
            "source_event_id": str(
                episode.get("scenario_metadata", {}).get("source_event_id", "unknown")
            ),
            "cav_crash": cav_crash,
            "log_contribution": log_contribution,
            "contribution": 0.0 if not cav_crash else math.exp(log_contribution),
        })
    if len(rows) != int(audit["complete_episode_count"]):
        raise ValueError(f"Experiment {name} row count disagrees with its audit")
    return rows


def _integer_allocation(scores: dict[str, float], budget: int) -> dict[str, int]:
    total = sum(scores.values())
    if budget == 0:
        return {key: 0 for key in scores}
    if total <= 0.0:
        scores = {key: 1.0 for key in scores}
        total = float(len(scores))
    exact = {key: budget * value / total for key, value in scores.items()}
    allocation = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = budget - sum(allocation.values())
    order = sorted(
        scores,
        key=lambda key: (exact[key] - allocation[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def build_allocation(
    experiments: dict[str, str | Path],
    additional_budget: int,
    minimum_per_template: int = 5,
    variance_floor_fraction: float = 0.01,
) -> dict:
    if not experiments:
        raise ValueError("At least one audited experiment is required")
    if additional_budget < 1 or minimum_per_template < 0:
        raise ValueError("Budgets must be positive and minimum_per_template non-negative")
    if not 0.0 < variance_floor_fraction <= 1.0:
        raise ValueError("variance_floor_fraction must lie in (0, 1]")
    rows = []
    template_sets = []
    for name, root in experiments.items():
        loaded = _load_rows(name, root)
        rows.extend(loaded)
        template_sets.append({row["template"] for row in loaded})
    if any(values != template_sets[0] for values in template_sets[1:]):
        raise ValueError("Experiments must cover the same template set")
    templates = sorted(template_sets[0])
    required_floor = minimum_per_template * len(templates)
    if required_floor > additional_budget:
        raise ValueError("additional_budget is smaller than the exploration floor")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["template"]].append(row)
    summaries = {}
    for template in templates:
        values = [row["contribution"] for row in grouped[template]]
        sources = sorted({row["source_event_id"] for row in grouped[template]})
        if len(sources) != 1:
            raise ValueError(f"Template {template} maps to multiple source events")
        summaries[template] = {
            "source_event_id": sources[0],
            "pilot_sample_count": len(values),
            "pilot_crash_count": sum(row["cav_crash"] for row in grouped[template]),
            "pilot_raw_collision_rate": sum(row["cav_crash"] for row in grouped[template]) / len(values),
            "pilot_weighted_contribution_mean": mean(values),
            "pilot_weighted_contribution_std": stdev(values) if len(values) > 1 else 0.0,
        }
    maximum_std = max(
        item["pilot_weighted_contribution_std"] for item in summaries.values()
    )
    variance_floor = maximum_std * variance_floor_fraction
    scores = {
        template: max(item["pilot_weighted_contribution_std"], variance_floor)
        for template, item in summaries.items()
    }
    remainder = additional_budget - required_floor
    extra = _integer_allocation(scores, remainder)
    for template, item in summaries.items():
        item["allocation_score"] = scores[template]
        item["additional_rollouts"] = minimum_per_template + extra[template]

    return {
        "schema_version": 1,
        "scope": (
            "Planning allocation from validation pilots only; not a final risk estimate "
            "or permission to inspect the locked test split."
        ),
        "experiment_names": list(experiments),
        "template_count": len(templates),
        "pilot_episode_count": len(rows),
        "additional_budget": additional_budget,
        "minimum_per_template": minimum_per_template,
        "variance_floor_fraction": variance_floor_fraction,
        "equal_template_mixture_weighted_mean": mean(
            item["pilot_weighted_contribution_mean"] for item in summaries.values()
        ),
        "allocation_total": sum(
            item["additional_rollouts"] for item in summaries.values()
        ),
        "templates": summaries,
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
    parser.add_argument("--experiment", action="append", required=True, type=_parse_experiment)
    parser.add_argument("--additional_budget", required=True, type=int)
    parser.add_argument("--minimum_per_template", type=int, default=5)
    parser.add_argument("--variance_floor_fraction", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiments = {}
    for name, path in args.experiment:
        if name in experiments:
            parser.error(f"Duplicate experiment name: {name}")
        experiments[name] = path
    result = build_allocation(
        experiments,
        additional_budget=args.additional_budget,
        minimum_per_template=args.minimum_per_template,
        variance_floor_fraction=args.variance_floor_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
