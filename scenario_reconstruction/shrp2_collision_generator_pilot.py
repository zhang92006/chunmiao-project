"""Build a fixed, auditable SHRP2 pilot for MultiBV collision generation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable


def build_collision_generator_pilot(
    bridge_summary_path: str | Path,
    rollout_root: str | Path,
    output_dir: str | Path,
    pilot_count: int = 10,
) -> dict:
    """Aggregate prior rollouts and select a deterministic train-only pilot.

    Existing collisions are positive controls. Remaining places are filled from
    collision-free templates using source TTC, context blocking, and conflict
    diversity. The produced manifest is directly consumable by
    ``run_template_manifest``.
    """
    if pilot_count < 1:
        raise ValueError("pilot_count must be positive")
    bridge_summary_path = Path(bridge_summary_path)
    rollout_root = Path(rollout_root)
    output_dir = Path(output_dir)
    with bridge_summary_path.open("r", encoding="utf-8") as stream:
        bridge = json.load(stream)

    templates = {}
    for record in bridge.get("records", []):
        if record.get("status") != "template_created" or record.get("split") != "train":
            continue
        template_path = Path(record["template_path"])
        with template_path.open("r", encoding="utf-8") as stream:
            template = json.load(stream)
        metadata = template.get("bridge_metadata", {})
        event_id = int(metadata.get("source_event_id", record["event_id"]))
        adaptive = metadata.get("source_adaptive_critical_window", {})
        templates[event_id] = {
            "source_event_id": event_id,
            "split": "train",
            "category": metadata.get("source_category", record.get("category")),
            "conflict": metadata.get("source_conflict", "unknown"),
            "template_path": str(template_path),
            "initial_primary_ttc_s": _finite_or_none(adaptive.get("primary_ttc_s")),
            "initial_primary_gap_m": _finite_or_none(adaptive.get("primary_gap_m")),
            "context_blocking_potential": bool(
                adaptive.get("context_blocking_potential", False)
            ),
        }

    aggregate = defaultdict(lambda: {
        "rollout_count": 0,
        "crash_count": 0,
        "safe_count": 0,
        "minimum_observed_ttc_s": None,
        "minimum_observed_distance_m": None,
        "maximum_observed_criticality": None,
        "crash_log_importance_weights": [],
    })
    for outcome in ("crash", "tested_and_safe"):
        for episode_path in sorted((rollout_root / outcome).glob("*.json")):
            with episode_path.open("r", encoding="utf-8") as stream:
                episode = json.load(stream)
            metadata = episode.get("scenario_metadata", {})
            event_id = metadata.get("source_event_id")
            if event_id is None:
                continue
            event_id = int(event_id)
            stats = aggregate[event_id]
            stats["rollout_count"] += 1
            stats["crash_count" if outcome == "crash" else "safe_count"] += 1
            stats["minimum_observed_ttc_s"] = _minimum(
                stats["minimum_observed_ttc_s"], _numbers(episode.get("ttc_step_info", {}))
            )
            stats["minimum_observed_distance_m"] = _minimum(
                stats["minimum_observed_distance_m"],
                _numbers(episode.get("distance_step_info", {})),
            )
            stats["maximum_observed_criticality"] = _maximum(
                stats["maximum_observed_criticality"],
                _numbers(episode.get("criticality_step_info", {})),
            )
            if outcome == "crash":
                value = _finite_or_none(episode.get("log_importance_weight"))
                if value is not None:
                    stats["crash_log_importance_weights"].append(value)

    diagnostics = []
    for event_id, template in sorted(templates.items()):
        stats = dict(aggregate[event_id])
        rollouts = stats["rollout_count"]
        stats["empirical_collision_rate"] = (
            stats["crash_count"] / rollouts if rollouts else None
        )
        diagnostics.append({**template, **stats})

    selected = select_pilot_records(diagnostics, pilot_count)
    selected_ids = {record["source_event_id"] for record in selected}
    manifest_records = [
        {
            "event_id": event_id,
            "split": "train",
            "category": templates[event_id]["category"],
            "status": "template_created",
            "template_path": templates[event_id]["template_path"],
            "pilot_role": next(
                record["pilot_role"] for record in selected
                if record["source_event_id"] == event_id
            ),
        }
        for event_id in sorted(selected_ids)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "per_template_diagnostics.json"
    manifest_path = output_dir / "collision_generator_pilot_manifest.json"
    summary_path = output_dir / "collision_generator_pilot_summary.json"
    _write_json(diagnostics_path, {"schema_version": 1, "records": diagnostics})
    _write_json(manifest_path, {"schema_version": 1, "records": manifest_records})
    summary = {
        "schema_version": 1,
        "bridge_summary": str(bridge_summary_path),
        "rollout_root": str(rollout_root),
        "train_template_count": len(diagnostics),
        "pilot_count": len(selected),
        "positive_control_count": sum(
            record["pilot_role"] == "positive_control" for record in selected
        ),
        "selected_source_event_ids": [
            record["source_event_id"] for record in selected
        ],
        "selected_records": selected,
        "diagnostics_path": str(diagnostics_path),
        "pilot_manifest_path": str(manifest_path),
    }
    _write_json(summary_path, summary)
    return summary


def select_pilot_records(diagnostics: list[dict], pilot_count: int) -> list[dict]:
    """Choose positive controls plus diverse high-potential negative controls."""
    positives = sorted(
        (record for record in diagnostics if record["crash_count"] > 0),
        key=lambda record: (
            -record["empirical_collision_rate"],
            -record["crash_count"],
            record["source_event_id"],
        ),
    )
    selected = [dict(record, pilot_role="positive_control") for record in positives[:pilot_count]]
    remaining = pilot_count - len(selected)
    if remaining <= 0:
        return selected

    negatives = sorted(
        (record for record in diagnostics if record["crash_count"] == 0),
        key=_negative_potential_key,
    )
    chosen_ids = {record["source_event_id"] for record in selected}
    seen_conflicts = {record["conflict"] for record in selected}
    diverse = []
    for record in negatives:
        if record["conflict"] not in seen_conflicts:
            diverse.append(record)
            seen_conflicts.add(record["conflict"])
        if len(diverse) == remaining:
            break
    for record in negatives:
        if len(diverse) == remaining:
            break
        if record["source_event_id"] not in chosen_ids | {
            item["source_event_id"] for item in diverse
        }:
            diverse.append(record)
    selected.extend(dict(record, pilot_role="search_candidate") for record in diverse)
    return selected


def _negative_potential_key(record: dict) -> tuple:
    ttc = record.get("initial_primary_ttc_s")
    return (
        not record.get("context_blocking_potential", False),
        math.inf if ttc is None else ttc,
        -(record.get("maximum_observed_criticality") or 0.0),
        record["source_event_id"],
    )


def _numbers(value) -> Iterable[float]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numbers(child)
    else:
        number = _finite_or_none(value)
        if number is not None:
            yield number


def _minimum(current: float | None, values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return current
    candidate = min(values)
    return candidate if current is None else min(current, candidate)


def _maximum(current: float | None, values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return current
    candidate = max(values)
    return candidate if current is None else max(current, candidate)


def _finite_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fixed train-only pilot for MultiBV collision search."
    )
    parser.add_argument("--bridge_summary", required=True)
    parser.add_argument("--rollout_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot_count", type=int, default=10)
    args = parser.parse_args()
    result = build_collision_generator_pilot(
        args.bridge_summary, args.rollout_root, args.output, args.pilot_count
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
