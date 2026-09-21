"""Audit three independent importance-weight records in a D2RL training pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any


def audit_training_pool(
    index_path: str | Path,
    workspace_root: str | Path = ".",
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Cross-check pool index, episode log ledger, and actual step weights."""
    index_path = Path(index_path)
    workspace_root = Path(workspace_root)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_log_weights = payload.get("log_weights", payload)
    failures: list[str] = []
    records: list[dict[str, Any]] = []

    for raw_path, raw_index_log_weight in indexed_log_weights.items():
        episode_path = Path(raw_path)
        if not episode_path.is_absolute():
            episode_path = workspace_root / episode_path
        if not episode_path.is_file():
            failures.append(f"missing episode: {raw_path}")
            continue
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        try:
            index_log_weight = float(raw_index_log_weight)
            episode_audit = validate_episode_probability_ledger(
                episode, tolerance=tolerance
            )
            episode_log_weight = episode_audit["episode_log_weight"]
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"invalid episode ledger {raw_path}: {exc}")
            continue

        errors = {
            "index_vs_episode": abs(index_log_weight - episode_log_weight),
            "ledger_vs_episode": episode_audit["ledger_vs_episode_error"],
            "step_weights_vs_episode": episode_audit[
                "step_weights_vs_episode_error"
            ],
        }
        for check, error in errors.items():
            if error is None:
                continue
            if not math.isfinite(error) or error > tolerance:
                failures.append(f"{raw_path}: {check} error={error}")
        failures.extend(
            f"{raw_path}: {issue}" for issue in episode_audit["failures"]
        )
        metadata = episode.get("scenario_metadata", {})
        records.append({
            "path": raw_path,
            "batch": _batch_name(Path(raw_path)),
            "source_event_id": str(metadata.get("source_event_id", "unknown")),
            "log_importance_weight": episode_log_weight,
            "errors": errors,
            "audit_passed": (
                not episode_audit["failures"]
                and errors["index_vs_episode"] <= tolerance
            ),
        })

    by_batch: dict[str, dict[str, Any]] = {}
    for batch in sorted({record["batch"] for record in records}):
        selected = [record for record in records if record["batch"] == batch]
        by_batch[batch] = {
            "episode_count": len(selected),
            "inconsistent_episode_count": sum(
                not record["audit_passed"]
                for record in selected
            ),
            "source_event_counts": dict(sorted(Counter(
                record["source_event_id"] for record in selected
            ).items())),
        }

    logs_by_source: dict[str, list[float]] = defaultdict(list)
    for record in records:
        logs_by_source[record["source_event_id"]].append(
            record["log_importance_weight"]
        )
    return {
        "schema_version": 1,
        "index_path": str(index_path),
        "indexed_episode_count": len(indexed_log_weights),
        "checked_episode_count": len(records),
        "tolerance": tolerance,
        "consistent_episode_count": sum(
            record["audit_passed"]
            for record in records
        ),
        "maximum_absolute_errors": {
            check: max(
                (
                    record["errors"][check]
                    for record in records
                    if record["errors"][check] is not None
                ),
                default=0.0,
            )
            for check in (
                "index_vs_episode",
                "ledger_vs_episode",
                "step_weights_vs_episode",
            )
        },
        "by_batch": by_batch,
        "importance_weight_diagnostics": _weight_diagnostics(
            [
                record["log_importance_weight"]
                for record in records
                if record["audit_passed"]
            ]
        ),
        "by_source_event": {
            source: _weight_diagnostics([
                record["log_importance_weight"]
                for record in records
                if record["source_event_id"] == source and record["audit_passed"]
            ])
            for source in sorted(logs_by_source)
        },
        "audit_passed": not failures and len(records) == len(indexed_log_weights),
        "failures": failures,
    }


def validate_episode_probability_ledger(
    episode: dict[str, Any], tolerance: float = 1e-7
) -> dict[str, Any]:
    """Validate that a training episode has positive support and complete logs."""
    episode_log_weight = float(episode["log_importance_weight"])
    if not math.isfinite(episode_log_weight):
        raise ValueError("episode log importance weight must be finite")
    ledger_log_weight = _sum_probability_ledger(
        episode["log_probability_step_info"]
    )
    ledger_error = abs(ledger_log_weight - episode_log_weight)
    failures = []
    if ledger_error > tolerance:
        failures.append(
            f"probability ledger disagrees with episode total: error={ledger_error}"
        )

    step_error = None
    try:
        step_weight_log = _sum_step_weights(episode["weight_step_info"])
        step_error = abs(step_weight_log - episode_log_weight)
        if step_error > tolerance:
            failures.append(
                f"step weights disagree with episode total: error={step_error}"
            )
    except ValueError as exc:
        failures.append(str(exc))

    raw_weight = float(episode.get("weight_episode", 0.0))
    if not math.isfinite(raw_weight) or raw_weight < 0:
        failures.append("raw episode weight must be finite and non-negative")
    elif raw_weight == 0.0:
        if episode_log_weight >= math.log(sys.float_info.min):
            failures.append(
                "raw episode weight is zero although the logged weight is in normal range"
            )
    elif abs(math.log(raw_weight) - episode_log_weight) > tolerance:
        failures.append("raw episode weight disagrees with episode log weight")

    return {
        "episode_log_weight": episode_log_weight,
        "ledger_vs_episode_error": ledger_error,
        "step_weights_vs_episode_error": step_error,
        "audit_passed": not failures,
        "failures": failures,
    }


def _sum_probability_ledger(ledger: dict[str, Any]) -> float:
    if not isinstance(ledger, dict):
        raise TypeError("log_probability_step_info must be an object")
    values = []
    for step in ledger.values():
        if not isinstance(step, dict) or "log_importance_weight" not in step:
            raise ValueError("probability ledger step lacks log_importance_weight")
        values.append(float(step["log_importance_weight"]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("probability ledger contains a non-finite value")
    return sum(values)


def _sum_step_weights(step_weights: dict[str, Any]) -> float:
    if not isinstance(step_weights, dict):
        raise TypeError("weight_step_info must be an object")
    logs = []
    for step in step_weights.values():
        value = step.get("joint") if isinstance(step, dict) else step
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("step importance weight must be positive and finite")
        logs.append(math.log(value))
    return sum(logs)


def _batch_name(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3 and parts[-2] in {"crash", "tested_and_safe", "rejected"}:
        return parts[-3]
    return path.parent.name


def _weight_diagnostics(log_weights: list[float]) -> dict[str, Any]:
    if not log_weights:
        return {
            "episode_count": 0,
            "effective_sample_size": 0.0,
            "effective_sample_size_ratio": 0.0,
            "largest_normalized_weight": None,
        }
    peak = max(log_weights)
    scaled = [math.exp(value - peak) for value in log_weights]
    total = sum(scaled)
    normalized = [value / total for value in scaled]
    ess = 1.0 / sum(value * value for value in normalized)
    return {
        "episode_count": len(log_weights),
        "effective_sample_size": ess,
        "effective_sample_size_ratio": ess / len(log_weights),
        "largest_normalized_weight": max(normalized),
        "minimum_log_importance_weight": min(log_weights),
        "maximum_log_importance_weight": max(log_weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_path")
    parser.add_argument("--workspace_root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    result = audit_training_pool(
        args.index_path,
        workspace_root=args.workspace_root,
        tolerance=args.tolerance,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
