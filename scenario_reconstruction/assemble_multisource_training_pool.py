"""Assemble immutable episode indexes into a source-audited D2RL pool."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .prepare_training_data import write_importance_weight_diagnostics


def assemble_training_pool(
    experiment_roots: list[str | Path], output_dir: str | Path
) -> dict:
    """Merge training-ready crash indexes without copying episode JSON files."""
    if not experiment_roots:
        raise ValueError("At least one experiment root is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_weights = {}
    merged_log_weights = {}
    source_counts = Counter()
    provenance = []
    duplicate_count = 0
    for raw_root in experiment_roots:
        root = Path(raw_root)
        with (root / "crash_weight_dict.json").open("r", encoding="utf-8") as stream:
            weights = json.load(stream)
        with (root / "crash_log_weight_dict.json").open("r", encoding="utf-8") as stream:
            log_payload = json.load(stream)
        log_weights = log_payload.get("log_weights", log_payload)
        accepted = 0
        for raw_path, weight in weights.items():
            path = Path(raw_path)
            key = path.as_posix()
            if key in merged_weights:
                duplicate_count += 1
                continue
            with path.open("r", encoding="utf-8") as stream:
                episode = json.load(stream)
            metadata = episode.get("scenario_metadata", {})
            if metadata.get("source_split") != "train":
                raise ValueError(f"Non-train episode cannot enter training pool: {path}")
            if (
                metadata.get("collision_search_only") is True
                or metadata.get("not_for_d2rl_training") is True
            ):
                raise ValueError(f"Search-only episode cannot enter training pool: {path}")
            source = metadata.get("source_event_id")
            if source is None:
                raise ValueError(f"Episode has no source_event_id: {path}")
            log_weight = float(log_weights[key])
            if not math.isfinite(log_weight):
                raise ValueError(f"Episode has non-finite log weight: {path}")
            merged_weights[key] = weight
            merged_log_weights[key] = log_weight
            source_counts[str(source)] += 1
            accepted += 1
        provenance.append({
            "experiment_root": str(root),
            "indexed_crash_count": len(weights),
            "accepted_unique_count": accepted,
        })
    _write_json(output_dir / "crash_weight_dict.json", merged_weights)
    _write_json(output_dir / "crash_log_weight_dict.json", {
        "schema_version": 1,
        "sampling_weight": "exp(log_importance_weight - max_log_weight)",
        "log_weights": merged_log_weights,
    })
    diagnostics = write_importance_weight_diagnostics(output_dir, merged_weights)
    summary = {
        "schema_version": 1,
        "episode_storage": "external_paths_no_copy",
        "training_split_only": True,
        "crash_episode_count": len(merged_weights),
        "source_event_count": len(source_counts),
        "crash_count_by_source_event": dict(sorted(source_counts.items())),
        "duplicate_path_count": duplicate_count,
        "provenance": provenance,
        "importance_weight_diagnostics": diagnostics,
        "recommended_training_sampler": "uniform_source",
        "scientific_boundary": (
            "uniform_source is a diversity-focused training distribution; "
            "do not use it as an unbiased naturalistic collision-rate estimator"
        ),
    }
    _write_json(output_dir / "training_pool_summary.json", summary)
    return summary


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble multiple training-ready crash indexes without copying episodes."
    )
    parser.add_argument("--experiment_root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = assemble_training_pool(args.experiment_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
