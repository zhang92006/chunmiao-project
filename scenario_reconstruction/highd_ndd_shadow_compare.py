"""Compare strict-D2RL and free-flow highD shadows on paired SUMO states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _episode_paths(root: Path) -> dict[str, Path]:
    paths = {
        str(path.relative_to(root)): path
        for folder in ("crash", "tested_and_safe")
        for path in (root / folder).glob("*.json")
    }
    if not paths:
        raise ValueError(f"No saved episode JSON files found: {root}")
    return paths


def _probability_summary(
    strict_values: list[float], extension_values: list[float]
) -> dict[str, Any]:
    strict = np.asarray(strict_values, dtype=float)
    extension = np.asarray(extension_values, dtype=float)
    delta = extension - strict
    if len(strict) == 0:
        return {
            "record_count": 0,
            "strict_mean_lane_change_probability": None,
            "extension_mean_lane_change_probability": None,
            "mean_probability_delta": None,
            "extension_probability_quantiles": None,
        }
    return {
        "record_count": len(strict),
        "strict_mean_lane_change_probability": float(strict.mean()),
        "extension_mean_lane_change_probability": float(extension.mean()),
        "mean_probability_delta": float(delta.mean()),
        "extension_probability_quantiles": {
            name: float(value)
            for name, value in zip(
                ("minimum", "p50", "p90", "maximum"),
                np.quantile(extension, (0, 0.5, 0.9, 1)),
            )
        },
    }


def compare_shadow_roots(
    strict_root: str | Path,
    extension_root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    strict_root = Path(strict_root)
    extension_root = Path(extension_root)
    strict_paths = _episode_paths(strict_root)
    extension_paths = _episode_paths(extension_root)
    if strict_paths.keys() != extension_paths.keys():
        raise ValueError("Strict and extension episode paths do not match")

    grouped: dict[str, tuple[list[float], list[float]]] = {
        name: ([], [])
        for name in ("all", "no_current_leader", "current_leader", "other")
    }
    maximum_original_pdf_error = 0.0
    maximum_weight_error = 0.0
    maximum_log_weight_error = 0.0
    longitudinal_source_mismatch_count = 0
    fallback_mismatch_count = 0
    paired_records = 0
    for relative_path in sorted(strict_paths):
        strict_episode = json.loads(
            strict_paths[relative_path].read_text(encoding="utf-8")
        )
        extension_episode = json.loads(
            extension_paths[relative_path].read_text(encoding="utf-8")
        )
        strict_metadata = strict_episode.get("scenario_metadata", {})
        extension_metadata = extension_episode.get("scenario_metadata", {})
        for key in ("template_id", "source_event_id", "source_split", "simulation_seed"):
            if strict_metadata.get(key) != extension_metadata.get(key):
                raise ValueError(f"Episode metadata mismatch for {key}: {relative_path}")
        maximum_weight_error = max(
            maximum_weight_error,
            abs(
                float(strict_episode.get("weight_episode", 0.0))
                - float(extension_episode.get("weight_episode", 0.0))
            ),
        )
        maximum_log_weight_error = max(
            maximum_log_weight_error,
            abs(
                float(strict_episode.get("log_importance_weight", 0.0))
                - float(extension_episode.get("log_importance_weight", 0.0))
            ),
        )
        strict_steps = strict_episode.get("highd_ndd_shadow_step_info", {})
        extension_steps = extension_episode.get("highd_ndd_shadow_step_info", {})
        if strict_steps.keys() != extension_steps.keys():
            raise ValueError(f"Shadow step keys do not match: {relative_path}")
        for step in strict_steps:
            strict_actors = strict_steps[step]
            extension_actors = extension_steps[step]
            if strict_actors.keys() != extension_actors.keys():
                raise ValueError(
                    f"Shadow actor keys do not match: {relative_path} step={step}"
                )
            for actor_id in strict_actors:
                strict_record = strict_actors[actor_id]
                extension_record = extension_actors[actor_id]
                if strict_record.get("vehicle_id") != extension_record.get("vehicle_id"):
                    raise ValueError(
                        f"Shadow vehicle IDs do not match: {relative_path} "
                        f"step={step} actor={actor_id}"
                    )
                longitudinal_source_mismatch_count += int(
                    strict_record.get("longitudinal_source")
                    != extension_record.get("longitudinal_source")
                )
                fallback_mismatch_count += int(
                    bool(strict_record.get("fallback"))
                    != bool(extension_record.get("fallback"))
                )
                original_error = float(np.max(np.abs(
                    np.asarray(strict_record["original_pdf"], dtype=float)
                    - np.asarray(extension_record["original_pdf"], dtype=float)
                )))
                maximum_original_pdf_error = max(
                    maximum_original_pdf_error, original_error
                )
                strict_probability = float(
                    strict_record["highd_lane_change_probability"]
                )
                extension_probability = float(
                    extension_record["highd_lane_change_probability"]
                )
                if strict_record["lateral_source"] == "original_structure_no_current_leader":
                    group = "no_current_leader"
                elif strict_record["lateral_source"] == "highd_adjacent_context":
                    group = "current_leader"
                else:
                    group = "other"
                for target in ("all", group):
                    grouped[target][0].append(strict_probability)
                    grouped[target][1].append(extension_probability)
                paired_records += 1

    tolerance = 1e-12
    result = {
        "schema_version": 1,
        "status": "paired_read_only_shadow_comparison",
        "strict_root": strict_root.name,
        "extension_root": extension_root.name,
        "paired_episode_count": len(strict_paths),
        "paired_record_count": paired_records,
        "maximum_original_pdf_error": maximum_original_pdf_error,
        "maximum_weight_error": maximum_weight_error,
        "maximum_log_weight_error": maximum_log_weight_error,
        "longitudinal_source_mismatch_count": longitudinal_source_mismatch_count,
        "fallback_mismatch_count": fallback_mismatch_count,
        "by_current_leader_presence": {
            name: _probability_summary(*values)
            for name, values in grouped.items()
        },
        "paired_inputs_and_weights_passed": (
            maximum_original_pdf_error <= tolerance
            and maximum_weight_error <= tolerance
            and maximum_log_weight_error <= tolerance
            and longitudinal_source_mismatch_count == 0
            and fallback_mismatch_count == 0
        ),
        "scope": (
            "Read-only probability comparison on paired states; this does not measure "
            "closed-loop lane-change frequency or authorize runtime NDD replacement."
        ),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strict_root")
    parser.add_argument("extension_root")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(compare_shadow_roots(
        args.strict_root, args.extension_root, args.output
    ), indent=2))


if __name__ == "__main__":
    main()
