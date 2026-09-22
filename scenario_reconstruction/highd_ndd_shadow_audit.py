"""Audit read-only highD shadow records produced by reconstructed SUMO runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def audit_shadow(root: str | Path, maximum_fallback_rate: float = 0.05) -> dict:
    root = Path(root)
    if not 0 <= maximum_fallback_rate <= 1:
        raise ValueError("maximum_fallback_rate must lie in [0, 1]")
    episode_paths = sorted(
        path
        for folder in ("crash", "tested_and_safe")
        for path in (root / folder).glob("*.json")
    )
    if not episode_paths:
        raise ValueError("No saved episode JSON files found")
    records = []
    episode_checks = []
    longitudinal_sources = Counter()
    lateral_sources = Counter()
    fallback_count = 0
    skipped = Counter()
    maximum_probability_sum_error = 0.0
    for path in episode_paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        metadata = episode.get("scenario_metadata", {}).get("highd_ndd_shadow")
        if not metadata or metadata.get("mode") != "read_only_shadow":
            raise ValueError(f"Missing read-only shadow metadata: {path}")
        if metadata.get("runtime_actions_changed") or metadata.get("importance_weights_changed"):
            raise ValueError(f"Shadow metadata declares a runtime mutation: {path}")
        step_info = episode.get("highd_ndd_shadow_step_info", {})
        for actors in episode.get("highd_ndd_shadow_skipped_step_info", {}).values():
            for record in actors.values():
                skipped[
                    f"{record.get('reason')}:{record.get('controller_type')}"
                ] += 1
        for actors in step_info.values():
            for record in actors.values():
                original = np.asarray(record["original_pdf"], dtype=float)
                candidate = np.asarray(record["highd_shadow_pdf"], dtype=float)
                for name, pdf in (("original", original), ("highd", candidate)):
                    if pdf.shape != (33,) or not np.isfinite(pdf).all() or np.any(pdf < 0):
                        raise ValueError(f"Invalid {name} PDF in {path}")
                    maximum_probability_sum_error = max(
                        maximum_probability_sum_error, abs(float(pdf.sum()) - 1.0)
                    )
                fallback_count += int(record["fallback"])
                longitudinal_sources[record["longitudinal_source"]] += 1
                lateral_sources[record["lateral_source"]] += 1
                records.append(record)
        episode_checks.append({
            "path": str(path.relative_to(root)),
            "weight_episode": episode.get("weight_episode"),
            "log_importance_weight": episode.get("log_importance_weight"),
            "shadow_step_count": len(step_info),
        })
    if not records:
        raise ValueError("Episodes contain no shadow records")

    def quantiles(key):
        values = np.asarray([record[key] for record in records], dtype=float)
        return {
            name: float(value)
            for name, value in zip(
                ("minimum", "p50", "p90", "maximum"),
                np.quantile(values, (0, 0.5, 0.9, 1)),
            )
        }

    fallback_rate = fallback_count / len(records)
    probability_integrity_passed = maximum_probability_sum_error <= 1e-9
    coverage_gate_passed = fallback_rate <= maximum_fallback_rate
    result = {
        "schema_version": 1,
        "status": "read_only_shadow_audit",
        "episode_count": len(episode_paths),
        "record_count": len(records),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "maximum_fallback_rate": maximum_fallback_rate,
        "skipped_record_counts": dict(skipped),
        "longitudinal_source_counts": dict(longitudinal_sources),
        "lateral_source_counts": dict(lateral_sources),
        "maximum_probability_sum_error": maximum_probability_sum_error,
        "l1_distance": quantiles("l1_distance"),
        "kl_original_to_highd": quantiles("kl_original_to_highd"),
        "mean_original_lane_change_probability": float(np.mean([
            record["original_lane_change_probability"] for record in records
        ])),
        "mean_highd_lane_change_probability": float(np.mean([
            record["highd_lane_change_probability"] for record in records
        ])),
        "episode_checks": episode_checks,
        "probability_integrity_passed": probability_integrity_passed,
        "coverage_gate_passed": coverage_gate_passed,
        "audit_passed": probability_integrity_passed and coverage_gate_passed,
        "scope": (
            "Probability shadow only; this audit does not establish closed-loop "
            "distributional equivalence or authorize importance-weight replacement."
        ),
    }
    (root / "highd_ndd_shadow_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--maximum_fallback_rate", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(audit_shadow(args.root, args.maximum_fallback_rate), indent=2))


if __name__ == "__main__":
    main()
