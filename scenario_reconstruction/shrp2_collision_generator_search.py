"""Search-only dual-BV epsilon grid for the SHRP2 collision generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_template_manifest import run_template_manifest


def run_epsilon_grid(
    pilot_manifest: str | Path,
    output_root: str | Path,
    primary_epsilons: list[float],
    context_epsilons: list[float],
    repeats: int,
) -> dict:
    """Run an isolated factorized grid and summarize raw collision yield.

    These episodes are adaptive-search observations and must not be used as the
    final training sample. After choosing a frozen proposal, draw fresh episodes
    in a new directory.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    _validate_grid(primary_epsilons, "primary_epsilons")
    _validate_grid(context_epsilons, "context_epsilons")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = {
        "search_only": True,
        "training_use_allowed": False,
        "reason": "proposal parameters were selected using these outcomes",
    }
    _write_json(output_root / "SEARCH_ONLY_DO_NOT_TRAIN.json", marker)

    trials = []
    for primary in primary_epsilons:
        for context in context_epsilons:
            name = f"primary_{primary:.0e}_context_{context:.0e}".replace("+", "")
            trial_dir = output_root / name
            summary = run_template_manifest(
                pilot_manifest,
                experiment_path=trial_dir,
                split="train",
                repeats=repeats,
                epsilon={"BV_primary": primary, "BV_context": context},
                proposal_mode="factorized",
            )
            raw_crashes = len(list((trial_dir / "crash").glob("*.json")))
            raw_safe = len(list((trial_dir / "tested_and_safe").glob("*.json")))
            completed = raw_crashes + raw_safe
            trials.append({
                "epsilon_primary": primary,
                "epsilon_context": context,
                "attempted": summary["attempted"],
                "successful_runs": summary["successful_runs"],
                "raw_crashes": raw_crashes,
                "raw_safe": raw_safe,
                "raw_collision_rate": raw_crashes / completed if completed else None,
                "training_ready_crashes": summary["training_ready_crashes"],
                "importance_weight_diagnostics": summary.get(
                    "importance_weight_diagnostics"
                ),
                "experiment_path": str(trial_dir),
            })
    ranked = sorted(
        trials,
        key=lambda item: (
            -(item["raw_collision_rate"] or 0.0),
            -item["training_ready_crashes"],
            item["epsilon_primary"],
            item["epsilon_context"],
        ),
    )
    result = {
        "schema_version": 1,
        **marker,
        "pilot_manifest": str(pilot_manifest),
        "repeats_per_template": repeats,
        "trial_count": len(trials),
        "recommended_frozen_candidate": ranked[0] if ranked else None,
        "trials": trials,
    }
    _write_json(output_root / "epsilon_search_summary.json", result)
    return result


def _validate_grid(values: list[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError(f"all {name} values must lie strictly between zero and one")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the search-only dual-BV epsilon grid.")
    parser.add_argument("pilot_manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon_primary", action="append", type=float, required=True)
    parser.add_argument("--epsilon_context", action="append", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    result = run_epsilon_grid(
        args.pilot_manifest,
        args.output,
        args.epsilon_primary,
        args.epsilon_context,
        args.repeats,
    )
    print("Collision-generator epsilon search finished.")
    print(f"trial_count={result['trial_count']}")
    print(json.dumps(result["recommended_frozen_candidate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
