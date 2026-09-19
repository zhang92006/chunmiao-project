"""Bounded search-only CEM over two-BV actions and intervention timing."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .run_template import run_template


DEFAULT_SPACE = {
    "start_time_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "context_delay_s": [-0.5, 0.0, 0.5, 1.0],
    "primary_duration_s": [0.5, 1.0, 1.5, 2.0],
    "context_duration_s": [0.5, 1.0, 1.5, 2.0],
    "primary_action_id": [2, 7, 12, 17, 22],
    "context_action_id": [0, 1, 2, 12, 22, 27, 32],
}


def run_collision_action_cem(
    pilot_manifest: str | Path,
    output_root: str | Path,
    generations: int = 4,
    population: int = 16,
    elite_fraction: float = 0.25,
    smoothing: float = 0.25,
    rollouts_per_candidate: int = 1,
    seed: int = 20260919,
    include_positive_controls: bool = False,
    search_space: dict[str, list[Any]] | None = None,
    source_event_ids: set[int] | None = None,
) -> dict:
    """Search each pilot source independently and persist after every generation."""
    _validate_settings(
        generations, population, elite_fraction, smoothing, rollouts_per_candidate
    )
    space = search_space or DEFAULT_SPACE
    _validate_space(space)
    pilot_manifest = Path(pilot_manifest)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "SEARCH_ONLY_DO_NOT_TRAIN.json", {
        "search_only": True,
        "training_use_allowed": False,
        "reason": "CEM conditions on rollout outcomes and forced actions have no frozen q(a|s)",
    })
    with pilot_manifest.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    records = [
        record for record in manifest.get("records", [])
        if (include_positive_controls or record.get("pilot_role") != "positive_control")
        and (
            source_event_ids is None
            or int(record["event_id"]) in source_event_ids
        )
    ]
    if not records:
        raise ValueError("No pilot records selected for CEM")

    all_results = []
    next_episode_id = 0
    for record_index, record in enumerate(records):
        event_id = int(record["event_id"])
        rng = np.random.default_rng(seed + event_id + record_index)
        distributions = {
            name: np.full(len(values), 1.0 / len(values), dtype=float)
            for name, values in space.items()
        }
        source_result = {
            "source_event_id": event_id,
            "pilot_role": record.get("pilot_role"),
            "source_template": record["template_path"],
            "generations": [],
        }
        source_root = output_root / f"event_{event_id}"
        template_root = source_root / "templates"
        episode_root = source_root / "episodes"
        template_root.mkdir(parents=True, exist_ok=True)
        best = None
        for generation in range(generations):
            candidates = []
            for candidate_index in range(population):
                parameters = _sample_parameters(space, distributions, rng)
                candidate_id = f"e{event_id}_g{generation:02d}_c{candidate_index:03d}"
                template_path = template_root / f"{candidate_id}.json"
                candidate = build_search_template(
                    record["template_path"], template_path, candidate_id, parameters
                )
                scores = []
                target_collisions = 0
                episode_records = []
                for repeat in range(rollouts_per_candidate):
                    episode_id = next_episode_id
                    next_episode_id += 1
                    try:
                        run_template(
                            str(template_path),
                            episode_id,
                            str(episode_root),
                            epsilon=1.0,
                            proposal_mode="naturalistic",
                        )
                        metrics = _read_episode_metrics(episode_root, episode_id)
                        status = "ok"
                    except Exception as exc:
                        metrics = _empty_metrics()
                        status = f"error:{type(exc).__name__}:{exc}"
                    score = _score_metrics(metrics)
                    scores.append(score)
                    target_collisions += int(metrics["target_collision"])
                    episode_records.append({
                        "episode": episode_id,
                        "repeat": repeat,
                        "status": status,
                        **metrics,
                        "score": score,
                    })
                result = {
                    "candidate_id": candidate_id,
                    "parameters": parameters,
                    "template_path": str(template_path),
                    "mean_score": float(np.mean(scores)),
                    "target_collision_count": target_collisions,
                    "target_collision_rate": target_collisions / rollouts_per_candidate,
                    "episodes": episode_records,
                    "search_only": candidate["bridge_metadata"]["collision_search_only"],
                }
                candidates.append(result)
                if best is None or _candidate_rank(result) > _candidate_rank(best):
                    best = result
            elite_count = max(1, int(math.ceil(population * elite_fraction)))
            elites = sorted(candidates, key=_candidate_rank, reverse=True)[:elite_count]
            distributions = _update_distributions(
                space, distributions, elites, smoothing
            )
            generation_result = {
                "generation": generation,
                "elite_count": elite_count,
                "target_collision_count": sum(
                    item["target_collision_count"] for item in candidates
                ),
                "best_candidate": max(candidates, key=_candidate_rank),
                "updated_distributions": _serializable_distributions(
                    space, distributions
                ),
                "candidates": candidates,
            }
            source_result["generations"].append(generation_result)
            source_result["best_candidate"] = best
            _write_json(source_root / "cem_search_summary.json", source_result)
        all_results.append(source_result)
        _write_json(output_root / "cem_search_summary.json", {
            "schema_version": 1,
            "search_only": True,
            "training_use_allowed": False,
            "completed_source_count": len(all_results),
            "requested_source_count": len(records),
            "source_results": all_results,
        })

    result = {
        "schema_version": 1,
        "search_only": True,
        "training_use_allowed": False,
        "pilot_manifest": str(pilot_manifest),
        "source_count": len(records),
        "generations": generations,
        "population": population,
        "rollouts_per_candidate": rollouts_per_candidate,
        "attempted_rollouts": len(records) * generations * population * rollouts_per_candidate,
        "sources_with_target_collision": sum(
            item.get("best_candidate", {}).get("target_collision_count", 0) > 0
            for item in all_results
        ),
        "source_results": all_results,
    }
    _write_json(output_root / "cem_search_summary.json", result)
    return result


def build_search_template(
    source_path: str | Path,
    output_path: str | Path,
    candidate_id: str,
    parameters: dict[str, Any],
) -> dict:
    with Path(source_path).open("r", encoding="utf-8") as stream:
        candidate = json.load(stream)
    candidate = copy.deepcopy(candidate)
    candidate["template_id"] = f"{candidate['template_id']}_cem_{candidate_id}"
    candidate["description"] = (
        "Search-only CEM candidate with declared forced BV actions. "
        "It is not a D2RL training episode and has no frozen proposal probability."
    )
    tags = list(candidate.get("tags", []))
    for tag in ("collision_search_only", "not_for_d2rl_training"):
        if tag not in tags:
            tags.append(tag)
    candidate["tags"] = tags
    context_start = max(
        0.0,
        float(parameters["start_time_s"]) + float(parameters["context_delay_s"]),
    )
    candidate["events"] = [
        _search_event(
            "BV_primary",
            parameters["primary_action_id"],
            parameters["start_time_s"],
            parameters["primary_duration_s"],
            candidate_id,
        ),
        _search_event(
            "BV_context",
            parameters["context_action_id"],
            context_start,
            parameters["context_duration_s"],
            candidate_id,
        ),
    ]
    candidate["perturbations"] = []
    candidate.setdefault("bridge_metadata", {}).update({
        "collision_search_only": True,
        "not_for_d2rl_training": True,
        "collision_search_candidate_id": candidate_id,
        "collision_search_parameters": parameters,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, candidate)
    return candidate


def _search_event(
    actor: str, action_id: int, start_time: float, duration: float, candidate_id: str
) -> dict:
    action = _action_command(int(action_id))
    return {
        "type": "forced_bv_action",
        "actor": actor,
        "start_time": float(start_time),
        "duration": float(duration),
        "params": {
            **action,
            "action_id": int(action_id),
            "apply_once": False,
            "search_only": True,
            "not_for_d2rl_training": True,
            "collision_search_candidate_id": candidate_id,
        },
    }


def _action_command(action_id: int) -> dict[str, Any]:
    if action_id == 0:
        return {"lateral": "left", "longitudinal": 0.0}
    if action_id == 1:
        return {"lateral": "right", "longitudinal": 0.0}
    if 2 <= action_id <= 32:
        return {
            "lateral": "central",
            "longitudinal": float(-4.0 + 0.2 * (action_id - 2)),
        }
    raise ValueError(f"Unsupported BV action id: {action_id}")


def _sample_parameters(space, distributions, rng) -> dict[str, Any]:
    result = {}
    for name, values in space.items():
        index = int(rng.choice(len(values), p=distributions[name]))
        result[name] = values[index]
    return result


def _update_distributions(space, current, elites, smoothing, minimum=0.02):
    updated = {}
    for name, values in space.items():
        counts = np.asarray([
            sum(elite["parameters"][name] == value for elite in elites)
            for value in values
        ], dtype=float)
        empirical = counts / counts.sum()
        probability = smoothing * current[name] + (1.0 - smoothing) * empirical
        probability = np.maximum(probability, minimum)
        updated[name] = probability / probability.sum()
    return updated


def _serializable_distributions(space, distributions) -> dict:
    return {
        name: [
            {"value": value, "probability": float(probability)}
            for value, probability in zip(values, distributions[name])
        ]
        for name, values in space.items()
    }


def _read_episode_metrics(episode_root: Path, episode_id: int) -> dict:
    crash_path = episode_root / "crash" / f"{episode_id}.json"
    safe_path = episode_root / "tested_and_safe" / f"{episode_id}.json"
    path = crash_path if crash_path.is_file() else safe_path
    if not path.is_file():
        return _empty_metrics()
    with path.open("r", encoding="utf-8") as stream:
        episode = json.load(stream)
    collision_ids = [str(value) for value in episode.get("collision_id") or []]
    collision = bool(episode.get("collision_result"))
    return {
        "logged_episode_path": str(path),
        "collision": collision,
        "target_collision": collision and "CAV" in collision_ids,
        "collision_ids": collision_ids,
        "minimum_ttc_s": _minimum_number(episode.get("ttc_step_info", {})),
        "minimum_distance_m": _minimum_number(
            episode.get("distance_step_info", {})
        ),
    }


def _empty_metrics() -> dict:
    return {
        "logged_episode_path": None,
        "collision": False,
        "target_collision": False,
        "collision_ids": [],
        "minimum_ttc_s": None,
        "minimum_distance_m": None,
    }


def _score_metrics(metrics: dict) -> float:
    score = 100.0 if metrics["target_collision"] else 0.0
    ttc = metrics.get("minimum_ttc_s")
    distance = metrics.get("minimum_distance_m")
    if ttc is not None:
        score += max(0.0, 5.0 - max(0.0, ttc)) * 4.0
    if distance is not None:
        score += max(0.0, 10.0 - max(0.0, distance)) * 2.0
    if metrics["collision"] and not metrics["target_collision"]:
        score -= 25.0
    return float(score)


def _candidate_rank(candidate: dict) -> tuple:
    return (
        candidate["target_collision_rate"],
        candidate["mean_score"],
    )


def _minimum_number(value) -> float | None:
    numbers = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
        else:
            try:
                number = float(current)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numbers.append(number)
    return min(numbers) if numbers else None


def _validate_settings(generations, population, elite_fraction, smoothing, repeats):
    if generations < 1 or population < 2 or repeats < 1:
        raise ValueError("generations/repeats must be positive and population at least two")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must lie in (0, 1]")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must lie in [0, 1)")


def _validate_space(space: dict[str, list[Any]]) -> None:
    missing = set(DEFAULT_SPACE) - set(space)
    if missing:
        raise ValueError(f"Search space is missing: {sorted(missing)}")
    if any(not values for values in space.values()):
        raise ValueError("Every search-space variable requires at least one value")
    for name in ("primary_action_id", "context_action_id"):
        for action_id in space[name]:
            _action_command(int(action_id))


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run search-only per-source CEM over two-BV action timing."
    )
    parser.add_argument("pilot_manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite_fraction", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--rollouts_per_candidate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--include_positive_controls", action="store_true")
    parser.add_argument(
        "--source_event_id",
        action="append",
        type=int,
        default=None,
        help="Optionally restrict CEM to selected source ids; repeat as needed.",
    )
    args = parser.parse_args()
    result = run_collision_action_cem(
        args.pilot_manifest,
        args.output,
        generations=args.generations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        smoothing=args.smoothing,
        rollouts_per_candidate=args.rollouts_per_candidate,
        seed=args.seed,
        include_positive_controls=args.include_positive_controls,
        source_event_ids=(
            set(args.source_event_id) if args.source_event_id is not None else None
        ),
    )
    print("Collision action CEM search finished.")
    print(f"attempted_rollouts={result['attempted_rollouts']}")
    print(f"sources_with_target_collision={result['sources_with_target_collision']}")


if __name__ == "__main__":
    main()
