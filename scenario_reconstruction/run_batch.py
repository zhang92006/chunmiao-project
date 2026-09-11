from __future__ import annotations

import argparse
import json
from pathlib import Path

from .augment_templates import generate_template_variants, write_variants
from .prepare_training_data import prepare_crash_weight_dict, prepare_safe_weight_dict
from .run_template import run_template
from .templates import load_template


def run_batch(
    template_path: str,
    count: int,
    output_root: str,
    seed: int = 0,
    max_ndd_possi: float | None = 0.01,
    target_end_time: float = 1.4,
    target_collision_actor: str = "BV_cut_in",
    multi_bv: bool = True,
    multi_bv_num: int = 2,
) -> dict:
    output_dir = Path(output_root)
    variants_dir = output_dir / "variants"
    experiment_dir = output_dir / "episodes"
    variants_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("crash", "tested_and_safe", "rejected"):
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

    template = load_template(template_path)
    variants = generate_template_variants(template, count, seed=seed)
    variant_paths = write_variants(variants, variants_dir)

    results = []
    for index, variant_path in enumerate(variant_paths):
        try:
            weight = run_template(
                str(variant_path),
                episode=index,
                experiment_path=str(experiment_dir),
            )
            results.append(
                {
                    "episode": index,
                    "variant": str(variant_path),
                    "status": "ok",
                    "weight_result": weight,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "episode": index,
                    "variant": str(variant_path),
                    "status": "error",
                    "error": str(exc),
                }
            )

    crash_weight_dict = prepare_crash_weight_dict(
        experiment_dir,
        max_ndd_possi=max_ndd_possi,
        multi_bv=multi_bv,
        agent_num=multi_bv_num,
    )
    safe_weight_dict = prepare_safe_weight_dict(
        experiment_dir,
        multi_bv=multi_bv,
        agent_num=multi_bv_num,
    )
    outcomes = _collect_episode_outcomes(
        experiment_dir,
        target_end_time=target_end_time,
        target_collision_actor=target_collision_actor,
    )
    summary = {
        "template": template_path,
        "count": count,
        "seed": seed,
        "output_root": str(output_dir),
        "successful_runs": sum(1 for item in results if item["status"] == "ok"),
        "failed_runs": sum(1 for item in results if item["status"] != "ok"),
        "training_ready_crashes": len(crash_weight_dict),
        "training_ready_safe": len(safe_weight_dict),
        "multi_bv": multi_bv,
        "multi_bv_num": multi_bv_num,
        "target_end_time": target_end_time,
        "target_collision_actor": target_collision_actor,
        "outcomes": outcomes,
        "results": results,
    }
    summary_path = output_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=4)
    return summary


def _collect_episode_outcomes(
    experiment_dir: Path,
    target_end_time: float,
    target_collision_actor: str,
) -> list[dict]:
    outcomes = []
    for path in sorted((experiment_dir / "crash").glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        outcomes.append(
            _episode_outcome(
                path,
                episode,
                target_end_time=target_end_time,
                target_collision_actor=target_collision_actor,
            )
        )
    for path in sorted((experiment_dir / "tested_and_safe").glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        outcomes.append(
            _episode_outcome(
                path,
                episode,
                target_end_time=target_end_time,
                target_collision_actor=target_collision_actor,
            )
        )
    return sorted(outcomes, key=lambda item: item["closeness_score"])


def _episode_outcome(
    path: Path,
    episode: dict,
    target_end_time: float,
    target_collision_actor: str,
) -> dict:
    min_ttc = min(episode.get("ttc_step_info", {"": 10000}).values())
    min_distance = min(episode.get("distance_step_info", {"": 10000}).values())
    end_time = float(episode.get("episode_info", {}).get("end_time", 10000))
    collision_ids = episode.get("collision_id") or []
    collision_match = target_collision_actor in collision_ids
    collision_snapshot = _extract_collision_snapshot(episode, target_collision_actor)
    event_key = next(iter(episode.get("drl_obs_step_info", {}) or {}), None)
    ndd_possi = None
    reward_at_099 = None
    if event_key is not None:
        ndd_possi = episode.get("ndd_step_info", {}).get(event_key)
        if ndd_possi is not None:
            ndd_for_reward = (
                ndd_possi.get("joint", 1.0)
                if isinstance(ndd_possi, dict)
                else ndd_possi
            )
            reward_at_099 = 100 - (float(ndd_for_reward) / (1 - 0.99)) * 500 * 100
    collision_penalty = 0 if episode.get("collision_result") else 10
    actor_penalty = 0 if collision_match else 5
    snapshot_penalty = _collision_snapshot_penalty(collision_snapshot)
    closeness_score = (
        abs(end_time - target_end_time)
        + min(min_ttc, 5) / 5
        + min(max(min_distance, 0), 20) / 20
        + collision_penalty
        + actor_penalty
        + snapshot_penalty
    )
    return {
        "file": str(path),
        "collision_result": episode.get("collision_result"),
        "collision_id": collision_ids,
        "end_time": end_time,
        "min_ttc": min_ttc,
        "min_distance": min_distance,
        "event_key": event_key,
        "ndd_possi": ndd_possi,
        "reward_at_epsilon_0.99": reward_at_099,
        "collision_snapshot": collision_snapshot,
        "collision_snapshot_penalty": snapshot_penalty,
        "closeness_score": closeness_score,
    }


def _extract_collision_snapshot(
    episode: dict,
    target_actor: str,
    lookback_steps: int = 2,
) -> dict | None:
    av_obs = episode.get("av_obs", {})
    if not av_obs:
        return None
    end_time = float(episode.get("episode_info", {}).get("end_time", 0.0))
    timestep = _nearest_timestep(av_obs, end_time)
    if timestep is None:
        return None

    snapshot = av_obs[timestep]
    ego = snapshot.get("Ego")
    actor = _find_vehicle_obs(snapshot, target_actor)
    if ego is None or actor is None:
        return None

    previous = _previous_risk_state(
        episode,
        current_timestep=timestep,
        lookback_steps=lookback_steps,
    )
    rel_x = actor["position"][0] - ego["position"][0]
    rel_y = actor["position"][1] - ego["position"][1]
    rel_speed = actor["velocity"] - ego["velocity"]
    return {
        "timestep": timestep,
        "ego": _compact_vehicle_state(ego),
        "actor": _compact_vehicle_state(actor),
        "relative_longitudinal_distance": rel_x,
        "relative_lateral_distance": rel_y,
        "relative_speed": rel_speed,
        "previous_ttc": previous.get("ttc"),
        "previous_distance": previous.get("distance"),
    }


def _nearest_timestep(av_obs: dict, target_time: float) -> str | None:
    if not av_obs:
        return None
    return min(av_obs.keys(), key=lambda key: abs(float(key) - target_time))


def _find_vehicle_obs(snapshot: dict, vehicle_id: str) -> dict | None:
    for value in snapshot.values():
        if isinstance(value, dict) and value.get("veh_id") == vehicle_id:
            return value
    return None


def _compact_vehicle_state(vehicle: dict) -> dict:
    return {
        "veh_id": vehicle.get("veh_id"),
        "position": vehicle.get("position"),
        "velocity": vehicle.get("velocity"),
        "lane_index": vehicle.get("lane_index"),
        "acceleration": vehicle.get("acceleration"),
        "heading": vehicle.get("heading"),
    }


def _previous_risk_state(
    episode: dict,
    current_timestep: str,
    lookback_steps: int,
) -> dict:
    ttc_info = episode.get("ttc_step_info", {})
    distance_info = episode.get("distance_step_info", {})
    numeric_steps = sorted(float(key) for key in ttc_info.keys())
    current = float(current_timestep)
    previous_steps = [step for step in numeric_steps if step <= current]
    if not previous_steps:
        return {}
    selected = previous_steps[max(0, len(previous_steps) - 1 - lookback_steps)]
    selected_key = min(ttc_info.keys(), key=lambda key: abs(float(key) - selected))
    return {
        "timestep": selected_key,
        "ttc": ttc_info.get(selected_key),
        "distance": distance_info.get(selected_key),
    }


def _collision_snapshot_penalty(snapshot: dict | None) -> float:
    if snapshot is None:
        return 5.0
    rel_x_penalty = min(abs(snapshot["relative_longitudinal_distance"]), 20.0) / 20.0
    rel_y_penalty = min(abs(snapshot["relative_lateral_distance"]), 8.0) / 8.0
    rel_speed_penalty = min(abs(snapshot["relative_speed"]), 20.0) / 20.0
    lane_penalty = 0.0
    ego_lane = snapshot["ego"].get("lane_index")
    actor_lane = snapshot["actor"].get("lane_index")
    if ego_lane is not None and actor_lane is not None:
        lane_penalty = min(abs(ego_lane - actor_lane), 2) * 0.5
    return rel_x_penalty + rel_y_penalty + rel_speed_penalty + lane_penalty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perturbed scenario variants and prepare training data."
    )
    parser.add_argument("template", help="Path to the base scenario template.")
    parser.add_argument("--count", type=int, default=10, help="Number of variants.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output_root",
        default="data_analysis/raw_data/ScenarioReconstructionBatch",
        help="Directory to store variants, episodes, and summary.",
    )
    parser.add_argument(
        "--max_ndd_possi",
        type=float,
        default=0.01,
        help="Filter training-ready crashes by maximum NDD probability.",
    )
    parser.add_argument(
        "--target_end_time",
        type=float,
        default=1.4,
        help="Reference crash end time used for closeness scoring.",
    )
    parser.add_argument(
        "--target_collision_actor",
        default="BV_cut_in",
        help="Reference collision actor used for closeness scoring.",
    )
    parser.add_argument(
        "--legacy_single_bv",
        action="store_true",
        help="Prepare legacy single-BV training data instead of MultiBV data.",
    )
    parser.add_argument(
        "--multi_bv_num",
        type=int,
        default=2,
        help="Number of controlled BVs expected in generated training records.",
    )
    args = parser.parse_args()

    summary = run_batch(
        args.template,
        count=args.count,
        output_root=args.output_root,
        seed=args.seed,
        max_ndd_possi=args.max_ndd_possi,
        target_end_time=args.target_end_time,
        target_collision_actor=args.target_collision_actor,
        multi_bv=not args.legacy_single_bv,
        multi_bv_num=args.multi_bv_num,
    )
    print("Batch finished.")
    print(f"successful_runs={summary['successful_runs']}")
    print(f"failed_runs={summary['failed_runs']}")
    print(f"training_ready_crashes={summary['training_ready_crashes']}")
    print(f"training_ready_safe={summary['training_ready_safe']}")
    print(f"summary={Path(args.output_root) / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
