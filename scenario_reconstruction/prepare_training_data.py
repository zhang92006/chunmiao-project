from __future__ import annotations

import argparse
import json
from math import isfinite, log
from pathlib import Path

from .importance import stable_weight_diagnostics


def prepare_crash_weight_dict(
    experiment_path: str | Path,
    threshold: float = 0.1,
    min_criticality: float = 0.0,
    max_ndd_possi: float | None = None,
    multi_bv: bool = False,
    agent_num: int = 2,
) -> dict[str, list[float]]:
    experiment_dir = Path(experiment_path)
    crash_dir = experiment_dir / "crash"
    if not crash_dir.is_dir():
        raise FileNotFoundError(f"Crash directory not found: {crash_dir}")

    crash_weight_dict: dict[str, list[float]] = {}
    # Keep the legacy raw-weight index for existing consumers, but write a
    # separate log-domain index for samplers. A long K-BV rollout can have a
    # valid importance weight below floating-point resolution; using that raw
    # value directly as random.choices weights makes sampling collapse.
    crash_log_weight_dict: dict[str, float] = {}
    for crash_json_path in sorted(crash_dir.glob("*.json")):
        with crash_json_path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        weight_episode = float(episode["weight_episode"])
        if weight_episode < threshold and _is_training_ready_episode(
            episode,
            min_criticality=min_criticality,
            max_ndd_possi=max_ndd_possi,
            multi_bv=multi_bv,
            agent_num=agent_num,
        ):
            normalized_path = crash_json_path.as_posix()
            crash_weight_dict[normalized_path] = [weight_episode, weight_episode]
            crash_log_weight_dict[normalized_path] = _episode_log_weight(episode)

    output_path = experiment_dir / "crash_weight_dict.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(crash_weight_dict, stream, indent=4)
    log_output_path = experiment_dir / "crash_log_weight_dict.json"
    with log_output_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "sampling_weight": "exp(log_importance_weight - max_log_weight)",
                "log_weights": crash_log_weight_dict,
            },
            stream,
            indent=4,
        )
    write_importance_weight_diagnostics(experiment_dir, crash_weight_dict)
    return crash_weight_dict


def _episode_log_weight(episode: dict) -> float:
    """Return a finite episode log-weight, including legacy episode files."""
    log_weight = episode.get("log_importance_weight")
    if log_weight is not None and isfinite(float(log_weight)):
        return float(log_weight)
    raw_weight = float(episode.get("weight_episode", 0.0))
    if not isfinite(raw_weight) or raw_weight <= 0.0:
        raise ValueError("Training-ready crash episode has no finite importance weight")
    return log(raw_weight)


def write_importance_weight_diagnostics(
    experiment_dir: Path,
    crash_weight_dict: dict[str, list[float]],
) -> dict:
    """Write stable ESS diagnostics for the exact crash pool passed to training."""
    records = []
    by_source: dict[str, list[dict]] = {}
    missing_log_weight_count = 0
    for raw_path in crash_weight_dict:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        log_weight = episode.get("log_importance_weight")
        if log_weight is None:
            raw_weight = float(episode.get("weight_episode", 0.0))
            if not isfinite(raw_weight) or raw_weight <= 0.0:
                missing_log_weight_count += 1
                continue
            log_weight = log(raw_weight)
            missing_log_weight_count += 1
        record = {
            "episode_path": path.as_posix(),
            "log_importance_weight": float(log_weight),
            "proposal_modes": list(episode.get("proposal_modes", [])),
        }
        records.append(record)
        source = str(episode.get("scenario_metadata", {}).get("source_event_id", "unknown"))
        by_source.setdefault(source, []).append(record)

    result = {
        "schema_version": 1,
        "pool": "training_ready_crashes",
        "missing_log_weight_count": missing_log_weight_count,
        "overall": stable_weight_diagnostics(records),
        "by_source_event": {
            source: stable_weight_diagnostics(source_records)
            for source, source_records in sorted(by_source.items())
        },
        "proposal_mode_counts": _proposal_mode_counts(records),
    }
    output_path = experiment_dir / "importance_weight_diagnostics.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=4)
    return result


def _proposal_mode_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        modes = record.get("proposal_modes", [])
        if not modes:
            counts["unknown"] = counts.get("unknown", 0) + 1
            continue
        for mode in modes:
            counts[str(mode)] = counts.get(str(mode), 0) + 1
    return counts


def prepare_safe_weight_dict(
    experiment_path: str | Path,
    default_weight: float = 1.0,
    multi_bv: bool = False,
    agent_num: int = 2,
) -> dict[str, list[float]]:
    experiment_dir = Path(experiment_path)
    safe_dir = experiment_dir / "tested_and_safe"
    if not safe_dir.is_dir():
        raise FileNotFoundError(f"Safe directory not found: {safe_dir}")

    safe_weight_dict: dict[str, list[float]] = {}
    for safe_json_path in sorted(safe_dir.glob("*.json")):
        with safe_json_path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        metadata = episode.get("scenario_metadata", {})
        if (
            metadata.get("collision_search_only") is True
            or metadata.get("not_for_d2rl_training") is True
        ):
            continue
        if multi_bv:
            if not _is_training_ready_episode(episode, multi_bv=True, agent_num=agent_num):
                continue
        normalized_path = safe_json_path.as_posix()
        safe_weight_dict[normalized_path] = [default_weight]

    output_path = experiment_dir / "safe_weight_dict.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(safe_weight_dict, stream, indent=4)
    return safe_weight_dict


def _is_training_ready_episode(
    episode: dict,
    min_criticality: float = 0.0,
    max_ndd_possi: float | None = None,
    multi_bv: bool = False,
    agent_num: int = 2,
) -> bool:
    metadata = episode.get("scenario_metadata", {})
    if (
        metadata.get("collision_search_only") is True
        or metadata.get("not_for_d2rl_training") is True
    ):
        return False
    weight_step_info = episode.get("weight_step_info", {})
    drl_obs_step_info = episode.get("drl_obs_step_info", {})
    criticality_step_info = episode.get("criticality_step_info", {})
    ndd_step_info = episode.get("ndd_step_info", {})
    if not weight_step_info or not drl_obs_step_info:
        return False
    for timestep, weight in weight_step_info.items():
        obs = drl_obs_step_info.get(timestep)
        criticality = float(criticality_step_info.get(timestep, 0.0))
        ndd_possi = ndd_step_info.get(timestep)
        if multi_bv:
            if _is_training_ready_multibv_step(
                obs=obs,
                weight=weight,
                ndd_possi=ndd_possi,
                criticality=criticality,
                max_ndd_possi=max_ndd_possi,
                min_criticality=min_criticality,
                agent_num=agent_num,
            ):
                return True
        else:
            has_ndd = ndd_possi is not None
            ndd_in_range = has_ndd and (
                max_ndd_possi is None or float(ndd_possi) <= max_ndd_possi
            )
            if (
                obs is not None
                and len(obs) == 10
                and float(weight) < 0.999
                and criticality > min_criticality
                and ndd_in_range
            ):
                return True
    return False


def _is_training_ready_multibv_step(
    obs: dict | None,
    weight: dict,
    ndd_possi: dict | None,
    criticality: float,
    max_ndd_possi: float | None,
    min_criticality: float,
    agent_num: int,
) -> bool:
    if not isinstance(obs, dict) or not isinstance(weight, dict) or not isinstance(ndd_possi, dict):
        return False
    joint_obs = obs.get("joint")
    per_agent_obs = obs.get("per_agent")
    per_agent_weight = weight.get("per_agent")
    per_agent_ndd = ndd_possi.get("per_agent")
    if not isinstance(joint_obs, list) or len(joint_obs) != 6 + 4 * agent_num:
        return False
    if not isinstance(per_agent_obs, list) or len(per_agent_obs) != agent_num:
        return False
    if any(not isinstance(item, list) or len(item) != 10 for item in per_agent_obs):
        return False
    if not isinstance(per_agent_weight, list) or len(per_agent_weight) != agent_num:
        return False
    if not isinstance(per_agent_ndd, list) or len(per_agent_ndd) != agent_num:
        return False
    joint_weight = float(weight.get("joint", 1.0))
    joint_ndd = float(ndd_possi.get("joint", 1.0))
    ndd_in_range = max_ndd_possi is None or joint_ndd <= max_ndd_possi
    return joint_weight < 0.999 and criticality > min_criticality and ndd_in_range


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reconstructed crash data for D2RL training."
    )
    parser.add_argument("experiment_path", help="Path containing crash/*.json.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Keep crash episodes with weight_episode below this threshold.",
    )
    parser.add_argument(
        "--min_criticality",
        type=float,
        default=0.0,
        help="Keep episodes with at least one training step above this criticality.",
    )
    parser.add_argument(
        "--max_ndd_possi",
        type=float,
        default=None,
        help="Optionally keep episodes with at least one training step below this NDD probability.",
    )
    parser.add_argument(
        "--multi_bv",
        action="store_true",
        help="Require MultiBV K-agent training records instead of legacy single-BV records.",
    )
    parser.add_argument(
        "--agent_num",
        type=int,
        default=2,
        help="Number of controlled BVs required when --multi_bv is set.",
    )
    parser.add_argument(
        "--include_safe_weight_dict",
        action="store_true",
        help="Also write safe_weight_dict.json for tested_and_safe episodes.",
    )
    args = parser.parse_args()

    crash_weight_dict = prepare_crash_weight_dict(
        args.experiment_path,
        threshold=args.threshold,
        min_criticality=args.min_criticality,
        max_ndd_possi=args.max_ndd_possi,
        multi_bv=args.multi_bv,
        agent_num=args.agent_num,
    )
    print(f"Prepared {len(crash_weight_dict)} crash episodes.")
    print(f"Wrote {Path(args.experiment_path) / 'crash_weight_dict.json'}")
    print(f"Wrote {Path(args.experiment_path) / 'crash_log_weight_dict.json'}")
    if args.include_safe_weight_dict:
        safe_weight_dict = prepare_safe_weight_dict(
            args.experiment_path,
            multi_bv=args.multi_bv,
            agent_num=args.agent_num,
        )
        print(f"Prepared {len(safe_weight_dict)} safe episodes.")
        print(f"Wrote {Path(args.experiment_path) / 'safe_weight_dict.json'}")


if __name__ == "__main__":
    main()
