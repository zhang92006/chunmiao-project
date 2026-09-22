"""Audit conditional independence of synchronized highD vehicle actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .highd_lane_change_baseline import _target_rate_mask
from .highd_ndd_baseline import _axis, _load_split_manifest, _nearest_index


ACTION_NAMES = ("decelerate", "steady", "accelerate")


def _action_index(acceleration: np.ndarray, thresholds: list[float]) -> np.ndarray:
    lower, upper = map(float, thresholds)
    return np.where(acceleration < lower, 0, np.where(acceleration > upper, 2, 1))


def _pair_rows(
    source_root: Path,
    recording_id: str,
    *,
    source_hz: int,
    target_hz: int,
) -> pd.DataFrame:
    columns = [
        "frame", "id", "xVelocity", "xAcceleration", "dhw",
        "precedingXVelocity", "precedingId", "laneId",
    ]
    tracks = pd.read_csv(
        source_root / "data" / f"{recording_id}_tracks.csv", usecols=columns
    )
    frames = tracks["frame"].to_numpy(dtype=np.int64)
    tracks = tracks.loc[_target_rate_mask(frames, source_hz, target_hz)].copy()
    followers = tracks.loc[tracks["precedingId"] > 0].copy()
    leaders = tracks[["frame", "id", "xVelocity", "xAcceleration", "laneId"]].rename(
        columns={
            "id": "precedingId",
            "xVelocity": "leaderVelocity",
            "xAcceleration": "leaderAcceleration",
            "laneId": "leaderLaneId",
        }
    )
    return followers.merge(leaders, on=["frame", "precedingId"], how="inner")


def _pair_indices(
    pairs: pd.DataFrame,
    axes: dict[str, np.ndarray],
    acceleration_range: list[float],
    action_thresholds: list[float],
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    speed = np.abs(pairs["xVelocity"].to_numpy(dtype=float))
    leader_speed = np.abs(pairs["leaderVelocity"].to_numpy(dtype=float))
    gap = pairs["dhw"].to_numpy(dtype=float)
    range_rate = leader_speed - speed
    follower_sign = np.where(pairs["xVelocity"].to_numpy() >= 0.0, 1.0, -1.0)
    leader_sign = np.where(pairs["leaderVelocity"].to_numpy() >= 0.0, 1.0, -1.0)
    follower_acceleration = pairs["xAcceleration"].to_numpy(dtype=float) * follower_sign
    leader_acceleration = pairs["leaderAcceleration"].to_numpy(dtype=float) * leader_sign
    acc_low, acc_high = map(float, acceleration_range)
    eligible = (
        (pairs["laneId"].to_numpy() == pairs["leaderLaneId"].to_numpy())
        & (speed >= axes["speed"][0])
        & (speed <= axes["speed"][-1])
        & (leader_speed >= axes["speed"][0])
        & (leader_speed <= axes["speed"][-1])
        & (gap >= axes["gap"][0])
        & (gap <= axes["gap"][-1])
        & (range_rate >= axes["range_rate"][0])
        & (range_rate <= axes["range_rate"][-1])
        & (follower_acceleration >= acc_low)
        & (follower_acceleration <= acc_high)
        & (leader_acceleration >= acc_low)
        & (leader_acceleration <= acc_high)
    )
    states = tuple(
        _nearest_index(values[eligible], axes[name])
        for name, values in (
            ("speed", speed), ("gap", gap), ("range_rate", range_rate)
        )
    )
    follower_action = _action_index(follower_acceleration[eligible], action_thresholds)
    leader_action = _action_index(leader_acceleration[eligible], action_thresholds)
    return states, follower_action, leader_action, eligible


def _joint_probability(counts: np.ndarray, alpha: float) -> np.ndarray:
    total = counts.sum(axis=(-2, -1), keepdims=True)
    return (counts + alpha) / (total + alpha * 9.0)


def _factorized_probability(counts: np.ndarray, alpha: float) -> np.ndarray:
    follower = counts.sum(axis=-1)
    leader = counts.sum(axis=-2)
    follower = (follower + alpha) / (follower.sum(axis=-1, keepdims=True) + 3 * alpha)
    leader = (leader + alpha) / (leader.sum(axis=-1, keepdims=True) + 3 * alpha)
    return follower[..., :, None] * leader[..., None, :]


def _conditional_mutual_information(counts: np.ndarray, minimum_state_count: int) -> float:
    flat = counts.reshape((-1, 3, 3)).astype(float)
    totals = flat.sum(axis=(1, 2))
    selected = totals >= minimum_state_count
    flat = flat[selected]
    totals = totals[selected]
    if not len(flat):
        return 0.0
    weighted = 0.0
    total_observations = float(totals.sum())
    for table, state_total in zip(flat, totals):
        joint = table / state_total
        first = joint.sum(axis=1, keepdims=True)
        second = joint.sum(axis=0, keepdims=True)
        independent = first * second
        nonzero = joint > 0
        weighted += float(
            state_total * np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero]))
        )
    return weighted / total_observations


def _evaluate_recording(
    source_root: Path,
    recording_id: str,
    axes: dict[str, np.ndarray],
    counts: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    pairs = _pair_rows(
        source_root,
        recording_id,
        source_hz=int(config["source_frequency_hz"]),
        target_hz=int(config["target_frequency_hz"]),
    )
    states, follower, leader, eligible = _pair_indices(
        pairs, axes, config["acceleration_range"], config["action_thresholds"]
    )
    alpha = float(config["laplace_alpha"])
    joint = _joint_probability(counts, alpha)
    factorized = _factorized_probability(counts, alpha)
    action_indices = states + (follower, leader)
    joint_selected = joint[action_indices]
    factorized_selected = factorized[action_indices]
    state_seen = counts.sum(axis=(-2, -1))[states] > 0
    n = len(follower)
    joint_nll = -float(np.log(joint_selected).sum()) / n if n else None
    factorized_nll = -float(np.log(factorized_selected).sum()) / n if n else None
    return {
        "recording_id": recording_id,
        "synchronized_preceding_pairs": len(pairs),
        "eligible_pair_actions": n,
        "seen_state_pair_actions": int(state_seen.sum()),
        "state_coverage": float(state_seen.mean()) if n else None,
        "joint_mean_negative_log_likelihood": joint_nll,
        "factorized_mean_negative_log_likelihood": factorized_nll,
        "factorization_nll_penalty": (
            factorized_nll - joint_nll if n else None
        ),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(record["eligible_pair_actions"] for record in records)
    if not total:
        return {"recording_count": len(records), "eligible_pair_actions": 0}
    weighted = lambda key: sum(
        record[key] * record["eligible_pair_actions"] for record in records
    ) / total
    return {
        "recording_count": len(records),
        "eligible_pair_actions": total,
        "state_coverage": sum(r["seen_state_pair_actions"] for r in records) / total,
        "joint_mean_negative_log_likelihood": weighted(
            "joint_mean_negative_log_likelihood"
        ),
        "factorized_mean_negative_log_likelihood": weighted(
            "factorized_mean_negative_log_likelihood"
        ),
        "factorization_nll_penalty": weighted("factorization_nll_penalty"),
        "recordings_with_positive_factorization_penalty": sum(
            record["factorization_nll_penalty"] > 0 for record in records
        ),
        "per_recording": records,
    }


def audit_joint_actions(
    source_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_root = Path(source_root)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_split_manifest(manifest_path)
    axes = {
        name: _axis(config["state_grid"][name])
        for name in ("speed", "gap", "range_rate")
    }
    counts = np.zeros(tuple(len(axis) for axis in axes.values()) + (3, 3), dtype=np.uint32)
    train_split = str(config.get("train_split", "train"))
    for recording_id in splits[train_split]:
        pairs = _pair_rows(
            source_root,
            recording_id,
            source_hz=int(config["source_frequency_hz"]),
            target_hz=int(config["target_frequency_hz"]),
        )
        states, follower, leader, _ = _pair_indices(
            pairs, axes, config["acceleration_range"], config["action_thresholds"]
        )
        np.add.at(counts, states + (follower, leader), 1)

    model_path = output_dir / "highd_joint_longitudinal_action_v1.npz"
    alpha = float(config["laplace_alpha"])
    np.savez_compressed(
        model_path,
        **{f"{name}_axis": axis for name, axis in axes.items()},
        action_names=np.array(ACTION_NAMES),
        joint_counts=counts,
        joint_probability=_joint_probability(counts, alpha).astype(np.float32),
        factorized_probability=_factorized_probability(counts, alpha).astype(np.float32),
    )
    evaluation = {
        split: _aggregate([
            _evaluate_recording(source_root, recording_id, axes, counts, config)
            for recording_id in ids
        ])
        for split, ids in splits.items()
    }
    cmi = _conditional_mutual_information(
        counts, int(config.get("minimum_state_count_for_cmi", 100))
    )
    validation_penalty = evaluation.get("validation", {}).get(
        "factorization_nll_penalty"
    )
    threshold = float(config.get("material_nll_penalty", 0.01))
    summary = {
        "schema_version": 1,
        "status": "joint_dependency_audit_not_runtime_enabled",
        "dataset": "highD-v1.0",
        "pair_definition": "same-lane vehicle and its recorded preceding vehicle",
        "action_definition": ACTION_NAMES,
        "conditional_mutual_information_nats": cmi,
        "material_nll_penalty_threshold": threshold,
        "factorized_assumption_supported": (
            validation_penalty is not None and validation_penalty <= threshold
        ),
        "model_path": str(model_path),
        "evaluation": evaluation,
        "limitations": [
            "The audit covers synchronized same-lane longitudinal actions only.",
            "Adjacent-lane cooperative lane changes require a separate conditional model.",
            "A failed independence check does not require a dense full joint table; a conditional factorization may be used.",
        ],
    }
    (output_dir / "joint_action_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = audit_joint_actions(
        args.source_root, args.manifest, args.output, config
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
