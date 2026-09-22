"""Fit a highD lane-change decision baseline without enabling it at runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .highd_ndd_baseline import _axis, _load_split_manifest, _nearest_index


ACTION_NAMES = ("left", "stay", "right")


def _target_rate_mask(frames: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    return (
        ((frames + 1) * target_hz) // source_hz
        > (frames * target_hz) // source_hz
    )


def _direction_map(source_root: Path, recording_id: str) -> dict[int, int]:
    metadata = pd.read_csv(
        source_root / "data" / f"{recording_id}_tracksMeta.csv",
        usecols=["id", "drivingDirection"],
    )
    return dict(zip(metadata["id"].astype(int), metadata["drivingDirection"].astype(int)))


def _decision_rows(
    source_root: Path,
    recording_id: str,
    *,
    source_hz: int,
    target_hz: int,
    decision_lead_s: float,
) -> pd.DataFrame:
    columns = [
        "frame", "id", "xVelocity", "dhw", "precedingXVelocity",
        "precedingId", "laneId",
    ]
    tracks = pd.read_csv(
        source_root / "data" / f"{recording_id}_tracks.csv", usecols=columns
    )
    directions = _direction_map(source_root, recording_id)
    selected_groups: list[pd.DataFrame] = []
    lead_frames = int(round(decision_lead_s * source_hz))
    for vehicle_id, group in tracks.groupby("id", sort=False):
        group = group.sort_values("frame").copy()
        frames = group["frame"].to_numpy(dtype=np.int64)
        lanes = group["laneId"].to_numpy(dtype=np.int64)
        keep = _target_rate_mask(frames, source_hz, target_hz)
        actions = np.ones(len(group), dtype=np.int8)
        transitions = np.flatnonzero(lanes[1:] != lanes[:-1]) + 1
        for transition in transitions:
            target_frame = frames[transition] - lead_frames
            decision = int(np.searchsorted(frames, target_frame, side="right") - 1)
            if decision < 0 or decision >= transition:
                continue
            direction = directions[int(vehicle_id)]
            lane_delta = int(lanes[transition] - lanes[transition - 1])
            is_left = (direction == 1 and lane_delta > 0) or (
                direction == 2 and lane_delta < 0
            )
            actions[decision] = 0 if is_left else 2
            keep[decision] = True
            # Once a lane-change decision has been made, intermediate frames
            # are execution states rather than fresh keep-lane decisions.
            keep[decision + 1:transition + 1] = False
        group = group.loc[keep].copy()
        group["action_index"] = actions[keep]
        selected_groups.append(group)
    if not selected_groups:
        return tracks.iloc[0:0].assign(action_index=np.array([], dtype=np.int8))
    return pd.concat(selected_groups, ignore_index=True)


def _state_action(
    rows: pd.DataFrame,
    axes: dict[str, np.ndarray],
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    speed = np.abs(rows["xVelocity"].to_numpy(dtype=float))
    has_leader = rows["precedingId"].to_numpy(dtype=np.int64) > 0
    gap = rows["dhw"].to_numpy(dtype=float)
    leader_speed = np.abs(rows["precedingXVelocity"].to_numpy(dtype=float))
    range_rate = leader_speed - speed
    gap = np.where(has_leader, gap, axes["gap"][-1])
    range_rate = np.where(has_leader, range_rate, 0.0)
    eligible = (
        np.isfinite(speed)
        & (speed >= axes["speed"][0])
        & (speed <= axes["speed"][-1])
        & (gap >= axes["gap"][0])
        & (gap <= axes["gap"][-1])
        & (range_rate >= axes["range_rate"][0])
        & (range_rate <= axes["range_rate"][-1])
    )
    indices = tuple(
        _nearest_index(values[eligible], axes[name])
        for name, values in (
            ("speed", speed), ("gap", gap), ("range_rate", range_rate)
        )
    )
    actions = rows["action_index"].to_numpy(dtype=np.int64)[eligible]
    return indices, actions, eligible


def _probabilities(counts: np.ndarray, alpha: float) -> np.ndarray:
    return ((counts + alpha) / (counts.sum(axis=-1, keepdims=True) + 3 * alpha)).astype(
        np.float32
    )


def _evaluate(
    source_root: Path,
    recording_ids: list[str],
    axes: dict[str, np.ndarray],
    counts: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    probabilities = _probabilities(counts, float(config["laplace_alpha"]))
    state_totals = counts.sum(axis=-1)
    metrics = Counter()
    log_likelihood = 0.0
    brier_sum = 0.0
    lane_change_log_likelihood = 0.0
    stay_log_likelihood = 0.0
    observed_lane_change_probability_sum = 0.0
    for recording_id in recording_ids:
        rows = _decision_rows(
            source_root,
            recording_id,
            source_hz=int(config["source_frequency_hz"]),
            target_hz=int(config["target_frequency_hz"]),
            decision_lead_s=float(config["decision_lead_s"]),
        )
        indices, actions, eligible = _state_action(rows, axes)
        distribution = probabilities[indices]
        selected = distribution[np.arange(len(actions)), actions]
        seen = state_totals[indices] > 0
        metrics["sampled_decision_rows"] += len(rows)
        metrics["eligible_rows"] += len(actions)
        metrics["seen_state_rows"] += int(seen.sum())
        for index, name in enumerate(ACTION_NAMES):
            metrics[f"{name}_actions"] += int((actions == index).sum())
        metrics["top1_correct_rows"] += int(
            (np.argmax(distribution, axis=-1) == actions).sum()
        )
        log_likelihood += float(np.log(selected).sum())
        lane_change = actions != 1
        lane_change_log_likelihood += float(np.log(selected[lane_change]).sum())
        stay_log_likelihood += float(np.log(selected[~lane_change]).sum())
        observed_lane_change_probability_sum += float(selected[lane_change].sum())
        brier_sum += float(
            (np.square(distribution).sum(axis=-1) - 2.0 * selected + 1.0).sum()
        )
    n = metrics["eligible_rows"]
    lane_changes = metrics["left_actions"] + metrics["right_actions"]
    return {
        **dict(metrics),
        "state_coverage": metrics["seen_state_rows"] / n if n else None,
        "mean_negative_log_likelihood": -log_likelihood / n if n else None,
        "mean_brier_score": brier_sum / n if n else None,
        "top1_action_accuracy": metrics["top1_correct_rows"] / n if n else None,
        "lane_change_rate_per_1000_decisions": 1000.0 * lane_changes / n if n else None,
        "lane_change_mean_negative_log_likelihood": (
            -lane_change_log_likelihood / lane_changes if lane_changes else None
        ),
        "stay_mean_negative_log_likelihood": (
            -stay_log_likelihood / metrics["stay_actions"]
            if metrics["stay_actions"] else None
        ),
        "mean_probability_of_observed_lane_change": (
            observed_lane_change_probability_sum / lane_changes
            if lane_changes else None
        ),
    }


def fit_lane_change_baseline(
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
        name: _axis(config["grid"][name])
        for name in ("speed", "gap", "range_rate")
    }
    counts = np.zeros(tuple(len(axes[name]) for name in axes) + (3,), dtype=np.uint32)
    train_split = str(config.get("train_split", "train"))
    for recording_id in splits[train_split]:
        rows = _decision_rows(
            source_root,
            recording_id,
            source_hz=int(config["source_frequency_hz"]),
            target_hz=int(config["target_frequency_hz"]),
            decision_lead_s=float(config["decision_lead_s"]),
        )
        indices, actions, _ = _state_action(rows, axes)
        np.add.at(counts, indices + (actions,), 1)

    model_path = output_dir / "highd_lane_change_ndd_v1.npz"
    np.savez_compressed(
        model_path,
        **{f"{name}_axis": axis for name, axis in axes.items()},
        action_names=np.array(ACTION_NAMES),
        counts=counts,
        probability=_probabilities(counts, float(config["laplace_alpha"])),
    )
    evaluation_splits = config.get("evaluation_splits", list(splits))
    unknown_splits = sorted(set(evaluation_splits) - set(splits))
    if unknown_splits:
        raise ValueError(f"Unknown evaluation splits: {unknown_splits}")
    evaluation = {
        split: _evaluate(source_root, splits[split], axes, counts, config)
        for split in evaluation_splits
    }
    summary = {
        "schema_version": 1,
        "status": "pilot_lane_change_baseline_not_runtime_enabled",
        "dataset": "highD-v1.0",
        "target_domain": "highway_20_to_40_mps",
        "probability_semantics": "p_highD(left, stay, right | speed, gap, range_rate)",
        "label_semantics": (
            f"discrete action {float(config['decision_lead_s'])} seconds before "
            "the observed lane-boundary crossing"
        ),
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "recordings_by_split": splits,
        "source_frequency_hz": int(config["source_frequency_hz"]),
        "target_frequency_hz": int(config["target_frequency_hz"]),
        "decision_lead_s": float(config["decision_lead_s"]),
        "laplace_alpha": float(config["laplace_alpha"]),
        "evaluation_splits": evaluation_splits,
        "occupied_states": int(np.count_nonzero(counts.sum(axis=-1))),
        "possible_states": int(np.prod(counts.shape[:-1])),
        "evaluation": evaluation,
        "limitations": [
            "The decision label is placed before the observed lane-boundary crossing.",
            "This first model uses one-leader state only; adjacent-lane vehicle state is not included.",
            "The model is not enabled in SUMO/NADE and is not used for importance weights.",
        ],
    }
    (output_dir / "lane_change_summary.json").write_text(
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
    summary = fit_lane_change_baseline(
        args.source_root, args.manifest, args.output, config
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
