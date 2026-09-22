"""Fit an offline highD lane-change model with adjacent-lane context and backoff."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .highd_lane_change_baseline import _decision_rows, _direction_map
from .highd_ndd_baseline import _load_split_manifest


def _bin(values: np.ndarray, boundaries: list[float]) -> np.ndarray:
    return np.digitize(values, np.asarray(boundaries, dtype=float), right=False)


def hierarchical_probability(
    positive: np.ndarray,
    total: np.ndarray,
    index: np.ndarray,
    base_probability: np.ndarray,
    concentration: float,
) -> np.ndarray:
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    return (
        positive[index] + concentration * base_probability
    ) / (total[index] + concentration)


def _merge_neighbor(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    id_column: str,
    prefix: str,
) -> pd.DataFrame:
    renamed = lookup.rename(columns={
        "id": id_column,
        "x": f"{prefix}_x",
        "width": f"{prefix}_width",
        "xVelocity": f"{prefix}_xVelocity",
    })
    return rows[["frame", id_column]].merge(
        renamed, how="left", on=["frame", id_column], sort=False
    )


def _neighbor_features(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    side: str,
    travel_sign: np.ndarray,
    ego_speed: np.ndarray,
    gap_boundaries: list[float],
    relative_speed_boundaries: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    front = _merge_neighbor(rows, lookup, f"{side}PrecedingId", f"{side}_front")
    rear = _merge_neighbor(rows, lookup, f"{side}FollowingId", f"{side}_rear")
    ego_center = rows["x"].to_numpy(float) + rows["width"].to_numpy(float) / 2
    front_present = rows[f"{side}PrecedingId"].to_numpy(np.int64) > 0
    rear_present = rows[f"{side}FollowingId"].to_numpy(np.int64) > 0

    front_center = front[f"{side}_front_x"].to_numpy(float) + (
        front[f"{side}_front_width"].to_numpy(float) / 2
    )
    rear_center = rear[f"{side}_rear_x"].to_numpy(float) + (
        rear[f"{side}_rear_width"].to_numpy(float) / 2
    )
    front_gap = travel_sign * (front_center - ego_center) - (
        front[f"{side}_front_width"].to_numpy(float)
        + rows["width"].to_numpy(float)
    ) / 2
    rear_gap = travel_sign * (ego_center - rear_center) - (
        rear[f"{side}_rear_width"].to_numpy(float)
        + rows["width"].to_numpy(float)
    ) / 2
    front_rr = np.abs(front[f"{side}_front_xVelocity"].to_numpy(float)) - ego_speed
    rear_rr = ego_speed - np.abs(rear[f"{side}_rear_xVelocity"].to_numpy(float))
    if np.any(front_present & ~np.isfinite(front_center)):
        raise ValueError(f"Missing {side} front-neighbor row at a referenced frame")
    if np.any(rear_present & ~np.isfinite(rear_center)):
        raise ValueError(f"Missing {side} rear-neighbor row at a referenced frame")

    front_gap_category = np.zeros(len(rows), dtype=np.int16)
    rear_gap_category = np.zeros(len(rows), dtype=np.int16)
    front_rr_category = np.zeros(len(rows), dtype=np.int16)
    rear_rr_category = np.zeros(len(rows), dtype=np.int16)
    front_gap_category[front_present] = 1 + _bin(
        np.maximum(front_gap[front_present], 0), gap_boundaries
    )
    rear_gap_category[rear_present] = 1 + _bin(
        np.maximum(rear_gap[rear_present], 0), gap_boundaries
    )
    front_rr_category[front_present] = 1 + _bin(
        front_rr[front_present], relative_speed_boundaries
    )
    rear_rr_category[rear_present] = 1 + _bin(
        rear_rr[rear_present], relative_speed_boundaries
    )
    alongside = (rows[f"{side}AlongsideId"].to_numpy(np.int64) > 0).astype(np.int16)
    return (
        front_gap_category,
        front_rr_category,
        rear_gap_category,
        rear_rr_category,
        alongside,
    )


def _recording_arrays(
    source_root: Path,
    recording_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = _decision_rows(
        source_root,
        recording_id,
        source_hz=int(config["source_frequency_hz"]),
        target_hz=int(config["target_frequency_hz"]),
        decision_lead_s=float(config["decision_lead_s"]),
        execution_tail_s=float(config["execution_tail_s"]),
    ).reset_index(drop=True)
    required = {
        "x", "width", "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
        "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"highD context columns missing: {missing}")
    tracks = pd.read_csv(
        source_root / "data" / f"{recording_id}_tracks.csv",
        usecols=["frame", "id", "x", "width", "xVelocity", "laneId"],
    )
    lookup = tracks[["frame", "id", "x", "width", "xVelocity"]]
    direction_map = _direction_map(source_root, recording_id)
    vehicle_direction = tracks["id"].map(direction_map).to_numpy(np.int8)
    lane_sets = {
        direction: set(tracks.loc[vehicle_direction == direction, "laneId"].astype(int))
        for direction in (1, 2)
    }
    directions = rows["id"].map(direction_map).to_numpy(np.int8)
    travel_sign = np.where(directions == 1, -1.0, 1.0)
    ego_speed = np.abs(rows["xVelocity"].to_numpy(float))
    has_leader = rows["precedingId"].to_numpy(np.int64) > 0
    current_gap = np.where(
        has_leader, rows["dhw"].to_numpy(float), float(config["maximum_gap_m"])
    )
    current_rr = np.where(
        has_leader,
        np.abs(rows["precedingXVelocity"].to_numpy(float)) - ego_speed,
        0.0,
    )
    eligible = (
        np.isfinite(ego_speed)
        & (ego_speed >= config["speed_range_mps"][0])
        & (ego_speed <= config["speed_range_mps"][1])
        & (current_gap >= 0)
        & (current_gap <= config["maximum_gap_m"])
        & (current_rr >= config["relative_speed_range_mps"][0])
        & (current_rr <= config["relative_speed_range_mps"][1])
    )
    actions_before_filter = rows["action_index"].to_numpy(np.int8)
    original_d2rl_unsupported = eligible & ~has_leader
    if config.get("require_current_leader_for_lateral", False):
        eligible &= has_leader
    excluded_by_original_structure = int(original_d2rl_unsupported.sum())
    positive_excluded_by_original_structure = int(
        ((actions_before_filter != 1) & original_d2rl_unsupported).sum()
    )
    rows = rows.loc[eligible].reset_index(drop=True)
    directions = directions[eligible]
    travel_sign = travel_sign[eligible]
    ego_speed = ego_speed[eligible]
    current_gap = current_gap[eligible]
    current_rr = current_rr[eligible]
    actions = rows["action_index"].to_numpy(np.int8)

    speed_index = _bin(ego_speed, config["speed_boundaries_mps"])
    gap_index = _bin(current_gap, config["current_gap_boundaries_m"])
    rr_index = _bin(current_rr, config["relative_speed_boundaries_mps"])
    base_shape = (
        len(config["speed_boundaries_mps"]) + 1,
        len(config["current_gap_boundaries_m"]) + 1,
        len(config["relative_speed_boundaries_mps"]) + 1,
    )
    base_index = np.ravel_multi_index(
        (speed_index, gap_index, rr_index), base_shape
    )

    side_data = {}
    context_shape = base_shape + (
        len(config["target_gap_boundaries_m"]) + 2,
        len(config["relative_speed_boundaries_mps"]) + 2,
        len(config["target_gap_boundaries_m"]) + 2,
        len(config["relative_speed_boundaries_mps"]) + 2,
        2,
    )
    for side, action_index in (("left", 0), ("right", 2)):
        target_delta = np.where(
            directions == 1,
            1 if side == "left" else -1,
            -1 if side == "left" else 1,
        )
        target_lane = rows["laneId"].to_numpy(np.int64) + target_delta
        available = np.fromiter(
            (
                int(target_lane[i]) in lane_sets[int(directions[i])]
                for i in range(len(rows))
            ),
            dtype=bool,
            count=len(rows),
        )
        context = _neighbor_features(
            rows,
            lookup,
            side,
            travel_sign,
            ego_speed,
            config["target_gap_boundaries_m"],
            config["relative_speed_boundaries_mps"],
        )
        context_index = np.ravel_multi_index(
            (speed_index, gap_index, rr_index) + context, context_shape
        )
        side_data[side] = {
            "available": available,
            "base_index": base_index,
            "context_index": context_index,
            "positive": actions == action_index,
        }
    return {
        "actions": actions,
        "base_shape": base_shape,
        "context_shape": context_shape,
        "sides": side_data,
        "sampled_rows": len(rows),
        "excluded_by_original_no_leader_structure": excluded_by_original_structure,
        "positive_excluded_by_original_no_leader_structure": (
            positive_excluded_by_original_structure
        ),
    }


def _fit_counts(
    source_root: Path,
    recording_ids: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    base_total = base_positive = context_total = context_positive = None
    diagnostics = Counter()
    for recording_id in recording_ids:
        data = _recording_arrays(source_root, recording_id, config)
        if base_total is None:
            base_total = np.zeros(np.prod(data["base_shape"]), dtype=np.uint64)
            base_positive = np.zeros_like(base_total)
            context_total = np.zeros(np.prod(data["context_shape"]), dtype=np.uint64)
            context_positive = np.zeros_like(context_total)
        diagnostics["decision_rows"] += data["sampled_rows"]
        diagnostics["excluded_by_original_no_leader_structure"] += data[
            "excluded_by_original_no_leader_structure"
        ]
        diagnostics["positive_excluded_by_original_no_leader_structure"] += data[
            "positive_excluded_by_original_no_leader_structure"
        ]
        for side in ("left", "right"):
            values = data["sides"][side]
            mask = values["available"]
            diagnostics[f"{side}_available_opportunities"] += int(mask.sum())
            diagnostics[f"{side}_unavailable_opportunities"] += int((~mask).sum())
            diagnostics[f"{side}_positive_actions"] += int(values["positive"].sum())
            diagnostics[f"{side}_positive_while_unavailable"] += int(
                (values["positive"] & ~mask).sum()
            )
            np.add.at(base_total, values["base_index"][mask], 1)
            np.add.at(base_positive, values["base_index"][mask], values["positive"][mask])
            np.add.at(context_total, values["context_index"][mask], 1)
            np.add.at(
                context_positive,
                values["context_index"][mask],
                values["positive"][mask],
            )
    return {
        "base_total": base_total,
        "base_positive": base_positive,
        "context_total": context_total,
        "context_positive": context_positive,
        "diagnostics": dict(diagnostics),
    }


def _candidate_probabilities(
    data: dict[str, Any],
    counts: dict[str, Any],
    base_concentration: float,
    context_concentration: float,
    maximum_total_lane_change_probability: float,
    probability_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if probability_scale <= 0:
        raise ValueError("probability_scale must be positive")
    global_probability = (
        counts["base_positive"].sum() + 0.5
    ) / (counts["base_total"].sum() + 1.0)
    base_probability_table = (
        counts["base_positive"] + base_concentration * global_probability
    ) / (counts["base_total"] + base_concentration)
    result = []
    for side in ("left", "right"):
        values = data["sides"][side]
        base_probability = base_probability_table[values["base_index"]]
        probability = hierarchical_probability(
            counts["context_positive"],
            counts["context_total"],
            values["context_index"],
            base_probability,
            context_concentration,
        )
        probability = np.where(values["available"], probability, 0.0)
        result.append(probability)
    left, right = (result[0] * probability_scale, result[1] * probability_scale)
    total = left + right
    rescale = np.maximum(total / maximum_total_lane_change_probability, 1.0)
    return left / rescale, right / rescale


def _score_recordings(
    source_root: Path,
    recording_ids: list[str],
    config: dict[str, Any],
    counts: dict[str, Any],
    candidates: list[tuple[float, float]],
    probability_scale: float = 1.0,
) -> dict[str, Any]:
    accumulators = {
        f"base_{base:g}_context_{context:g}": Counter()
        for base, context in candidates
    }
    for recording_id in recording_ids:
        data = _recording_arrays(source_root, recording_id, config)
        actions = data["actions"]
        for base, context in candidates:
            name = f"base_{base:g}_context_{context:g}"
            left, right = _candidate_probabilities(
                data,
                counts,
                base,
                context,
                float(config["maximum_total_lane_change_probability"]),
                probability_scale,
            )
            stay = 1.0 - left - right
            selected = np.choose(actions, [left, stay, right])
            selected = np.maximum(selected, np.finfo(float).tiny)
            accumulator = accumulators[name]
            accumulator["decision_count"] += len(actions)
            accumulator["observed_lane_changes"] += int((actions != 1).sum())
            accumulator["predicted_lane_changes"] += float((left + right).sum())
            accumulator["negative_log_likelihood_sum"] += float(-np.log(selected).sum())
            distributions_square = left * left + stay * stay + right * right
            accumulator["brier_sum"] += float(
                (distributions_square - 2 * selected + 1).sum()
            )
    result = {}
    for name, values in accumulators.items():
        n = values["decision_count"]
        observed = values["observed_lane_changes"]
        result[name] = {
            "decision_count": int(n),
            "observed_lane_changes": int(observed),
            "predicted_lane_changes": values["predicted_lane_changes"],
            "predicted_to_observed_ratio": (
                values["predicted_lane_changes"] / observed if observed else None
            ),
            "mean_nll": values["negative_log_likelihood_sum"] / n,
            "mean_brier": values["brier_sum"] / n,
        }
    return result


def fit_context_model(
    source_root: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    splits = _load_split_manifest(Path(manifest_path))
    counts = _fit_counts(source_root, splits["train"], config)
    candidates = [
        (float(base), float(context))
        for base in config["base_concentrations"]
        for context in config["context_concentrations"]
    ]
    calibration = _score_recordings(
        source_root, splits["calibration"], config, counts, candidates
    )
    selected_name = min(calibration, key=lambda name: calibration[name]["mean_nll"])
    selected = next(
        candidate
        for candidate in candidates
        if f"base_{candidate[0]:g}_context_{candidate[1]:g}" == selected_name
    )
    selected_calibration = calibration[selected_name]
    probability_scale = (
        selected_calibration["observed_lane_changes"]
        / selected_calibration["predicted_lane_changes"]
    )
    validation_raw = _score_recordings(
        source_root, splits["validation"], config, counts, [selected]
    )
    validation_calibrated = _score_recordings(
        source_root,
        splits["validation"],
        config,
        counts,
        [selected],
        probability_scale=probability_scale,
    )
    validation_per_recording = {
        recording_id: _score_recordings(
            source_root,
            [recording_id],
            config,
            counts,
            [selected],
            probability_scale=probability_scale,
        )[selected_name]
        for recording_id in splits["validation"]
    }
    summary = {
        "schema_version": 1,
        "status": "offline_adjacent_lane_context_ablation_not_runtime_enabled",
        "probability_semantics": (
            "Each side is estimated conditionally on ego/current leader and target-lane "
            "front/rear/alongside context, then combined into left/stay/right."
        ),
        "recordings_by_split": splits,
        "config": config,
        "train_diagnostics": counts["diagnostics"],
        "occupied_base_states": int(np.count_nonzero(counts["base_total"])),
        "occupied_context_states": int(np.count_nonzero(counts["context_total"])),
        "selected_by_calibration_nll": selected_name,
        "probability_scale_from_calibration_rate": probability_scale,
        "calibration": calibration,
        "validation_before_rate_calibration": validation_raw[selected_name],
        "validation": validation_calibrated[selected_name],
        "validation_per_recording": validation_per_recording,
        "test_evaluated": False,
        "runtime_enabled": False,
        "limitations": [
            (
                "The primary model preserves the original D2RL rule that lateral "
                "lane-change decisions require a current-lane leader."
            ),
            "Observed highD free-flow lane changes are excluded and must be reported as unsupported coverage.",
            "The side-invariant model pools left and right opportunities to reduce sparsity.",
            "Neighbor relations use highD tracker IDs and are not driver-intention labels.",
            "Target gaps and relative speeds are coarsely binned with hierarchical backoff.",
            "This artifact is not yet used by SUMO or importance-weight calculations.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "highd_lane_change_context_v1.npz",
        base_total=counts["base_total"],
        base_positive=counts["base_positive"],
        context_total=counts["context_total"],
        context_positive=counts["context_positive"],
        selected_base_concentration=selected[0],
        selected_context_concentration=selected[1],
        probability_scale=probability_scale,
    )
    (output_dir / "context_summary.json").write_text(
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
    summary = fit_context_model(
        args.source_root, args.manifest, config, args.output
    )
    print(json.dumps({
        "selected": summary["selected_by_calibration_nll"],
        "validation": summary["validation"],
        "train_diagnostics": summary["train_diagnostics"],
    }, indent=2))


if __name__ == "__main__":
    main()
