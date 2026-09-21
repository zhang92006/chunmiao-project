"""Audit SHRP2 candidate events before fitting closed-loop trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .shrp2_collision import (
    associate_primary_target,
    load_category,
    project_split,
)
from .shrp2_reference_trajectory import (
    _actor_track,
    _collision_audit,
    _file_hash,
    _interp_angle,
    _quality_summary,
    _rotation,
    _wrap_array,
    validate_config,
)
from .shrp2_collision import front_bumper_to_center


def audit_candidates(
    source_root: str | Path,
    collision_config: dict[str, Any],
    reference_config: dict[str, Any],
    conflict: str = "leading",
) -> dict[str, Any]:
    """Rank event/target candidates using source trajectory consistency checks."""
    validate_config(reference_config)
    if conflict not in collision_config.get("conflict_types", {}):
        raise ValueError(f"Unknown conflict type: {conflict}")
    _, meta_path, data_path, metadata, data = load_category(source_root, "Crash")
    subset = metadata[
        (metadata["conflict"] == conflict)
        & (metadata["severity_first"] == 3)
        & (metadata["duration_enough"] == True)  # noqa: E712
    ]
    records = []
    for _, metadata_row in subset.sort_values("event_id").iterrows():
        event_id = int(metadata_row["event_id"])
        if project_split(
            event_id,
            collision_config["seed"],
            collision_config["split_fractions"],
        ) != collision_config["source_split"]:
            continue
        event_rows = data[data["event_id"] == event_id]
        association = associate_primary_target(
            event_rows,
            metadata_row,
            collision_config["maximum_target_alignment_dt_s"],
        )
        if association is None:
            records.append({
                "event_id": event_id,
                "status": "no_target_association",
            })
            continue
        if association["source_box_clearance_m"] > collision_config["maximum_source_box_clearance_m"]:
            records.append({
                "event_id": event_id,
                "target_id": association["target_id"],
                "source_box_clearance_m": association["source_box_clearance_m"],
                "status": "source_clearance_exceeds_limit",
            })
            continue
        target_id = int(association["target_id"])
        rows = event_rows[event_rows["target_id"] == target_id].replace(
            [np.inf, -np.inf], np.nan
        ).dropna(subset=[
            "time", "x_ego", "y_ego", "v_ego", "psi_ego",
            "x_sur", "y_sur", "v_sur", "psi_sur",
        ]).sort_values("time")
        try:
            quality, collision = _score_rows(rows, metadata_row, reference_config)
            ready = _is_reconstruction_ready(quality, collision, reference_config)
            records.append({
                "event_id": event_id,
                "target_id": target_id,
                "source_box_clearance_m": float(association["source_box_clearance_m"]),
                "source_sample_dt_s": float(association["source_sample_dt_s"]),
                "trajectory_start_s": float(rows["time"].min()),
                "trajectory_end_s": float(rows["time"].max()),
                "status": "audited",
                "quality": quality,
                "collision_audit": collision,
                "reconstruction_ready": ready,
                "ranking_score": _ranking_score(quality, collision, reference_config),
            })
        except ValueError as exc:
            records.append({
                "event_id": event_id,
                "target_id": target_id,
                "source_box_clearance_m": float(association["source_box_clearance_m"]),
                "status": "insufficient_trajectory_window",
                "reason": str(exc),
            })
    audited = [record for record in records if record["status"] == "audited"]
    ranked = sorted(
        audited,
        key=lambda record: (
            not record["reconstruction_ready"],
            record["ranking_score"],
            record["event_id"],
        ),
    )
    summary = {
        "schema_version": 1,
        "status": "SHRP2 candidate trajectory quality audit; no SUMO execution",
        "dataset": "SHRP2 NDS public bird's-eye trajectory reconstruction",
        "dataset_doi": "10.15787/VTT1/T7UUC1",
        "source_category": "Crash",
        "source_split": collision_config["source_split"],
        "conflict": conflict,
        "source_file_sha256": {
            "event_meta.csv": _file_hash(meta_path),
            "event_data.h5": _file_hash(data_path),
        },
        "candidate_count_in_split": len(records),
        "audited_count": len(audited),
        "reconstruction_ready_count": sum(record["reconstruction_ready"] for record in audited),
        "top_ranked_event_ids": [record["event_id"] for record in ranked[:10]],
        "records": [_compact_record(record) for record in records],
        "selection_rule": {
            "CAV": "position-speed consistency and heading consistency pass",
            "BV": "front-bumper speed RMSE and heading MAE pass",
            "event": "initial state is collision-free and collision time is within the configured error tolerance",
            "maximum_collision_time_error_s": reference_config["maximum_collision_time_error_s"],
            "hard_exclusions": [
                "missing target association",
                "source clearance above collision-config limit",
                "pre-impact history shorter than reference window",
            ],
        },
        "interpretation": [
            "This audit identifies source events suitable for soft fitting; it does not claim exact accident replay.",
            "The measured target position is kept at its front bumper; center conversion is not accepted as a hard truth when psi_sur is unstable.",
            "If reconstruction_ready_count is zero, review the top-ranked events and relax no threshold silently; change thresholds only in a documented ablation.",
        ],
    }
    return summary


def _compact_record(record):
    """Keep the committed audit readable while preserving selection evidence."""
    if record.get("status") != "audited":
        return record
    quality = record["quality"]
    bv_front = quality["BV_primary"].get("source_front_bumper_diagnostics", {})
    return {
        "event_id": record["event_id"],
        "target_id": record["target_id"],
        "source_box_clearance_m": record["source_box_clearance_m"],
        "source_sample_dt_s": record["source_sample_dt_s"],
        "trajectory_window_s": [record["trajectory_start_s"], record["trajectory_end_s"]],
        "status": record["status"],
        "reconstruction_ready": record["reconstruction_ready"],
        "ranking_score": record["ranking_score"],
        "collision": {
            "initial_collision": record["collision_audit"]["initial_collision"],
            "first_sampled_contact_time_s": record["collision_audit"]["first_sampled_contact_time_s"],
            "collision_in_reference_window": record["collision_audit"]["collision_in_reference_window"],
        },
        "quality": {
            "cav_speed_path_rmse_mps": quality["CAV"]["reported_vs_path_speed_rmse_mps"],
            "cav_heading_path_mae_rad": quality["CAV"]["reported_vs_path_heading_mae_rad"],
            "bv_center_speed_path_rmse_mps": quality["BV_primary"]["reported_vs_path_speed_rmse_mps"],
            "bv_front_bumper_speed_path_rmse_mps": bv_front.get("reported_vs_path_speed_rmse_mps"),
            "bv_front_bumper_heading_path_mae_rad": bv_front.get("reported_vs_path_heading_mae_rad"),
        },
    }


def _score_rows(rows, metadata_row, config):
    impact_s = float(metadata_row["impact_timestamp"]) / 1000.0
    history_s = float(config["history_s"])
    source_start_s = impact_s - history_s
    tolerance_s = 0.5 / float(config["sample_hz"])
    if float(rows["time"].min()) > source_start_s + tolerance_s:
        raise ValueError("pre-impact history is shorter than configured window")
    if float(rows["time"].max()) < impact_s - tolerance_s:
        raise ValueError("trajectory does not reach annotated impact")
    times = np.arange(0.0, history_s + tolerance_s, 1.0 / float(config["sample_hz"]))
    query = source_start_s + times
    raw_times = rows["time"].to_numpy(dtype=float)
    ego_impact = np.array([
        np.interp(impact_s, raw_times, rows["x_ego"]),
        np.interp(impact_s, raw_times, rows["y_ego"]),
    ])
    impact_heading = _interp_angle(
        np.array([impact_s]), raw_times, rows["psi_ego"].to_numpy(dtype=float)
    )[0]
    rotation = _rotation(-impact_heading)
    ego_world = np.column_stack((
        np.interp(query, raw_times, rows["x_ego"]),
        np.interp(query, raw_times, rows["y_ego"]),
    ))
    front_world = np.column_stack((
        np.interp(query, raw_times, rows["x_sur"]),
        np.interp(query, raw_times, rows["y_sur"]),
    ))
    center_samples = np.asarray([
        front_bumper_to_center(row.x_sur, row.y_sur, row.psi_sur, metadata_row["target_length"])
        for row in rows.itertuples()
    ])
    center_world = np.column_stack((
        np.interp(query, raw_times, center_samples[:, 0]),
        np.interp(query, raw_times, center_samples[:, 1]),
    ))
    ego = _actor_track(
        times,
        (ego_world - ego_impact) @ rotation.T,
        np.interp(query, raw_times, rows["v_ego"]),
        _wrap_array(_interp_angle(query, raw_times, rows["psi_ego"].to_numpy(dtype=float)) - impact_heading),
        None,
        config,
    )
    target = _actor_track(
        times,
        (center_world - ego_impact) @ rotation.T,
        np.interp(query, raw_times, rows["v_sur"]),
        _wrap_array(_interp_angle(query, raw_times, rows["psi_sur"].to_numpy(dtype=float)) - impact_heading),
        None,
        config,
    )
    target["source_front_bumper_xy_m"] = ((front_world - ego_impact) @ rotation.T).tolist()
    actors = {
        "CAV": {"length_m": float(metadata_row["ego_length"]), "width_m": float(metadata_row["ego_width"]), **ego},
        "BV_primary": {"length_m": float(metadata_row["target_length"]), "width_m": float(metadata_row["target_width"]), **target},
    }
    quality = {
        "CAV": _quality_summary(actors["CAV"], config),
        "BV_primary": _quality_summary(actors["BV_primary"], config, "source_front_bumper_xy_m"),
    }
    return quality, _collision_audit(times, actors)


def _is_reconstruction_ready(quality, collision, config):
    cav = quality["CAV"]
    bv = quality["BV_primary"]
    front = bv.get("source_front_bumper_diagnostics", {})
    front_speed = front.get("reported_vs_path_speed_rmse_mps")
    front_heading = front.get("reported_vs_path_heading_mae_rad")
    contact_time = collision.get("first_sampled_contact_time_s")
    contact_time_ok = (
        contact_time is not None
        and abs(contact_time - config["history_s"]) <= config["maximum_collision_time_error_s"]
    )
    return bool(
        not collision["initial_collision"]
        and collision["collision_in_reference_window"]
        and contact_time_ok
        and cav["position_speed_consistent"]
        and cav["heading_usable"]
        and front_speed is not None
        and front_heading is not None
        and front_speed <= 1.0
        and front_heading <= 0.35
    )


def _ranking_score(quality, collision, config):
    cav = quality["CAV"]
    bv = quality["BV_primary"]
    front = bv.get("source_front_bumper_diagnostics", {})
    front_speed = front.get("reported_vs_path_speed_rmse_mps")
    front_heading = front.get("reported_vs_path_heading_mae_rad")
    contact_time = collision.get("first_sampled_contact_time_s")
    timing_penalty = (
        abs(contact_time - config["history_s"]) if contact_time is not None else 10.0
    )
    return float(
        cav["reported_vs_path_speed_rmse_mps"]
        + (cav["reported_vs_path_heading_mae_rad"] or 1.0)
        + (front_speed if front_speed is not None else 10.0)
        + (front_heading if front_heading is not None else 3.14)
        + timing_penalty
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--collision_config", default="configs/shrp2_collision_pilot.json")
    parser.add_argument("--reference_config", default="configs/shrp2_reference_trajectory.json")
    parser.add_argument("--conflict", default="leading")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    collision_config = json.loads(Path(args.collision_config).read_text(encoding="utf-8"))
    reference_config = json.loads(Path(args.reference_config).read_text(encoding="utf-8"))
    summary = audit_candidates(args.source_root, collision_config, reference_config, args.conflict)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
