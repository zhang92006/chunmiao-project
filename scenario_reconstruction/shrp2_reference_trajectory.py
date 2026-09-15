"""Export an auditable SHRP2 pre-impact trajectory as a soft reconstruction reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .shrp2_collision import (
    front_bumper_to_center,
    load_category,
    signed_box_clearance,
)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Only SHRP2 reference schema_version 1 is supported")
    for name in (
        "history_s",
        "sample_hz",
        "heading_speed_threshold_mps",
        "maximum_speed_path_rmse_mps",
        "maximum_heading_path_mae_rad",
    ):
        value = config.get(name)
        if isinstance(value, bool) or value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if int(config["sample_hz"]) != config["sample_hz"]:
        raise ValueError("sample_hz must be an integer")


def build_reference(
    source_root: str | Path,
    seed_path: str | Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build full trajectory and concise quality summary for one SHRP2 seed."""
    validate_config(config)
    seed_path = Path(seed_path)
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    source = seed.get("source") or {}
    if "event_id" not in source or "associated_target_id" not in source:
        raise ValueError(
            "seed must contain source.event_id and source.associated_target_id"
        )

    _, meta_path, data_path, metadata, data = load_category(source_root, "Crash")
    event_id = int(source["event_id"])
    target_id = int(source["associated_target_id"])
    meta = metadata[metadata["event_id"] == event_id]
    if meta.empty:
        raise ValueError(f"No SHRP2 Crash metadata for event_id={event_id}")
    metadata_row = meta.iloc[0]
    rows = data[
        (data["event_id"] == event_id) & (data["target_id"] == target_id)
    ].replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "time", "x_ego", "y_ego", "v_ego", "psi_ego",
        "x_sur", "y_sur", "v_sur", "psi_sur",
    ]).sort_values("time")
    if rows.empty:
        raise ValueError(f"No finite SHRP2 rows for event={event_id}, target={target_id}")

    impact_source_s = float(metadata_row["impact_timestamp"]) / 1000.0
    history_s = float(config["history_s"])
    source_start_s = impact_source_s - history_s
    tolerance_s = 0.5 / float(config["sample_hz"])
    if float(rows["time"].min()) > source_start_s + tolerance_s:
        raise ValueError("Selected target does not cover the requested pre-impact history")
    if float(rows["time"].max()) < impact_source_s - tolerance_s:
        raise ValueError("Selected target does not reach the annotated impact time")

    times = np.arange(
        0.0, history_s + tolerance_s, 1.0 / float(config["sample_hz"])
    )
    query_times = source_start_s + times
    raw_times = rows["time"].to_numpy(dtype=float)
    ego_impact = np.array([
        np.interp(impact_source_s, raw_times, rows["x_ego"]),
        np.interp(impact_source_s, raw_times, rows["y_ego"]),
    ])
    ego_impact_heading = _interp_angle(
        np.array([impact_source_s]), raw_times, rows["psi_ego"].to_numpy(dtype=float)
    )[0]
    rotation = _rotation(-ego_impact_heading)

    ego_world = np.column_stack((
        np.interp(query_times, raw_times, rows["x_ego"]),
        np.interp(query_times, raw_times, rows["y_ego"]),
    ))
    target_front_world = np.column_stack((
        np.interp(query_times, raw_times, rows["x_sur"]),
        np.interp(query_times, raw_times, rows["y_sur"]),
    ))
    target_world_samples = np.asarray([
        front_bumper_to_center(row.x_sur, row.y_sur, row.psi_sur, metadata_row["target_length"])
        for row in rows.itertuples()
    ])
    target_world = np.column_stack((
        np.interp(query_times, raw_times, target_world_samples[:, 0]),
        np.interp(query_times, raw_times, target_world_samples[:, 1]),
    ))

    ego_track = _actor_track(
        times=times,
        xy=(ego_world - ego_impact) @ rotation.T,
        speed=np.interp(query_times, raw_times, rows["v_ego"]),
        heading=_wrap_array(_interp_angle(
            query_times, raw_times, rows["psi_ego"].to_numpy(dtype=float)
        ) - ego_impact_heading),
        reported_acceleration=(
            np.interp(query_times, raw_times, rows["acc_ego"])
            if "acc_ego" in rows and rows["acc_ego"].notna().any() else None
        ),
        config=config,
    )
    target_track = _actor_track(
        times=times,
        xy=(target_world - ego_impact) @ rotation.T,
        speed=np.interp(query_times, raw_times, rows["v_sur"]),
        heading=_wrap_array(_interp_angle(
            query_times, raw_times, rows["psi_sur"].to_numpy(dtype=float)
        ) - ego_impact_heading),
        reported_acceleration=None,
        config=config,
    )
    target_track["source_front_bumper_xy_m"] = (
        (target_front_world - ego_impact) @ rotation.T
    ).tolist()
    actors = {
        "CAV": {
            "role": "CAV",
            "length_m": float(metadata_row["ego_length"]),
            "width_m": float(metadata_row["ego_width"]),
            **ego_track,
        },
        "BV_primary": {
            "role": "BV",
            "source_target_id": target_id,
            "length_m": float(metadata_row["target_length"]),
            "width_m": float(metadata_row["target_width"]),
            **target_track,
        },
    }
    collision_audit = _collision_audit(times, actors)
    quality = {
        "CAV": _quality_summary(actors["CAV"], config),
        "BV_primary": _quality_summary(
            actors["BV_primary"], config, alternate_position_key="source_front_bumper_xy_m"
        ),
    }
    provenance = {
        "dataset": "SHRP2 NDS public bird's-eye trajectory reconstruction",
        "doi": source.get("doi", "10.15787/VTT1/T7UUC1"),
        "category": "Crash",
        "event_id": event_id,
        "associated_target_id": target_id,
        "source_split": source.get("source_split"),
        "event_meta_sha256": _file_hash(meta_path),
        "event_data_sha256": _file_hash(data_path),
    }
    reference = {
        "schema_version": 1,
        "record_type": "shrp2_preimpact_soft_reference",
        "status": (
            "measured trajectory reference for fitting and validation; not a forced "
            "SUMO replay, executable controller, or probability estimate"
        ),
        "source": provenance,
        "alignment": {
            "source_impact_timestamp_s": impact_source_s,
            "reference_time_s": [float(times[0]), float(times[-1])],
            "sample_hz": int(config["sample_hz"]),
            "origin": "CAV centroid interpolated at annotated impact",
            "x_axis": "CAV reported heading interpolated at annotated impact",
            "target_position": "front bumper converted to center before interpolation",
        },
        "actors": actors,
        "quality": quality,
        "collision_audit": collision_audit,
        "constraint_policy": {
            "hard_constraints": [
                "initial actor state",
                "collision actor pair",
                "annotated impact time tolerance",
                "impact relative position and relative speed",
            ],
            "soft_constraints": [
                "pre-impact position trajectory",
                "reported speed trajectory",
                "reported heading where quality permits",
                "derived acceleration profile",
            ],
            "never_claim": "pointwise trajectory equality is not required for closed-loop reconstruction",
        },
    }
    summary = {
        "schema_version": 1,
        "status": "SHRP2 pre-impact soft-reference extraction summary",
        "source": provenance,
        "sample_count": int(len(times)),
        "reference_window_s": [float(times[0]), float(times[-1])],
        "quality": quality,
        "collision_audit": collision_audit,
        "recommended_usage": {
            "position": {
                "CAV": "centroid position is usable as a soft fitting objective",
                "BV_primary": (
                    "use the measured front-bumper path as a soft objective; the converted "
                    "center path is diagnostic until target heading is reviewed"
                ),
            },
            "speed": "separate soft fitting objective; do not infer it from the target center path",
            "heading": {
                actor_id: (
                    "soft fitting objective" if metrics["heading_usable"]
                    else "diagnostic only until source trajectory is reviewed"
                )
                for actor_id, metrics in quality.items()
            },
            "impact": "hard event constraint",
            "sumo_execution": "fit controller/fault parameters; do not teleport actors through every point",
        },
    }
    return reference, summary


def _actor_track(times, xy, speed, heading, reported_acceleration, config):
    velocity = np.gradient(xy, times, axis=0)
    path_speed = np.linalg.norm(velocity, axis=1)
    path_heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    derived_acceleration = np.gradient(speed, times)
    track = {
        "time_s": times.tolist(),
        "xy_m": xy.tolist(),
        "reported_speed_mps": speed.tolist(),
        "path_speed_mps": path_speed.tolist(),
        "derived_acceleration_mps2": derived_acceleration.tolist(),
        "reported_heading_rad": heading.tolist(),
        "path_heading_rad": path_heading.tolist(),
    }
    if reported_acceleration is not None:
        track["reported_acceleration_mps2"] = reported_acceleration.tolist()
    return track


def _quality_summary(actor, config, alternate_position_key=None):
    speed = np.asarray(actor["reported_speed_mps"], dtype=float)
    path_speed = np.asarray(actor["path_speed_mps"], dtype=float)
    heading = np.asarray(actor["reported_heading_rad"], dtype=float)
    path_heading = np.asarray(actor["path_heading_rad"], dtype=float)
    moving = (speed >= config["heading_speed_threshold_mps"]) & (
        path_speed >= config["heading_speed_threshold_mps"]
    )
    heading_errors = np.abs(_wrap_array(heading[moving] - path_heading[moving]))
    speed_rmse = float(np.sqrt(np.mean((speed - path_speed) ** 2)))
    heading_mae = float(np.mean(heading_errors)) if heading_errors.size else None
    position_usable = speed_rmse <= config["maximum_speed_path_rmse_mps"]
    heading_usable = bool(
        position_usable
        and heading_mae is not None
        and heading_mae <= config["maximum_heading_path_mae_rad"]
    )
    result = {
        "reported_vs_path_speed_rmse_mps": speed_rmse,
        "reported_vs_path_heading_mae_rad": heading_mae,
        "heading_comparison_samples": int(heading_errors.size),
        "position_speed_consistent": bool(position_usable),
        "heading_usable": heading_usable,
        "rules": {
            "maximum_speed_path_rmse_mps": config["maximum_speed_path_rmse_mps"],
            "maximum_heading_path_mae_rad": config["maximum_heading_path_mae_rad"],
            "minimum_heading_speed_mps": config["heading_speed_threshold_mps"],
        },
    }
    if "reported_acceleration_mps2" in actor:
        reported = np.asarray(actor["reported_acceleration_mps2"], dtype=float)
        derived = np.asarray(actor["derived_acceleration_mps2"], dtype=float)
        result["reported_vs_derived_acceleration_rmse_mps2"] = float(
            np.sqrt(np.mean((reported - derived) ** 2))
        )
    if alternate_position_key:
        alternate_xy = np.asarray(actor[alternate_position_key], dtype=float)
        times = np.asarray(actor["time_s"], dtype=float)
        alternate_velocity = np.gradient(alternate_xy, times, axis=0)
        alternate_speed = np.linalg.norm(alternate_velocity, axis=1)
        alternate_heading = np.arctan2(alternate_velocity[:, 1], alternate_velocity[:, 0])
        alternate_moving = (speed >= config["heading_speed_threshold_mps"]) & (
            alternate_speed >= config["heading_speed_threshold_mps"]
        )
        alternate_heading_error = np.abs(_wrap_array(
            heading[alternate_moving] - alternate_heading[alternate_moving]
        ))
        result["source_front_bumper_diagnostics"] = {
            "reported_vs_path_speed_rmse_mps": float(
                np.sqrt(np.mean((speed - alternate_speed) ** 2))
            ),
            "reported_vs_path_heading_mae_rad": (
                float(np.mean(alternate_heading_error))
                if alternate_heading_error.size else None
            ),
            "heading_comparison_samples": int(alternate_heading_error.size),
            "note": (
                "SHRP2 x_sur/y_sur describe the target front bumper; converting to the "
                "vehicle center amplifies heading noise when psi_sur changes rapidly"
            ),
        }
    return result


def _collision_audit(times, actors):
    ego, target = actors["CAV"], actors["BV_primary"]
    clearances = []
    for index in range(len(times)):
        clearances.append(signed_box_clearance(
            ego["xy_m"][index], ego["reported_heading_rad"][index],
            ego["length_m"], ego["width_m"],
            target["xy_m"][index], target["reported_heading_rad"][index],
            target["length_m"], target["width_m"],
        ))
    first_index = next((index for index, value in enumerate(clearances) if value <= 0), None)
    return {
        "initial_collision": bool(clearances[0] <= 0),
        "collision_in_reference_window": first_index is not None,
        "first_sampled_contact_time_s": (
            float(times[first_index]) if first_index is not None else None
        ),
        "minimum_sampled_signed_clearance_m": float(min(clearances)),
        "method": "oriented boxes at resampled source states; no between-sample interpolation",
    }


def _interp_angle(query_times, source_times, angles):
    return _wrap_array(np.interp(query_times, source_times, np.unwrap(angles)))


def _wrap_array(values):
    values = np.asarray(values, dtype=float)
    return (values + math.pi) % (2 * math.pi) - math.pi


def _rotation(angle):
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--config", default="configs/shrp2_reference_trajectory.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    reference, summary = build_reference(args.source_root, args.seed, config)
    _write_json(args.output, reference)
    if args.summary_output:
        _write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
