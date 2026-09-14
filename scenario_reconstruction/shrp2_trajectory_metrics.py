"""Compute auditable SHRP2 seed and SUMO trajectory reproduction metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .shrp2_collision import front_bumper_to_center, load_category


def compute_metrics(
    source_root: str | Path,
    seed_path: str | Path,
    sumo_episode_path: str | Path,
    sumo_template_path: str | Path,
) -> dict[str, Any]:
    seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    episode = json.loads(Path(sumo_episode_path).read_text(encoding="utf-8"))
    template = json.loads(Path(sumo_template_path).read_text(encoding="utf-8"))
    source = seed.get("source")
    if not source or "event_id" not in source or "associated_target_id" not in source:
        raise ValueError(
            "seed must be an original SHRP2 kinematic scenario with source.event_id "
            "and source.associated_target_id; a SUMO bridge template is not valid here"
        )
    export_root, _, _, metadata, data = load_category(source_root, "Crash")
    meta = metadata[metadata["event_id"] == int(source["event_id"])]
    if meta.empty:
        raise ValueError(f"No SHRP2 metadata row for event_id={source['event_id']}")
    metadata_row = meta.iloc[0]
    event_rows = data[
        (data["event_id"] == int(source["event_id"]))
        & (data["target_id"] == int(source["associated_target_id"]))
    ].copy()
    if event_rows.empty:
        raise ValueError("No SHRP2 rows for the selected target")

    impact_s = float(metadata_row["impact_timestamp"]) / 1000.0
    reference_index = (event_rows["time"] - impact_s).abs().idxmin()
    reference = event_rows.loc[reference_index]
    reference_heading = float(reference["psi_ego"])
    reference_ego = np.array([float(reference["x_ego"]), float(reference["y_ego"])])
    source_times = np.asarray(seed["time_s"], dtype=float)
    impact_time = float(seed["impact_conditioning"]["requested_impact_time_s"])
    comparison_times = source_times[source_times <= impact_time + 1e-9]

    raw_tracks = _normalized_raw_tracks(
        event_rows, metadata_row, impact_s, reference_ego, reference_heading, impact_time
    )
    seed_tracks = _seed_tracks(seed)
    seed_metrics = {
        actor: _track_metrics(
            raw_tracks[actor], seed_tracks[actor], comparison_times
        )
        for actor in ("CAV", "BV_primary")
    }

    sumo_tracks = _sumo_tracks(episode, template)
    sumo_reference_tracks = _bridge_kinematic_tracks(template, sumo_tracks["times"])
    sumo_metrics = {
        actor: _track_metrics(
            sumo_tracks[actor], sumo_reference_tracks[actor], sumo_tracks["times"]
        )
        for actor in ("CAV", "BV_primary")
    }
    ttc = [float(value) for time, value in episode.get("ttc_step_info", {}).items() if float(time) <= impact_time]
    distances = [float(value) for time, value in episode.get("distance_step_info", {}).items() if float(time) <= impact_time]
    return {
        "status": "trajectory metrics; raw SHRP2 and SUMO closed-loop errors are reported separately",
        "source_root": str(export_root),
        "source_event_id": int(source["event_id"]),
        "source_target_id": int(source["associated_target_id"]),
        "source_split": source.get("source_split"),
        "comparison_window_s": [float(comparison_times.min()), float(comparison_times.max())],
        "comparison_sample_count": int(len(comparison_times)),
        "seed_vs_shrp2_raw": {
            "alignment": "translate to the SHRP2 CAV state nearest annotated impact and rotate by its heading; compare only pre-impact [0, impact_time]",
            "actors": seed_metrics,
        },
        "sumo_closed_loop_vs_bridge_kinematic_reference": {
            "alignment": "compare SUMO world coordinates against constant-speed, same-lane projection from the bridge template",
            "actors": sumo_metrics,
            "episode_collision": bool(episode.get("collision_result")),
            "episode_collision_ids": episode.get("collision_id") or [],
            "minimum_ttc_pre_target_s": min((value for value in ttc if value < 9999), default=None),
            "minimum_distance_pre_target_m": min(distances, default=None),
        },
        "interpretation": [
            "seed_vs_shrp2_raw measures fidelity lost when the source trajectory is replaced by constant-heading, constant-speed kinematics and impact conditioning.",
            "sumo_closed_loop_vs_bridge_kinematic_reference measures controller and map divergence, not raw SHRP2 replay error.",
            "No claim of full-dataset reproduction is made from this single high-quality rear-end event.",
        ],
    }


def _normalized_raw_tracks(rows, metadata_row, impact_s, reference_ego, reference_heading, impact_time):
    target_length = float(metadata_row["target_length"])
    rows = rows.sort_values("time")
    times = rows["time"].to_numpy(dtype=float) - impact_s + impact_time
    ego_xy, target_xy, ego_speed, target_speed, ego_heading, target_heading = [], [], [], [], [], []
    for _, row in rows.iterrows():
        ego_point = np.array([float(row["x_ego"]), float(row["y_ego"])]) - reference_ego
        target_center = front_bumper_to_center(
            float(row["x_sur"]), float(row["y_sur"]), float(row["psi_sur"]), target_length
        ) - reference_ego
        ego_xy.append(_rotate(ego_point, -reference_heading))
        target_xy.append(_rotate(target_center, -reference_heading))
        ego_speed.append(float(row["v_ego"]))
        target_speed.append(float(row["v_sur"]))
        ego_heading.append(_wrap(float(row["psi_ego"]) - reference_heading))
        target_heading.append(_wrap(float(row["psi_sur"]) - reference_heading))
    return {
        "CAV": {"times": times, "position": np.asarray(ego_xy), "speed": np.asarray(ego_speed), "heading": np.asarray(ego_heading)},
        "BV_primary": {"times": times, "position": np.asarray(target_xy), "speed": np.asarray(target_speed), "heading": np.asarray(target_heading)},
    }


def _seed_tracks(seed):
    tracks = {}
    for actor in seed["actors"]:
        tracks[actor["id"]] = {
            "times": np.asarray(seed["time_s"], dtype=float),
            "position": np.asarray(actor["xy_m"], dtype=float),
            "speed": np.full(len(seed["time_s"]), float(actor["speed_mps"])),
            "heading": np.full(len(seed["time_s"]), float(actor["heading_rad"])),
        }
    return tracks


def _sumo_tracks(episode, template):
    observations = {float(time): value for time, value in episode.get("av_obs", {}).items()}
    times = sorted(time for time in observations if time <= 4.0 + 1e-9)
    tracks = {actor: {"times": np.asarray(times), "position": [], "speed": [], "heading": []} for actor in ("CAV", "BV_primary")}
    for time in times:
        snapshot = observations[time]
        cav = snapshot["Ego"]
        lead = _find_observation_actor(snapshot, "BV_primary")
        if lead is None:
            raise ValueError(f"BV_primary missing from SUMO observation at t={time}")
        for actor, value in (("CAV", cav), ("BV_primary", lead)):
            tracks[actor]["position"].append(np.asarray(value["position"][:2], dtype=float))
            tracks[actor]["speed"].append(float(value["velocity"]))
            tracks[actor]["heading"].append(math.radians(float(value["heading"])))
    for actor in tracks:
        for key in ("position", "speed", "heading"):
            tracks[actor][key] = np.asarray(tracks[actor][key])
    return {"times": np.asarray(times), **tracks}


def _find_observation_actor(snapshot, actor_id):
    for slot in ("Lead", "Foll", "LeftLead", "LeftFoll", "RightLead", "RightFoll"):
        value = snapshot.get(slot)
        if value and value.get("veh_id") == actor_id:
            return value
    return None


def _bridge_kinematic_tracks(template, times):
    result = {}
    for actor in [template["ego"], next(item for item in template["actors"] if item["id"] == "BV_primary")]:
        result[actor["id"] if actor["id"] != "CAV" else "CAV"] = {
            "times": times,
            "position": np.column_stack((float(actor["position"]) + float(actor["speed"]) * times, np.full(len(times), 46.0))),
            "speed": np.full(len(times), float(actor["speed"])),
            "heading": np.full(len(times), math.pi / 2),
        }
    return result


def _track_metrics(actual, reference, times):
    sampled_position = _resample(actual, times, vector_key="position")
    sampled_speed = _resample(actual, times, vector_key="speed")
    sampled_heading = _resample(actual, times, vector_key="heading")
    reference_position = _resample(reference, times, vector_key="position")
    reference_speed = _resample(reference, times, vector_key="speed")
    reference_heading = _resample(reference, times, vector_key="heading")
    position_error = np.linalg.norm(sampled_position - reference_position, axis=1)
    speed_error = np.abs(sampled_speed - reference_speed)
    heading_error = np.abs(np.asarray([_wrap(value) for value in sampled_heading - reference_heading]))
    return {
        "ade_m": float(np.mean(position_error)),
        "fde_m": float(position_error[-1]),
        "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "speed_mae_mps": float(np.mean(speed_error)),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
        "heading_mae_rad": float(np.mean(heading_error)),
        "max_position_error_m": float(np.max(position_error)),
        "samples": int(len(position_error)),
    }


def _resample(track, times, vector_key=None):
    values = track[vector_key or "position"]
    source_times = np.asarray(track["times"], dtype=float)
    if values.ndim == 1:
        return np.interp(times, source_times, values)
    return np.column_stack([np.interp(times, source_times, values[:, index]) for index in range(values.shape[1])])


def _rotate(point, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([cosine * point[0] - sine * point[1], sine * point[0] + cosine * point[1]])


def _wrap(value):
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--sumo_episode", required=True)
    parser.add_argument("--sumo_template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compute_metrics(args.source_root, args.seed, args.sumo_episode, args.sumo_template)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
