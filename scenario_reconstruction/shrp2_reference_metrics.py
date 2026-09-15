"""Compare completed calibration episodes with an already exported SHRP2 reference.

No HDF5 or SUMO run is required. SUMO positions are front-bumper centers;
headings are converted from clockwise-from-north degrees to Cartesian radians.
The common pre-contact window prevents early termination from hiding error.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .shrp2_trajectory_metrics import _find_observation_actor, _track_metrics
from .shrp2_reference_trajectory import _rotation


def observation_tracks(episode, vehicle_length_m):
    records = sorted((float(t), s) for t, s in episode["av_obs"].items())
    if not records:
        raise ValueError("Episode has no logged observations")
    tracks = {k: {"times": [], "position": [], "speed": [], "heading": []} for k in ("CAV", "BV_front", "BV_center")}
    for time, snapshot in records:
        bv = _find_observation_actor(snapshot, "BV_primary")
        if bv is None:
            raise ValueError(f"BV_primary missing at t={time}")
        for name, value in (("CAV", snapshot["Ego"]), ("BV_front", bv), ("BV_center", bv)):
            heading = math.pi / 2 - math.radians(float(value["heading"]))
            position = np.asarray(value["position"][:2], dtype=float)
            if name != "BV_front":
                position = position - 0.5 * vehicle_length_m * np.array([math.cos(heading), math.sin(heading)])
            track = tracks[name]
            track["times"].append(time)
            track["position"].append(position)
            track["speed"].append(float(value["velocity"]))
            track["heading"].append(heading)
    for track in tracks.values():
        for key in track:
            track[key] = np.asarray(track[key], dtype=float)
        track["heading"] = np.unwrap(track["heading"])
    return tracks


def mapped_reference_tracks(reference, template, vehicle_length_m, lane_y_m):
    cav = reference["actors"]["CAV"]
    origin = np.asarray(cav["xy_m"][0])
    heading = float(cav["reported_heading_rad"][0])
    rotation = _rotation(-heading)
    # Anchor to declared initialization, not the first logged state or each actor.
    destination = np.array([float(template["ego"]["position"]) - 0.5 * vehicle_length_m, lane_y_m])
    tracks = {}
    for name, actor, position_key in (
        ("CAV", cav, "xy_m"),
        ("BV_front", reference["actors"]["BV_primary"], "source_front_bumper_xy_m"),
        ("BV_center", reference["actors"]["BV_primary"], "xy_m"),
    ):
        tracks[name] = {
            "times": np.asarray(actor["time_s"], dtype=float),
            "position": (np.asarray(actor[position_key]) - origin) @ rotation.T + destination,
            "speed": np.asarray(actor["reported_speed_mps"]),
            "heading": np.unwrap(np.asarray(actor["reported_heading_rad"]) - heading),
        }
    return tracks


def compare_selected(reference, summary, vehicle_length_m=5.0, lane_y_m=46.0):
    selected = summary.get("selected", [])
    if not selected:
        raise ValueError("No selected candidates to compare")
    reference_end = float(reference["actors"]["CAV"]["time_s"][-1])
    contact = reference["collision_audit"].get("first_sampled_contact_time_s")
    cutoff = min(reference_end, float(contact)) if contact is not None else reference_end
    cases = []
    for candidate in selected:
        episode = json.loads(Path(candidate["episode_path"]).read_text(encoding="utf-8"))
        template = json.loads(Path(candidate["template_path"]).read_text(encoding="utf-8"))
        actual = observation_tracks(episode, vehicle_length_m)
        expected = mapped_reference_tracks(reference, template, vehicle_length_m, lane_y_m)
        cases.append((candidate, template, actual, expected))
    start = max(float(a["CAV"]["times"][0]) for _, _, a, _ in cases)
    end = min(cutoff, *(float(a["CAV"]["times"][-1]) for _, _, a, _ in cases))
    if end <= start:
        raise ValueError("No shared pre-contact observation window")
    times = np.asarray(reference["actors"]["CAV"]["time_s"])
    times = times[(times >= start - 1e-8) & (times <= end + 1e-8)]
    outcomes = []
    for candidate, template, actual, expected in cases:
        outcomes.append({
            "candidate_index": candidate["candidate_index"],
            "gap_offset_m": candidate["gap_offset_m"],
            "collision_time_s": candidate["end_time_s"],
            "collision_time_error_s": candidate["collision_time_error_s"],
            "last_logged_observation_time_s": float(actual["CAV"]["times"][-1]),
            "common_window_metrics": {k: _track_metrics(actual[k], expected[k], times) for k in actual},
            "logged_t0_cav_front_displacement_from_template_m": float(actual["CAV"]["position"][0, 0] + 0.5 * vehicle_length_m - template["ego"]["position"]),
        })
    return {
        "status": "SHRP2 measured reference vs SUMO; common pre-contact window; not full-window replay fidelity",
        "source_event_id": reference["source"]["event_id"],
        "source_split": reference["source"].get("source_split"),
        "executed_count": summary["executed_count"],
        "target_collision_count": sum(bool(o.get("collision_result")) and {"CAV", "BV_primary"}.issubset(o.get("collision_ids", [])) for o in summary["outcomes"]),
        "selected_count": len(outcomes), "reference_contact_time_s": contact,
        "common_window_s": [float(times[0]), float(times[-1])], "common_sample_count": len(times),
        "mapping": {"SUMO_position": "front bumper center", "SUMO_vehicle_length_assumption_m": vehicle_length_m, "lane_y_m": lane_y_m, "alignment": "source initial CAV centroid and heading to declared template initial CAV center; no per-actor shift or time warping", "time": "use logged timestamps as stored; t=0 may already contain initialization motion"},
        "outcomes": outcomes,
        "limitations": [
            "BV_front is the primary position metric; BV_center remains diagnostic because conversion uses reported source heading.",
            "FDE is at the end of the common observed window, not at 4 seconds or at the collision state.",
            "No extrapolation of stopped episodes; no time alignment search to improve scores.",
            "Vehicle dimensions and log/physics time alignment still need explicit runtime calibration.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vehicle_length_m", type=float, default=5.0)
    parser.add_argument("--lane_y_m", type=float, default=46.0)
    args = parser.parse_args()
    if not math.isfinite(args.vehicle_length_m) or args.vehicle_length_m <= 0:
        parser.error("vehicle_length_m must be finite and positive")
    result = compare_selected(json.loads(Path(args.reference).read_text(encoding="utf-8")), json.loads(Path(args.summary).read_text(encoding="utf-8")), args.vehicle_length_m, args.lane_y_m)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
