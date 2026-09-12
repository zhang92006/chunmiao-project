"""Local-only highD lane-change windows; never redistribute derived trajectories.

x/y in highD are the upper-left bounding-box corner; width is longitudinal
length, height is lateral width. Output x is forward, y is left, in metres.
The CAV is the target-lane follower, NOT the lane-changing vehicle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_config(config):
    seen = set()
    for split, recordings in config["splits"].items():
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        for rec in recordings:
            rec = f"{int(rec):02d}"
            if rec in seen:
                raise ValueError(f"Recording {rec} leaks across or repeats within splits")
            seen.add(rec)
    for name in ("seconds_before", "seconds_after", "max_neighbor_distance_m"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("max_scenes_per_recording", "frame_stride", "max_agents"):
        if isinstance(config[name], bool) or int(config[name]) != config[name] or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["max_agents"] < 2:
        raise ValueError("At least the CAV and the lane changer are required")


def center(row):
    return np.array([float(row["x"]) + float(row["width"]) / 2,
                     float(row["y"]) + float(row["height"]) / 2])


def make_scene(rec, split, metadata, track_meta, tracks, changer_id, event_frame, config):
    fps = float(metadata["frameRate"])
    changer = tracks[changer_id]
    event = changer[event_frame]
    ego_id = int(event["followingId"])
    if not ego_id or ego_id not in tracks or ego_id == changer_id:
        raise ValueError("no_target_follower")
    start = event_frame - round(config["seconds_before"] * fps)
    end = event_frame + round(config["seconds_after"] * fps)
    frames = list(range(start, end + 1, config["frame_stride"]))
    if any(frame not in tracks[ego_id] or frame not in changer for frame in frames):
        raise ValueError("incomplete_primary_window")
    ego_event = tracks[ego_id].get(event_frame)
    if ego_event is None or int(ego_event["laneId"]) != int(event["laneId"]):
        raise ValueError("follower_not_in_target_lane")
    direction = int(track_meta[changer_id]["drivingDirection"])
    if int(track_meta[ego_id]["drivingDirection"]) != direction:
        raise ValueError("opposite_direction")
    sign = 1 if direction == 2 else -1
    axes = np.array([sign, -sign])
    origin = center(tracks[ego_id][frames[0]])
    gap = sign * (center(event)[0] - center(ego_event)[0])
    if not 0 < gap <= config["max_neighbor_distance_m"]:
        raise ValueError("target_follower_out_of_range")
    selected = [ego_id, changer_id]
    neighbors = []
    for identifier, track in tracks.items():
        if identifier in selected or event_frame not in track:
            continue
        if int(track_meta[identifier]["drivingDirection"]) != direction:
            continue
        distance = abs(center(track[event_frame])[0] - center(ego_event)[0])
        if distance <= config["max_neighbor_distance_m"] and all(f in track for f in frames):
            neighbors.append((distance, identifier))
    selected.extend(identifier for _, identifier in sorted(neighbors)[:config["max_agents"] - 2])
    actors = []
    for identifier in selected:
        samples = [tracks[identifier][frame] for frame in frames]
        xy = (np.array([center(row) for row in samples]) - origin) * axes
        velocity = np.array([[float(row["xVelocity"]), float(row["yVelocity"])] for row in samples]) * axes
        acceleration = np.array([[float(row["xAcceleration"]), float(row["yAcceleration"])] for row in samples]) * axes
        if not all(np.isfinite(a).all() for a in (xy, velocity, acceleration)):
            raise ValueError("nonfinite_state")
        actors.append({
            "id": str(identifier), "role": "CAV" if identifier == ego_id else "BV",
            "length": float(track_meta[identifier]["width"]),
            "width": float(track_meta[identifier]["height"]),
            "xy": xy.tolist(), "velocity": velocity.tolist(), "acceleration": acceleration.tolist(),
            "lane_id": [int(row["laneId"]) for row in samples],
        })
    marking_key = "lowerLaneMarkings" if direction == 2 else "upperLaneMarkings"
    boundaries = sorted((float(y) - origin[1]) * axes[1] for y in metadata[marking_key].split(";"))
    return {
        "schema_version": 1, "scene_id": f"highd_{rec}_{changer_id}_{event_frame}",
        "split": split, "recording_id": rec, "location_id": int(metadata["locationId"]),
        "source_frames": frames, "source_frame_rate": fps,
        "time": [(frame - start) / fps for frame in frames],
        "event_time": (event_frame - start) / fps, "changer_id": str(changer_id),
        "coordinate_system": "vehicle-centre, x-forward, y-left, metres; CAV starts at origin",
        "driving_direction": direction, "road_boundaries_y": boundaries,
        "actors": actors,
    }


def export_recording(data_dir, output, rec, split, config):
    files = {suffix: data_dir / f"{rec}_{suffix}.csv" for suffix in ("tracks", "tracksMeta", "recordingMeta")}
    metadata = next(read_rows(files["recordingMeta"]))
    track_meta = {int(row["id"]): row for row in read_rows(files["tracksMeta"])}
    changing_ids = {i for i, meta in track_meta.items() if int(meta["numLaneChanges"]) > 0}
    changing = {i: {} for i in changing_ids}
    for row in read_rows(files["tracks"]):
        identifier = int(row["id"])
        if identifier in changing:
            changing[identifier][int(row["frame"])] = row
    events, wanted_ids = [], set(changing_ids)
    neighbor_fields = ("followingId", "precedingId", "leftPrecedingId", "leftFollowingId",
                       "rightPrecedingId", "rightFollowingId", "leftAlongsideId", "rightAlongsideId")
    for identifier, track in changing.items():
        ordered = sorted(track)
        for before, after in zip(ordered, ordered[1:]):
            if after != before + 1 or track[before]["laneId"] == track[after]["laneId"]:
                continue
            # Discard fleeting changes: require 0.4 s in the new lane.
            count = max(1, round(float(metadata["frameRate"]) * 0.4))
            if any(f not in track or track[f]["laneId"] != track[after]["laneId"] for f in range(after, after + count)):
                continue
            events.append((identifier, after))
            wanted_ids.update(int(track[after][key]) for key in neighbor_fields)
    # A second streaming pass loads only candidate neighbours, not the full recording.
    tracks = dict(changing)
    for row in read_rows(files["tracks"]):
        identifier = int(row["id"])
        if identifier in wanted_ids and identifier not in changing:
            tracks.setdefault(identifier, {})[int(row["frame"])] = row
    random.Random(config["seed"] + int(rec)).shuffle(events)
    scenes, rejected, used_changers = [], Counter(), set()
    for identifier, frame in events:
        if identifier in used_changers:
            continue
        try:
            scene = make_scene(rec, split, metadata, track_meta, tracks, identifier, frame, config)
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        name = f"{split}/{scene['scene_id']}.json"
        write_json(output / name, scene)
        scenes.append({"scene_id": scene["scene_id"], "path": name, "split": split,
                       "recording_id": rec, "location_id": scene["location_id"],
                       "sha256": file_hash(output / name), "agent_count": len(scene["actors"])})
        used_changers.add(identifier)
        if len(scenes) >= config["max_scenes_per_recording"]:
            break
    return scenes, {"recording_id": rec, "split": split, "candidate_events": len(events),
                    "exported": len(scenes), "rejected_before_limit": dict(rejected),
                    "source_sha256": {path.name: file_hash(path) for path in files.values()}}


def export_dataset(data_dir, output, config):
    validate_config(config)
    data_dir, output = Path(data_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be absent or empty; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "dataset": config["dataset"], "config": config,
                "license_notice": "Local licensed use only; do not commit source or derived trajectories.",
                "scenes": [], "recordings": []}
    for split, recordings in config["splits"].items():
        for rec in recordings:
            rec = f"{int(rec):02d}"
            scenes, report = export_recording(data_dir, output, rec, split, config)
            manifest["scenes"].extend(scenes)
            manifest["recordings"].append(report)
            print(f"{split}/{rec}: {len(scenes)} scenes", flush=True)
    if not manifest["scenes"]:
        raise ValueError("No complete lane-change scenes were found")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Local highD data directory; not uploaded")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/highd_pilot.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    export_dataset(args.data_dir, args.output, config)


if __name__ == "__main__":
    main()
