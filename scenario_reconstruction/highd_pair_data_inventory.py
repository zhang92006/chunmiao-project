"""Inventory highD follower pairs and lane-change/rear-vehicle windows.

Descriptive data audit only: no fitting, scoring a model, SUMO, or split changes.
Windows are around lane-ID crossings, not verified maneuver start/end times.
Third-lane flags describe observed context; absence does not prove independence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


NEIGHBORS = [f"{side}{role}Id" for side in ("left", "right")
             for role in ("Preceding", "Alongside", "Following")]
COLUMNS = ["frame", "id", "laneId", "x", "width", "xVelocity", "xAcceleration",
           "dhw", "precedingXVelocity", "precedingId"] + NEIGHBORS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def lane_sets(meta):
    upper = len(str(meta["upperLaneMarkings"]).split(";")) - 1
    lower = len(str(meta["lowerLaneMarkings"]).split(";")) - 1
    return {1: set(range(2, upper + 2)),
            2: set(range(upper + 3, upper + lower + 3))}


def side_of_change(direction, delta):
    return "left" if (direction == 1 and delta > 0) or (direction == 2 and delta < 0) else "right"


def in_range(values, bounds):
    return np.isfinite(values) & (values >= bounds[0]) & (values <= bounds[1])


def longitudinal_domain(frame, config):
    speed = np.abs(frame["xVelocity"].to_numpy(float))
    gap = frame["dhw"].to_numpy(float)
    rr = np.abs(frame["precedingXVelocity"].to_numpy(float)) - speed
    has_leader = frame["precedingId"].to_numpy(int) > 0
    free = ~has_leader | (np.isfinite(gap) & (gap > config["gap_m"][1]))
    return in_range(speed, config["speed_mps"]) & (
        free | (in_range(gap, config["gap_m"]) & in_range(rr, config["range_rate_mps"])))


def full_window(track, start, end):
    if track is None:
        return None
    frames = track["frame"].to_numpy()
    left, right = np.searchsorted(frames, [start, end + 1])
    piece = track.iloc[left:right]
    if len(piece) != end - start + 1 or not np.array_equal(piece["frame"].to_numpy(), np.arange(start, end + 1)):
        return None
    return piece


def at_frame(track, frame):
    if track is None:
        return None
    frames = track["frame"].to_numpy()
    index = int(np.searchsorted(frames, frame))
    return track.iloc[index] if index < len(track) and frames[index] == frame else None


def following_counts(sampled, fps, target_hz, config):
    counts = Counter(sampled_vehicle_rows=len(sampled))
    same = sampled.loc[sampled["precedingId"] > 0].merge(
        sampled, left_on=["frame", "precedingId"], right_on=["frame", "id"],
        suffixes=("", "_leader"), how="inner", validate="many_to_one")
    counts["referenced_leader_synchronized_rows"] = len(same)
    same = same.loc[(same.laneId == same.laneId_leader) & (same.direction == same.direction_leader)].copy()
    counts["same_lane_pair_rows"] = len(same)
    if same.empty:
        return dict(counts)
    speed = np.abs(same.xVelocity.to_numpy(float))
    leader_speed = np.abs(same.xVelocity_leader.to_numpy(float))
    eligible = (in_range(speed, config["speed_mps"]) & in_range(leader_speed, config["speed_mps"])
                & in_range(same.dhw.to_numpy(float), config["gap_m"])
                & in_range(leader_speed - speed, config["range_rate_mps"]))
    counts["pair_state_in_domain_rows"] = int(eligible.sum())
    eligible &= in_range(same.xAcceleration.to_numpy(float) * same.travel_sign.to_numpy(), config["acceleration_mps2"])
    eligible &= in_range(same.xAcceleration_leader.to_numpy(float) * same.travel_sign_leader.to_numpy(), config["acceleration_mps2"])
    counts["pair_state_and_actions_in_domain_rows"] = int(eligible.sum())
    eligible &= same.longitudinal_domain.to_numpy(bool) & same.longitudinal_domain_leader.to_numpy(bool)
    counts["both_single_longitudinal_models_supported_rows"] = int(eligible.sum())
    eligible &= ~same.near_transition.to_numpy(bool) & ~same.near_transition_leader.to_numpy(bool)
    stable = same.loc[eligible].sort_values(["id", "precedingId", "frame"])
    counts["stable_pair_rows"] = len(stable)
    counts["stable_unique_directed_pairs"] = len(stable[["id", "precedingId"]].drop_duplicates())
    if stable.empty:
        return dict(counts)
    new = ((stable.id != stable.id.shift()) | (stable.precedingId != stable.precedingId.shift())
           | (stable.frame.diff() > np.ceil(fps / target_hz)))
    segments = stable.groupby(new.cumsum()).frame.agg(["min", "max", "count"])
    long_enough = (segments["max"] - segments["min"]) / fps >= config["following_minimum_span_s"]
    counts["stable_segments"] = len(segments)
    counts["stable_segments_at_least_2s"] = int(long_enough.sum())
    counts["rows_in_stable_segments_at_least_2s"] = int(segments.loc[long_enough, "count"].sum())
    return dict(counts)


def opportunity_counts(sampled, lanes):
    counts = Counter()
    for side in ("left", "right"):
        delta = np.where(sampled.direction.to_numpy() == 1, 1, -1) * (1 if side == "left" else -1)
        target = sampled.laneId.to_numpy() + delta
        available = np.zeros(len(sampled), dtype=bool)
        for direction in (1, 2):
            available |= (sampled.direction.to_numpy() == direction) & np.isin(target, list(lanes[direction]))
        valid = sampled.longitudinal_domain.to_numpy(bool) & ~sampled.near_transition.to_numpy(bool) & available
        candidates = sampled.loc[valid, ["frame", "id", "direction", f"{side}FollowingId"]].copy()
        candidates["target_lane"] = target[valid]
        counts[f"{side}_stable_available_side_rows"] = len(candidates)
        joined = candidates.merge(sampled[["frame", "id", "laneId", "direction", "longitudinal_domain"]],
            left_on=["frame", f"{side}FollowingId"], right_on=["frame", "id"], suffixes=("", "_rear"), how="inner")
        ok = ((joined.target_lane == joined.laneId) & (joined.direction == joined.direction_rear)
              & joined.longitudinal_domain)
        counts[f"{side}_stable_rows_with_supported_target_rear"] = int(ok.sum())
    return dict(counts)


def lane_change_counts(groups, directions, lanes, transitions, fps, config):
    counts = Counter()
    by_side = defaultdict(Counter)
    by_lane_pair = defaultdict(Counter)
    pre = round(config["lane_change_pre_s"] * fps)
    post = round(config["lane_change_post_s"] * fps)
    lead = round(config["anchor_before_crossing_s"] * fps)
    for event in transitions:
        vehicle, crossing, old_lane, target_lane = event
        direction = directions[vehicle]
        side = side_of_change(direction, target_lane - old_lane)
        c = Counter(crossings=1)
        pair = {old_lane, target_lane}
        lane_pair = f"direction{direction}:{min(pair)}-{max(pair)}"
        ego_track = groups[vehicle]
        anchor = at_frame(ego_track, crossing - lead)
        if anchor is None or int(anchor.laneId) != old_lane:
            c["anchor_missing_or_not_source_lane"] = 1
        else:
            c["anchor_valid"] = 1
            rear_id = int(anchor[f"{side}FollowingId"])
            rear = at_frame(groups.get(rear_id), crossing - lead)
            has_rear = rear is not None and directions[rear_id] == direction and int(rear.laneId) == target_lane
            if not has_rear:
                c["no_valid_target_rear_at_anchor"] = 1
            else:
                c["target_rear_at_anchor"] = 1
                if bool(anchor.longitudinal_domain) and bool(rear.longitudinal_domain):
                    c["both_single_models_supported_at_anchor"] = 1
                outside_lanes = lanes[direction] - pair
                if outside_lanes:
                    c["road_has_lanes_outside_selected_pair"] = 1
                third_near = False
                for actor in (anchor, rear):
                    for field in NEIGHBORS:
                        neighbor = at_frame(groups.get(int(actor[field])), crossing - lead)
                        if neighbor is not None and int(neighbor.laneId) in outside_lanes:
                            actor_center = float(actor.x + actor.width / 2)
                            neighbor_center = float(neighbor.x + neighbor.width / 2)
                            if abs(actor_center - neighbor_center) <= config["third_lane_distance_m"]:
                                third_near = True
                c["near_outside_lane_neighbor_at_anchor"] = int(third_near)
                ego_window = full_window(ego_track, crossing - pre, crossing + post)
                rear_window = full_window(groups[rear_id], crossing - pre, crossing + post)
                if ego_window is None:
                    c["ego_window_incomplete"] = 1
                elif rear_window is None:
                    c["rear_window_incomplete_given_ego_complete"] = 1
                else:
                    c["complete_pair_windows"] = 1
                    c["complete_pair_windows_with_third_neighbor_at_anchor"] = int(third_near)
                    all_lanes_inside = (ego_window.laneId.isin(pair).all() and rear_window.laneId.isin(pair).all())
                    ego_crossings = int(np.count_nonzero(np.diff(ego_window.laneId.to_numpy())))
                    simple = all_lanes_inside and ego_crossings == 1 and (rear_window.laneId == target_lane).all()
                    c["selected_actors_visit_outside_lane_in_window"] = int(not all_lanes_inside)
                    c["simple_pair_windows"] = int(simple)
                    both_speed = (in_range(np.abs(ego_window.xVelocity.to_numpy()), config["speed_mps"]).all()
                                  and in_range(np.abs(rear_window.xVelocity.to_numpy()), config["speed_mps"]).all())
                    c["complete_pair_windows_full_speed_domain"] = int(both_speed)
                    all_longitudinal = ego_window.longitudinal_domain.all() and rear_window.longitudinal_domain.all()
                    c["complete_pair_windows_full_longitudinal_state_domain"] = int(all_longitudinal)
                    acceleration_ok = (in_range(ego_window.xAcceleration.to_numpy() * ego_window.travel_sign.to_numpy(), config["acceleration_mps2"]).all()
                                       and in_range(rear_window.xAcceleration.to_numpy() * rear_window.travel_sign.to_numpy(), config["acceleration_mps2"]).all())
                    candidate = bool(simple and all_longitudinal and acceleration_ok)
                    c["strict_simple_in_domain_pair_windows"] = int(candidate)
                    c["strict_pair_windows_with_third_neighbor_at_anchor"] = int(candidate and third_near)
                    c["strict_pair_windows_without_observed_third_neighbor_at_anchor"] = int(candidate and not third_near)
                    if candidate:
                        c["strict_left" if side == "left" else "strict_right"] = 1
        counts.update(c)
        by_side[side].update(c)
        by_lane_pair[lane_pair].update(c)
    return {"counts": dict(counts), "by_side": {k: dict(v) for k, v in by_side.items()},
            "by_lane_pair": {k: dict(v) for k, v in by_lane_pair.items()}}


def inventory_frames(tracks, track_meta, meta, config):
    fps = int(meta["frameRate"])
    target_hz = int(config["target_hz"])
    if fps < target_hz or target_hz <= 0:
        raise ValueError("Invalid sampling frequency")
    directions = dict(zip(track_meta.id.astype(int), track_meta.drivingDirection.astype(int)))
    tracks = tracks.sort_values(["id", "frame"]).reset_index(drop=True).copy()
    if tracks.duplicated(["id", "frame"]).any():
        raise ValueError("Duplicate vehicle/frame")
    tracks["direction"] = tracks.id.map(directions)
    if not tracks.direction.isin([1, 2]).all():
        raise ValueError("Unknown vehicle driving direction")
    tracks["travel_sign"] = np.where(tracks.direction == 1, -1, 1)
    tracks["longitudinal_domain"] = longitudinal_domain(tracks, config)
    tracks["near_transition"] = False
    lanes = lane_sets(meta)
    groups = {}
    transitions = []
    guard = round(config["transition_guard_s"] * fps)
    invalid_crossings = 0
    for vehicle, track in tracks.groupby("id", sort=False):
        track = track.copy()
        f, lane = track.frame.to_numpy(), track.laneId.to_numpy()
        edges = np.flatnonzero(lane[1:] != lane[:-1]) + 1
        near = np.zeros(len(track), dtype=bool)
        for edge in edges:
            frame = int(f[edge])
            near |= np.abs(f - frame) <= guard
            if (f[edge] - f[edge - 1] == 1 and abs(lane[edge] - lane[edge - 1]) == 1
                    and lane[edge] in lanes[directions[int(vehicle)]] and lane[edge - 1] in lanes[directions[int(vehicle)]]):
                transitions.append((int(vehicle), frame, int(lane[edge - 1]), int(lane[edge])))
            else:
                invalid_crossings += 1
        track["near_transition"] = near
        groups[int(vehicle)] = track
    tracks = pd.concat(groups.values(), ignore_index=True)
    frames = tracks.frame.to_numpy(np.int64)
    sampled = tracks.loc[((frames + 1) * target_hz // fps) > (frames * target_hz // fps)].copy()
    return {"raw_rows": len(tracks), "observed_vehicles": len(groups),
            "invalid_or_nonadjacent_lane_transitions": invalid_crossings,
            "following": following_counts(sampled, fps, target_hz, config),
            "opportunities": opportunity_counts(sampled, lanes),
            "lane_change": lane_change_counts(groups, directions, lanes, transitions, fps, config)}


def usage_inventory(root, config):
    usage = defaultdict(list)
    hashes = {}
    for name in config["usage_manifests"]:
        hashes[name] = digest(root / name)
        for item in read_json(root / name)["recordings"]:
            usage[str(item["recording_id"]).zfill(2)].append({"manifest": name, "split": item.get("split", "unspecified")})
    primary = {str(r["recording_id"]).zfill(2): r["split"] for r in read_json(root / config["primary_manifest"])["recordings"]}
    return dict(usage), primary, hashes


def aggregate(records):
    groups = {}
    for dimension in ("all", "topology", "primary_split"):
        buckets = defaultdict(list)
        for record in records:
            buckets["all" if dimension == "all" else record[dimension]].append(record)
        groups[dimension] = {}
        for key, bucket in buckets.items():
            value = {"recording_count": len(bucket), "recording_ids": [r["recording_id"] for r in bucket]}
            for name in ("following", "opportunities", "lane_change"):
                c = Counter()
                for record in bucket:
                    c.update(record[name]["counts"] if name == "lane_change" else record[name])
                value[name] = dict(c)
            groups[dimension][key] = value
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--config", default="configs/highd_pair_data_inventory_v1.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recordings", nargs="+", help="Default: all 60; each recording is cached for resumption")
    args = parser.parse_args()
    root = Path.cwd()
    config = read_json(args.config)
    usage, primary, manifest_hashes = usage_inventory(root, config)
    signature = {"config_sha256": digest(args.config), "script_sha256": digest(__file__), "usage_manifest_sha256": manifest_hashes}
    metadata = []
    for path in sorted((args.source_root / "data").glob("*_recordingMeta.csv")):
        rec = path.name.split("_")[0]
        meta = pd.read_csv(path).iloc[0]
        tm = pd.read_csv(path.with_name(rec + "_tracksMeta.csv"))
        lanes = lane_sets(meta)
        metadata.append({"recording_id": rec, "topology": f"{len(lanes[1])}/{len(lanes[2])}",
                         "location_id": int(meta.locationId), "vehicles": len(tm),
                         "metadata_lane_changes": int(tm.numLaneChanges.sum()),
                         "primary_split": primary.get(rec, "not_in_primary_manifest"),
                         "known_usage": usage.get(rec, [])})
    args.output.mkdir(parents=True, exist_ok=True)
    header = {**signature, "config": config, "metadata": metadata,
              "scope": "Availability audit. Listed manifests are not an exhaustive use history. This scan observes event availability, not model performance; do not later call these counts unseen.",
              "window_definition": "Lane-ID crossing +/- 2 s (inclusive), not physical maneuver start/end. Fixed rear selected ~0.5 s before crossing. Future labels used for descriptive inventory only.",
              "third_lane_definition": "Referenced outside-pair adjacent neighbor within 115 m longitudinal center distance at anchor, for either actor. Proxy only; no absence-of-influence claim.",
              "opportunity_definition": "10Hz stable available-side rows outside +/-1s of any own lane-ID transition. Descriptive controls, not a calibrated decision-risk denominator; temporal correlation and selection remain.",
              "pair_domain_definition": "Strict LC windows require both actors' longitudinal state and acceleration domain throughout, one ego crossing, rear stays target lane. Not full 33-action or runtime acceptance."}
    header_path = args.output / "inventory_protocol.json"
    if header_path.exists():
        if read_json(header_path) != header:
            raise ValueError("Existing output protocol changed; use a fresh output")
    else:
        write_new(header_path, header)
    requested = [str(r).zfill(2) for r in args.recordings] if args.recordings else [r["recording_id"] for r in metadata]
    by_id = {r["recording_id"]: r for r in metadata}
    if len(requested) != len(set(requested)) or any(r not in by_id for r in requested):
        raise ValueError("Invalid recording selection")
    for rec in requested:
        start = time.monotonic()
        source = args.source_root / "data" / f"{rec}_tracks.csv"
        source_stamp = {"size_bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
        dest = args.output / "recordings" / f"{rec}.json"
        if dest.exists():
            cached = read_json(dest)
            if cached["source_stamp"] != source_stamp:
                raise ValueError("Source changed: " + rec)
            print(f"recording={rec} cached", flush=True)
            continue
        tracks = pd.read_csv(source, usecols=COLUMNS)
        tm = pd.read_csv(source.with_name(f"{rec}_tracksMeta.csv"))
        meta = pd.read_csv(source.with_name(f"{rec}_recordingMeta.csv")).iloc[0]
        result = inventory_frames(tracks, tm, meta, config)
        result.update(by_id[rec])
        result["source_stamp"] = source_stamp
        result["elapsed_s"] = round(time.monotonic() - start, 3)
        write_new(dest, result)
        print(json.dumps({"recording": rec, "elapsed_s": result["elapsed_s"],
                          "stable_pair_rows": result["following"].get("stable_pair_rows", 0),
                          "crossings": result["lane_change"]["counts"].get("crossings", 0),
                          "strict_pair_windows": result["lane_change"]["counts"].get("strict_simple_in_domain_pair_windows", 0)}), flush=True)
    records = [read_json(p) for p in sorted((args.output / "recordings").glob("*.json"))]
    summary = {"schema_version": 1, "signature": signature, "metadata_recording_count": len(metadata),
               "scanned_recording_count": len(records), "all_recordings_scanned": len(records) == len(metadata),
               "groups": aggregate(records), "scope": header["scope"]}
    summary_path = args.output / f"inventory_summary_{len(records):02d}.json"
    if summary_path.exists():
        if read_json(summary_path) != summary:
            raise ValueError("Summary changed; use a new output")
    else:
        write_new(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "scanned_recordings": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
