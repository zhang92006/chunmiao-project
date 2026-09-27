"""Export train-only pair data without changing NDD, splits, or SUMO.

Event-centred windows and prospective decision-clock labels are separate tables.
Future labels must never be used as online features. All trajectories stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import highd_pair_data_inventory as inventory


SOURCE_COLUMNS = inventory.COLUMNS + ["y", "height", "yVelocity", "yAcceleration", "followingId"]
REFERENCES = ["precedingId", "followingId"] + inventory.NEIGHBORS


def clock_mask(frames, fps, hz):
    if hz <= 0 or hz > fps:
        raise ValueError("Sampling rate must be positive and no greater than source rate")
    frames = np.asarray(frames, dtype=np.int64)
    return (frames + 1) * hz // fps > frames * hz // fps


def prepare(tracks, track_meta, meta, domain):
    """Current-state features plus explicitly named hindsight diagnostics."""
    directions = dict(zip(track_meta.id.astype(int), track_meta.drivingDirection.astype(int)))
    lanes = inventory.lane_sets(meta)
    fps = int(meta["frameRate"])
    tracks = tracks.sort_values(["id", "frame"]).reset_index(drop=True).copy()
    if tracks.duplicated(["id", "frame"]).any():
        raise ValueError("Duplicate vehicle/frame")
    tracks["direction"] = tracks.id.map(directions)
    if not tracks.direction.isin([1, 2]).all():
        raise ValueError("Unknown direction")
    tracks["travel_sign"] = np.where(tracks.direction == 1, -1, 1)
    tracks["speed_mps"] = np.abs(tracks.xVelocity)
    tracks["acceleration_mps2"] = tracks.xAcceleration * tracks.travel_sign
    tracks["longitudinal_domain"] = inventory.longitudinal_domain(tracks, domain)
    tracks["acceleration_domain"] = inventory.in_range(tracks.acceleration_mps2, domain["acceleration_mps2"])
    tracks["speed_domain"] = inventory.in_range(tracks.speed_mps, domain["speed_mps"])
    tracks["source_lane_valid"] = [int(l) in lanes[int(d)] for l, d in zip(tracks.laneId, tracks.direction)]
    groups, transitions = {}, []
    for vehicle, part in tracks.groupby("id", sort=False):
        part = part.copy()
        frames, lane = part.frame.to_numpy(), part.laneId.to_numpy()
        edges = np.flatnonzero(lane[1:] != lane[:-1]) + 1
        near = np.zeros(len(part), dtype=bool)
        last_cross = np.full(len(part), np.nan)
        for edge in edges:
            f = int(frames[edge])
            near |= np.abs(frames - f) <= round(domain["transition_guard_s"] * fps)
            last_cross[edge:] = (frames[edge:] - f) / fps
            if (frames[edge] - frames[edge - 1] == 1 and abs(lane[edge] - lane[edge - 1]) == 1
                    and lane[edge] in lanes[directions[int(vehicle)]]
                    and lane[edge - 1] in lanes[directions[int(vehicle)]]):
                transitions.append((int(vehicle), f, int(lane[edge - 1]), int(lane[edge])))
        part["diagnostic_near_crossing_hindsight"] = near
        part["seconds_since_observed_lane_change"] = last_cross
        groups[int(vehicle)] = part
    return pd.concat(groups.values(), ignore_index=True), groups, lanes, transitions


def following_samples(tracks, fps, domain):
    sampled = tracks.loc[clock_mask(tracks.frame, fps, domain["target_hz"])].copy()
    columns = ["frame", "id", "laneId", "direction", "longitudinal_domain", "acceleration_domain",
               "speed_mps", "diagnostic_near_crossing_hindsight"]
    joined = sampled.loc[sampled.precedingId > 0].merge(
        sampled[columns], left_on=["frame", "precedingId"], right_on=["frame", "id"],
        suffixes=("", "_leader"), how="inner", validate="many_to_one")
    joined = joined.loc[(joined.laneId == joined.laneId_leader) & (joined.direction == joined.direction_leader)].copy()
    valid = (joined.longitudinal_domain & joined.longitudinal_domain_leader
             & joined.acceleration_domain & joined.acceleration_domain_leader
             & inventory.in_range(joined.dhw, domain["gap_m"])
             & inventory.in_range(joined.speed_mps_leader - joined.speed_mps, domain["range_rate_mps"]))
    stable = valid & ~joined.diagnostic_near_crossing_hindsight & ~joined.diagnostic_near_crossing_hindsight_leader
    output = joined[["frame", "id", "precedingId", "dhw"]].rename(columns={"id": "follower_id", "precedingId": "leader_id"})
    output["both_state_and_action_supported"] = valid.to_numpy()
    output["diagnostic_stable_supported_hindsight"] = stable.to_numpy()
    output = output.sort_values(["follower_id", "leader_id", "frame"]).reset_index(drop=True)
    # Stable segmentation reproduces the availability audit, not an online gate.
    output["diagnostic_segment_id"] = -1
    subset = output.loc[output.diagnostic_stable_supported_hindsight]
    starts = ((subset.follower_id != subset.follower_id.shift()) | (subset.leader_id != subset.leader_id.shift())
              | (subset.frame.diff() > np.ceil(fps / domain["target_hz"])))
    output.loc[subset.index, "diagnostic_segment_id"] = starts.cumsum().to_numpy()
    return output


def decision_labels(track, anchors, fps, horizon_s, valid_lanes):
    """First observed crossing in (t,t+h]; unknown if coverage/geometry fails.

    Full outcome horizon is required even if an early crossing is observed.
    No rear future, acceleration or future domain condition selects a row.
    """
    frames, lane = track.frame.to_numpy(), track.laneId.to_numpy()
    horizon = round(horizon_s * fps)
    if horizon < 1:
        raise ValueError("Decision horizon too short")
    first = np.searchsorted(frames, anchors)
    last = np.searchsorted(frames, np.asarray(anchors) + horizon)
    in_bounds = last < len(frames)
    bounded = np.minimum(last, len(frames) - 1)
    gaps = np.r_[0, np.cumsum(np.diff(frames) != 1)]
    valid_geometry = np.isin(lane, list(valid_lanes))
    invalid = np.r_[0, np.cumsum(~valid_geometry)]
    known = (in_bounds & (frames[bounded] == np.asarray(anchors) + horizon)
             & (gaps[bounded] == gaps[first]) & ((invalid[bounded + 1] - invalid[first]) == 0))
    edges = np.flatnonzero(lane[1:] != lane[:-1]) + 1
    # Also invalidate skipped-lane transitions, not just invalid lane IDs.
    jumps = np.r_[0, np.cumsum(np.abs(np.diff(lane)) > 1)]
    known &= jumps[bounded] == jumps[first]
    next_index = np.searchsorted(edges, first, side="right")
    labels = np.full(len(anchors), "unknown", dtype=object)
    crossing_frames = np.full(len(anchors), -1, dtype=int)
    labels[known] = "stay"
    direction = int(track.direction.iloc[0])
    for j in np.flatnonzero(known & (next_index < len(edges))):
        edge = edges[next_index[j]]
        if edge <= last[j]:
            labels[j] = inventory.side_of_change(direction, lane[edge] - lane[edge - 1])
            crossing_frames[j] = frames[edge]
    return labels, crossing_frames


def decision_samples(tracks, groups, lanes, fps, config):
    pieces = []
    for vehicle, track in groups.items():
        part = track.loc[clock_mask(track.frame, fps, config["decision_hz"]) & track.source_lane_valid].copy()
        if part.empty:
            continue
        labels, crossings = decision_labels(track, part.frame.to_numpy(), fps, config["decision_horizon_s"],
                                            lanes[int(track.direction.iloc[0])])
        for side in ("left", "right"):
            delta = (1 if int(track.direction.iloc[0]) == 1 else -1) * (1 if side == "left" else -1)
            available = (part.laneId + delta).isin(lanes[int(track.direction.iloc[0])])
            one = part.loc[available, ["frame", "id", "laneId", "direction", "x", "width", "travel_sign",
                                       "speed_mps", "longitudinal_domain", f"{side}FollowingId"]].copy()
            one = one.rename(columns={f"{side}FollowingId": "rear_id", "laneId": "source_lane"})
            one["side"] = side
            one["target_lane"] = one.source_lane + delta
            one["label_first_crossing_side"] = labels[available]
            one["label_crossing_frame"] = crossings[available]
            one["label_side_crossing"] = np.where(labels[available] == "unknown", -1, (labels[available] == side).astype(int))
            pieces.append(one)
    base_columns = ["frame", "id", "source_lane", "direction", "x", "width", "travel_sign", "speed_mps",
                    "longitudinal_domain", "rear_id", "side", "target_lane", "label_first_crossing_side",
                    "label_crossing_frame", "label_side_crossing"]
    candidates = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=base_columns)
    rear_columns = ["frame", "id", "laneId", "direction", "x", "width", "speed_mps", "longitudinal_domain"]
    candidates = candidates.merge(tracks[rear_columns], left_on=["frame", "rear_id"], right_on=["frame", "id"],
                                  suffixes=("", "_rear"), how="left", validate="many_to_one")
    candidates["rear_valid_now"] = ((candidates.rear_id > 0) & (candidates.target_lane == candidates.laneId)
                                    & (candidates.direction == candidates.direction_rear))
    candidates["both_state_supported_now"] = (candidates.rear_valid_now & candidates.longitudinal_domain
                                               & candidates.longitudinal_domain_rear.fillna(False))
    candidates["rear_center_distance_m"] = ((candidates.x + candidates.width / 2)
                                             - (candidates.x_rear + candidates.width_rear / 2)) * candidates.travel_sign
    candidates["rear_gap_m"] = candidates.rear_center_distance_m - (candidates.width + candidates.width_rear) / 2
    candidates.loc[~candidates.rear_valid_now, ["rear_center_distance_m", "rear_gap_m"]] = np.nan
    candidates["decision_id"] = np.arange(len(candidates), dtype=np.int64)
    return candidates[["decision_id", "frame", "id", "source_lane", "target_lane", "direction", "side", "rear_id",
                       "rear_valid_now", "both_state_supported_now", "speed_mps", "rear_gap_m",
                       "label_first_crossing_side", "label_crossing_frame", "label_side_crossing"]]


def matched_controls(decisions, config):
    """Descriptive 1:1 strata controls, WITH replacement. Not prevalence data."""
    valid = decisions.loc[decisions.both_state_supported_now & (decisions.label_side_crossing >= 0)
                          & np.isfinite(decisions.rear_gap_m.to_numpy(float))].copy()
    valid["speed_bin"] = np.floor(valid.speed_mps.to_numpy(float) / config["matching_speed_bin_mps"]).astype(int)
    valid["rear_gap_bin"] = np.floor(valid.rear_gap_m.to_numpy(float) / config["matching_rear_gap_bin_m"]).astype(int)
    strata = ["direction", "source_lane", "side", "speed_bin", "rear_gap_bin"]
    controls = {key: group.decision_id.to_numpy() for key, group in valid.loc[valid.label_first_crossing_side == "stay"].groupby(strata)}
    rng = np.random.default_rng(config["matching_seed"])
    rows = []
    positives = valid.loc[valid.label_side_crossing == 1]
    for row in positives.itertuples():
        key = tuple(getattr(row, field) for field in strata)
        pool = controls.get(key, np.array([], dtype=int))
        rows.append({"positive_decision_id": row.decision_id,
                     "control_decision_id": int(rng.choice(pool)) if len(pool) else -1,
                     "candidate_control_count": len(pool),
                     "control_draw_probability": 1.0 / len(pool) if len(pool) else np.nan})
    return pd.DataFrame(rows, columns=["positive_decision_id", "control_decision_id", "candidate_control_count", "control_draw_probability"])


def event_samples(groups, lanes, transitions, fps, domain):
    events, rows = [], []
    pre, post, lead = [round(domain[k] * fps) for k in ("lane_change_pre_s", "lane_change_post_s", "anchor_before_crossing_s")]
    for event_id, (vehicle, crossing, old, target) in enumerate(transitions):
        ego_track = groups[vehicle]
        direction = int(ego_track.direction.iloc[0])
        side = inventory.side_of_change(direction, target - old)
        anchor_frame = crossing - lead
        anchor = inventory.at_frame(ego_track, anchor_frame)
        valid_anchor = anchor is not None and int(anchor.laneId) == old
        rear_id = int(anchor[f"{side}FollowingId"]) if valid_anchor else 0
        rear = inventory.at_frame(groups.get(rear_id), anchor_frame)
        valid_rear = rear is not None and int(rear.direction) == direction and int(rear.laneId) == target
        info = {"event_id": event_id, "id": vehicle, "rear_id": rear_id, "crossing_frame": crossing,
                "anchor_frame": anchor_frame, "source_lane": old, "target_lane": target, "side": side,
                "anchor_valid": valid_anchor, "rear_valid_at_anchor": valid_rear,
                "road_has_outside_lane": bool(lanes[direction] - {old, target}),
                "complete_window": False, "strict_compatible_diagnostic": False}
        if valid_anchor and valid_rear:
            frames = np.arange(crossing - pre, crossing + post + 1)
            a = ego_track.set_index("frame").reindex(frames)
            b = groups[rear_id].set_index("frame").reindex(frames)
            present_a, present_b = a.id.notna().to_numpy(), b.id.notna().to_numpy()
            complete = bool(present_a.all() and present_b.all())
            simple = complete and bool(a.laneId.isin([old, target]).all() and (b.laneId == target).all()
                                       and np.count_nonzero(np.diff(a.laneId)) == 1)
            state_ok = (a.longitudinal_domain.fillna(False) & b.longitudinal_domain.fillna(False)).to_numpy(bool)
            action_ok = (a.acceleration_domain.fillna(False) & b.acceleration_domain.fillna(False)).to_numpy(bool)
            info.update(complete_window=complete, strict_compatible_diagnostic=bool(simple and state_ok.all() and action_ok.all()))
            rows.append(pd.DataFrame({"event_id": event_id, "frame": frames, "id": vehicle, "rear_id": rear_id,
                                      "relative_time_to_crossing_s": (frames - crossing) / fps,
                                      "actor_observed": present_a, "rear_observed": present_b,
                                      "both_state_supported": state_ok, "both_action_supported": action_ok}))
        events.append(info)
    event_columns = ["event_id", "id", "rear_id", "crossing_frame", "anchor_frame", "source_lane", "target_lane", "side",
                     "anchor_valid", "rear_valid_at_anchor", "road_has_outside_lane", "complete_window", "strict_compatible_diagnostic"]
    window_columns = ["event_id", "frame", "id", "rear_id", "relative_time_to_crossing_s", "actor_observed", "rear_observed",
                      "both_state_supported", "both_action_supported"]
    return pd.DataFrame(events, columns=event_columns), pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=window_columns)


def observation_samples(tracks, windows, fps, domain, config):
    """Global sampled frames plus window actors AND their referenced neighbours."""
    selected = clock_mask(tracks.frame, fps, domain["target_hz"]) | clock_mask(tracks.frame, fps, config["decision_hz"])
    index = pd.MultiIndex.from_frame(tracks[["frame", "id"]])
    if not windows.empty:
        wanted = pd.concat([windows[["frame", "id"]], windows[["frame", "rear_id"]].rename(columns={"rear_id": "id"})]).drop_duplicates()
        positions = index.get_indexer(pd.MultiIndex.from_frame(wanted))
        selected[positions[positions >= 0]] = True
        # One-hop context of each selected actor, including third-lane neighbours.
        actors = tracks.iloc[positions[positions >= 0]]
        for field in REFERENCES:
            keys = pd.MultiIndex.from_arrays([actors.frame.to_numpy(), actors[field].to_numpy()])
            extra = index.get_indexer(keys)
            selected[extra[extra >= 0]] = True
    return tracks.loc[selected].copy()


def export_tables(tracks, track_meta, meta, domain, config):
    fps = int(meta["frameRate"])
    tracks, groups, lanes, transitions = prepare(tracks, track_meta, meta, domain)
    following = following_samples(tracks, fps, domain)
    decisions = decision_samples(tracks, groups, lanes, fps, config)
    events, windows = event_samples(groups, lanes, transitions, fps, domain)
    tables = {"following": following, "events": events, "windows": windows, "decisions": decisions,
              "matched_controls": matched_controls(decisions, config),
              "observations": observation_samples(tracks, windows, fps, domain, config)}
    summary = {name + "_rows": len(table) for name, table in tables.items()}
    summary.update(stable_supported_following_rows=int(following.diagnostic_stable_supported_hindsight.sum()),
                   complete_windows=int(events.complete_window.sum()), strict_windows=int(events.strict_compatible_diagnostic.sum()),
                   positive_side_decisions=int((decisions.label_side_crossing == 1).sum()),
                   unknown_side_decisions=int((decisions.label_side_crossing < 0).sum()),
                   matched_positive_decisions=int((tables["matched_controls"].control_decision_id >= 0).sum()))
    return tables, summary


def audit_tables(tables):
    """Check exported observation references and control labels before writing."""
    obs = tables["observations"]
    keys = pd.MultiIndex.from_frame(obs[["frame", "id"]])
    if not keys.is_unique:
        raise ValueError("Duplicate exported observation")
    def require(frame, identity):
        lookup = pd.MultiIndex.from_arrays([np.asarray(frame), np.asarray(identity)])
        if (keys.get_indexer(lookup) < 0).any():
            raise ValueError("Exported sample references an absent observation")
    following, windows, decisions = (tables[k] for k in ("following", "windows", "decisions"))
    for actor in ("follower_id", "leader_id"):
        require(following.frame, following[actor])
    for actor, mask in (("id", "actor_observed"), ("rear_id", "rear_observed")):
        valid = windows.loc[windows[mask].astype(bool)]
        require(valid.frame, valid[actor])
    require(decisions.frame, decisions.id)
    rear = decisions.loc[decisions.rear_valid_now.astype(bool)]
    require(rear.frame, rear.rear_id)
    matched = tables["matched_controls"]
    matched = matched.loc[matched.control_decision_id >= 0]
    indexed = decisions.set_index("decision_id")
    if (indexed.loc[matched.positive_decision_id, "label_side_crossing"] != 1).any():
        raise ValueError("Matched positive label disagrees")
    if (indexed.loc[matched.control_decision_id, "label_first_crossing_side"] != "stay").any():
        raise ValueError("Matched control is not a fully observed stay outcome")
    return {"passed": True, "observation_keys_unique": True, "sample_references_complete": True,
            "matched_labels_valid": True}


def select_recordings(manifest, requested):
    records = manifest["recordings"]
    ids = [str(r["recording_id"]).zfill(2) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recording assignment")
    train = [str(r["recording_id"]).zfill(2) for r in records if r["split"] == "train"]
    selected = [str(r).zfill(2) for r in requested] if requested else train
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(train):
        raise ValueError("This exporter only accepts unique recordings in the existing TRAIN split")
    return selected, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/highd_pair_sample_export_v1.json"))
    parser.add_argument("--recordings", nargs="+")
    args = parser.parse_args()
    config = inventory.read_json(args.config)
    for name in ("decision_hz", "decision_horizon_s", "matching_speed_bin_mps", "matching_rear_gap_bin_m"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError("Invalid config: " + name)
    domain = inventory.read_json(config["inventory_config"])
    freeze = inventory.read_json(config["freeze_contract"])
    manifest_path = Path(domain["primary_manifest"])
    manifest = inventory.read_json(manifest_path)
    selected, train = select_recordings(manifest, args.recordings)
    if config["allowed_split"] != "train" or set(train) != set(freeze["train_recordings"]):
        raise ValueError("Train split disagrees with frozen reference")
    if inventory.digest(manifest_path) != freeze["split_manifest"]["sha256"]:
        raise ValueError("Frozen split manifest hash disagrees")
    for field, grid in (("speed_mps", "speed"), ("gap_m", "gap"), ("range_rate_mps", "range_rate"), ("acceleration_mps2", "acceleration")):
        if domain[field] != freeze["grid"][grid][:2]:
            raise ValueError("Domain disagrees with frozen reference: " + field)
    protocol = {"schema_version": 1, "config": config, "domain": domain, "train_recordings": train,
                "source_root": str(args.source_root.resolve()), "frozen_reference_status": freeze["status"],
                "sha256": {str(path): inventory.digest(path) for path in [args.config, Path(config["inventory_config"]),
                            Path(config["freeze_contract"]), manifest_path, Path(__file__), Path(inventory.__file__)]},
                "scope": "Train-only offline export; no model fit/runtime acceptance. Decision labels predict lane-ID crossings, not verified maneuver initiation.",
                "features_forbidden": ["label_*", "diagnostic_*", "relative_time_to_crossing_s", "event membership", "complete_window"],
                "matching": "1:1 with replacement within recording/direction/source lane/side/5mps speed/20m rear gap strata; diagnostic only, NOT the occurrence-rate denominator.",
                "identity": "All IDs are recording-qualified. No within-recording train/test split. Windows overlap and decisions are temporally correlated.",
                "source_integrity": "Source CSV size+mtime only; output tables and code/configs SHA256 checked."}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "export_protocol.json"
    if protocol_path.exists():
        if inventory.read_json(protocol_path) != protocol:
            raise ValueError("Output protocol changed; use a fresh output directory")
    else:
        inventory.write_new(protocol_path, protocol)
    for rec in selected:
        started = time.monotonic()
        sources = {name: args.source_root / "data" / f"{rec}_{name}.csv" for name in ("tracks", "tracksMeta", "recordingMeta")}
        stamps = {name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for name, p in sources.items()}
        dest = args.output / "recordings" / rec
        done = dest / "summary.json"
        if done.exists():
            cached = inventory.read_json(done)
            if cached["source_stamps"] != stamps or any(inventory.digest(dest / k) != v for k, v in cached["output_sha256"].items()):
                raise ValueError("Source or cached output changed: " + rec)
            print(f"recording={rec} cached", flush=True)
            continue
        if dest.exists():
            raise ValueError("Incomplete recording output retained; use a fresh output root: " + str(dest))
        tracks = pd.read_csv(sources["tracks"], usecols=SOURCE_COLUMNS)
        tm = pd.read_csv(sources["tracksMeta"])
        meta = pd.read_csv(sources["recordingMeta"]).iloc[0]
        tables, summary = export_tables(tracks, tm, meta, domain, config)
        table_audit = audit_tables(tables)
        dest.mkdir(parents=True)
        hashes = {}
        for name, table in tables.items():
            table.insert(0, "recording_id", rec)
            table.insert(1, "split", "train")
            path = dest / f"{name}.csv.gz"
            table.to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
            hashes[path.name] = inventory.digest(path)
        result = {"recording_id": rec, "split": "train", "location_id": int(meta.locationId),
                  "frame_rate": int(meta.frameRate), "counts": summary,
                  "road_geometry": {"upper_lane_markings": str(meta.upperLaneMarkings),
                                    "lower_lane_markings": str(meta.lowerLaneMarkings),
                                    "lanes_by_direction": {str(k): sorted(v) for k, v in inventory.lane_sets(meta).items()}},
                  "table_audit": table_audit,
                  "source_stamps": stamps, "output_sha256": hashes, "elapsed_s": round(time.monotonic() - started, 3)}
        inventory.write_new(done, result)
        print(json.dumps({"recording_id": rec, "elapsed_s": result["elapsed_s"], **summary}), flush=True)
    records = [inventory.read_json(p) for p in sorted((args.output / "recordings").glob("*/summary.json"))]
    counts = Counter()
    for rec in records:
        counts.update(rec["counts"])
    result = {"schema_version": 1, "recording_ids": [r["recording_id"] for r in records], "counts": dict(counts),
              "all_train_recordings_exported": set(r["recording_id"] for r in records) == set(train)}
    summary_path = args.output / f"export_summary_{len(records):02d}.json"
    if not summary_path.exists():
        inventory.write_new(summary_path, result)
    elif inventory.read_json(summary_path) != result:
        raise ValueError("Aggregate changed; use fresh output")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
