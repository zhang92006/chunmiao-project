"""Export native highD local-window NDD validation scenes and observed references."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .highd_ndd_baseline import _load_split_manifest


def support_ids(scene, core_ids, config):
    """Add initial-frame context only; never require support vehicles' futures."""
    radius = float(config.get("support_range_m", 0))
    if radius <= 0:
        return [], {}
    core = scene[scene.id.isin(core_ids)]
    if len(core) != len(core_ids) or core.direction.nunique() != 1:
        raise ValueError("Core vehicles must be present in one driving direction")
    candidates = scene[(scene.direction == core.direction.iloc[0]) & ~scene.id.isin(core_ids)]
    candidates = candidates.loc[
        np.min(abs(candidates.x.to_numpy()[:, None] - core.x.to_numpy()), axis=1) <= radius]
    valid = candidates.xVelocity.abs().between(*config["speed_range_mps"])
    return sorted(int(i) for i in candidates.loc[valid, "id"]), {
        "support_speed_outside_domain": int((~valid).sum())}


def export_reference(source_root, split_manifest, config, output, anchor_manifest=None):
    length_mode = config.get("vehicle_length_mode", "fixed")
    if length_mode not in ("fixed", "source"):
        raise ValueError("vehicle_length_mode must be fixed or source")
    root = Path(source_root) / "data"
    splits = _load_split_manifest(Path(split_manifest))
    allowed = ("validation",) if config.get("purpose") == "frozen_model_validation" else ("train", "calibration")
    for recording, split in config["recordings"].items():
        if split not in allowed or recording not in splits.get(split, []):
            raise ValueError("Native pilot must respect its development or frozen-validation split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config["seed"])
    records = []
    exclusions = Counter()
    anchors = {}
    if anchor_manifest:
        source = json.loads(Path(anchor_manifest).read_text(encoding="utf-8"))
        for record in source["records"]:
            template = json.loads(Path(record["template_path"]).read_text(encoding="utf-8"))
            meta = template["bridge_metadata"]
            recording = str(meta["source_recording"])
            if config["recordings"].get(recording) != meta["source_split"]:
                raise ValueError("Anchor recording/split does not match config")
            if float(template["duration"]) != float(config["duration_s"]):
                raise ValueError("Anchored comparison must preserve the duration")
            anchors.setdefault(recording, []).append(template)
    for recording, split in config["recordings"].items():
        metadata = pd.read_csv(root / f"{recording}_recordingMeta.csv").iloc[0]
        if any(len(str(metadata[k]).split(";")) != 3
               for k in ("upperLaneMarkings", "lowerLaneMarkings")):
            raise ValueError("Pilot requires exactly two lanes in each highD direction")
        rate = float(metadata["frameRate"])
        horizon = int(round(config["duration_s"] * rate))
        tracks = pd.read_csv(root / f"{recording}_tracks.csv")
        vehicle_metadata = pd.read_csv(root / f"{recording}_tracksMeta.csv").set_index("id")
        direction = vehicle_metadata["drivingDirection"].to_dict()
        tracks["direction"] = tracks["id"].map(direction)
        groups = {int(i): g.set_index("frame") for i, g in tracks.groupby("id")}
        lane_sets = {d: sorted(tracks.loc[tracks.direction == d, "laneId"].unique())
                     for d in (1, 2)}
        if any(len(lanes) != 2 for lanes in lane_sets.values()):
            raise ValueError("Observed lanes do not match two-lane metadata")
        anchored = {(t["bridge_metadata"]["source_frame"], t["bridge_metadata"]["source_ego_id"]): t
                    for t in anchors.get(recording, [])}
        if anchor_manifest and not anchored:
            raise ValueError(f"No anchors for recording {recording}")
        candidates = np.array(list(anchored)) if anchor_manifest else tracks.loc[
            tracks.frame % config["frame_stride"] == 0, ["frame", "id"]].to_numpy()
        if not anchor_manifest:
            rng.shuffle(candidates)
        candidate_frames = {int(c[0]) for c in candidates}
        by_frame = {int(f): g for f, g in tracks.groupby("frame") if f in candidate_frames}
        used_egos = set()
        count = 0
        for frame, ego_id in candidates:
            frame, ego_id = int(frame), int(ego_id)
            if ego_id in used_egos:
                continue
            ego = groups[ego_id].loc[frame]
            sign = -1 if int(ego.direction) == 1 else 1
            full_scene = by_frame[frame]
            anchor = anchored.get((frame, ego_id))
            if anchor:
                evaluation = anchor["bridge_metadata"].get(
                    "evaluation_actor_ids", [a["id"] for a in anchor["actors"]])
                core_ids = [ego_id] + [int(i.removeprefix("BV_")) for i in evaluation]
                scene = full_scene[full_scene.id.isin(core_ids)]
                if len(scene) != len(core_ids):
                    raise ValueError("Anchor lost a core vehicle")
            else:
                scene = full_scene[(full_scene.direction == ego.direction)
                    & ((full_scene.x - ego.x).abs() <= config["neighbor_range_m"])]
            if len(scene) - 1 < config["minimum_bvs"]:
                exclusions["too_few_neighbors"] += 1
                continue
            if not scene.xVelocity.abs().between(*config["speed_range_mps"]).all():
                exclusions["initial_speed_outside_domain"] += 1
                continue
            ids = [ego_id] + sorted(int(i) for i in scene.id if i != ego_id)
            core_ids = list(ids)
            window_frames = np.arange(frame, frame + horizon + 1)
            if any(not np.isin(window_frames, groups[i].index).all() for i in ids):
                exclusions["incomplete_observed_future"] += 1
                continue
            added, rejected = support_ids(full_scene, core_ids, config)
            exclusions.update(rejected)
            ids += added
            # Preserve relative front-bumper positions; log the explicit length mapping.
            ego_front = float(ego.x + (ego.width if sign == 1 else 0))
            lane_map = dict(zip(lane_sets[int(ego.direction)], (0, 1) if sign == -1 else (1, 0)))
            specs, references, lengths = [], [], {}
            for n, vehicle_id in enumerate(ids):
                first = groups[vehicle_id].loc[frame]
                front = float(first.x + (first.width if sign == 1 else 0))
                sim_id = "CAV" if n == 0 else f"BV_{vehicle_id}"
                lengths[sim_id] = float(first.width) if length_mode == "source" else config["vehicle_length_m"]
                specs.append({"id": sim_id, "role": "CAV" if n == 0 else "BV",
                    "route": "route_0", "lane_index": lane_map[int(first.laneId)],
                    "position": 400 + sign * (front - ego_front),
                    "speed": abs(float(first.xVelocity)), "controller": "IDM"})
                if vehicle_id not in core_ids:
                    # Support is simulated, not scored against a censored raw future.
                    continue
                future = groups[vehicle_id].loc[window_frames]
                for f, row in future.iterrows():
                    v = abs(float(row.xVelocity))
                    gap = float(row.dhw) if int(row.precedingId) > 0 else None
                    closing = v - abs(float(row.precedingXVelocity)) if gap is not None else 0
                    references.append({"time": (int(f) - frame) / rate, "vehicle_id": sim_id,
                        "speed_mps": v, "acceleration_mps2": sign * float(row.xAcceleration),
                        "lane_index": lane_map[int(row.laneId)], "gap_m": gap,
                        "source_leader_id": int(row.precedingId) if gap is not None else None,
                        "leader_id": ("CAV" if int(row.precedingId) == ego_id else f"BV_{int(row.precedingId)}")
                            if gap is not None else None,
                        "source_leader_in_template": int(row.precedingId) in ids if gap is not None else None,
                        "headway_s": gap / v if gap is not None and v > 0 else None,
                        "ttc_s": gap / closing if gap is not None and closing > 0 else None})
            overlap = False
            for lane in (0, 1):
                ordered = sorted((s for s in specs if s["lane_index"] == lane), key=lambda s: s["position"])
                overlap |= any(b["position"] - a["position"] < lengths[b["id"]]
                               for a, b in zip(ordered, ordered[1:]))
            if overlap:
                exclusions["overlap_after_length_mapping"] += 1
                continue
            if any(s["position"] < lengths[s["id"]] or s["position"] > 1000 for s in specs):
                exclusions["outside_sumo_initialization_region"] += 1
                continue
            template_id = f"highd_naturalistic_{recording}_{frame}_{ego_id}"
            template = {"template_id": template_id, "description": "Native highD local-window NDD pilot",
                "map": "2Lane", "route": "route_0", "duration": config["duration_s"],
                "ego": specs[0], "actors": specs[1:], "events": [], "perturbations": [],
                "tags": ["highd", split], "bridge_metadata": {
                    "source_split": split, "source_recording": recording,
                    "source_frame": frame, "source_ego_id": ego_id,
                    "evaluation_actor_ids": [f"BV_{i}" for i in core_ids if i != ego_id],
                    "support_actor_ids": [f"BV_{i}" for i in added],
                    "observation_range_m": config.get("observation_range_m", 115.0),
                    "source_lengths_m": {str(i): float(groups[i].loc[frame].width) for i in ids},
                    "vehicle_length_mode": length_mode, "vehicle_lengths_m": lengths,
                    "mapped_length_m": config["vehicle_length_m"] if length_mode == "fixed" else None,
                    "source_record_type": "highd_native_local_window"}}
            template_path = output / f"{template_id}.json"
            template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
            reference_path = output / f"{template_id}_reference.json"
            reference_path.write_text(json.dumps({"snapshots": references,
                "evaluation_actor_ids": template["bridge_metadata"]["evaluation_actor_ids"]}, indent=2), encoding="utf-8")
            records.append({"status": "template_created", "split": split,
                "support_count": len(added), "evaluation_bv_count": len(core_ids) - 1,
                "template_path": str(template_path), "reference_path": str(reference_path)})
            used_egos.add(ego_id)
            count += 1
            if not anchor_manifest and count >= config["scenes_per_recording"]:
                break
        if anchor_manifest and count != len(anchored):
            raise ValueError("Anchored export rejected a scene; inspect inputs, do not silently change the cohort")
    manifest = {"schema_version": 2, "config": config, "records": records,
                "anchor_manifest_sha256": hashlib.sha256(Path(anchor_manifest).read_bytes()).hexdigest()
                    if anchor_manifest else None,
                "split_manifest_sha256": hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest(),
                "exclusions": dict(exclusions)}
    (output / "native_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"exported": len(records), "by_split": dict(Counter(r["split"] for r in records)),
            "exclusions": dict(exclusions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor_manifest", help="Preserve an existing cohort's core vehicles and initial frames")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(export_reference(args.source_root, args.split_manifest, config, args.output,
                                     args.anchor_manifest), indent=2))


if __name__ == "__main__":
    main()
