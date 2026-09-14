"""Pre-register high-conflict-potential labels from native trajectory geometry.

This is retrospective offline scene mining, not online risk prediction.  Labels
do not use a planner rollout, generated candidate, or held-out test recording.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .highd import file_hash, validate_config as validate_highd_config, write_json
from .trajectory_benchmark import box_clearance, validate_scene


ALLOWED_SPLITS = ("train", "validation")


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Only conflict diagnostic schema_version 1 is supported")
    positive = ("max_event_headway_s", "min_event_closing_speed_mps",
                "max_overlap_longitudinal_clearance_m", "min_intervention_lead_s")
    for name in positive:
        if not np.isfinite(config.get(name)) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    prefix = config.get("observed_prefix_s")
    if not np.isfinite(prefix) or prefix < 0:
        raise ValueError("observed_prefix_s must be finite and nonnegative")
    if not isinstance(config.get("require_changer_ahead_at_event"), bool):
        raise ValueError("require_changer_ahead_at_event must be boolean")


def conflict_features(scene, config):
    """Compute native-trajectory features and an auditable diagnostic label."""
    validate_scene(scene)
    validate_config(config)
    actors = {actor["id"]: actor for actor in scene["actors"]}
    ego = next(actor for actor in scene["actors"] if actor["role"] == "CAV")
    changer = actors[scene["changer_id"]]
    time = np.asarray(scene["time"], dtype=float)
    ego_xy, changer_xy = np.asarray(ego["xy"]), np.asarray(changer["xy"])
    prefix = float(config["observed_prefix_s"])
    future = time > prefix + 1e-9
    if not np.any(future):
        raise ValueError("Observed prefix leaves no future samples")

    half_width = (ego["width"] + changer["width"]) / 2
    half_length = (ego["length"] + changer["length"]) / 2
    lateral_overlap = np.abs(changer_xy[:, 1] - ego_xy[:, 1]) <= half_width
    future_overlap = future & lateral_overlap
    longitudinal_clearance = np.abs(changer_xy[:, 0] - ego_xy[:, 0]) - half_length
    front_net_gap = changer_xy[:, 0] - ego_xy[:, 0] - half_length
    box_separation = box_clearance(
        ego_xy, changer_xy, [ego["length"], ego["width"]],
        np.array([changer["length"], changer["width"]]))

    event_time = float(scene["event_time"])
    event_index = int(np.argmin(np.abs(time - event_time)))
    if abs(time[event_index] - event_time) > 1e-8:
        raise ValueError("Event time is not represented in the scene samples")
    overlap_indices = np.flatnonzero(future_overlap)
    first_overlap_index = int(overlap_indices[0]) if len(overlap_indices) else None
    first_overlap_time = float(time[first_overlap_index]) if first_overlap_index is not None else None
    lead = first_overlap_time - prefix if first_overlap_time is not None else None
    min_overlap_clearance = (float(np.min(longitudinal_clearance[future_overlap]))
                             if np.any(future_overlap) else None)
    interaction = scene["interaction"]
    event_headway = interaction.get("time_headway_s")
    event_closing = float(interaction["closing_speed_mps"])
    changer_ahead = bool(front_net_gap[event_index] > 0)

    criteria = {
        "event_headway_within_limit": bool(event_headway is not None and
                                             event_headway <= config["max_event_headway_s"]),
        "event_closing_speed_above_minimum": bool(event_closing >= config["min_event_closing_speed_mps"]),
        "changer_ahead_at_event": bool(changer_ahead or not config["require_changer_ahead_at_event"]),
        "future_lateral_overlap": bool(np.any(future_overlap)),
        "enough_intervention_lead": bool(lead is not None and lead >= config["min_intervention_lead_s"]),
        "overlap_longitudinal_clearance_within_limit": bool(
            min_overlap_clearance is not None and
            min_overlap_clearance <= config["max_overlap_longitudinal_clearance_m"]),
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "scene_id": scene["scene_id"], "recording_id": scene["recording_id"],
        "location_id": scene["location_id"], "split": scene["split"],
        "interaction_stratum": interaction["stratum"], "event_time_s": event_time,
        "event_net_gap_m": float(interaction["net_gap_m"]),
        "event_time_headway_s": event_headway,
        "event_closing_speed_mps": event_closing, "event_ttc_s": interaction.get("ttc_s"),
        "changer_ahead_at_event": changer_ahead,
        "has_future_lateral_overlap": bool(np.any(future_overlap)),
        "first_future_lateral_overlap_time_s": first_overlap_time,
        "intervention_lead_to_overlap_s": float(lead) if lead is not None else None,
        "front_net_gap_at_first_overlap_m": (float(front_net_gap[first_overlap_index])
                                              if first_overlap_index is not None else None),
        "minimum_longitudinal_clearance_during_overlap_m": min_overlap_clearance,
        "minimum_native_target_future_clearance_m": float(np.min(box_separation[future])),
        "native_target_future_collision": bool(np.any(box_separation[future] <= 0)),
        "criteria": criteria, "high_conflict_potential": not failed,
        "exclusion_reasons": failed,
    }


def _load_scene(manifest_path, manifest, item, split):
    path = (manifest_path.parent / item["path"]).resolve()
    if not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("Scene path escapes dataset directory")
    if file_hash(path) != item["sha256"]:
        raise ValueError(f"Scene checksum mismatch: {item['scene_id']}")
    scene = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f"{int(rec):02d}" for rec in manifest["config"]["splits"][split]}
    if (scene["split"] != split or scene["recording_id"] not in allowed
            or item["split"] != split or scene["scene_id"] != item["scene_id"]):
        raise ValueError("Scene split or identity disagrees with manifest")
    return scene


def _quantiles(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return None
    return {name: float(value) for name, value in zip(
        ("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1]))}


def summarize(records, config, manifest_sha256):
    selected = [record for record in records if record["high_conflict_potential"]]
    return {
        "status": "retrospective offline diagnostic label; not online prediction or accident ground truth",
        "config": config, "manifest_sha256": manifest_sha256,
        "splits": sorted({record["split"] for record in records}),
        "scene_count": len(records), "recording_count": len({r["recording_id"] for r in records}),
        "high_conflict_potential_scenes": len(selected),
        "high_conflict_potential_recordings": len({r["recording_id"] for r in selected}),
        "native_target_future_collision_scenes": sum(r["native_target_future_collision"] for r in records),
        "by_split": {split: {
            "scene_count": sum(r["split"] == split for r in records),
            "high_conflict_potential_scenes": sum(
                r["split"] == split and r["high_conflict_potential"] for r in records),
        } for split in sorted({record["split"] for record in records})},
        "by_interaction_stratum": dict(Counter(r["interaction_stratum"] for r in selected)),
        "exclusion_reason_counts": dict(Counter(
            reason for record in records for reason in record["exclusion_reasons"])),
        "feature_quantiles": {
            "event_net_gap_m": _quantiles([r["event_net_gap_m"] for r in records]),
            "event_time_headway_s": _quantiles([r["event_time_headway_s"] for r in records]),
            "event_closing_speed_mps": _quantiles([r["event_closing_speed_mps"] for r in records]),
            "intervention_lead_to_overlap_s": _quantiles([r["intervention_lead_to_overlap_s"] for r in records]),
            "minimum_longitudinal_clearance_during_overlap_m": _quantiles(
                [r["minimum_longitudinal_clearance_during_overlap_m"] for r in records]),
            "minimum_native_target_future_clearance_m": _quantiles(
                [r["minimum_native_target_future_clearance_m"] for r in records]),
        },
        "selected_scene_ids": [record["scene_id"] for record in selected],
    }


def run_diagnostics(manifest_path, output, config, splits=ALLOWED_SPLITS):
    validate_config(config)
    splits = tuple(dict.fromkeys(splits))
    if not splits or any(split not in ALLOWED_SPLITS for split in splits):
        raise ValueError("Conflict diagnostics only allow train and validation; test remains locked")
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_highd_config(manifest["config"])
    items = [item for item in manifest["scenes"] if item["split"] in splits]
    records = [conflict_features(
        _load_scene(manifest_path, manifest, item, item["split"]), config) for item in items]
    if not records:
        raise ValueError("No scenes found for requested splits")
    summary = summarize(records, config, file_hash(manifest_path))
    write_json(output / "records.json", records)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/conflict_diagnostics.json")
    parser.add_argument("--splits", nargs="+", choices=ALLOWED_SPLITS, default=list(ALLOWED_SPLITS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_diagnostics(
        args.manifest, args.output, json.loads(Path(args.config).read_text(encoding="utf-8")), args.splits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
