"""Export measured SHRP2 multi-BV scene seeds for later SUMO/NADE mapping.

This module deliberately stops at a source-frame scene seed. It does not infer
lanes, invent actions, or claim that a geometrically selected target is the
verified crash participant. The resulting records are therefore suitable for
scenario initialization and split-aware sampling, not direct D2RL training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .shrp2_collision import load_category, locate_export_online, project_split
from .shrp2_diffusion_data_audit import build_pair_window, validate_config
from .shrp2_reference_trajectory import (
    _actor_track,
    _interp_angle,
    _quality_summary,
    _rotation,
    _wrap_array,
)


def _finite_track(rows, config, anchor_s, origin, rotation, reference_yaw):
    required = [
        "time", "x_sur", "y_sur", "v_sur", "psi_sur",
    ]
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    rows = rows.sort_values("time").drop_duplicates("time", keep="first")
    if len(rows) < 3:
        raise ValueError("insufficient_context_samples")
    start = anchor_s - float(config["history_s"])
    raw = rows["time"].to_numpy(dtype=float)
    if raw[0] > start + 1e-8 or raw[-1] < anchor_s - 1e-8:
        raise ValueError("insufficient_context_history")
    left = max(0, int(np.searchsorted(raw, start)) - 1)
    right = min(len(raw), int(np.searchsorted(raw, anchor_s, side="right")) + 1)
    if np.max(np.diff(raw[left:right])) > config["maximum_interpolation_gap_s"] + 1e-8:
        raise ValueError("context_source_gap_exceeds_limit")
    if (rows["v_sur"].to_numpy(dtype=float) < 0).any():
        raise ValueError("negative_context_speed")
    times = np.arange(
        0.0,
        float(config["history_s"]) + 0.5 / float(config["sample_hz"]),
        1.0 / float(config["sample_hz"]),
    )
    query = start + times
    xy = np.column_stack(
        [np.interp(query, raw, rows[key]) for key in ("x_sur", "y_sur")]
    )
    return _actor_track(
        times,
        (xy - origin) @ rotation.T,
        np.interp(query, raw, rows["v_sur"]),
        _wrap_array(
            _interp_angle(query, raw, rows["psi_sur"].to_numpy()) - reference_yaw
        ),
        None,
        config,
    )


def _target_distance_at_anchor(rows, anchor_s):
    finite = rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time", "x_ego", "y_ego", "x_sur", "y_sur"]
    )
    if finite.empty:
        raise ValueError("missing_context_anchor")
    row = finite.loc[(finite["time"] - anchor_s).abs().idxmin()]
    return float(
        np.hypot(float(row["x_sur"]) - float(row["x_ego"]),
                 float(row["y_sur"]) - float(row["y_ego"]))
    )


def build_multibv_seed(window, event_rows, metadata_row, config, bv_count=2):
    """Add nearest measured context BVs to one quality-passed pair window."""
    if bv_count < 2:
        raise ValueError("bv_count must be at least 2")
    primary_id = int(window["source"]["target_id"])
    anchor_s = float(metadata_row["impact_timestamp"]) / 1000.0
    start_s = anchor_s - float(config["history_s"])
    ego_rows = event_rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time", "x_ego", "y_ego", "psi_ego"]
    ).sort_values("time").drop_duplicates("time", keep="first")
    if ego_rows.empty:
        raise ValueError("missing_cav_track")
    raw = ego_rows["time"].to_numpy(dtype=float)
    if raw[0] > start_s + 1e-8 or raw[-1] < anchor_s - 1e-8:
        raise ValueError("insufficient_cav_history")
    origin = np.array([
        np.interp(start_s, raw, ego_rows[key]) for key in ("x_ego", "y_ego")
    ])
    yaw = float(_interp_angle(
        np.array([start_s]), raw, ego_rows["psi_ego"].to_numpy(dtype=float)
    )[0])
    rotation = _rotation(-yaw)

    distances = []
    grouped = event_rows.groupby("target_id", sort=True)
    for target_id, rows in grouped:
        target_id = int(target_id)
        if target_id == primary_id:
            continue
        try:
            distances.append((_target_distance_at_anchor(rows, anchor_s), target_id))
        except ValueError:
            continue
    distances.sort(key=lambda item: (item[0], item[1]))
    selected = [(primary_id, "BV_primary")]
    context_needed = bv_count - 1
    for _, target_id in distances[:context_needed]:
        selected.append((target_id, f"BV_context_{len(selected)}"))
    if len(selected) != bv_count:
        raise ValueError("insufficient_context_targets")

    tracks = [np.asarray(window["states"], dtype=float)[:, 1, :]]
    masks = [np.asarray(window["state_mask"], dtype=bool)[:, 1, :]]
    actor_records = [{
        "id": "BV_primary",
        "role": "primary_risk_bv",
        "source_target_id": primary_id,
        "state_channels": window["state_channels"],
    }]
    context_quality = {}
    for target_id, actor_id in selected[1:]:
        track = _finite_track(
            grouped.get_group(target_id), config, anchor_s, origin, rotation, yaw
        )
        quality = _quality_summary(track, config)
        if not quality["position_speed_consistent"]:
            raise ValueError("context_quality_fail")
        tracks.append(np.column_stack([
            track["xy_m"], track["reported_speed_mps"], track["reported_heading_rad"]
        ]))
        context_mask = np.ones_like(tracks[-1], dtype=bool)
        context_mask[:, 3] = quality["heading_usable"]
        masks.append(context_mask)
        context_quality[actor_id] = quality
        actor_records.append({
            "id": actor_id,
            "role": "context_bv",
            "source_target_id": target_id,
            "state_channels": window["state_channels"],
        })

    states = np.stack([
        np.asarray(window["states"], dtype=float)[:, 0, :], *tracks
    ], axis=1)
    state_mask = np.stack([
        np.asarray(window["state_mask"], dtype=bool)[:, 0, :], *masks
    ], axis=1)
    actor_records.insert(0, {
        "id": "CAV",
        "role": "CAV",
        "state_channels": window["state_channels"],
    })
    source = dict(window["source"])
    source.update({
        "target_role": "primary geometric target plus nearest measured context targets",
        "context_target_ids": [target_id for target_id, _ in selected[1:]],
        "observed_target_count": int(event_rows["target_id"].nunique()),
    })
    return {
        "record_type": "shrp2_measured_multibv_seed_v1",
        "time_s": window["time_s"],
        "state_channels": window["state_channels"],
        "position_reference": ["CAV_centroid", "BV_front_bumper"],
        "states": states.tolist(),
        "state_mask": state_mask.tolist(),
        "actors": actor_records,
        "source": source,
        "condition": {
            "initial_state": states[0].tolist(),
            "initial_state_mask": state_mask[0].tolist(),
            "critical_time_s": float(window["condition"]["critical_time_s"]),
            "mapping_status": "source_frame_only; lane and route mapping pending",
        },
        "quality": {
            "primary": window["quality"]["BV"],
            "contexts": context_quality,
        },
        "training_window_ready": True,
        "drl_training_ready": False,
        "limitations": [
            "Context vehicles are nearest measured targets, not verified accident participants.",
            "The record has no inferred lane, route, policy action, NDD probability, or importance weight.",
        ],
    }


def _read_windows(audit_root):
    windows = []
    for split in ("train", "validation", "test"):
        path = Path(audit_root) / split / "windows.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                windows.append(json.loads(line))
    return windows


def export_multibv_seeds(source_root, audit_root, output, config, bv_count=2):
    """Export context-augmented seeds, loading one SHRP2 category at a time."""
    validate_config(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty multi-BV seed output directory")
    windows = _read_windows(audit_root)
    if not windows:
        raise FileNotFoundError("No audit windows found under audit_root")
    by_category = defaultdict(list)
    for window in windows:
        by_category[window["source"]["category"]].append(window)
    counts = Counter()
    exclusions = Counter()
    records = []
    for category, category_windows in sorted(by_category.items()):
        _, _, _, metadata, data = load_category(source_root, category)
        metadata_by_id = metadata.drop_duplicates("event_id").set_index("event_id")
        for window in category_windows:
            event_id = int(window["source"]["event_id"])
            try:
                rows = data[data["event_id"] == event_id]
                meta = metadata_by_id.loc[event_id]
                seed = build_multibv_seed(window, rows, meta, config, bv_count=bv_count)
                split = window["source"]["split"]
                destination = output / split / "seeds.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(seed, ensure_ascii=False, allow_nan=False) + "\n")
                records.append({
                    "event_id": event_id,
                    "category": category,
                    "split": split,
                    "context_target_ids": seed["source"]["context_target_ids"],
                })
                counts[split] += 1
            except (KeyError, ValueError) as exc:
                exclusions[str(exc)] += 1
        del metadata, data
    summary = {
        "schema_version": 1,
        "record_type": "shrp2_measured_multibv_seed_v1",
        "dataset_doi": config["dataset_doi"],
        "source_root": str(locate_export_online(source_root)),
        "audit_root": str(audit_root),
        "bv_count": bv_count,
        "window_count": len(windows),
        "exported_count": len(records),
        "exported_by_split": dict(counts),
        "excluded_count": sum(exclusions.values()),
        "exclusion_reasons": dict(exclusions),
        "drl_training_ready": False,
        "limitations": [
            "Seeds remain in the measured CAV frame; SUMO lane/route mapping is pending.",
            "They contain no policy action or D2RL importance-weight labels.",
            "Event-level split is inherited from the audit manifest.",
        ],
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "seed_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--audit_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/shrp2_diffusion_data_audit.json")
    parser.add_argument("--bv_count", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = export_multibv_seeds(
        args.source_root, args.audit_root, args.output, config, bv_count=args.bv_count
    )
    print(json.dumps({
        key: result[key]
        for key in ("window_count", "exported_count", "exported_by_split", "excluded_count")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
