"""Inventory public SHRP2 categories and optionally export quality-gated pair windows.

The default reads metadata only. --scan_trajectories reads one HDF5 category
at a time; --export_windows writes measured references, never SUMO episodes.
Category labels and geometric contacts are kept as separate observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .shrp2_collision import (
    DATA_COLUMNS, associate_primary_target, front_bumper_to_center,
    load_category, locate_export_online, project_split,
)
from .shrp2_reference_trajectory import (
    _actor_track, _collision_audit, _file_hash, _interp_angle, _quality_summary,
    _rotation, _wrap_array, validate_config as validate_reference_config,
)


def validate_config(config):
    validate_reference_config(config)
    fractions = config.get("split_fractions", {})
    if set(fractions) != {"train", "validation", "test"} or not all(
        isinstance(v, (float, int)) and not isinstance(v, bool)
        and math.isfinite(v) and 0 < v < 1 for v in fractions.values()
    ) or not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("split_fractions must be positive and sum to one")
    if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    for key in ("maximum_interpolation_gap_s", "maximum_target_alignment_dt_s"):
        if not isinstance(config.get(key), (int, float)) or isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")


def build_pair_window(rows, meta, config):
    """Resample one measured target track in the initial CAV frame, with masks."""
    required = sorted(DATA_COLUMNS - {"event_id", "target_id"})
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    rows = rows.sort_values("time").drop_duplicates("time", keep="first")
    if len(rows) < 3:
        raise ValueError("insufficient_finite_samples")
    anchor = float(meta["impact_timestamp"]) / 1000.0
    start = anchor - config["history_s"]
    raw = rows["time"].to_numpy(dtype=float)
    # No extrapolation and no interpolation across missing source segments.
    if raw[0] > start + 1e-8 or raw[-1] < anchor - 1e-8:
        raise ValueError("insufficient_critical_history")
    left = max(0, int(np.searchsorted(raw, start)) - 1)
    right = min(len(raw), int(np.searchsorted(raw, anchor, side="right")) + 1)
    if np.max(np.diff(raw[left:right])) > config["maximum_interpolation_gap_s"] + 1e-8:
        raise ValueError("source_gap_exceeds_limit")
    if (rows[["v_ego", "v_sur"]].to_numpy() < 0).any():
        raise ValueError("negative_reported_speed")
    times = np.arange(0.0, config["history_s"] + 0.5 / config["sample_hz"], 1.0 / config["sample_hz"])
    query = start + times
    origin = np.array([np.interp(start, raw, rows[k]) for k in ("x_ego", "y_ego")])
    yaw = float(_interp_angle(np.array([start]), raw, rows["psi_ego"].to_numpy())[0])
    rotation = _rotation(-yaw)
    tracks = {}
    for name, suffix in (("CAV", "ego"), ("BV", "sur")):
        xy = np.column_stack([np.interp(query, raw, rows[f"{k}_{suffix}"]) for k in ("x", "y")])
        tracks[name] = _actor_track(
            times, (xy - origin) @ rotation.T,
            np.interp(query, raw, rows[f"v_{suffix}"]),
            _wrap_array(_interp_angle(query, raw, rows[f"psi_{suffix}"].to_numpy()) - yaw),
            None, config,
        )
    quality = {name: _quality_summary(track, config) for name, track in tracks.items()}
    ready = all(q["position_speed_consistent"] for q in quality.values())
    geometry = None
    dims = np.array([meta[k] for k in ("ego_length", "ego_width", "target_length", "target_width")], dtype=float)
    if np.isfinite(dims).all() and (dims > 0).all():
        bv = tracks["BV"]
        center = np.array([front_bumper_to_center(x, y, h, dims[2]) for (x, y), h in zip(bv["xy_m"], bv["reported_heading_rad"])])
        center_track = _actor_track(times, center, np.asarray(bv["reported_speed_mps"]), np.asarray(bv["reported_heading_rad"]), None, config)
        center_quality = _quality_summary(center_track, config)
        geometry = _collision_audit(times, {
            "CAV": {**tracks["CAV"], "length_m": dims[0], "width_m": dims[1]},
            "BV_primary": {**center_track, "length_m": dims[2], "width_m": dims[3]},
        })
        geometry["center_heading_consistent"] = bool(center_quality["heading_usable"] and quality["CAV"]["heading_usable"])
        geometry["dimensions_rule"] = "metadata target dimensions; geometric association is not an independently verified actor identity"
    states = np.stack([
        np.column_stack([t["xy_m"], t["reported_speed_mps"], t["reported_heading_rad"]])
        for t in tracks.values()
    ], axis=1)
    mask = np.ones_like(states, dtype=bool)
    for i, q in enumerate(quality.values()):
        mask[:, i, 3] = q["heading_usable"]
    return {
        "record_type": "shrp2_measured_pair_window_v1",
        "time_s": times.tolist(), "state_channels": ["x_m", "y_m", "speed_mps", "heading_rad"],
        "position_reference": ["CAV_centroid", "BV_front_bumper"],
        "states": states.tolist(), "state_mask": mask.tolist(),
        "derived_tangential_acceleration_mps2": np.column_stack([t["derived_acceleration_mps2"] for t in tracks.values()]).tolist(),
        "condition": {"initial_state": states[0].tolist(), "initial_state_mask": mask[0].tolist(), "critical_time_s": float(times[-1])},
        "frame": {"origin": "CAV centroid at window start", "x_axis": "CAV heading at window start"},
        "quality": quality, "training_window_ready": ready, "geometry_diagnostic": geometry,
    }


def audit_dataset(source_root, output, config, scan_trajectories=False, export_windows=False, max_events_per_category=None):
    """Count independent event IDs across categories; splits depend on ID only."""
    import pandas as pd
    validate_config(config)
    if export_windows and not scan_trajectories:
        raise ValueError("export_windows requires scan_trajectories")
    if max_events_per_category is not None and max_events_per_category <= 0:
        raise ValueError("max_events_per_category must be positive")
    if scan_trajectories:
        try:
            import tables  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Trajectory scanning needs PyTables; run this command with a Python environment containing pandas and tables") from exc
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty audit output directory")
    root = locate_export_online(source_root) / "SafetyCriticalTestSet"
    folders = sorted(p.parent for p in root.glob("*/event_meta.csv"))
    if not folders:
        raise ValueError("No category metadata found")
    output.mkdir(parents=True, exist_ok=True)
    global_ids, training_ids = set(), set()
    memberships = defaultdict(set)
    inventory = []
    for folder in folders:
        metadata = pd.read_csv(folder / "event_meta.csv")
        if "event_id" not in metadata or metadata["event_id"].isna().any():
            raise ValueError(f"{folder.name}: missing event IDs")
        ids = set(int(i) for i in metadata["event_id"])
        global_ids.update(ids)
        for eid in ids:
            memberships[eid].add(folder.name)
        record = {
            "category": folder.name, "metadata_rows": len(metadata), "unique_event_count": len(ids),
            "duplicate_event_rows": len(metadata) - len(ids),
            "event_meta_sha256": _file_hash(folder / "event_meta.csv"),
            "has_trajectory_hdf5": (folder / "event_data.h5").is_file(),
            "events_by_split": dict(Counter(project_split(i, config["seed"], config["split_fractions"]) for i in ids)),
            "events_by_conflict": dict(Counter(metadata.drop_duplicates("event_id")["conflict"].fillna("unknown"))) if "conflict" in metadata else {},
            "trajectory_scan_status": "not_scanned",
        }
        print(f"category={folder.name} metadata_events={len(ids)}", flush=True)
        if scan_trajectories:
            _, _, hdf_path, metadata, data = load_category(source_root, folder.name)
            record["event_data_sha256"] = _file_hash(hdf_path)
            by_event = data.groupby("event_id", sort=False)
            selected = metadata.drop_duplicates("event_id").sort_values("event_id")
            if max_events_per_category is not None:
                selected = selected.head(max_events_per_category)
            counts = Counter()
            exclusion_reasons = Counter()
            geometry_contact_pairs = 0
            pair_count_by_split = Counter()
            with (output / f"{folder.name}_pair_audit.jsonl").open("w", encoding="utf-8") as audit_stream:
                for _, meta in selected.iterrows():
                    eid = int(meta["event_id"])
                    split = project_split(eid, config["seed"], config["split_fractions"])
                    decision = {"category": folder.name, "event_id": eid, "split": split}
                    try:
                        if eid not in by_event.groups:
                            raise ValueError("missing_event_trajectory")
                        event_rows = by_event.get_group(eid)
                        association = associate_primary_target(event_rows, meta, config["maximum_target_alignment_dt_s"])
                        if association is None:
                            raise ValueError("no_geometric_target_association")
                        tid = int(association["target_id"])
                        window = build_pair_window(event_rows[event_rows["target_id"] == tid], meta, config)
                        window["source"] = {
                            "category": folder.name, "event_id": eid, "target_id": tid, "split": split,
                            "conflict": str(meta["conflict"]), "first": str(meta.get("first", "unknown")),
                            "second": str(meta.get("second", "unknown")),
                            "label_scope": "source event category, not a verified crash label for this target",
                            "target_role": "geometric_nearest_target_at_critical_timestamp",
                            "observed_target_count": int(event_rows["target_id"].nunique()),
                        }
                        window["condition"].update({"event_category": folder.name, "conflict": str(meta["conflict"]), "actor_dimensions_m": [[float(meta[k]) for k in ("ego_length", "ego_width")], [float(meta[k]) for k in ("target_length", "target_width")]]})
                        decision.update({"target_id": tid, "status": "quality_pass" if window["training_window_ready"] else "quality_fail", "quality": window["quality"], "geometry_diagnostic": window["geometry_diagnostic"]})
                        geometry = window["geometry_diagnostic"]
                        if geometry and geometry["center_heading_consistent"] and not geometry["initial_collision"] and geometry["first_sampled_contact_time_s"] is not None and abs(geometry["first_sampled_contact_time_s"] - config["history_s"]) <= config["maximum_collision_time_error_s"] + 1e-8:
                            geometry_contact_pairs += 1
                        if window["training_window_ready"]:
                            training_ids.add(eid)
                            pair_count_by_split[split] += 1
                            if export_windows:
                                destination = output / split / "windows.jsonl"
                                destination.parent.mkdir(exist_ok=True)
                                with destination.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps(window, ensure_ascii=False, allow_nan=False) + "\n")
                    except (ValueError, KeyError) as exc:
                        decision.update({"status": "excluded", "reason": str(exc)})
                        exclusion_reasons[str(exc)] += 1
                    counts[decision["status"]] += 1
                    audit_stream.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
            record.update({"trajectory_scan_status": "sampled" if len(selected) < len(ids) else "complete", "scanned_events": len(selected), "pair_status_counts": dict(counts), "exclusion_reasons": dict(exclusion_reasons), "geometry_consistent_timed_contact_pairs": geometry_contact_pairs, "quality_pass_pairs_by_split": dict(pair_count_by_split)})
            del by_event, data
        inventory.append(record)
    summary = {
        "schema_version": 1, "status": "measured public SHRP2 data audit; no diffusion training or SUMO execution",
        "mode": "trajectories" if scan_trajectories else "metadata_only", "config": config,
        "unique_event_count": len(global_ids), "category_event_count_sum": sum(r["unique_event_count"] for r in inventory),
        "cross_category_duplicate_event_count": sum(len(c) > 1 for c in memberships.values()),
        "unique_events_by_split": dict(Counter(project_split(i, config["seed"], config["split_fractions"]) for i in global_ids)),
        "quality_pass_unique_event_count": len(training_ids) if scan_trajectories else None,
        "quality_pass_unique_events_by_split": dict(Counter(project_split(i, config["seed"], config["split_fractions"]) for i in training_ids)) if scan_trajectories else None,
        "export_windows": export_windows, "inventory": inventory,
        "limitations": [
            "Counts of metadata events are not counts of usable training trajectories.",
            "Quality-pass windows need not collide; NearCrash is retained as a distinct source category.",
            "The same event ID always shares one split across categories and targets; trip/driver grouping needs additional identifiers.",
            "Exports contain measured pairs, not complete multi-agent scenes or invented lane/map labels.",
            "Geometric nearest-target association and contact diagnostics do not verify accident participant identity.",
            "Heading channels are masked when path-heading consistency fails; fit normalization on train only.",
        ],
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/shrp2_diffusion_data_audit.json")
    parser.add_argument("--scan_trajectories", action="store_true")
    parser.add_argument("--export_windows", action="store_true")
    parser.add_argument("--max_events_per_category", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = audit_dataset(args.source_root, args.output, config, args.scan_trajectories, args.export_windows, args.max_events_per_category)
    print(json.dumps({k: result[k] for k in ("mode", "unique_event_count", "quality_pass_unique_event_count", "quality_pass_unique_events_by_split")}, indent=2))


if __name__ == "__main__":
    main()
