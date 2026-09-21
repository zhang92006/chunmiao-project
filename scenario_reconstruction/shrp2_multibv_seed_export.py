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


def _anchor_state(
    rows,
    anchor_s,
    origin,
    rotation,
    reference_yaw,
    maximum_alignment_dt_s=None,
):
    """Return one finite context state at the requested initialization timestamp."""
    finite = rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time", "x_sur", "y_sur", "v_sur", "psi_sur"]
    )
    if finite.empty:
        raise ValueError("missing_context_anchor")
    index = (finite["time"] - anchor_s).abs().idxmin()
    row = finite.loc[index]
    if (
        maximum_alignment_dt_s is not None
        and abs(float(row["time"]) - float(anchor_s))
        > float(maximum_alignment_dt_s) + 1e-8
    ):
        raise ValueError("context_initialization_alignment_exceeds_limit")
    if float(row["v_sur"]) < 0:
        raise ValueError("negative_context_speed")
    xy = np.array([float(row["x_sur"]), float(row["y_sur"])])
    heading = float(row["psi_sur"])
    return [
        *((xy - origin) @ rotation.T),
        float(row["v_sur"]),
        float(_wrap_array(np.asarray([heading - reference_yaw]))[0]),
    ]


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


def _context_anchor_time_delta(rows, anchor_s):
    """Return the nearest usable context-observation offset from ``anchor_s``."""
    finite = rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time", "x_sur", "y_sur", "v_sur", "psi_sur"]
    )
    if finite.empty:
        raise ValueError("missing_context_anchor")
    index = (finite["time"] - float(anchor_s)).abs().idxmin()
    return abs(float(finite.loc[index, "time"]) - float(anchor_s))


def _time_aligned_context_target_ids(
    grouped, primary_id, anchor_s, maximum_alignment_dt_s
):
    """List only context tracks that can support a measured anchor state."""
    aligned = []
    for target_id, rows in grouped:
        target_id = int(target_id)
        if target_id == int(primary_id):
            continue
        try:
            if _context_anchor_time_delta(rows, anchor_s) <= float(
                maximum_alignment_dt_s
            ) + 1e-8:
                aligned.append(target_id)
        except ValueError:
            continue
    return aligned


def _window_frame_index(window, requested_time_s):
    """Return an existing audited window frame at the requested relative time."""
    times = np.asarray(window["time_s"], dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("window lacks time_s")
    index = int(np.argmin(np.abs(times - float(requested_time_s))))
    if len(times) == 1:
        tolerance = 1e-8
    else:
        tolerance = float(np.min(np.diff(times))) / 2.0 + 1e-8
    if abs(float(times[index]) - float(requested_time_s)) > tolerance:
        raise ValueError("initialization_time_not_on_window_grid")
    return index


def validate_adaptive_selection_config(selection_config, history_s):
    """Validate the small, explicit policy used to choose an observed seed time."""
    if selection_config.get("schema_version") != 1:
        raise ValueError("Only adaptive critical-window schema_version 1 is supported")
    if selection_config.get(
        "context_selection_mode", "all_time_aligned_contexts"
    ) != "all_time_aligned_contexts":
        raise ValueError("Only context_selection_mode='all_time_aligned_contexts' is supported")
    offsets = selection_config.get("candidate_offsets_before_critical_s")
    if not isinstance(offsets, list) or not offsets:
        raise ValueError("candidate_offsets_before_critical_s must be a non-empty list")
    parsed_offsets = [float(value) for value in offsets]
    if (not np.isfinite(parsed_offsets).all() or any(
        value < 0.0 or value > float(history_s) for value in parsed_offsets
    )):
        raise ValueError("candidate offsets must lie in [0, history_s]")
    if len(set(parsed_offsets)) != len(parsed_offsets):
        raise ValueError("candidate offsets must be unique")
    for key in (
        "same_lane_lateral_threshold_m",
        "minimum_closing_speed_mps",
        "minimum_ttc_s",
        "maximum_ttc_s",
        "target_ttc_s",
        "escape_blocking_longitudinal_distance_m",
    ):
        value = selection_config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be positive")
    if selection_config["minimum_ttc_s"] >= selection_config["maximum_ttc_s"]:
        raise ValueError("minimum_ttc_s must be smaller than maximum_ttc_s")
    if not (
        selection_config["minimum_ttc_s"]
        <= selection_config["target_ttc_s"]
        <= selection_config["maximum_ttc_s"]
    ):
        raise ValueError("target_ttc_s must lie within the TTC interval")


def _adaptive_candidate_metrics(seed, selection_config):
    """Measure a source-state candidate without claiming a counterfactual crash.

    The primary BV must be in front of and closing on the CAV in its mapped
    source lane.  The adjacent BV is deliberately a ranking feature rather
    than a hard requirement: demanding a perfectly blocking vehicle at every
    timestamp would discard most independent SHRP2 events before SUMO can test
    their actual interaction.
    """
    cav, primary, context = np.asarray(seed["condition"]["initial_state"], dtype=float)[:3]
    lane_threshold = float(selection_config["same_lane_lateral_threshold_m"])
    gap_m = float(primary[0] - cav[0])
    closing_speed_mps = float(cav[2] - primary[2])
    primary_lateral_offset_m = float(primary[1] - cav[1])
    context_lateral_offset_m = float(context[1] - cav[1])
    same_lane = abs(primary_lateral_offset_m) <= lane_threshold
    context_adjacent_lane = abs(context_lateral_offset_m) > lane_threshold
    context_longitudinal_distance_m = abs(float(context[0] - cav[0]))
    context_blocking_potential = bool(
        context_adjacent_lane
        and context_longitudinal_distance_m
        <= float(selection_config["escape_blocking_longitudinal_distance_m"])
    )
    ttc_s = math.inf
    if gap_m > 0.0 and closing_speed_mps > 0.0:
        ttc_s = gap_m / closing_speed_mps

    eligible = bool(
        same_lane
        and gap_m > 0.0
        and closing_speed_mps >= float(selection_config["minimum_closing_speed_mps"])
        and float(selection_config["minimum_ttc_s"]) <= ttc_s <= float(selection_config["maximum_ttc_s"])
    )
    if not same_lane:
        reason = "primary_not_same_lane"
    elif gap_m <= 0.0:
        reason = "primary_not_ahead"
    elif closing_speed_mps < float(selection_config["minimum_closing_speed_mps"]):
        reason = "primary_not_closing"
    elif ttc_s < float(selection_config["minimum_ttc_s"]):
        reason = "primary_ttc_too_short"
    elif ttc_s > float(selection_config["maximum_ttc_s"]):
        reason = "primary_ttc_too_long"
    else:
        reason = "eligible"

    # Lower is better.  A potentially blocking adjacent vehicle improves the
    # rank but never fabricates one or rejects a valid primary conflict.
    score = abs(ttc_s - float(selection_config["target_ttc_s"])) if eligible else math.inf
    if eligible and not context_blocking_potential:
        score += float(selection_config.get("nonblocking_context_penalty", 0.25))
    return {
        "eligible": eligible,
        "reason": reason,
        "score": float(score),
        "primary_gap_m": gap_m,
        "primary_closing_speed_mps": closing_speed_mps,
        "primary_ttc_s": None if not math.isfinite(ttc_s) else float(ttc_s),
        "primary_lateral_offset_m": primary_lateral_offset_m,
        "context_lateral_offset_m": context_lateral_offset_m,
        "context_longitudinal_distance_m": context_longitudinal_distance_m,
        "context_adjacent_lane": context_adjacent_lane,
        "context_blocking_potential": context_blocking_potential,
    }


def select_adaptive_multibv_seed(
    window, event_rows, metadata_row, config, selection_config, bv_count=2
):
    """Choose one observed pre-critical initialization per independent event."""
    validate_adaptive_selection_config(selection_config, config["history_s"])
    candidates, reasons = [], Counter()
    primary_id = int(window["source"]["target_id"])
    grouped = event_rows.groupby("target_id", sort=True)
    anchor_s = float(metadata_row["impact_timestamp"]) / 1000.0
    # Prefer an earlier start only when physical scores are otherwise identical.
    for offset_s in sorted(
        (float(value) for value in selection_config["candidate_offsets_before_critical_s"]),
        reverse=True,
    ):
        initialization_s = anchor_s - offset_s
        aligned_context_ids = _time_aligned_context_target_ids(
            grouped, primary_id, initialization_s,
            config["maximum_target_alignment_dt_s"],
        )
        if not aligned_context_ids:
            reasons["context_initialization_alignment_exceeds_limit"] += 1
            continue
        # Search every time-aligned vehicle, not merely the nearest vehicle
        # whose observation might turn out to be stale at this timestamp.
        for context_id in aligned_context_ids:
            try:
                seed = build_multibv_seed(
                    window, event_rows, metadata_row, config, bv_count=bv_count,
                    context_mode="anchor_only", initialization_offset_s=offset_s,
                    context_target_ids=[context_id],
                )
            except ValueError as exc:
                reasons[str(exc)] += 1
                continue
            metrics = _adaptive_candidate_metrics(seed, selection_config)
            if not metrics["eligible"]:
                reasons[metrics["reason"]] += 1
                continue
            candidates.append((
                0 if metrics["context_blocking_potential"] else 1,
                metrics["score"],
                -offset_s,
                context_id,
                len(aligned_context_ids),
                seed,
                metrics,
            ))
    if not candidates:
        detail = ",".join(sorted(reasons)) or "no_candidate"
        raise ValueError(f"adaptive_no_eligible_candidate:{detail}")
    _, _, _, selected_context_id, aligned_context_count, selected, metrics = min(
        candidates, key=lambda item: item[:4]
    )
    selected["condition"]["adaptive_critical_window"] = {
        "selection_schema_version": 1,
        "candidate_offsets_before_critical_s": selection_config[
            "candidate_offsets_before_critical_s"
        ],
        "selection_score": metrics.pop("score"),
        "context_selection_mode": "all_time_aligned_contexts",
        "selected_context_target_id": int(selected_context_id),
        "time_aligned_context_target_count": int(aligned_context_count),
        **metrics,
    }
    return selected


def build_multibv_seed(
    window,
    event_rows,
    metadata_row,
    config,
    bv_count=2,
    context_mode="full",
    initialization_offset_s=0.0,
    context_target_ids=None,
):
    """Add nearest measured context BVs to one quality-passed pair window."""
    if bv_count < 2:
        raise ValueError("bv_count must be at least 2")
    if context_mode not in {"full", "anchor_only"}:
        raise ValueError("context_mode must be 'full' or 'anchor_only'")
    initialization_offset_s = float(initialization_offset_s)
    history_s = float(config["history_s"])
    if not np.isfinite(initialization_offset_s) or not 0.0 <= initialization_offset_s <= history_s:
        raise ValueError("initialization_offset_s must lie in [0, history_s]")
    if context_mode == "full" and initialization_offset_s != 0.0:
        raise ValueError(
            "initialization_offset_s requires context_mode='anchor_only'; "
            "full-history bridge selection is not changed implicitly"
        )
    primary_id = int(window["source"]["target_id"])
    anchor_s = float(metadata_row["impact_timestamp"]) / 1000.0
    start_s = anchor_s - history_s
    initialization_s = anchor_s - initialization_offset_s
    initialization_time_s = history_s - initialization_offset_s
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

    context_reference_s = initialization_s if context_mode == "anchor_only" else anchor_s
    grouped = event_rows.groupby("target_id", sort=True)
    selected = [(primary_id, "BV_primary")]
    context_needed = bv_count - 1
    if context_target_ids is None:
        distances, unaligned_count = [], 0
        for target_id, rows in grouped:
            target_id = int(target_id)
            if target_id == primary_id:
                continue
            try:
                if context_mode == "anchor_only" and _context_anchor_time_delta(
                    rows, context_reference_s
                ) > float(config["maximum_target_alignment_dt_s"]) + 1e-8:
                    unaligned_count += 1
                    continue
                distances.append((
                    _target_distance_at_anchor(rows, context_reference_s), target_id
                ))
            except ValueError:
                continue
        distances.sort(key=lambda item: (item[0], item[1]))
        selected_ids = [target_id for _, target_id in distances[:context_needed]]
    else:
        selected_ids = [int(target_id) for target_id in context_target_ids]
        if len(selected_ids) != context_needed or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("context_target_ids must contain one unique target per context BV")
        if primary_id in selected_ids or any(target_id not in grouped.groups for target_id in selected_ids):
            raise ValueError("context_target_ids contains an unavailable target")
        unaligned_count = 0
    for target_id in selected_ids:
        selected.append((target_id, f"BV_context_{len(selected)}"))
    if len(selected) != bv_count:
        if context_mode == "anchor_only" and unaligned_count:
            raise ValueError("context_initialization_alignment_exceeds_limit")
        raise ValueError("insufficient_context_targets")

    if context_mode == "full":
        times = window["time_s"]
        cav_states = np.asarray(window["states"], dtype=float)[:, 0, :]
        cav_masks = np.asarray(window["state_mask"], dtype=bool)[:, 0, :]
        tracks = [np.asarray(window["states"], dtype=float)[:, 1, :]]
        masks = [np.asarray(window["state_mask"], dtype=bool)[:, 1, :]]
    else:
        frame_index = _window_frame_index(window, initialization_time_s)
        times = [float(window["time_s"][frame_index])]
        cav_states = np.asarray(window["states"], dtype=float)[frame_index, 0, :][None, :]
        cav_masks = np.asarray(window["state_mask"], dtype=bool)[frame_index, 0, :][None, :]
        tracks = [np.asarray(window["states"], dtype=float)[frame_index, 1, :][None, :]]
        masks = [np.asarray(window["state_mask"], dtype=bool)[frame_index, 1, :][None, :]]
    actor_records = [{
        "id": "BV_primary",
        "role": "primary_risk_bv",
        "source_target_id": primary_id,
        "state_channels": window["state_channels"],
    }]
    context_quality = {}
    for target_id, actor_id in selected[1:]:
        if context_mode == "full":
            track = _finite_track(
                grouped.get_group(target_id), config, anchor_s, origin, rotation, yaw
            )
            quality = _quality_summary(track, config)
            if not quality["position_speed_consistent"]:
                raise ValueError("context_quality_fail")
            context_state = np.column_stack([
                track["xy_m"], track["reported_speed_mps"], track["reported_heading_rad"]
            ])
            context_mask = np.ones_like(context_state, dtype=bool)
            context_mask[:, 3] = quality["heading_usable"]
        else:
            context_state = np.asarray([_anchor_state(
                grouped.get_group(target_id),
                initialization_s,
                origin,
                rotation,
                yaw,
                config["maximum_target_alignment_dt_s"],
            )], dtype=float)
            context_mask = np.ones_like(context_state, dtype=bool)
            quality = {"mode": "anchor_only", "position_speed_consistent": None,
                       "heading_usable": True}
        tracks.append(context_state)
        masks.append(context_mask)
        context_quality[actor_id] = quality
        actor_records.append({
            "id": actor_id,
            "role": "context_bv",
            "source_target_id": target_id,
            "state_channels": window["state_channels"],
        })

    states = np.stack([
        cav_states, *tracks
    ], axis=1)
    state_mask = np.stack([
        cav_masks, *masks
    ], axis=1)
    actor_records.insert(0, {
        "id": "CAV",
        "role": "CAV",
        "state_channels": window["state_channels"],
    })
    source = dict(window["source"])
    source.update({
        "target_role": (
            "primary geometric target plus explicitly selected measured contexts"
            if context_target_ids is not None
            else "primary geometric target plus nearest time-aligned measured contexts"
        ),
        "context_target_ids": [target_id for target_id, _ in selected[1:]],
        "observed_target_count": int(event_rows["target_id"].nunique()),
    })
    return {
        "record_type": "shrp2_measured_multibv_seed_v1",
        "time_s": times,
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
            "initialization_time_s": float(times[0]),
            "initialization_offset_before_critical_s": initialization_offset_s,
            "mapping_status": "source_frame_only; lane and route mapping pending",
            "context_mode": context_mode,
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


def export_multibv_seeds(
    source_root,
    audit_root,
    output,
    config,
    bv_count=2,
    context_mode="full",
    initialization_offset_s=0.0,
    adaptive_selection_config=None,
):
    """Export context-augmented seeds, loading one SHRP2 category at a time."""
    validate_config(config)
    if adaptive_selection_config is not None:
        if context_mode != "anchor_only":
            raise ValueError("adaptive selection requires context_mode='anchor_only'")
        if float(initialization_offset_s) != 0.0:
            raise ValueError(
                "adaptive selection chooses its own offset; omit initialization_offset_s"
            )
        validate_adaptive_selection_config(adaptive_selection_config, config["history_s"])
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
    selected_offsets = Counter()
    selected_context_modes = Counter()
    selected_blocking_contexts = Counter()
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
                if adaptive_selection_config is None:
                    seed = build_multibv_seed(
                        window, rows, meta, config, bv_count=bv_count,
                        context_mode=context_mode,
                        initialization_offset_s=initialization_offset_s,
                    )
                else:
                    seed = select_adaptive_multibv_seed(
                        window, rows, meta, config, adaptive_selection_config,
                        bv_count=bv_count,
                    )
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
                    "initialization_offset_before_critical_s": seed["condition"][
                        "initialization_offset_before_critical_s"
                    ],
                })
                counts[split] += 1
                if adaptive_selection_config is not None:
                    selected_offsets[
                        str(seed["condition"]["initialization_offset_before_critical_s"])
                    ] += 1
                    adaptive = seed["condition"].get("adaptive_critical_window", {})
                    selected_context_modes[str(adaptive.get("context_selection_mode"))] += 1
                    selected_blocking_contexts[str(bool(
                        adaptive.get("context_blocking_potential", False)
                    )).lower()] += 1
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
        "context_mode": context_mode,
        "initialization_offset_s": float(initialization_offset_s),
        "adaptive_selection": adaptive_selection_config,
        "adaptive_selected_by_offset_s": dict(selected_offsets),
        "adaptive_selected_context_modes": dict(selected_context_modes),
        "adaptive_selected_context_blocking_potential": dict(selected_blocking_contexts),
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
    parser.add_argument(
        "--context_mode",
        choices=("full", "anchor_only"),
        default="full",
        help="Use full 4-second context history or one aligned anchor state per actor.",
    )
    parser.add_argument(
        "--adaptive_selection_config",
        help=(
            "JSON policy for selecting one observed 1--3 s pre-critical "
            "anchor per event. Requires --context_mode anchor_only."
        ),
    )
    parser.add_argument(
        "--initialization_offset_s",
        type=float,
        default=0.0,
        help=(
            "For anchor_only seeds, initialize this many seconds before the "
            "critical timestamp; must lie in [0, history_s]."
        ),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    adaptive_selection_config = None
    if args.adaptive_selection_config:
        adaptive_selection_config = json.loads(
            Path(args.adaptive_selection_config).read_text(encoding="utf-8")
        )
    result = export_multibv_seeds(
        args.source_root, args.audit_root, args.output, config,
        bv_count=args.bv_count, context_mode=args.context_mode,
        initialization_offset_s=args.initialization_offset_s,
        adaptive_selection_config=adaptive_selection_config,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "window_count",
            "exported_count",
            "exported_by_split",
            "excluded_count",
            "initialization_offset_s",
            "adaptive_selection",
            "adaptive_selected_by_offset_s",
            "adaptive_selected_context_modes",
            "adaptive_selected_context_blocking_potential",
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
