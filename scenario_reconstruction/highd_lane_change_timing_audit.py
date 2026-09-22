"""Audit highD lane-change timing against the local D2RL/SUMO action interface."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .highd_ndd_baseline import _load_split_manifest


def _centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float, copy=True)
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _onset_to_crossing_seconds(
    frames: np.ndarray,
    lanes: np.ndarray,
    lateral_velocity: np.ndarray,
    transition: int,
    *,
    source_hz: int,
    threshold_mps: float,
    smoothing_window_s: float,
    maximum_lookback_s: float,
    endpoint_tolerance_s: float,
) -> float | None:
    """Return a reproducible signal onset, not a driver-intention label."""
    direction = int(np.sign(lanes[transition] - lanes[transition - 1]))
    if direction == 0:
        return None
    lookback = max(1, int(round(maximum_lookback_s * source_hz)))
    start = max(0, transition - lookback)
    signed_velocity = direction * lateral_velocity[start : transition + 1]
    smoothed = _centered_mean(
        signed_velocity, max(1, int(round(smoothing_window_s * source_hz)))
    )
    active = np.isfinite(smoothed) & (smoothed >= threshold_mps)
    endpoint = len(active) - 1
    if not active[endpoint]:
        tolerance = max(0, int(round(endpoint_tolerance_s * source_hz)))
        candidates = np.flatnonzero(active[max(0, endpoint - tolerance) : endpoint + 1])
        if not len(candidates):
            return None
        endpoint = max(0, endpoint - tolerance) + int(candidates[-1])
    onset = endpoint
    while onset > 0 and active[onset - 1]:
        onset -= 1
    duration = (frames[transition] - frames[start + onset]) / source_hz
    if duration <= 0 or duration > maximum_lookback_s:
        return None
    return float(duration)


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    levels = (0.1, 0.25, 0.5, 0.75, 0.9)
    result = np.quantile(np.asarray(values, dtype=float), levels)
    return {f"p{int(level * 100)}": float(value) for level, value in zip(levels, result)}


def audit_highd_timing(
    source_root: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_root = Path(source_root)
    splits = _load_split_manifest(Path(manifest_path))
    thresholds = [float(value) for value in config["lateral_velocity_thresholds_mps"]]
    by_split: dict[str, Any] = {}
    for split, recording_ids in splits.items():
        values = {threshold: [] for threshold in thresholds}
        transition_count = 0
        for recording_id in recording_ids:
            tracks = pd.read_csv(
                source_root / "data" / f"{recording_id}_tracks.csv",
                usecols=["frame", "id", "laneId", "yVelocity"],
            )
            for _, group in tracks.groupby("id", sort=False):
                frames = group["frame"].to_numpy(dtype=np.int64)
                lanes = group["laneId"].to_numpy(dtype=np.int64)
                lateral_velocity = group["yVelocity"].to_numpy(dtype=float)
                transitions = np.flatnonzero(lanes[1:] != lanes[:-1]) + 1
                transition_count += len(transitions)
                for transition in transitions:
                    for threshold in thresholds:
                        duration = _onset_to_crossing_seconds(
                            frames,
                            lanes,
                            lateral_velocity,
                            int(transition),
                            source_hz=int(config["source_frequency_hz"]),
                            threshold_mps=threshold,
                            smoothing_window_s=float(config["smoothing_window_s"]),
                            maximum_lookback_s=float(config["maximum_lookback_s"]),
                            endpoint_tolerance_s=float(config["endpoint_tolerance_s"]),
                        )
                        if duration is not None:
                            values[threshold].append(duration)
        by_split[split] = {
            "lane_boundary_transitions": transition_count,
            "threshold_results": {
                str(threshold): {
                    "detected_onsets": len(values[threshold]),
                    "missing_onsets": transition_count - len(values[threshold]),
                    "onset_to_boundary_crossing_s": _quantiles(values[threshold]),
                    "fraction_between_0_5_and_1_5_s": (
                        float(np.mean((np.asarray(values[threshold]) >= 0.5)
                                      & (np.asarray(values[threshold]) <= 1.5)))
                        if values[threshold] else None
                    ),
                }
                for threshold in thresholds
            },
        }
    return {
        "schema_version": 1,
        "status": "offline_timing_audit_not_runtime_enabled",
        "manifest_path": str(manifest_path),
        "config": config,
        "by_split": by_split,
        "interpretation": {
            "signal_onset": (
                "First contiguous threshold crossing of smoothed signed yVelocity; "
                "not an observed driver-intention time."
            ),
            "simulation_compatible_label": (
                "Place the discrete action one measured SUMO lane-boundary lead before "
                "the highD laneId transition."
            ),
        },
    }


def probe_sumo_lane_change(
    sumo_binary: str | Path,
    sumo_config: str | Path,
    *,
    step_size_s: float,
    lane_change_duration_s: float,
) -> dict[str, Any]:
    """Measure boundary and center timing for the exact local sublane command."""
    import libsumo as traci

    binary = str(Path(sumo_binary).resolve())
    config_path = str(Path(sumo_config).resolve())
    traci.start([
        binary,
        "-c", config_path,
        "--step-length", str(step_size_s),
        "--lateral-resolution", "0.25",
        "--no-warnings", "true",
    ])
    try:
        vehicle_id = "highd_timing_probe"
        traci.vehicle.add(
            vehicle_id,
            "route_0",
            typeID="IDM",
            departLane="0",
            departPos="100",
            departSpeed="30",
        )
        traci.simulationStep()
        command_time = float(traci.simulation.getTime())
        initial_lane = traci.vehicle.getLaneID(vehicle_id)
        traci.vehicle.setLaneChangeMode(vehicle_id, 0)
        lane_width = float(traci.lane.getWidth(initial_lane))
        lateral_acceleration = float(
            traci.vehicle.getParameter(vehicle_id, "laneChangeModel.lcAccelLat")
        )
        b = lateral_acceleration * lane_change_duration_s
        c = lateral_acceleration * lane_width
        discriminant = b * b - 4 * c
        if discriminant < 0:
            raise ValueError("Configured lane-change duration is dynamically infeasible")
        maximum_lateral_speed = (-math.sqrt(discriminant) + b) / 2
        traci.vehicle.setMaxSpeedLat(vehicle_id, maximum_lateral_speed)
        traci.vehicle.changeSublane(vehicle_id, lane_width)
        boundary_time = None
        center_time = None
        samples = []
        max_steps = int(math.ceil(2 * lane_change_duration_s / step_size_s)) + 3
        for _ in range(max_steps):
            traci.simulationStep()
            current_time = float(traci.simulation.getTime())
            lane = traci.vehicle.getLaneID(vehicle_id)
            lateral_position = float(traci.vehicle.getLateralLanePosition(vehicle_id))
            samples.append({
                "time_s": current_time - command_time,
                "lane_id": lane,
                "lateral_lane_position_m": lateral_position,
            })
            if boundary_time is None and lane != initial_lane:
                boundary_time = current_time - command_time
            if lane != initial_lane and abs(lateral_position) <= 0.01:
                center_time = current_time - command_time
                break
        return {
            "step_size_s": step_size_s,
            "lane_change_duration_s": lane_change_duration_s,
            "lane_width_m": lane_width,
            "maximum_lateral_speed_mps": maximum_lateral_speed,
            "boundary_crossing_after_command_s": boundary_time,
            "target_lane_center_after_command_s": center_time,
            "samples": samples,
        }
    finally:
        traci.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sumo_binary")
    parser.add_argument("--sumo_config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = audit_highd_timing(args.source_root, args.manifest, config)
    if args.sumo_binary or args.sumo_config:
        if not (args.sumo_binary and args.sumo_config):
            raise ValueError("--sumo_binary and --sumo_config must be supplied together")
        binary = shutil.which(args.sumo_binary) or args.sumo_binary
        result["sumo_probe"] = probe_sumo_lane_change(
            binary,
            args.sumo_config,
            step_size_s=float(config["runtime_step_size_s"]),
            lane_change_duration_s=float(config["runtime_lane_change_duration_s"]),
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "timing_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
