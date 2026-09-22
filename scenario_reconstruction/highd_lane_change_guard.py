"""Opt-in geometric envelope for the current constant-speed lane-change command.

This conservative check is a candidate execution model, not a collision guarantee
or a calibrated human-driver policy. It changes the target NDE if enabled.
"""
from __future__ import annotations

import numpy as np


def validate_config(config):
    if config.get("mode") != "constant_velocity_gap_envelope_v1":
        raise ValueError("Unknown lane-change guard mode")
    horizon = float(config["duration_s"])
    margin = float(config["minimum_gap_m"])
    if not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Guard needs positive duration and nonnegative finite gap")


def gap_envelope(gap, gap_rate, config):
    validate_config(config)
    gap, gap_rate = float(gap), float(gap_rate)
    if not np.isfinite(gap) or not np.isfinite(gap_rate):
        raise ValueError("Nonfinite neighbor geometry")
    end = gap + gap_rate * config["duration_s"]
    reason = None
    if gap <= config["minimum_gap_m"]:
        reason = "initial_gap_not_clear"
    elif end <= config["minimum_gap_m"]:
        reason = "predicted_gap_not_clear"
    return {"initial_gap_m": gap, "predicted_end_gap_m": end,
            "minimum_predicted_gap_m": min(gap, end), "reason": reason}


def blocked_sides(obs, config):
    validate_config(config)
    speed = float(obs["Ego"]["velocity"])
    details = {}
    for side in ("left", "right"):
        entries = []
        for relation in ("Lead", "Foll"):
            neighbor = obs.get(side.capitalize() + relation)
            if neighbor is None:
                continue
            rr = float(neighbor["velocity"]) - speed
            if relation == "Foll":
                rr = -rr
            envelope = gap_envelope(neighbor["distance"], rr, config)
            if envelope["reason"]:
                entries.append({"neighbor_id": neighbor.get("veh_id"), "relation": relation, **envelope})
        if entries:
            details[side] = entries
    return details
