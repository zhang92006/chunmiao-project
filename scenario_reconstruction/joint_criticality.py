"""Lightweight, pair-aware NADE criticality for two jointly controlled BVs.

The original NADE tree search evaluates one BV against the CAV at a time.  This
module evaluates a bounded two-second action-pair surrogate and returns one
criticality array per BV.  The arrays remain factorized after marginalisation,
so the existing per-agent importance-sampling contract remains valid.
"""
from __future__ import annotations

import numpy as np

from conf import conf


def pairwise_joint_criticality_arrays(
    full_obs: dict,
    first_id: str,
    second_id: str,
    first_ndd_pdf,
    second_ndd_pdf,
    horizon_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return NDD-weighted marginal criticality arrays for one BV pair.

    Each entry integrates a joint challenge over the other BV's naturalistic
    action distribution.  A zero result means the pair cannot create a CAV
    conflict under the bounded kinematic surrogate; it is deliberately not
    turned into an artificial multi-agent label.
    """
    if "CAV" not in full_obs or first_id not in full_obs or second_id not in full_obs:
        raise KeyError("full_obs must contain CAV and both requested BVs")
    first_pdf = _normalised_pdf(first_ndd_pdf)
    second_pdf = _normalised_pdf(second_ndd_pdf)
    if len(first_pdf) != len(second_pdf):
        raise ValueError("Both BV action PDFs must have equal length")

    cav = full_obs["CAV"]
    first = full_obs[first_id]
    second = full_obs[second_id]
    first_states = [_predict_bv_state(first, action, horizon_s) for action in range(len(first_pdf))]
    second_states = [_predict_bv_state(second, action, horizon_s) for action in range(len(second_pdf))]

    challenge = np.zeros((len(first_pdf), len(second_pdf)), dtype=float)
    for first_action, first_state in enumerate(first_states):
        for second_action, second_state in enumerate(second_states):
            challenge[first_action, second_action] = _joint_challenge(
                cav, first, second, first_state, second_state, horizon_s
            )

    # This is the multi-agent analogue of NADE's ``NDD PDF * challenge``.
    # Marginalisation preserves a factorised proposal for the legacy sampler.
    first_array = first_pdf * np.dot(challenge, second_pdf)
    second_array = second_pdf * np.dot(first_pdf, challenge)
    debug = {
        "pair": [first_id, second_id],
        "horizon_s": float(horizon_s),
        "max_challenge": float(np.max(challenge)) if challenge.size else 0.0,
        "first_total": float(np.sum(first_array)),
        "second_total": float(np.sum(second_array)),
    }
    return first_array, second_array, debug


def _normalised_pdf(pdf) -> np.ndarray:
    values = np.asarray(pdf, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("NDD PDF must be a finite non-negative vector")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("NDD PDF must have positive mass")
    return values / total


def _predict_bv_state(vehicle: dict, action_id: int, horizon_s: float) -> dict:
    lane = int(vehicle["lane_index"])
    acceleration = 0.0
    if action_id == 0:
        lane += 1
    elif action_id == 1:
        lane -= 1
    else:
        acceleration = float(conf.BV_ACTIONS[action_id])
    lane = min(max(lane, 0), len(conf.lane_list) - 1)
    position = vehicle["position"]
    speed = float(vehicle["velocity"])
    return {
        "x": float(position[0]) + speed * horizon_s + 0.5 * acceleration * horizon_s**2,
        "speed": max(0.0, speed + acceleration * horizon_s),
        "lane": lane,
    }


def _joint_challenge(cav, first, second, first_state, second_state, horizon_s: float) -> float:
    cav_x = float(cav["position"][0]) + float(cav["velocity"]) * horizon_s
    cav_lane = int(cav["lane_index"])
    first_risk = _cav_conflict_risk(cav_x, cav_lane, first_state)
    second_risk = _cav_conflict_risk(cav_x, cav_lane, second_state)
    # A pair is more hazardous when both actions place vehicles near the CAV,
    # or when they simultaneously occupy the same CAV lane with short spacing.
    joint = max(first_risk, second_risk)
    if first_risk > 0.0 and second_risk > 0.0:
        joint = min(1.0, joint + 0.5 * min(first_risk, second_risk))
    if first_state["lane"] == second_state["lane"] == cav_lane:
        bv_gap = abs(first_state["x"] - second_state["x"]) - float(conf.LENGTH)
        if bv_gap <= 0.0:
            joint = min(1.0, joint + 0.25)
    return float(joint)


def _cav_conflict_risk(cav_x: float, cav_lane: int, state: dict) -> float:
    if state["lane"] != cav_lane:
        return 0.0
    gap = abs(float(state["x"]) - cav_x) - float(conf.LENGTH)
    if gap <= 0.0:
        return 1.0
    # Restrict the joint surrogate to short, physically relevant interactions.
    return float(max(0.0, min(1.0, 1.0 - gap / 20.0)))
