"""Direction-aware, pair-aware NADE criticality for jointly controlled BVs.

The pair surrogate keeps NADE's factorised importance-sampling contract, but
scores action pairs by signed longitudinal clearance and a bounded forward
simulation of the tested CAV's emergency-braking response.
"""
from __future__ import annotations

import numpy as np

from conf import conf


DEFAULT_HORIZON_S = 4.0
SIMULATION_STEP_S = 0.1
EMERGENCY_BRAKE_MPS2 = -4.0
REACTION_TTC_S = 5.0
NEAR_COLLISION_CLEARANCE_M = 2.0
LANE_CHANGE_RESPONSE_S = 1.0
LANE_CHANGE_BLOCKING_CLEARANCE_M = 5.0


def pairwise_joint_criticality_arrays(
    full_obs: dict,
    first_id: str,
    second_id: str,
    first_ndd_pdf,
    second_ndd_pdf,
    horizon_s: float = DEFAULT_HORIZON_S,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return NDD-weighted action marginals for a physically directed BV pair.

    A leading BV receives challenge only when its manoeuvre closes the *net*
    CAV-to-BV gap despite CAV emergency braking. Thus a leading BV accelerating
    away is not treated as risky merely because it is geometrically nearby.
    """
    first_array, second_array, debug, _ = pairwise_joint_criticality_details(
        full_obs, first_id, second_id, first_ndd_pdf, second_ndd_pdf, horizon_s
    )
    return first_array, second_array, debug


def pairwise_joint_criticality_details(
    full_obs: dict,
    first_id: str,
    second_id: str,
    first_ndd_pdf,
    second_ndd_pdf,
    horizon_s: float = DEFAULT_HORIZON_S,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Return marginals plus the full action-pair grid for joint sampling."""
    if "CAV" not in full_obs or first_id not in full_obs or second_id not in full_obs:
        raise KeyError("full_obs must contain CAV and both requested BVs")
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    first_pdf = _normalised_pdf(first_ndd_pdf)
    second_pdf = _normalised_pdf(second_ndd_pdf)
    if len(first_pdf) != len(second_pdf):
        raise ValueError("Both BV action PDFs must have equal length")

    first_states = [_predict_bv_state(full_obs[first_id], action) for action in range(len(first_pdf))]
    second_states = [_predict_bv_state(full_obs[second_id], action) for action in range(len(second_pdf))]
    challenge, clearance, collision, escape_blocked = _joint_challenge_grid(
        full_obs["CAV"], first_states, second_states, horizon_s
    )
    first_array = first_pdf * np.dot(challenge, second_pdf)
    second_array = second_pdf * np.dot(first_pdf, challenge)
    best_pair = np.unravel_index(int(np.argmax(challenge)), challenge.shape)
    naturalistic_pdf = np.outer(first_pdf, second_pdf)
    debug = {
        "pair": [first_id, second_id],
        "horizon_s": float(horizon_s),
        "max_challenge": float(np.max(challenge)) if challenge.size else 0.0,
        "first_total": float(np.sum(first_array)),
        "second_total": float(np.sum(second_array)),
        "first_best_action": int(np.argmax(first_array)),
        "second_best_action": int(np.argmax(second_array)),
        "max_challenge_action_pair": [int(best_pair[0]), int(best_pair[1])],
        "max_challenge_minimum_clearance_m": float(clearance[best_pair]),
        "collision_action_pair_count": int(np.sum(collision)),
        "escape_blocking_action_pair_count": int(np.sum(escape_blocked)),
        "joint_critical_mass": float(np.sum(naturalistic_pdf * challenge)),
    }
    details = {
        "first_ndd_pdf": first_pdf,
        "second_ndd_pdf": second_pdf,
        "naturalistic_pdf": naturalistic_pdf,
        "challenge": challenge,
    }
    return first_array, second_array, debug, details


def joint_pair_proposal(details: dict, epsilon: float) -> dict | None:
    """Build a valid correlated IS proposal for one BV action pair.

    ``epsilon`` retains the project's existing meaning: the naturalistic
    mixture mass.  Thus ``epsilon=0.001`` samples almost entirely from the
    challenge-conditioned joint proposal while retaining full NDD support.
    """
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("joint proposal epsilon must lie strictly between zero and one")
    naturalistic = np.asarray(details["naturalistic_pdf"], dtype=float)
    challenge = np.asarray(details["challenge"], dtype=float)
    if naturalistic.shape != challenge.shape or naturalistic.ndim != 2:
        raise ValueError("joint naturalistic and challenge grids must be equal 2-D arrays")
    critical_unnormalized = naturalistic * np.clip(challenge, 0.0, None)
    critical_mass = float(np.sum(critical_unnormalized))
    if not np.isfinite(critical_mass) or critical_mass <= 0.0:
        return None
    critical = critical_unnormalized / critical_mass
    proposal = (1.0 - epsilon) * critical + epsilon * naturalistic
    proposal /= float(np.sum(proposal))
    return {
        "naturalistic_pdf": naturalistic,
        "critical_pdf": critical,
        "proposal_pdf": proposal,
        "critical_mass": critical_mass,
        "epsilon": epsilon,
    }


def sample_joint_action_pair(proposal: dict, rng=None) -> dict:
    """Sample one ordered action pair and return its auditable IS terms."""
    proposal_pdf = np.asarray(proposal["proposal_pdf"], dtype=float)
    naturalistic_pdf = np.asarray(proposal["naturalistic_pdf"], dtype=float)
    if proposal_pdf.shape != naturalistic_pdf.shape or proposal_pdf.ndim != 2:
        raise ValueError("joint proposal and naturalistic PDFs must be equal 2-D arrays")
    sampler = np.random if rng is None else rng
    flat_index = int(sampler.choice(proposal_pdf.size, p=proposal_pdf.reshape(-1)))
    first_action, second_action = np.unravel_index(flat_index, proposal_pdf.shape)
    naturalistic_probability = float(naturalistic_pdf[first_action, second_action])
    proposal_probability = float(proposal_pdf[first_action, second_action])
    return {
        "action_pair": [int(first_action), int(second_action)],
        "naturalistic_probability": naturalistic_probability,
        "proposal_probability": proposal_probability,
        "importance_weight": naturalistic_probability / proposal_probability,
        "critical_mass": float(proposal["critical_mass"]),
        "epsilon": float(proposal["epsilon"]),
    }


def _normalised_pdf(pdf) -> np.ndarray:
    values = np.asarray(pdf, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("NDD PDF must be a finite non-negative vector")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("NDD PDF must have positive mass")
    return values / total


def _predict_bv_state(vehicle: dict, action_id: int) -> dict:
    lane = int(vehicle["lane_index"])
    acceleration = 0.0
    if action_id == 0:
        lane += 1
    elif action_id == 1:
        lane -= 1
    else:
        acceleration = float(conf.BV_ACTIONS[action_id])
    return {
        "x": float(vehicle["position"][0]),
        "speed": max(0.0, float(vehicle["velocity"])),
        "lane": min(max(lane, 0), len(conf.lane_list) - 1),
        "acceleration": acceleration,
    }


def _joint_challenge_grid(
    cav: dict,
    first_states: list[dict],
    second_states: list[dict],
    horizon_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate every action pair in one vectorised forward simulation."""
    count = len(first_states)
    shape = (count, len(second_states))
    first_x, first_speed, first_lane, first_acceleration = _state_grid(first_states, 0, shape)
    second_x, second_speed, second_lane, second_acceleration = _state_grid(second_states, 1, shape)
    cav_x = np.full(shape, float(cav["position"][0]), dtype=float)
    cav_speed = np.full(shape, max(0.0, float(cav["velocity"])), dtype=float)
    cav_lane = int(cav["lane_index"])
    escape_lane = 1 - cav_lane if len(conf.lane_list) == 2 else None
    minimum_clearance = np.full(shape, np.inf, dtype=float)
    collision = np.zeros(shape, dtype=bool)
    escape_blocked_during_danger = np.zeros(shape, dtype=bool)
    steps = max(1, int(np.ceil(horizon_s / SIMULATION_STEP_S)))

    for step_index in range(steps):
        first_is_lead = (first_lane == cav_lane) & (first_x > cav_x)
        second_is_lead = (second_lane == cav_lane) & (second_x > cav_x)
        first_lead_x = np.where(first_is_lead, first_x, np.inf)
        second_lead_x = np.where(second_is_lead, second_x, np.inf)
        choose_first = first_lead_x <= second_lead_x
        lead_x = np.minimum(first_lead_x, second_lead_x)
        lead_speed = np.where(choose_first, first_speed, second_speed)
        has_lead = np.isfinite(lead_x)
        clearance_to_lead = lead_x - cav_x - float(conf.LENGTH)
        closing_speed = cav_speed - lead_speed
        time_to_collision = np.full(shape, np.inf, dtype=float)
        np.divide(
            clearance_to_lead,
            closing_speed,
            out=time_to_collision,
            where=closing_speed > 0.0,
        )
        needs_emergency_brake = has_lead & (
            (clearance_to_lead <= 0.0)
            | ((closing_speed > 0.0) & (time_to_collision <= REACTION_TTC_S))
        )
        cav_x, cav_speed = _integrate_grid(
            cav_x, cav_speed, np.where(needs_emergency_brake, EMERGENCY_BRAKE_MPS2, 0.0)
        )
        first_x, first_speed = _integrate_grid(first_x, first_speed, first_acceleration)
        second_x, second_speed = _integrate_grid(second_x, second_speed, second_acceleration)

        first_current_lane_clearance = np.where(
            first_lane == cav_lane, np.abs(first_x - cav_x) - float(conf.LENGTH), np.inf
        )
        second_current_lane_clearance = np.where(
            second_lane == cav_lane, np.abs(second_x - cav_x) - float(conf.LENGTH), np.inf
        )
        current_lane_clearance = np.minimum(
            first_current_lane_clearance, second_current_lane_clearance
        )
        if escape_lane is None:
            escape_blocked = np.ones(shape, dtype=bool)
        else:
            first_escape_clearance = np.where(
                first_lane == escape_lane,
                np.abs(first_x - cav_x) - float(conf.LENGTH),
                np.inf,
            )
            second_escape_clearance = np.where(
                second_lane == escape_lane,
                np.abs(second_x - cav_x) - float(conf.LENGTH),
                np.inf,
            )
            escape_clearance = np.minimum(first_escape_clearance, second_escape_clearance)
            escape_blocked = escape_clearance <= LANE_CHANGE_BLOCKING_CLEARANCE_M

        elapsed_s = (step_index + 1) * SIMULATION_STEP_S
        danger = current_lane_clearance <= NEAR_COLLISION_CLEARANCE_M
        escape_blocked_during_danger |= danger & escape_blocked
        can_escape = (elapsed_s > LANE_CHANGE_RESPONSE_S) & ~escape_blocked
        effective_clearance = np.where(can_escape, np.inf, current_lane_clearance)
        minimum_clearance = np.minimum(minimum_clearance, effective_clearance)
        collision |= (current_lane_clearance <= 0.0) & ~can_escape

    challenge = np.where(
        collision,
        1.0,
        np.maximum(0.0, 1.0 - minimum_clearance / NEAR_COLLISION_CLEARANCE_M),
    )
    challenge[~np.isfinite(minimum_clearance)] = 0.0
    return challenge, minimum_clearance, collision, escape_blocked_during_danger


def _state_grid(states: list[dict], axis: int, shape: tuple[int, int]):
    keys = ("x", "speed", "lane", "acceleration")
    result = []
    for key in keys:
        values = np.asarray([state[key] for state in states], dtype=float)
        values = values[:, None] if axis == 0 else values[None, :]
        result.append(np.broadcast_to(values, shape).copy())
    return result


def _integrate_grid(x: np.ndarray, speed: np.ndarray, acceleration) -> tuple[np.ndarray, np.ndarray]:
    next_speed = np.maximum(0.0, speed + np.asarray(acceleration, dtype=float) * SIMULATION_STEP_S)
    return x + (speed + next_speed) * 0.5 * SIMULATION_STEP_S, next_speed
