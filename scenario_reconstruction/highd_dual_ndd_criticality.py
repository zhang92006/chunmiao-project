"""CAV-directed behavior-policy proposal, not a learned/calibrated crash model.

The frozen natural partition/P is untouched. Score its already selected pair
using a short deterministic IDM/kinematic surrogate, then use H proportional
to P times challenge. This is a collection baseline for the new 93-action
contract, not the original paper's 33-action tree search or final D2RL policy.
"""
from __future__ import annotations

import numpy as np

from .highd_dual_ndd_d2rl import OBSERVATION_CONTRACT, observation
from .highd_dual_ndd_proposal import ConditionalChainProposal


DEFAULT_CONFIG = {
    "horizon_s": 4., "step_s": .1, "nominal_lane_change_s": 4.,
    "near_clearance_m": 2., "minimum_critical_mass": 1e-6,
    "minimum_action_contrast": 1e-6,
    "cav_accel": 2., "cav_decel": 4., "cav_emergency_decel": 9.,
    "cav_tau": .2, "cav_min_gap": 2., "cav_desired_speed": 40.,
}


def validate_config(config):
    result = {**DEFAULT_CONFIG, **(config or {})}
    if set(result) != set(DEFAULT_CONFIG):
        raise ValueError("Unknown criticality configuration field")
    if any(not np.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("Criticality parameters must be finite and positive")
    if result["step_s"] > result["horizon_s"]:
        raise ValueError("Prediction step exceeds horizon")
    return result


def challenge_grid(actors, cav_id, actor_ids, lane_centers, config=None, locks=None, now=0.):
    """Return 93x93 CAV challenge and collision flags before action sampling.

    BV accelerations are held over the horizon as a continuation hypothesis,
    NOT as an instruction to SUMO (which still resamples every 0.1 seconds).
    New starts use a nominal continuous lateral path; the actual shared random
    motion kernel is not replaced. Existing commands use their current waypoint.
    Unselected vehicles participate in the CAV's response, not in the search.
    """
    cfg = validate_config(config)
    if len(set(actor_ids)) != 2 or cav_id in actor_ids:
        raise ValueError("Expected two distinct BVs, separate from the CAV")
    by_id = {a["id"]: a for a in actors}
    if not set((cav_id, *actor_ids)).issubset(by_id):
        raise ValueError("Missing CAV or BV")
    shape = (93, 93)
    code = np.arange(93)
    locks = locks or {}
    xs, speeds, accelerations, paths = {}, {}, {}, {}
    for actor in actors:
        key = actor["id"]
        xs[key] = np.full(shape, actor["center_x"], dtype=float)
        speeds[key] = np.full(shape, actor["speed"], dtype=float)
        accel = np.full(shape, actor["acceleration"], dtype=float)
        target = np.full(shape, actor["center_y"], dtype=float)
        duration = cfg["nominal_lane_change_s"]
        if key in actor_ids:
            values = code[:, None] if key == actor_ids[0] else code[None, :]
            accel = np.broadcast_to(-4 + .2 * (values // 3), shape).copy()
            lane = actor["lane"] + np.where(values % 3 == 1, 1, np.where(values % 3 == 2, -1, 0))
            # Illegal starts have zero mass in P. Never turn them into legal Q.
            target = np.where(values % 3 == 0, actor["center_y"],
                              np.asarray(lane_centers)[np.clip(lane, 0, 1)])
        if key in locks:
            event = locks[key]
            target = event.get("waypoints_y", [event.get("outbound_y", actor["center_y"])])[event.get("waypoint_index", 0)]
            duration = max(.1, event["diagnostic_duration_s"] - (now - event["start_time_s"]))
        accelerations[key] = accel
        paths[key] = (actor["center_y"], target, duration)
    cav = by_id[cav_id]
    challenge = np.zeros(shape)
    collision = np.zeros(shape, dtype=bool)
    min_clearance = np.full(shape, np.inf)
    elapsed = 0.
    while elapsed < cfg["horizon_s"] - 1e-10:
        dt = min(cfg["step_s"], cfg["horizon_s"] - elapsed)
        ys = {key: start + (target - start) * min(1., elapsed / duration)
              for key, (start, target, duration) in paths.items()}
        gap, lead_speed = np.full(shape, np.inf), np.zeros(shape)
        for key, other in by_id.items():
            if key == cav_id:
                continue
            overlaps = np.abs(ys[key] - cav["center_y"]) < (other["width"] + cav["width"]) / 2
            dx = xs[key] - xs[cav_id]
            net = dx - (other["length"] + cav["length"]) / 2
            choose = overlaps & (dx > 0) & (net < gap)
            gap = np.where(choose, net, gap)
            lead_speed = np.where(choose, speeds[key], lead_speed)
        v = speeds[cav_id]
        desired_gap = cfg["cav_min_gap"] + np.maximum(0., v * cfg["cav_tau"] +
                      v * (v - lead_speed) / (2 * np.sqrt(cfg["cav_accel"] * cfg["cav_decel"])))
        a_cav = cfg["cav_accel"] * (1 - (v / cfg["cav_desired_speed"]) ** 4 -
                                     (desired_gap / np.maximum(gap, .01)) ** 2)
        accelerations[cav_id] = np.clip(a_cav, -cfg["cav_emergency_decel"], cfg["cav_accel"])
        for key in by_id:
            new_v = np.maximum(0., speeds[key] + accelerations[key] * dt)
            xs[key] += (speeds[key] + new_v) * .5 * dt
            speeds[key] = new_v
        elapsed += dt
        for key, other in by_id.items():
            if key == cav_id:
                continue
            start, target, duration = paths[key]
            y = start + (target - start) * min(1., elapsed / duration)
            overlaps = np.abs(y - cav["center_y"]) < (other["width"] + cav["width"]) / 2
            clearance = np.abs(xs[key] - xs[cav_id]) - (other["length"] + cav["length"]) / 2
            min_clearance = np.minimum(min_clearance, np.where(overlaps, clearance, np.inf))
            collision |= overlaps & (clearance <= 0)
            challenge = np.maximum(challenge, np.where(overlaps,
                np.clip(1 - clearance / cfg["near_clearance_m"], 0, 1), 0))
    return challenge, collision, min_clearance


class CAVCriticalityProvider:
    def __init__(self, epsilon=(.1, .1), config=None):
        self.epsilon = epsilon
        self.config = validate_config(config)
        self.locks = {}
        self.records = []

    def __call__(self, scene, candidate, distribution):
        if distribution.natural.shape != (93, 93) or not distribution.structured:
            raise ValueError("CAV scorer requires the structured 93-action encoding")
        actors = [{"id": int(r.id), "center_x": -r.x - r.width / 2,
                   "center_y": r.y + r.height / 2, "length": r.width, "width": r.height,
                   "lane": int(r.laneId) - 2, "speed": r.speed_mps,
                   "acceleration": r.acceleration_mps2} for _, r in scene.latest.iterrows()]
        score, collision, clearance = challenge_grid(actors, scene.cav_id, distribution.actor_ids,
            [scene.centers[2], scene.centers[3]], self.config, self.locks, scene.last_decision / 10)
        p = distribution.natural
        supported = p > 0
        mass = float(np.sum(p * score))
        contrast = float(np.ptp(score[supported]))
        active = mass >= self.config["minimum_critical_mass"] and contrast >= self.config["minimum_action_contrast"]
        diagnostic = {"decision_tick": scene.last_decision, "actor_ids": list(distribution.actor_ids),
                      "critical_mass": mass, "supported_action_contrast": contrast,
                      "supported_predicted_collision_action_pairs": int(np.sum(collision & supported)),
                      "minimum_predicted_cav_clearance_m": float(clearance[supported].min()) if np.isfinite(clearance[supported].min()) else None,
                      "status": "critical_proposal" if active else "noncritical_natural"}
        self.records.append(diagnostic)
        if not active:
            return None  # Actual Q=P, without unused/synthetic training actions.
        proposal = ConditionalChainProposal(p, p * score / mass, self.epsilon)
        proposal.training_observation = {"contract": OBSERVATION_CONTRACT,
                                         "joint": observation(scene, distribution.actor_ids)}
        proposal.criticality = mass
        return proposal
