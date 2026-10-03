"""Opt-in bridge from dual NDD draws to the existing centralized D2RL replay.

Policy outputs remain two epsilon values. New structured action IDs and new
observation semantics must NOT be loaded with historical SHRP2 checkpoints.
"""
from __future__ import annotations

import numpy as np

from .highd_dual_ndd_proposal import ConditionalChainProposal, tilted_joint


OBSERVATION_CONTRACT = "highd_dual_kinematic14_v1"


def observation(scene, ids):
    """6 CAV + 4 per ordered BV, causal; only NN inputs are normalized/clipped."""
    cav = scene._own(scene.cav_id).iloc[0]
    has_front = cav.precedingId > 0 and cav.dhw <= 115
    result = [cav.speed_mps / 40, cav.acceleration_mps2 / 4, cav.laneId - 2,
              float(has_front), cav.dhw / 115 if has_front else 1.,
              (abs(cav.precedingXVelocity) - cav.speed_mps) / 10 if has_front else 0.]
    for actor in ids:
        own = scene._own(actor).iloc[0]
        dx = -(own.x + own.width / 2 - cav.x - cav.width / 2)
        result.extend([dx / 115, own.speed_mps / 40, own.laneId - cav.laneId, own.acceleration_mps2 / 4])
    values = np.asarray(result, dtype=float)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("Invalid causal policy observation")
    return np.clip(values, -5, 5).tolist()


def make_provider(score_function, epsilon=(.5, .5), policy=None):
    """Accept a pre-sampling joint log-score and optional fresh policy callable.

    The scorer returns (93x93 log-score, scalar criticality). This module does
    not substitute a learned collision critic or silently reuse 33-action IDs.
    """
    def provider(scene, candidate, distribution):
        obs = observation(scene, distribution.actor_ids)
        eps = epsilon if policy is None else policy(obs)
        score, criticality = score_function(scene, candidate, distribution)
        if not np.isfinite(criticality) or criticality < 0:
            raise ValueError("Criticality must be finite and non-negative")
        result = ConditionalChainProposal(distribution.natural, tilted_joint(distribution.natural, score), eps)
        result.training_observation = {"contract": OBSERVATION_CONTRACT, "joint": obs}
        result.criticality = float(criticality)
        return result
    return provider


def acceleration_tilt_probe(scene, candidate, distribution):
    """Interface positive control ONLY, not a calibrated collision criticality.

    Change both sampled acceleration laws to exercise two trainable epsilons.
    This artificial bounded score is never presented as natural driving data.
    """
    a = -4 + .2 * (np.arange(93) // 3)
    return .5 * (a[:, None] - a[None, :]), 1.


def episode_record(episode):
    """Convert a complete unforced rollout; keep actual CAV outcome, not labels.

    Step records are for the existing single-critical-step replay objective.
    Full raw scene logs retain all discarded steps and the full episode weight;
    a densified training reward is NOT the episode-level risk estimator.
    """
    protocol = episode["protocol"]
    if protocol["forced_start"]:
        raise ValueError("Forced lateral controls are not a natural/proposal training rollout")
    if episode["termination"] not in ("duration_reached", "collision"):
        raise ValueError("Incomplete/support-exit episodes cannot become safe training labels")
    cav_id = str(protocol["config"]["cav_id"])
    collision_ids = [] if not isinstance(episode["failure"], dict) else episode["failure"].get("colliding_actor_ids", [])
    if episode["termination"] == "collision" and cav_id not in map(str, collision_ids):
        raise ValueError("A BV-only collision truncates CAV exposure; do not label it safe")
    result = {"schema_version": 1, "collision_result": cav_id in map(str, collision_ids),
              "scenario_metadata": {"natural_target_sha256": protocol.get("natural_target_sha256", protocol["bundle_sha256"]),
                                    "observation_contract": OBSERVATION_CONTRACT,
                                    "source_event_id": protocol["config"].get("scenario_id", "configured_simulator_initial_state"),
                                    "source_kind": "configured_simulator_initial_state_not_recorded_highD_event",
                                    "scope": "opt-in interface candidate, not empirical NDD acceptance"},
              "full_episode_log_importance_weight": sum(d["log_importance_ratio"] for d in episode["decisions"])}
    fields = ("weight_step_info", "ndd_step_info", "drl_obs_step_info", "drl_epsilon_step_info",
              "real_epsilon_step_info", "criticality_step_info", "controlled_bv_ids_step_info")
    result.update({key: {} for key in fields})
    for record in episode["decisions"]:
        units = [u for u in record["units"] if "proposal_components" in u]
        if len(units) > 1:
            raise ValueError("K=2 replay currently accepts only one biased pair per decision")
        if not units:
            continue
        unit = units[0]
        obs = unit.get("training_observation", {})
        if obs.get("contract") != OBSERVATION_CONTRACT:
            raise ValueError("Missing or different training observation contract")
        key, draw = str(record["decision_tick"]), unit["draw"]
        values = [draw["weight_record"], draw["ndd_record"], obs,
                  draw["weight_record"]["generation_epsilon"], draw["weight_record"]["generation_epsilon"],
                  unit["criticality"], list(unit["actor_ids"])]
        for field, value in zip(fields, values):
            result[field][key] = value
    if not result["weight_step_info"]:
        raise ValueError("No conditional-proposal decision to export")
    return result
