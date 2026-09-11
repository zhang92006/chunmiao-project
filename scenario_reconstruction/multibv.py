from __future__ import annotations

from dataclasses import dataclass
from math import log10
from typing import Iterable

import numpy as np

from conf import conf


@dataclass(frozen=True)
class MultiBVSelection:
    ids: list[str]
    joint_obs: list[float]
    per_agent_obs: list[list[float]]


def select_multibv_training_actors(env, primary_actor_id: str, agent_num: int = 2) -> MultiBVSelection | None:
    """Select the primary event actor plus nearby BVs and build K-BV observations."""
    cav = env.vehicle_list.get("CAV")
    if cav is None:
        return None

    candidates = []
    for actor in env.scenario_template.actors:
        if actor.id in env.vehicle_list:
            vehicle = env.vehicle_list[actor.id]
            candidates.append((actor.id, _distance_to_cav(env, vehicle)))

    ordered_ids = []
    if primary_actor_id in env.vehicle_list:
        ordered_ids.append(primary_actor_id)
    for vehicle_id, _ in sorted(candidates, key=lambda item: item[1]):
        if vehicle_id not in ordered_ids:
            ordered_ids.append(vehicle_id)

    selected_ids = ordered_ids[:agent_num]
    if len(selected_ids) < agent_num:
        return None

    full_obs = _collect_full_obs(env, selected_ids)
    joint_obs = build_multibv_joint_obs(
        full_obs=full_obs,
        selected_bv_ids=selected_ids,
        episode_weight=env.info_extractor.episode_log.get("weight_episode", 1.0),
    )
    per_agent_obs = [
        build_multibv_joint_obs(
            full_obs=full_obs,
            selected_bv_ids=[vehicle_id],
            episode_weight=env.info_extractor.episode_log.get("weight_episode", 1.0),
        )
        for vehicle_id in selected_ids
    ]
    return MultiBVSelection(selected_ids, joint_obs, per_agent_obs)


def build_multibv_joint_obs(
    full_obs: dict,
    selected_bv_ids: Iterable[str],
    episode_weight: float,
) -> list[float]:
    selected_bv_ids = list(selected_bv_ids)
    cav = full_obs["CAV"]
    cav_position = list(cav["position"])
    cav_speed = float(cav["velocity"])
    bv_info = []
    for vehicle_id in selected_bv_ids:
        vehicle = full_obs[vehicle_id]
        vehicle_position = list(vehicle["position"])
        relative_position = [
            float(vehicle_position[0] - cav_position[0]),
            float(vehicle_position[1] - cav_position[1]),
        ]
        relative_speed = float(vehicle["velocity"] - cav_speed)
        predicted_relative_x = relative_position[0] + relative_speed
        bv_info.extend(relative_position + [relative_speed, predicted_relative_x])

    controlled_bv_num = len(selected_bv_ids)
    common_lb, common_ub = _common_obs_bounds()
    vehicle_lb = [-20, -8, -10, -20] * controlled_bv_num
    vehicle_ub = [20, 8, 10, 20] * controlled_bv_num
    criticality_flag = 1.0
    criticality_value = _safe_log10(_estimate_joint_criticality(full_obs, selected_bv_ids))
    raw_obs = np.array(
        cav_position
        + [cav_speed]
        + [_safe_log10(max(float(episode_weight), 1e-30))]
        + [criticality_flag]
        + [criticality_value]
        + bv_info,
        dtype=np.float32,
    )
    lb_array = np.array(common_lb + vehicle_lb, dtype=np.float32)
    ub_array = np.array(common_ub + vehicle_ub, dtype=np.float32)
    normalized = 2 * (raw_obs - lb_array) / (ub_array - lb_array) - 1
    return np.clip(normalized, -5, 5).astype(np.float32).tolist()


def _collect_full_obs(env, selected_bv_ids: list[str]) -> dict:
    full_obs = {"CAV": env.vehicle_list["CAV"].observation.information["Ego"]}
    for vehicle_id in selected_bv_ids:
        full_obs[vehicle_id] = env.vehicle_list[vehicle_id].observation.information["Ego"]
    return full_obs


def _distance_to_cav(env, vehicle) -> float:
    cav = env.vehicle_list.get("CAV")
    if cav is None:
        return float("inf")
    cav_position = cav.observation.information["Ego"]["position"]
    vehicle_position = vehicle.observation.information["Ego"]["position"]
    return float(np.linalg.norm(np.array(vehicle_position) - np.array(cav_position)))


def _estimate_joint_criticality(full_obs: dict, selected_bv_ids: list[str]) -> float:
    cav = full_obs["CAV"]
    values = []
    for vehicle_id in selected_bv_ids:
        vehicle = full_obs[vehicle_id]
        distance = float(vehicle["position"][0] - cav["position"][0] - 5.0)
        closing_speed = float(cav["velocity"] - vehicle["velocity"])
        if distance > 0 and closing_speed > 0:
            ttc = distance / closing_speed
        else:
            ttc = 5.0
        values.append(max(0.01, min(1.0, 1.0 - min(ttc, 5.0) / 5.0)))
    return max(values) if values else 0.01


def _common_obs_bounds() -> tuple[list[float], list[float]]:
    if conf.simulation_config["map"] == "2LaneLong":
        cav_position_lb, cav_position_ub = [400, 40], [4400, 50]
    else:
        cav_position_lb, cav_position_ub = [400, 40], [800, 50]
    return (
        cav_position_lb + [0, -30, 0, -16],
        cav_position_ub + [20, 0, 1, 0],
    )


def _safe_log10(value: float) -> float:
    return float(log10(max(value, 1e-30)))
