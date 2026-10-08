"""Short opt-in two-lane SUMO exercise of the assembled dual NDD.

Legacy bundles use a fixed diagnostic duration; opt-in fitted process bundles
use their declared shared P/Q kernel. Neither implies behavior acceptance.
No original controller, probability artifact or map is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import hashlib

import numpy as np

from .highd_dual_ndd_bundle import DualNDDBundle
from .highd_dual_ndd_runtime import SceneNDD, UnsupportedState, audit_decision
from .highd_pair_data_inventory import digest, read_json, write_new


class DecoupledExecutor:
    def __init__(self, connection, lane_centers, config, motion_model=None, motion_seed=7):
        self.connection = connection
        self.centers = lane_centers
        self.config = config
        self.locks = {}
        self.events = []
        self.commands = []
        self.motion_model = motion_model
        self.motion_rng = np.random.default_rng(np.random.SeedSequence([motion_seed, 257]))

    def _lateral_command(self, key, distance, duration):
        vehicle = self.connection.vehicle
        accel = self.config["lateral_acceleration_mps2"]
        discriminant = (accel * duration) ** 2 - 4 * accel * abs(distance)
        if discriminant < 0:
            raise ValueError("Configured lateral motion is infeasible")
        maximum_speed = (accel * duration - np.sqrt(discriminant)) / 2
        vehicle.setParameter(key, "laneChangeModel.lcAccelLat", str(accel))
        vehicle.setMaxSpeedLat(key, float(maximum_speed))
        vehicle.changeSublane(key, float(distance))
        return float(maximum_speed)

    def start(self, actor, lateral_choice, now, source="sampled", motion_context=None):
        if actor in self.locks or lateral_choice not in (1, 2):
            raise ValueError("A locked lateral maneuver cannot be restarted")
        vehicle = self.connection.vehicle
        key = str(actor)
        lane = int(vehicle.getLaneIndex(key))
        target = lane + (1 if lateral_choice == 1 else -1)
        if target not in (0, 1):
            raise ValueError("Illegal lateral initiation")
        y = float(vehicle.getPosition(key)[1])
        distance = self.centers[target] - y
        duration = self.config["lateral_duration_diagnostic_s"]
        process = None
        leg_duration = duration
        end_lane = target
        if self.motion_model is not None:
            if self.motion_model.get("kernel") == "conditional_paired_motion_v1":
                from .highd_lane_motion_context import sample, audit_sample
                if motion_context is None:
                    raise ValueError("Conditional motion requires the actual pre-command traffic context")
                process = sample(self.motion_model, self.motion_rng, abs(self.centers[1] - self.centers[0]),
                                 distance, self.config["lateral_acceleration_mps2"], motion_context)
                process["common_transition_log_density"] = audit_sample(process, self.motion_model)
            else:
                from .highd_lane_motion import sample
                process = sample(self.motion_model, self.motion_rng, abs(self.centers[1] - self.centers[0]),
                                 distance, self.config["lateral_acceleration_mps2"])
            duration = process["requested_duration_s"]
            leg_duration = (duration - .24) / process["legs"]
            distance = np.sign(distance) * process["outbound_distance_m"]
            if process["legs"] == 2:
                end_lane = lane
            if "waypoint_distances_m" in process:
                leg_duration = process["leg_durations_s"][0]
                end_lane = target if process["outcome"] == "completed" else lane
        event = {"actor_id": actor, "source": source, "start_time_s": now, "origin_lane": lane,
                 "target_lane": target, "start_y": y, "diagnostic_duration_s": duration,
                 "stable_since": None,
                 "status": "active", "motion_sample": process, "stage": "outbound", "end_lane": end_lane,
                 "outbound_y": y + distance, "leg_duration_s": leg_duration, "crossed_target_lane": False}
        if process and "waypoint_distances_m" in process:
            sign = 1 if target > lane else -1
            event["waypoints_y"] = [y + sign * v for v in process["waypoint_distances_m"]]
            event["waypoint_index"] = 0
            event["endpoint_y"] = event["waypoints_y"][-1]
            # Do not silently map a donor endpoint into another lane.
            if abs(event["endpoint_y"] - self.centers[end_lane]) >= abs(self.centers[1] - self.centers[0]) / 2:
                raise ValueError("Paired endpoint lies outside its declared outcome lane")
        event["maximum_lateral_speed_mps"] = self._lateral_command(key, distance, leg_duration)
        self.locks[actor] = event
        self.events.append(event)

    def observe(self, now):
        vehicle = self.connection.vehicle
        for actor, event in list(self.locks.items()):
            key = str(actor)
            lane = int(vehicle.getLaneIndex(key))
            y = float(vehicle.getPosition(key)[1])
            event["crossed_target_lane"] |= lane == event["target_lane"]
            process = event.get("motion_sample")
            if "waypoints_y" in event:
                i = event["waypoint_index"]
                if i + 1 < len(event["waypoints_y"]):
                    if abs(y - event["waypoints_y"][i]) <= .03 and abs(float(vehicle.getLateralSpeed(key))) < .2:
                        event["waypoint_index"] += 1
                        self._lateral_command(key, event["waypoints_y"][i + 1] - y, process["leg_durations_s"][i + 1])
                        event.update(stage="returning", reversal_time_s=now)
                    if now - event["start_time_s"] > process["requested_duration_s"] + 2:
                        raise RuntimeError("Paired waypoint motion failed to resolve")
                    continue
            elif process and process["legs"] == 2 and event["stage"] == "outbound":
                if abs(y - event["outbound_y"]) <= .03 and abs(float(vehicle.getLateralSpeed(key))) < .2:
                    self._lateral_command(key, event["start_y"] - y, event["leg_duration_s"])
                    event.update(stage="returning", reversal_time_s=now)
                if now - event["start_time_s"] > process["requested_duration_s"] + 2:
                    raise RuntimeError("Sampled outbound motion failed to resolve")
                continue
            end_y = event.get("endpoint_y", event["start_y"] if process and process["legs"] == 2 else self.centers[event["target_lane"]])
            center_error = abs(y - end_y)
            stable = (lane == event["end_lane"] and center_error <= (.03 if process and process["legs"] == 2 else .6)
                      and abs(float(vehicle.getLateralSpeed(key))) < .2)
            if "endpoint_y" in event:
                stable = lane == event["end_lane"] and center_error <= .03 and abs(float(vehicle.getLateralSpeed(key))) < .2
            if not stable:
                event["stable_since"] = None
            elif event["stable_since"] is None:
                event["stable_since"] = now
            elif now - event["stable_since"] >= .2 - 1e-9:
                status = "completed"
                if process and process["outcome"] != "completed":
                    status = ("returned_after_crossing" if event["crossed_target_lane"] else
                              "within_lane_adjustment" if process["outcome"] == "within_lane_adjustment" else "returned_before_crossing")
                event.update(status=status, end_time_s=now, actual_lock_duration_s=now - event["start_time_s"])
                del self.locks[actor]
            deadline = process["requested_duration_s"] + 2 if process else self.config["maximum_lock_s"]
            if actor in self.locks and now - event["start_time_s"] > deadline:
                raise RuntimeError("Lateral command did not complete; stop diagnostic instead of silently resetting it")

    def apply(self, actions, now):
        if len({a["actor_id"] for a in actions}) != len(actions):
            raise ValueError("Duplicate commands for one actor")
        for action in actions:
            actor = action["actor_id"]
            was_locked = actor in self.locks
            if action["lateral_choice"] > 0:
                self.start(actor, action["lateral_choice"], now, motion_context=action.get("motion_context"))
            # Crucially: no 'central'/recenter command and no new lateral call
            # while continuing a lock. Only acceleration is refreshed at 10 Hz.
            self.connection.vehicle.setAcceleration(str(actor), float(action["acceleration_mps2"]), .1)
            self.commands.append({"time_s": now, **action, "was_lateral_locked": was_locked,
                                  "lateral_locked_after": actor in self.locks, "duration_s": .1})


def snapshot(connection):
    vehicle = connection.vehicle
    result = []
    for name in sorted(vehicle.getIDList(), key=int):
        position = vehicle.getPosition(name)
        length = float(vehicle.getLength(name))
        result.append({"id": int(name), "center_x": float(position[0]) - length / 2,
                       "center_y": float(position[1]), "length": length, "width": float(vehicle.getWidth(name)),
                       "lane": int(vehicle.getLaneIndex(name)), "speed": float(vehicle.getSpeed(name)),
                       "acceleration": float(vehicle.getAcceleration(name)),
                       "lateral_speed": float(vehicle.getLateralSpeed(name))})
    return result


def run(output, config, seed=7, forced_start=False, proposal_epsilon=None,
        proposal_mode="interface_probe", criticality_config=None):
    # Keep SUMO out of offline imports and unit tests.
    import traci
    import sumolib
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh pilot output")
    output.mkdir(parents=True)
    bundle = DualNDDBundle(config["bundle_manifest"])
    if bundle.motion_model:
        contract = bundle.manifest["runtime_contract"]
        if not config.get("use_motion_model") or any(config.get(k) != contract[k] for k in ("relation_mode", "lateral_acceleration_mps2")):
            raise ValueError("Pilot differs from the declared motion/geometry target")
    provider = None
    if proposal_mode not in ("interface_probe", "cav_criticality_v1"):
        raise ValueError("Unknown proposal mode")
    if proposal_epsilon is not None:
        if proposal_mode == "cav_criticality_v1":
            from .highd_dual_ndd_criticality import CAVCriticalityProvider
            provider = CAVCriticalityProvider(proposal_epsilon, criticality_config)
        else:
            from .highd_dual_ndd_d2rl import make_provider, acceleration_tilt_probe
            provider = make_provider(acceleration_tilt_probe, epsilon=proposal_epsilon)
    protocol = {"config": config, "seed": seed, "forced_start": forced_start,
                "proposal_epsilon": proposal_epsilon,
                "proposal_scorer": ("cav_criticality_v1" if proposal_mode == "cav_criticality_v1" else
                                    "acceleration_tilt_interface_probe_not_collision_model") if provider else None,
                "map_sha256": digest(config["sumo_net"]), "bundle_sha256": digest(config["bundle_manifest"]),
                "runtime_code_sha256": digest(Path(__file__).with_name("highd_dual_ndd_runtime.py")),
                "pilot_code_sha256": digest(__file__), "python": platform.python_version(),
                "scope": "Opt-in candidate execution; shared motion P/Q kernel when enabled. No formal training or risk-efficiency claim."}
    target = {k: protocol[k] for k in ("bundle_sha256", "runtime_code_sha256", "pilot_code_sha256")}
    target.update({k: config.get(k) for k in ("relation_mode", "lateral_acceleration_mps2", "use_motion_model", "lateral_duration_diagnostic_s")})
    protocol["natural_target_sha256"] = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
    if provider is not None and proposal_mode == "cav_criticality_v1":
        protocol["criticality_config"] = provider.config
        protocol["criticality_code_sha256"] = digest(Path(__file__).with_name("highd_dual_ndd_criticality.py"))
    write_new(output / "protocol.json", protocol)
    cmd = [sumolib.checkBinary("sumo"), "-c", config["sumo_config"], "--step-length", ".02",
           "--lateral-resolution", ".1", "--step-method.ballistic", "true", "--seed", str(seed),
           "--collision.action", "warn", "--collision.mingap-factor", "0", "--time-to-teleport", "-1",
           "--no-step-log", "true", "--duration-log.disable", "true"]
    label = "dual_ndd_pilot"
    traci.start(cmd, label=label)
    connection = traci.getConnection(label)
    decisions, observations, endpoint_errors = [], [], []
    termination = "duration_reached"
    failure = None
    executor = None
    scene = None
    try:
        protocol["sumo_version"] = connection.getVersion()
        centers = [float(connection.lane.getShape(f"0to1_{k}")[0][1]) for k in (0, 1)]
        controlled = [a["id"] for a in config["actors"] if a["id"] != config["cav_id"]]
        for actor in config["actors"]:
            connection.vehicle.add(str(actor["id"]), "route_0", typeID="IDM", departLane=str(actor["lane"]),
                                   departPos=str(actor["front_x"]), departSpeed=str(actor["speed"]))
        connection.simulationStep()
        if set(connection.vehicle.getIDList()) != {str(a["id"]) for a in config["actors"]}:
            raise ValueError("Not all initial vehicles were inserted")
        for actor in controlled:
            connection.vehicle.setSpeedMode(str(actor), 0)
            connection.vehicle.setLaneChangeMode(str(actor), 0)
        # CAV is an unchanged IDM follower; disable its autonomous lane changes
        # to keep this narrow process diagnostic interpretable.
        connection.vehicle.setLaneChangeMode(str(config["cav_id"]), 0)
        scene = SceneNDD(bundle, controlled, config["cav_id"], centers, config["maximum_pairs"], seed,
                         config.get("relation_mode", "legacy_center_v1"))
        executor = DecoupledExecutor(connection, centers, config, bundle.motion_model, seed)
        if provider is not None and proposal_mode == "cav_criticality_v1":
            provider.locks = executor.locks
        total_steps = round(config["duration_s"] / .02)
        previous = None
        for step in range(total_steps + 1):
            now = step / 50
            if step % 2 == 0:
                actors = snapshot(connection)
                observations.append({"time_s": now, "actors": actors})
                scene.observe(step // 2, actors)
                executor.observe(now)
            if step % 5 == 0:
                if previous is not None:
                    for action in previous["actions"]:
                        actor = action["actor_id"]
                        realized = (connection.vehicle.getSpeed(str(actor)) - previous["speeds"][actor]) / .1
                        endpoint_errors.append({"actor_id": actor, "time_s": now,
                                                "command": action["acceleration_mps2"], "realized_mean": realized,
                                                "error": abs(realized - action["acceleration_mps2"])})
                if step == total_steps:
                    break
                if forced_start and step == 50:
                    executor.start(config["forced_start_actor_id"], 1, now, "forced_process_positive_control")
                record = scene.decide(step // 5, executor.locks, proposal_provider=provider)
                record["time_s"] = now
                record["probability_reconstruction_error"] = audit_decision(record)
                record["speeds"] = {actor: float(connection.vehicle.getSpeed(str(actor))) for actor in controlled}
                executor.apply(record["actions"], now)
                decisions.append(record)
                previous = record
            connection.simulationStep()
            collisions = list(connection.simulation.getCollidingVehiclesIDList())
            if collisions:
                termination = "collision"
                failure = {"colliding_actor_ids": collisions, "time_s": (step + 1) / 50,
                           "detector": "SUMO", "actors": snapshot(connection)}
                break
    except UnsupportedState as exc:
        termination, failure = "left_frozen_state_support", str(exc)
    except Exception as exc:
        termination, failure = "execution_error", repr(exc)
    finally:
        connection.close()
    events = executor.events if executor else []
    commands = executor.commands if executor else []
    rows = {"protocol": protocol, "termination": termination, "failure": failure,
            "decisions": decisions, "observations": observations, "commands": commands,
            "lateral_events": events, "endpoint_checks": endpoint_errors,
            "behavior_criticality": getattr(provider, "records", [])}
    write_new(output / "episode.json", rows)
    replay_path = None
    cav_collision = termination == "collision" and str(config["cav_id"]) in (failure or {}).get("colliding_actor_ids", [])
    has_proposal = any("proposal_components" in u for d in decisions for u in d["units"])
    if has_proposal and not forced_start and (termination == "duration_reached" or cav_collision):
        from .highd_dual_ndd_d2rl import episode_record
        replay = episode_record(rows)
        replay_path = str(output / "d2rl_episode.json")
        write_new(replay_path, replay)
    selected = Counter(c["relation"] for d in decisions for c in d["selected_pairs"])
    changing_while_locked = 0
    last_accel = {}
    for command in commands:
        actor = command["actor_id"]
        if command["was_lateral_locked"] and actor in last_accel and last_accel[actor] != command["acceleration_mps2"]:
            changing_while_locked += 1
        last_accel[actor] = command["acceleration_mps2"]
    result = {"seed": seed, "mode": ("forced_positive_control" if forced_start else
              "cav_criticality_behavior_collection" if provider and proposal_mode == "cav_criticality_v1" else
              "conditional_proposal_interface_probe" if provider else "natural_command_sampling_diagnostic"),
              "behavior_status_counts": dict(Counter(r["status"] for r in getattr(provider, "records", []))),
              "raw_cav_collision": cav_collision,
              "d2rl_replay_path": replay_path,
              "conditional_proposal_decisions": sum("proposal_components" in u for d in decisions for u in d["units"]),
              "termination": termination, "failure": failure, "decision_count": len(decisions),
              "controlled_bv_action_count": len(commands), "selected_pair_relation_counts": dict(selected),
              "unpaired_single_action_count": sum(len(u["actor_ids"]) == 1 for d in decisions for u in d["units"]),
              "sampled_lateral_starts": sum(e["source"] == "sampled" for e in events),
              "forced_lateral_starts": sum(e["source"] != "sampled" for e in events),
              "completed_lateral_maneuvers": sum(e["status"] == "completed" for e in events),
              "lateral_outcome_counts": dict(Counter(e["status"] for e in events)),
              "sampled_motion_outcome_counts": dict(Counter(e["motion_sample"]["outcome"] for e in events if e.get("motion_sample"))),
              "motion_duration_physical_truncation_mean": float(np.mean([e["motion_sample"]["removed_duration_mass"] for e in events if e.get("motion_sample")])) if any(e.get("motion_sample") for e in events) else None,
              "longitudinal_updates_during_lateral_lock": sum(c["was_lateral_locked"] for c in commands),
              "acceleration_changes_during_lateral_lock": changing_while_locked,
              "max_probability_reconstruction_error": max((d["probability_reconstruction_error"] for d in decisions), default=0.),
              "max_realized_mean_acceleration_error": max((e["error"] for e in endpoint_errors), default=None),
              "max_log_importance_ratio_abs": max((abs(d["log_importance_ratio"]) for d in decisions), default=0.),
              "causal_phase_counts": dict(Counter(u["phase"] for d in decisions for u in d["units"] if u.get("phase") is not None)),
              "parallel_reference_decisions": sum(bool(r["parallel_ids"]) for d in decisions for r in d.get("reference_relations", [])),
              "observed_completed_events": len(scene.ended_events) if scene else 0,
              "lateral_duration_model": (bundle.motion_model.get("kernel", "censor_aware_outcome_lognormal_candidate") if bundle.motion_model else "fixed 4s execution diagnostic, not highD-calibrated"),
              "importance_weights_usable_for_risk_estimation": False,
              "runtime_accepted": False}
    write_new(output / "pilot_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if termination == "execution_error":
        raise RuntimeError(failure)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/highd_dual_ndd_pilot_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forced_start", action="store_true")
    parser.add_argument("--proposal_epsilon", type=float, nargs=2, default=None,
                        help="Natural mixture masses for the explicitly selected proposal mode")
    parser.add_argument("--proposal_mode", choices=("interface_probe", "cav_criticality_v1"), default="interface_probe")
    args = parser.parse_args()
    run(args.output, read_json(args.config), args.seed, args.forced_start, args.proposal_epsilon, args.proposal_mode)


if __name__ == "__main__":
    main()
