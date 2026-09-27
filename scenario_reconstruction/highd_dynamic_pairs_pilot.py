"""Versioned SUMO harness for dynamic multi-pair P and a single biased unit.

The loop is derived from the frozen pilot; executor/snapshot and fitted kernels
are reused unchanged. Keeping a separate entry preserves all old target hashes.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from .highd_dual_ndd_bundle import DualNDDBundle
from .highd_dual_ndd_pilot import DecoupledExecutor, snapshot
from .highd_dual_ndd_runtime import UnsupportedState
from .highd_dynamic_pairs import DynamicSceneNDD, audit_dynamic_decision, PARTITION_VERSION, INTERVENTION_VERSION
from .highd_pair_data_inventory import digest, write_new

def run(output, config, seed=7, forced_start=False, proposal_epsilon=None,
        proposal_mode="cav_criticality_v1", criticality_config=None):
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
    if proposal_mode != "cav_criticality_v1":
        raise ValueError("Unknown proposal mode")
    if proposal_epsilon is not None:
        from .highd_dual_ndd_criticality import CAVCriticalityProvider
        provider = CAVCriticalityProvider(proposal_epsilon, criticality_config)
    protocol = {"config": config, "seed": seed, "forced_start": forced_start,
                "proposal_epsilon": proposal_epsilon,
                "proposal_scorer": "cav_criticality_v1" if provider else None,
                "map_sha256": digest(config["sumo_net"]), "bundle_sha256": digest(config["bundle_manifest"]),
                "runtime_code_sha256": digest(Path(__file__).with_name("highd_dual_ndd_runtime.py")),
                "pilot_code_sha256": digest(__file__), "python": platform.python_version(),
                "scope": "Opt-in candidate execution; shared motion P/Q kernel when enabled. No formal training or risk-efficiency claim."}
    target = {k: protocol[k] for k in ("bundle_sha256", "runtime_code_sha256", "pilot_code_sha256")}
    protocol["partition_contract"] = {"version": PARTITION_VERSION, "maximum_pairs": config["maximum_pairs"],
                                      "intervention_version": INTERVENTION_VERSION}
    protocol["dynamic_runtime_sha256"] = digest(Path(__file__).with_name("highd_dynamic_pairs.py"))
    target.update(partition_contract=protocol["partition_contract"], dynamic_runtime_sha256=protocol["dynamic_runtime_sha256"],
                  map_sha256=protocol["map_sha256"],
                  frozen_executor_sha256=digest(Path(__file__).with_name("highd_dual_ndd_pilot.py")))
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
        scene = DynamicSceneNDD(bundle, controlled, config["cav_id"], centers, config["maximum_pairs"], seed,
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
                record["probability_reconstruction_error"] = audit_dynamic_decision(record)
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
    result.update(partition_version=PARTITION_VERSION,
                  pair_count_histogram=dict(Counter(str(len(d["selected_pairs"])) for d in decisions)),
                  intervention_count_histogram=dict(Counter(str(sum("proposal_components" in u for u in d["units"])) for d in decisions)),
                  reconfiguration_decisions=sum(bool(d["pair_transitions"]["added"] or d["pair_transitions"]["removed"])
                                               for d in decisions[1:]),
                  natural_target_sha256=protocol["natural_target_sha256"])
    write_new(output / "pilot_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if termination == "execution_error":
        raise RuntimeError(failure)
    return result
