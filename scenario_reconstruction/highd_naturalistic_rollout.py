"""Opt-in highD naturalistic SUMO pilot, with auditable executed probabilities.

This standalone NDE runner controls every template BV. Its outputs are validation
logs, not NADE/D2RL training episodes. Uncovered states use the original NDD.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import platform

import numpy as np

from .highd_ndd_shadow import HighDShadowNDD
from .highd_lane_change_guard import blocked_sides, validate_config as validate_guard
from .templates import load_template


def actor_rng(seed, actor_id):
    """Independent stable streams: adding support actors must not shift core draws."""
    identity = int.from_bytes(hashlib.sha256(actor_id.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, identity]))


def execution_pdf(candidate, obs, guard_config=None):
    """Apply road legality and renormalise BEFORE drawing an action."""
    pdf = np.asarray(candidate, dtype=float).copy()
    if pdf.shape != (33,) or not np.isfinite(pdf).all() or np.any(pdf < 0):
        raise ValueError("Expected a finite nonnegative 33-action PDF")
    for index, side in enumerate(("left", "right")):
        if not obs["Ego"][f"could_drive_adjacent_lane_{side}"]:
            pdf[index] = 0.0
    if guard_config is not None:
        rejected = blocked_sides(obs, guard_config)
        for index, side in enumerate(("left", "right")):
            if side in rejected:
                pdf[index] = 0.0
    if pdf.sum() <= 0:
        raise ValueError("No legal probability mass")
    return pdf / pdf.sum()


def validate_template_for_ndd(template):
    if template.events:
        raise ValueError("Naturalistic validation requires templates without forced/fault events")
    if template.bridge_metadata.get("source_split") == "test" or "test" in template.tags:
        raise ValueError("Development calibration must not use test templates")
    if template.ego.id != "CAV" or not template.actors:
        raise ValueError("Expected CAV and at least one BV")
    for vehicle in [template.ego, *template.actors]:
        if not 20 <= vehicle.speed <= 40:
            raise ValueError("Initial speeds must be within 20--40 m/s")
    if any(not actor.id.startswith("BV_") for actor in template.actors):
        raise ValueError("All background IDs must start with BV_")
    lengths = template.bridge_metadata.get("vehicle_lengths_m")
    if lengths is not None:
        ids = {v.id for v in [template.ego, *template.actors]}
        if set(lengths) != ids or any(not np.isfinite(v) or v <= 0 for v in lengths.values()):
            raise ValueError("Explicit vehicle lengths must be positive and cover exactly all actors")


def _summary(records, snapshots, guard_config=None):
    failures = []
    probability_error = 0.0
    for row in records:
        pdf = np.asarray(row["executed_pdf"], dtype=float)
        if pdf.shape != (33,) or not np.isfinite(pdf).all() or np.any(pdf < 0):
            failures.append("invalid executed PDF")
            continue
        probability_error = max(probability_error, abs(float(pdf.sum()) - 1))
        p = float(pdf[row["action_id"]])
        if p <= 0 or not np.isclose(p, row["p_action"], rtol=1e-12, atol=0):
            failures.append("sampled action probability mismatch")
        if row["p_action"] != row["q_action"]:
            failures.append("naturalistic p != q")
        obs = row["observation"]
        reconstructed = execution_pdf(row["candidate_pdf"], obs, guard_config)
        if not np.allclose(pdf, reconstructed, atol=1e-12, rtol=0):
            failures.append("executed PDF disagrees with legality transform")
        if guard_config is not None:
            expected = blocked_sides(obs, guard_config)
            if row.get("lane_change_guard") != expected:
                failures.append("lane-change guard diagnostics mismatch")
        if row["action_id"] < 2:
            side = ("left", "right")[row["action_id"]]
            if not obs["Ego"][f"could_drive_adjacent_lane_{side}"]:
                failures.append("sampled unavailable lane")
        if not row["applied"]:
            failures.append("sampled command was not applied")
    bv_rows = [r for r in snapshots if r["vehicle_id"].startswith("BV_")]
    previous = {}
    lane_crossings = 0
    observed_seconds = 0.0
    for row in bv_rows:
        prior = previous.get(row["vehicle_id"])
        if prior is not None:
            observed_seconds += row["time"] - prior["time"]
            lane_crossings += int(row["lane_index"] != prior["lane_index"])
        previous[row["vehicle_id"]] = row

    def describe(key):
        values = [r[key] for r in bv_rows if r[key] is not None and np.isfinite(r[key])]
        if not values:
            return {"count": 0}
        return {"count": len(values), "mean": float(np.mean(values)),
                "p05": float(np.quantile(values, .05)),
                "p50": float(np.quantile(values, .5)),
                "p95": float(np.quantile(values, .95))}

    if not records:
        failures.append("no sampled decisions")
    if probability_error > 1e-9:
        failures.append("PDF not normalized")
    fallback = sum(r["fallback"] for r in records)
    return {
        "decision_count": len(records), "fallback_count": fallback,
        "fallback_rate": fallback / len(records) if records else None,
        "sampled_lane_changes": sum(r["action_id"] < 2 for r in records),
        "sampled_no_leader_lane_changes": sum(
            r["action_id"] < 2 and r["observation"].get("Lead") is None for r in records),
        "guard_enabled": guard_config is not None,
        "guard_adjusted_decisions": sum(any(
            side in r.get("lane_change_guard", {}) and r["candidate_pdf"][i] > 0
            and r["observation"]["Ego"][f"could_drive_adjacent_lane_{side}"]
            for i, side in enumerate(("left", "right"))) for r in records),
        "observed_lane_crossings": lane_crossings,
        "bv_observed_seconds": observed_seconds,
        "lane_crossings_per_vehicle_hour": lane_crossings * 3600 / observed_seconds
            if observed_seconds else None,
        "speed_boundary_adjusted_commands": sum(r["speed_boundary_adjusted"] for r in records),
        "longitudinal_sources": dict(Counter(r["longitudinal_source"] for r in records)),
        "lane_occupancy_samples": dict(Counter(str(r["lane_index"]) for r in bv_rows)),
        "distributions": {k: describe(k) for k in
            ("speed_mps", "acceleration_mps2", "headway_s", "ttc_s", "gap_m")},
        "maximum_probability_sum_error": probability_error,
        "probability_audit_passed": not failures,
        "failures": sorted(set(failures)),
    }


def audit_episode(path, model=None):
    episode = json.loads(Path(path).read_text(encoding="utf-8"))
    if episode["metadata"]["mode"] != "highd_naturalistic_closed_loop":
        raise ValueError("Not a highD naturalistic episode")
    summary = _summary(episode["decisions"], episode["snapshots"],
                       episode["metadata"].get("lane_change_guard_config"))
    lengths = episode["metadata"].get("vehicle_lengths_m", {})
    for row in episode["snapshots"]:
        if row["vehicle_id"] in lengths and not np.isclose(
                row.get("vehicle_length_m", float("nan")), lengths[row["vehicle_id"]], atol=1e-9, rtol=0):
            summary["failures"].append("executed vehicle length differs from template")
    maximum_model_error = 0.0
    if model is not None:
        for key in ("longitudinal_model_sha256", "context_model_sha256", "context_config_sha256"):
            if episode["metadata"][key] != model.metadata[key]:
                raise ValueError(f"Artifact mismatch: {key}")
        for row in episode["decisions"]:
            reconstructed = model.compare(row["observation"], np.asarray(row["original_pdf"]))
            maximum_model_error = max(maximum_model_error, float(np.max(np.abs(
                np.asarray(reconstructed["highd_shadow_pdf"]) - row["candidate_pdf"]))))
            if reconstructed["fallback"] != row["fallback"]:
                summary["failures"].append("model fallback reconstruction mismatch")
        if maximum_model_error > 1e-12:
            summary["failures"].append("model probability reconstruction mismatch")
    summary["model_reconstruction_checked"] = model is not None
    summary["maximum_model_reconstruction_error"] = maximum_model_error if model else None
    summary["failures"] = sorted(set(summary["failures"]))
    summary["probability_audit_passed"] = not summary["failures"]
    return summary


def run_naturalistic(template_path, model, output, seed=7, guard_config=None):
    # Keep runtime dependencies out of offline audit/test imports.
    from controller.nddcontroller import NDDController
    from envs.nde import NDE
    from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
    from mtlsp.logger.infoextractor import InfoExtractor
    from mtlsp.simulator import Simulator, traci
    from mtlsp.vehicle.vehicle import Vehicle
    import utils

    template = load_template(template_path)
    validate_template_for_ndd(template)
    if guard_config is not None:
        validate_guard(guard_config)
        if guard_config["duration_s"] != 1.0:
            raise ValueError("Guard duration must match the current 1s lane-change command")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    rngs = {actor.id: actor_rng(seed, actor.id) for actor in template.actors}
    # Explicitly bind the adapter to the simulator's 31 longitudinal actions.
    actual_axis = np.array([utils.action_id_to_action_command(i)["longitudinal"]
                            for i in range(2, 33)])
    if not np.allclose(model.acceleration_axis, actual_axis, atol=1e-8, rtol=0):
        raise ValueError("highD acceleration axis differs from the SUMO action interface")

    class Controller(NDDController):
        def step(self):
            if self.vehicle.controlled_duration != 0 or self.vehicle.controlled_flag:
                return
            obs = copy.deepcopy(self.vehicle.observation.information)
            _, _, original = NDDController.static_get_ndd_pdf(obs=obs)
            candidate = model.compare(obs, np.asarray(original))
            pdf = execution_pdf(candidate["highd_shadow_pdf"], obs, guard_config)
            action_id = int(rngs[self.vehicle.id].choice(33, p=pdf))
            self.action = utils.action_id_to_action_command(action_id)
            speed = float(obs["Ego"]["velocity"])
            acceleration = self.action["longitudinal"]
            # Vehicle.act uses a one-second speed-bound check even on 0.1s commands.
            bounded = float(np.clip(acceleration, self.vehicle.v_low - speed,
                                    self.vehicle.v_high - speed))
            row = {
                "time": float(self.vehicle.simulator.get_time()),
                "vehicle_id": self.vehicle.id, "observation": obs,
                "original_pdf": np.asarray(original, dtype=float).tolist(),
                "candidate_pdf": candidate["highd_shadow_pdf"],
                "executed_pdf": pdf.tolist(), "action_id": action_id,
                "p_action": float(pdf[action_id]), "q_action": float(pdf[action_id]),
                "action": dict(self.action), "bounded_acceleration_mps2": bounded,
                "speed_boundary_adjusted": abs(bounded - acceleration) > 1e-9,
                "fallback": candidate["fallback"],
                "longitudinal_source": candidate["longitudinal_source"],
                "lateral_source": candidate["lateral_source"], "applied": False,
                "lane_change_guard": blocked_sides(obs, guard_config) if guard_config else {},
            }
            self.vehicle.simulator.env.decisions.append(row)
            self.pending_record = row

    class BVGlobal(DummyGlobalController):
        def step(self):
            self.reset_control_and_action_state()
            for vehicle_id in sorted(self.controllable_veh_id_list):
                vehicle = self.env.vehicle_list[vehicle_id]
                vehicle.controller.pending_record = None
                vehicle.controller.step()
                vehicle.update()
                if vehicle.controller.pending_record is not None:
                    vehicle.controller.pending_record["applied"] = bool(vehicle.controlled_flag)

    class Logger(InfoExtractor):
        def get_snapshot_info(self, control_info):
            self.env.record_state()

        def get_terminate_info(self, stop, reason, additional_info):
            # Include the final state after the last SUMO action.
            self.env.record_state()
            self.env.termination = {"reason": reason, "details": additional_info}

    class Environment(NDE):
        def __init__(self):
            self.decisions = []
            self.snapshots = []
            self.termination = None
            super().__init__(BVController=Controller, BVGlobalController=BVGlobal,
                             info_extractor=Logger)

        def generate_traffic_flow(self, init_info=None):
            ego = template.ego
            self.generate_av(speed=ego.speed, position=ego.position, route=ego.route,
                             av_lane_id=f"0to1_{ego.lane_index}")
            lengths = template.bridge_metadata.get("vehicle_lengths_m", {})
            if ego.id in lengths:
                traci.vehicle.setLength(ego.id, lengths[ego.id])
            for actor in template.actors:
                vehicle = Vehicle(id=actor.id, controller=Controller(), routeID=actor.route,
                    simulator=self.simulator, initial_speed=actor.speed,
                    initial_position=actor.position, initial_lane_id=f"0to1_{actor.lane_index}")
                self.simulator._add_vehicle_to_sumo(vehicle, typeID="IDM")
                if actor.id in lengths:
                    traci.vehicle.setLength(actor.id, lengths[actor.id])
                vehicle.install_controller(Controller())
                self.vehicle_list.add_vehicles([vehicle])

        def record_state(self):
            for vehicle in self.vehicle_list:
                obs = vehicle.observation.information
                ego = obs["Ego"]
                lead = obs.get("Lead")
                speed = float(ego["velocity"])
                gap = float(lead["distance"]) if lead else None
                closing = speed - float(lead["velocity"]) if lead else 0
                self.snapshots.append({
                    "time": float(self.simulator.get_time()), "vehicle_id": vehicle.id,
                    "speed_mps": speed, "acceleration_mps2": float(ego["acceleration"]),
                    "lane_index": int(ego["lane_index"]), "gap_m": gap,
                    "leader_id": lead["veh_id"] if lead else None,
                    "vehicle_length_m": float(self.simulator.get_vehicle_length(vehicle.id)),
                    "headway_s": gap / speed if gap is not None and speed > 0 else None,
                    "ttc_s": gap / closing if gap is not None and closing > 0 else None,
                })

        def _terminate_check(self):
            collisions = sorted(set(self.simulator.detected_crash()))
            if collisions:
                return {1: "vehicle collision"}, True, {"collision_id": collisions}
            reason, stop, info = super()._terminate_check()
            if stop:
                return reason, stop, info
            if self.simulator.get_time() + 1e-9 >= template.duration:
                return {5: "scenario duration reached"}, True, {}
            return None, False, {}

    env = Environment()
    sim = Simulator(sumo_net_file_path="./maps/2LaneHighway/2LaneHighway.net.xml",
        sumo_config_file_path="./maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg",
        num_tries=50, step_size=.1, action_step_size=.1, lc_duration=1,
        sublane_flag=True, gui_flag=False, output=[], experiment_path=str(output))
    expected_range = template.bridge_metadata.get("observation_range_m", 115.0)
    if sim.config["max_obs_range"] != expected_range:
        raise ValueError("Native reference range does not match runtime observations")
    sim.bind_env(env)
    sim.simulation_seed = seed
    try:
        sim.run(0)
    finally:
        sim.stop()
    summary = _summary(env.decisions, env.snapshots, guard_config)
    metadata = {
        **model.metadata, "mode": "highd_naturalistic_closed_loop",
        "runtime_actions_changed": True,
        "controlled_vehicles": "all template BVs; CAV remains IDM",
        "target_distribution": "highD with logged original-NDD fallback",
        "lane_change_guard_config": guard_config,
        "execution_transform": "road_legality_only" if guard_config is None else "road_legality_and_gap_envelope",
        "target_distribution_note": "If guarded, p=q refers ONLY to the guarded hybrid NDE, not the unguarded model. Target support changes; no unbiasedness claim for the old NDE.",
        "weight_semantics": "p=q for this hybrid NDE, unit command-trajectory weight",
        "weight_episode": 1.0, "log_importance_weight": 0.0,
        "not_for_d2rl_training": True,
        "horizon_diagnostic": template.bridge_metadata.get("horizon_diagnostic"),
        "template_id": template.template_id,
        "template_sha256": hashlib.sha256(Path(template_path).read_bytes()).hexdigest(),
        "source_split": template.bridge_metadata.get("source_split"),
        "seed": seed, "step_size_s": .1, "lane_change_duration_s": 1.0,
        "observation_range_m": float(sim.config["max_obs_range"]),
        "sampling_rng": "per_actor_sha256_seedsequence_pcg64_v1",
        "evaluation_actor_ids": template.bridge_metadata.get(
            "evaluation_actor_ids", [actor.id for actor in template.actors]),
        "support_actor_ids": template.bridge_metadata.get("support_actor_ids", []),
        "vehicle_length_mode": template.bridge_metadata.get("vehicle_length_mode", "fixed"),
        "vehicle_lengths_m": template.bridge_metadata.get("vehicle_lengths_m", {}),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
    }
    episode = {"metadata": metadata, "termination": env.termination,
               "decisions": env.decisions, "snapshots": env.snapshots, "summary": summary}
    (output / "naturalistic_episode.json").write_text(
        json.dumps(episode, indent=2, allow_nan=False), encoding="utf-8")
    summary = audit_episode(output / "naturalistic_episode.json", model)
    (output / "naturalistic_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"template_id": template.template_id, "seed": seed,
            "termination": env.termination, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--split", choices=("train", "calibration", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--longitudinal_model", required=True)
    parser.add_argument("--context_model", required=True)
    parser.add_argument("--context_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lane_change_guard", help="Opt-in candidate geometry constraint; changes target NDE")
    args = parser.parse_args()
    guard_config = json.loads(Path(args.lane_change_guard).read_text(encoding="utf-8")) if args.lane_change_guard else None
    if guard_config is not None:
        validate_guard(guard_config)
        if guard_config["duration_s"] != 1.0:
            parser.error("Guard duration must match the current 1s lane-change command")
    if args.limit < 1 or args.repeats < 1:
        parser.error("limit and repeats must be positive")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = [r for r in manifest["records"]
               if r.get("status") == "template_created" and r.get("split") == args.split][:args.limit]
    if not records:
        parser.error("No eligible templates")
    # Validate the full selected list before starting SUMO.
    for r in records:
        validate_template_for_ndd(load_template(r["template_path"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = HighDShadowNDD(args.longitudinal_model, args.context_model, args.context_config)
    results = []
    for repeat in range(args.repeats):
        for i, record in enumerate(records):
            index = len(results)
            result = run_naturalistic(record["template_path"], model,
                output / f"episode_{index:04d}", args.seed + repeat * len(records) + i, guard_config)
            results.append(result)
            print(json.dumps({"completed": len(results), "decisions": result["decision_count"],
                              "lane_crossings": result["observed_lane_crossings"],
                              "audit_passed": result["probability_audit_passed"]}), flush=True)
            summary = {"schema_version": 1, "split": args.split,
                       "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                       "completed": len(results), "planned": len(records) * args.repeats,
                       "results": results,
                       "scope": "conditional-template naturalistic NDD pilot; not population validation"}
            (output / "naturalistic_summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if not all(r["probability_audit_passed"] for r in results):
        raise SystemExit("Naturalistic probability audit failed")


if __name__ == "__main__":
    main()
