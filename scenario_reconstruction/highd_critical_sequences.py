"""Audit fixed-behavior episodes and retain every critical decision in order.

No SUMO rollout, NDD refit, outcome relabeling, single-step selection, or
importance-weight clipping. Empty safe sequences remain in the exposure count.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from d2rl_training.conditional_chain import replay_weight
from d2rl_training.highd_event_contract import FIRST_EVENT_DEFINITION, FIRST_SEQUENCE, target_digest, with_first_event
from .highd_dual_ndd_d2rl import OBSERVATION_CONTRACT
from .highd_dual_ndd_runtime import audit_decision
from .highd_pair_data_inventory import digest, read_json, write_new


def close(actual, expected, name):
    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-9, rtol=1e-10):
        raise ValueError(f"{name} disagrees: {actual} versus {expected}")


def extract(episode, *, allow_censored=False):
    protocol = episode["protocol"]
    if protocol["forced_start"]:
        raise ValueError("Forced controls are not behavior data")
    termination = episode["termination"]
    ids = (episode.get("failure") or {}).get("colliding_actor_ids", []) if isinstance(episode.get("failure"), dict) else []
    crash = str(protocol["config"]["cav_id"]) in map(str, ids)
    censored = termination == "collision" and not crash
    if termination not in ("duration_reached", "collision") or (censored and not allow_censored):
        raise ValueError("Censored/incomplete episode cannot become a complete training sequence")
    if termination == "collision" and len(set(map(str, ids))) < 2:
        raise ValueError("Collision requires identifiable colliding actors")
    if termination == "duration_reached" and crash:
        raise ValueError("Conflicting outcome")
    end = episode["failure"]["time_s"] if termination == "collision" else protocol["config"]["duration_s"]
    steps, full_log_weight, max_error = [], 0., 0.
    expected_actors = sorted(a["id"] for a in protocol["config"]["actors"] if a["id"] != protocol["config"]["cav_id"])
    for tick, decision in enumerate(episode["decisions"]):
        if decision["decision_tick"] != tick:
            raise ValueError("Missing or unordered raw decision; cannot reconnect silently")
        close(decision["time_s"], tick / 10, "decision time")
        max_error = max(max_error, audit_decision(decision))
        p_log, q_log, active_units = 0., 0., []
        actors = []
        for unit in decision["units"]:
            draw = unit["draw"]
            actions = draw["actions"]
            actors.extend(a["actor_id"] for a in actions)
            indices = [a["acceleration_index"] for a in actions]
            p = (unit["longitudinal_joint"][indices[0]][indices[1]] if len(indices) == 2 else
                 unit["longitudinal_probability"][indices[0]])
            for a, lateral in zip(actions, unit["lateral_conditionals"]):
                if a["action_index"] != a["acceleration_index"] * 3 + a["lateral_choice"]:
                    raise ValueError("Wrong structured action encoding")
                p *= lateral[a["acceleration_index"]][a["lateral_choice"]]
            q = draw["proposal_joint_probability"]
            if p <= 0 or q <= 0:
                raise ValueError("Sampled action has zero P or Q")
            # Log comparisons retain relative sensitivity for rare probabilities.
            close(np.log(p), np.log(draw["natural_joint_probability"]), "sampled P")
            close(np.log(p), draw["log_natural_joint_probability"], "log P")
            close(np.log(q), draw["log_proposal_joint_probability"], "log Q")
            close(np.log(p) - np.log(q), draw["log_importance_ratio"], "unit log P/Q")
            p_log += np.log(p)
            q_log += np.log(q)
            if "proposal_components" in unit:
                active_units.append(unit)
            else:
                close(np.log(p), np.log(q), "unbiased-unit probability")
        if sorted(actors) != expected_actors or len(active_units) > 1:
            raise ValueError("Every BV must appear once; at most one pair can be biased")
        close(p_log, decision["log_natural_probability"], "scene log P")
        close(q_log, decision["log_proposal_probability"], "scene log Q")
        close(p_log - q_log, decision["log_importance_ratio"], "scene log P/Q")
        full_log_weight += p_log - q_log
        if not active_units:
            continue
        unit = active_units[0]
        draw = unit["draw"]
        obs = unit["training_observation"]
        if obs["contract"] != OBSERVATION_CONTRACT or np.asarray(obs["joint"]).shape != (14,):
            raise ValueError("Wrong observation contract")
        if not np.isfinite(obs["joint"]).all() or unit["criticality"] <= 0:
            raise ValueError("Invalid critical observation")
        weight = replay_weight(draw["weight_record"], draw["weight_record"]["generation_epsilon"], draw["ndd_record"])
        close(np.log(weight), decision["log_importance_ratio"], "critical replay weight")
        steps.append({"raw_tick": tick, "time_s": decision["time_s"], "observation": obs["joint"],
                      "actor_ids": unit["actor_ids"], "relation": unit["relation"],
                      "criticality": unit["criticality"], "actions": draw["actions"],
                      "weight_record": draw["weight_record"], "ndd_record": draw["ndd_record"],
                      "generation_log_weight": float(np.log(weight))})
    if not episode["decisions"] or end <= episode["decisions"][-1]["time_s"]:
        raise ValueError("Invalid terminal time")
    if termination == "duration_reached":
        close(len(episode["decisions"]) / 10, end, "complete exposure length")
    for i, step in enumerate(steps):
        step["next_raw_tick"] = steps[i + 1]["raw_tick"] if i + 1 < len(steps) else None
        step["delta_time_s"] = (steps[i + 1]["time_s"] if i + 1 < len(steps) else end) - step["time_s"]
    sequence_log_weight = sum(s["generation_log_weight"] for s in steps)
    close(sequence_log_weight, full_log_weight, "full versus densified log weight")
    return {"schema_version": 1, "natural_target_sha256": protocol["natural_target_sha256"],
            "observation_contract": OBSERVATION_CONTRACT, "collision_result": None if censored else crash,
            "outcome_complete": not censored, "colliding_actor_ids": ids,
            "cav_id": protocol["config"]["cav_id"], "horizon_s": protocol["config"]["duration_s"],
            "source_scenario_id": protocol["config"]["scenario_id"], "seed": protocol["seed"],
            "termination": termination, "end_time_s": end,
            "raw_decision_count": len(episode["decisions"]), "steps": steps,
            "generation_log_weight": float(full_log_weight), "max_probability_error": max_error,
            "max_sequence_log_weight_error": abs(sequence_log_weight - full_log_weight)}


def weighted_diagnostics(records):
    if any(r["collision_result"] is None for r in records):
        raise ValueError("Unknown CAV outcomes cannot be treated as safe; use collection audit")
    crashes = [r for r in records if r["collision_result"]]
    if not crashes:
        return {"attempted": len(records), "crashes": 0, "crash_contribution_ess": None,
                "conditional_weighted_collision_mean": 0., "largest_normalized_crash_contribution": None}
    logs = np.asarray([r["generation_log_weight"] for r in crashes])
    weights = np.exp(logs - logs.max())
    normalized = weights / weights.sum()
    log_mean = float(logs.max() + np.log(weights.sum()) - np.log(len(records)))
    return {"attempted": len(records), "crashes": len(crashes),
            "conditional_weighted_collision_mean": float(np.exp(log_mean)),
            "log_conditional_weighted_collision_mean": log_mean,
            "crash_contribution_ess": float(1 / np.sum(normalized ** 2)),
            "largest_normalized_crash_contribution": float(normalized.max()),
            "by_crash": [{"scenario": r["source_scenario_id"], "seed": r["seed"],
                          "log_weight": float(w), "normalized_contribution": float(n)}
                         for r, w, n in zip(crashes, logs, normalized)]}


def _read_collection(collection, *, allow_censored=False):
    collection = Path(collection)
    source = read_json(collection / "collection_summary.json")
    protocol = read_json(collection / "collection_protocol.json")
    by_mode, grouped, seen = defaultdict(list), defaultdict(Counter), set()
    # Validate the complete manifest before publishing any new training files.
    for run in source["runs"]:
        key = (run["mode"], run["scenario_id"], run["seed"])
        if key in seen:
            raise ValueError("Duplicate episode in collection")
        seen.add(key)
        episode = read_json(run["episode_path"])
        if episode["protocol"]["seed"] != run["seed"]:
            raise ValueError("Seed mismatch")
        if "config" in protocol:
            spec = protocol["config"]
            scene = next(s for s in spec["scenarios"] if s["id"] == run["scenario_id"])
            if episode["protocol"]["config"]["actors"] != scene["actors"]:
                raise ValueError("Initial scene differs from collection protocol")
            close(episode["protocol"]["config"]["duration_s"], spec["duration_s"], "collection horizon")
        sequence = extract(episode, allow_censored=allow_censored)
        if sequence["source_scenario_id"] != run["scenario_id"] or (sequence["collision_result"] is True) != run["raw_cav_collision"]:
            raise ValueError("Manifest outcome/source differs from raw episode")
        if run["mode"] == "natural" and sequence["steps"]:
            raise ValueError("Natural control contains biased actions")
        if run["mode"] == "behavior" and episode["protocol"]["proposal_scorer"] != "cav_criticality_v1":
            raise ValueError("Do not train from the acceleration-tilt interface probe")
        sequence.update(raw_episode_path=run["episode_path"], raw_episode_sha256=digest(run["episode_path"]))
        by_mode[run["mode"]].append(sequence)
        grouped[run["scenario_id"]][run["mode"] + "_crashes"] += int(sequence["collision_result"] is True)
    expected = {(mode, name, protocol["seed"] + k) for mode in protocol["modes"]
                for name in protocol["scenario_ids"] for k in range(protocol["repeats"])}
    if seen != expected:
        raise ValueError("Incomplete initial-scene exposure mixture")
    targets = {s["natural_target_sha256"] for records in by_mode.values() for s in records}
    if len(targets) != 1 or not by_mode["behavior"]:
        raise ValueError("Mixed natural targets or missing behavior pool")
    return by_mode, grouped, seen, targets.pop()


def censored_diagnostics(records):
    """Observed contributions only; never estimate a full-horizon risk by omission."""
    if not records:
        raise ValueError("No attempted episodes")
    complete = [r for r in records if r["collision_result"] is not None]
    unknown = [r for r in records if r["collision_result"] is None]
    observed = weighted_diagnostics(complete)
    mean = observed["conditional_weighted_collision_mean"] * len(complete) / len(records)
    by_source = defaultdict(float)
    for crash in observed.get("by_crash", []):
        by_source[crash["scenario"]] += crash["normalized_contribution"]
    return {"attempted": len(records), "complete_outcome_count": len(complete),
            "observed_cav_crashes": observed["crashes"],
            "full_horizon_no_crash_count": len(complete) - observed["crashes"],
            "bv_only_censored_count": len(unknown),
            "observed_cav_collision_fraction": observed["crashes"] / len(records),
            "observed_crash_weighted_contribution_per_attempt": mean,
            "conditional_weighted_collision_mean": None if unknown else mean,
            "observed_crash_contribution_ess": observed["crash_contribution_ess"],
            "largest_normalized_observed_crash_contribution": observed["largest_normalized_crash_contribution"],
            "normalized_observed_crash_contribution_by_source": dict(by_source),
            "by_observed_crash": sorted(observed.get("by_crash", []), key=lambda r: -r["normalized_contribution"]),
            "censored_episodes": [{"scenario": s["source_scenario_id"], "seed": s["seed"],
                                   "end_time_s": s["end_time_s"], "colliding_actor_ids": s["colliding_actor_ids"],
                                   "critical_steps": len(s["steps"]), "prefix_log_weight": s["generation_log_weight"]}
                                  for s in unknown]}


def audit_collection(collection, output):
    """Save all audited prefixes without publishing an incomplete training manifest."""
    collection, output = Path(collection), Path(output)
    if output.exists():
        raise ValueError("Use a fresh audit output; existing data are preserved")
    by_mode, _, seen, target = _read_collection(collection, allow_censored=True)
    all_records = [s for seq in by_mode.values() for s in seq]
    source_counts = defaultdict(Counter)
    for mode, records in by_mode.items():
        for record in records:
            label = "censored" if record["collision_result"] is None else "crashes" if record["collision_result"] else "full_horizon_no_crash"
            source_counts[record["source_scenario_id"]][mode + "_" + label] += 1
    behavior = by_mode["behavior"]
    summary = {"schema_version": 1, "probability_audit_passed": True,
               "full_horizon_outcomes_complete": all(s["outcome_complete"] for s in all_records),
               "episode_count": len(seen), "natural_target_sha256": target,
               "by_mode": {mode: censored_diagnostics(records) for mode, records in by_mode.items()},
               "source_outcomes": {k: dict(v) for k, v in source_counts.items()},
               "behavior_raw_decisions_including_censored": sum(s["raw_decision_count"] for s in behavior),
               "behavior_critical_decisions": sum(len(s["steps"]) for s in behavior),
               "behavior_complete_nonempty_sequences": sum(bool(s["steps"]) and s["outcome_complete"] for s in behavior),
               "behavior_complete_empty_sequences": sum(not s["steps"] and s["outcome_complete"] for s in behavior),
               "maximum_probability_error": max(s["max_probability_error"] for s in all_records),
               "maximum_sequence_log_weight_error": max(s["max_sequence_log_weight_error"] for s in all_records),
               "training_manifest_written": False,
               "scope": "Configured scenes; observed crash contributions exclude unknown future outcomes. "
                        "No full-horizon risk or policy-efficiency claim when censored episodes exist."}
    write_new(output / "audited_prefixes.json", {
        "contract": "highd_censor_aware_audit_only_v1", "natural_target_sha256": target,
        "collection_protocol_sha256": digest(collection / "collection_protocol.json"),
        "sequences_by_mode": dict(by_mode)})
    summary["audited_prefixes_sha256"] = digest(output / "audited_prefixes.json")
    write_new(output / "collection_audit.json", summary)
    return summary


def prepare(collection, output):
    collection, output = Path(collection), Path(output)
    if output.exists():
        raise ValueError("Use a fresh sequence output; existing data are preserved")
    by_mode, grouped, seen, target = _read_collection(collection)
    data = {"schema_version": 1, "contract": "highd_full_critical_sequence_v1",
            "natural_target_sha256": target, "observation_contract": OBSERVATION_CONTRACT,
            "collection_protocol_sha256": digest(collection / "collection_protocol.json"),
            "scope": "Training-interface pilot on configured scenes, not held-out policy evaluation",
            "sequences": by_mode["behavior"]}
    stats = {mode: weighted_diagnostics(records) for mode, records in by_mode.items()}
    behavior = by_mode["behavior"]
    summary = {"audit_passed": True, "episode_count": len(seen), "by_mode": stats,
               "source_crashes": {k: dict(v) for k, v in grouped.items()},
               "behavior_raw_decisions": sum(s["raw_decision_count"] for s in behavior),
               "behavior_critical_decisions": sum(len(s["steps"]) for s in behavior),
               "nonempty_sequences": sum(bool(s["steps"]) for s in behavior),
               "empty_sequences_retained": sum(not s["steps"] for s in behavior),
               "crash_sequence_lengths": [len(s["steps"]) for s in behavior if s["collision_result"]],
               "safe_nonempty_sequence_lengths": [len(s["steps"]) for s in behavior if s["steps"] and not s["collision_result"]],
               "maximum_probability_error": max(s["max_probability_error"] for seq in by_mode.values() for s in seq),
               "maximum_sequence_log_weight_error": max(s["max_sequence_log_weight_error"] for seq in by_mode.values() for s in seq),
               "formal_generalization_evidence": False}
    write_new(output / "sequences.json", data)
    write_new(output / "audit_summary.json", summary)
    write_new(output / "sequence_manifest.json", {"data_path": str((output / "sequences.json").resolve()),
              "sha256": digest(output / "sequences.json"), "summary": summary})
    return summary


def prepare_first_collision(collection, output):
    """Opt-in competing-event dataset; preserves original unknown outcomes."""
    collection, output = Path(collection), Path(output)
    if output.exists():
        raise ValueError("Use a fresh first-collision output")
    by_mode, _, seen, target = _read_collection(collection, allow_censored=True)
    protocol = read_json(collection / "collection_protocol.json")
    scenes = sorted((s for s in protocol["config"]["scenarios"] if s["id"] in protocol["scenario_ids"]), key=lambda s: s["id"])
    experiment = {"natural_target_sha256": target, "event_definition": FIRST_EVENT_DEFINITION,
                  "initial_distribution": {"kind": "uniform_configured_scenes", "scenarios": scenes}}
    by_mode = {mode: [with_first_event(s) for s in records] for mode, records in by_mode.items()}
    stats = {}
    for mode, records in by_mode.items():
        # The copy is only a view for the existing IS arithmetic, never a stored
        # full-horizon label. Competing outcomes still have collision_result=None.
        view = [dict(s, collision_result=s["event_result"]) for s in records]
        stats[mode] = weighted_diagnostics(view)
        stats[mode]["conditional_weighted_cav_first_collision_mean"] = stats[mode].pop("conditional_weighted_collision_mean")
        if "log_conditional_weighted_collision_mean" in stats[mode]:
            stats[mode]["log_conditional_weighted_cav_first_collision_mean"] = stats[mode].pop("log_conditional_weighted_collision_mean")
        stats[mode]["endpoint_counts"] = dict(Counter(s["event_type"] for s in records))
    behavior = by_mode["behavior"]
    data = {"schema_version": 1, "contract": FIRST_SEQUENCE,
            "natural_target_sha256": target, "observation_contract": OBSERVATION_CONTRACT,
            "event_definition": FIRST_EVENT_DEFINITION, "experiment_target": experiment,
            "experiment_target_sha256": target_digest(experiment),
            "collection_protocol_sha256": digest(collection / "collection_protocol.json"),
            "scope": "CAV involvement at the first collision within 8s, not full-horizon secondary-collision risk",
            "sequences": behavior}
    summary = {"audit_passed": True, "episode_count": len(seen), "by_mode": stats,
               "contract": FIRST_SEQUENCE, "event_definition": FIRST_EVENT_DEFINITION,
               "natural_target_sha256": target, "experiment_target_sha256": data["experiment_target_sha256"],
               "behavior_episode_count": len(behavior),
               "behavior_critical_decisions": sum(len(s["steps"]) for s in behavior),
               "nonempty_sequences": sum(bool(s["steps"]) for s in behavior),
               "empty_sequences_retained": sum(not s["steps"] for s in behavior),
               "full_horizon_unknown_outcomes_retained": sum(s["collision_result"] is None for s in behavior),
               "maximum_probability_error": max(s["max_probability_error"] for seq in by_mode.values() for s in seq),
               "maximum_sequence_log_weight_error": max(s["max_sequence_log_weight_error"] for seq in by_mode.values() for s in seq),
               "formal_generalization_evidence": False}
    write_new(output / "sequences.json", data)
    write_new(output / "audit_summary.json", summary)
    write_new(output / "sequence_manifest.json", {"data_path": str((output / "sequences.json").resolve()),
              "sha256": digest(output / "sequences.json"), "summary": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--audit_only", action="store_true", help="Retain censored prefixes without creating a training manifest")
    endpoint.add_argument("--cav_first_collision", action="store_true", help="Explicit 8s first-collision event; BV-only is a competing outcome")
    args = parser.parse_args()
    action = audit_collection if args.audit_only else prepare_first_collision if args.cav_first_collision else prepare
    print(json.dumps(action(args.collection, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
