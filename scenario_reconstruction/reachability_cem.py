"""Bounded CEM reachability diagnostic for the timed-action trajectory space.

It is an optimizer upper-bound diagnostic, not D2RL, a collision-rate estimator,
or a proof that a collision is globally impossible when none is found.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time as clock

import numpy as np

from .conflict_diagnostics import _load_scene
from .highd import file_hash, validate_config as validate_highd_config, write_json
from .trajectory_benchmark import (evaluate, ranking, scene_seed,
                                   validate_benchmark_config)


ALLOWED_SPLITS = ("train", "validation")
GROUPS = ("all", "high_conflict")


def validate_cem_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Only reachability CEM schema_version 1 is supported")
    if not isinstance(config.get("seed"), int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must lie in [0, 2**32)")
    tiers = config.get("budget_tiers")
    if (not isinstance(tiers, list) or not tiers or any(isinstance(value, bool) or int(value) != value
            for value in tiers) or any(value < 1 for value in tiers) or tiers != sorted(set(tiers))):
        raise ValueError("budget_tiers must be increasing unique positive integers")
    population = config.get("population_size")
    if isinstance(population, bool) or int(population) != population or population < 2:
        raise ValueError("population_size must be an integer of at least two")
    fraction = config.get("elite_fraction")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("elite_fraction must lie in (0, 1]")
    for name in ("initial_std_fraction", "minimum_std_fraction"):
        if not np.isfinite(config.get(name)) or not 0 < config[name] <= 1:
            raise ValueError(f"{name} must lie in (0, 1]")
    if config["minimum_std_fraction"] > config["initial_std_fraction"]:
        raise ValueError("minimum_std_fraction cannot exceed initial_std_fraction")


def action_bounds(trajectory_config):
    lower = np.array([-trajectory_config["max_longitudinal_action_delta_mps2"],
                      -trajectory_config["max_lateral_action_delta_mps2"],
                      0.0, trajectory_config["min_action_hold_s"]])
    upper = np.array([trajectory_config["max_longitudinal_action_delta_mps2"],
                      trajectory_config["max_lateral_action_delta_mps2"],
                      trajectory_config["max_action_start_delay_s"],
                      trajectory_config["max_action_hold_s"]])
    reference = np.array([0.0, 0.0, 0.0, trajectory_config["min_action_hold_s"]])
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
        raise ValueError("Invalid timed-action bounds")
    return lower, upper, reference


def _score(result):
    """A scalar only for CEM sampling; final choice uses the public lexicographic rank."""
    if not result["feasible"]:
        return 1e9 + 1e6 * len(result["violations"])
    if result["target_future_collision"]:
        return -1e6 + 1e-3 * result["parameter_l2"]
    if result["near_miss"]:
        return -1e3 + result["parameter_l2"]
    return result["target_future_min_clearance_m"] + 1e-3 * result["parameter_l2"]


def _snapshot(best, state, tier):
    return {
        "budget": tier,
        "evaluations": state["evaluations"],
        "feasible_candidates": state["feasible_candidates"],
        "target_future_collision_candidates": state["collision_candidates"],
        "feasible_target_future_collision_candidates": state["feasible_collision_candidates"],
        "feasible_near_miss_candidates": state["near_miss_candidates"],
        "first_feasible_target_collision_evaluation": state["first_collision_evaluation"],
        "violation_counts": dict(state["violations"]),
        "selected": copy.deepcopy(best),
    }


def cem_reachability(scene, trajectory_config, cem_config, seed):
    """Run one deterministic CEM trajectory and retain exact cumulative budget snapshots."""
    validate_benchmark_config(trajectory_config)
    validate_cem_config(cem_config)
    lower, upper, reference = action_bounds(trajectory_config)
    started = clock.perf_counter()
    tiers = list(cem_config["budget_tiers"])
    rng = np.random.default_rng(seed)
    span = upper - lower
    mean = (lower + upper) / 2
    std = span * cem_config["initial_std_fraction"]
    minimum_std = span * cem_config["minimum_std_fraction"]
    state = {"evaluations": 0, "feasible_candidates": 0, "collision_candidates": 0,
             "feasible_collision_candidates": 0, "near_miss_candidates": 0,
             "first_collision_evaluation": None, "violations": Counter()}
    best, snapshots = None, []

    def evaluate_candidate(params):
        nonlocal best
        result = evaluate(scene, params, trajectory_config, "action_delta")
        result["cem_score"] = float(_score(result))
        state["evaluations"] += 1
        state["feasible_candidates"] += int(result["feasible"])
        state["collision_candidates"] += int(result["target_future_collision"])
        state["feasible_collision_candidates"] += int(
            result["feasible"] and result["target_future_collision"])
        state["near_miss_candidates"] += int(result["feasible"] and result["near_miss"])
        state["violations"].update(result["violations"])
        if result["feasible"] and result["target_future_collision"] and state["first_collision_evaluation"] is None:
            state["first_collision_evaluation"] = state["evaluations"]
        if result["feasible"] and (best is None or ranking(result, trajectory_config) < ranking(best, trajectory_config)):
            best = result
        return result

    evaluate_candidate(reference)
    while state["evaluations"] < tiers[-1]:
        next_tier = next(tier for tier in tiers if tier > state["evaluations"])
        count = min(cem_config["population_size"], next_tier - state["evaluations"])
        samples = np.clip(rng.normal(mean, std, size=(count, len(lower))), lower, upper)
        population = [evaluate_candidate(sample) for sample in samples]
        feasible = [result for result in population if result["feasible"]]
        eligible = feasible if feasible else population
        elite_count = max(1, int(np.ceil(len(eligible) * cem_config["elite_fraction"])))
        elite = sorted(eligible, key=lambda result: result["cem_score"])[:elite_count]
        elite_params = np.asarray([result["params"] for result in elite])
        mean = np.mean(elite_params, axis=0)
        std = np.maximum(np.std(elite_params, axis=0), minimum_std)
        if state["evaluations"] == next_tier:
            snapshots.append(_snapshot(best, state, next_tier))
    return {
        "method": "timed_action_cem_reachability", "seed": seed,
        "parameterization": "action_delta",
        "parameter_names": ["ax_mps2", "ay_mps2", "start_delay_s", "hold_s"],
        "parameter_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "snapshots": snapshots,
        "elapsed_s": clock.perf_counter() - started,
    }


def _load_selected_ids(diagnostic_summary, manifest_sha256):
    diagnostic_summary = Path(diagnostic_summary)
    summary = json.loads(diagnostic_summary.read_text(encoding="utf-8"))
    if summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Diagnostic summary was generated from a different manifest")
    identifiers = summary.get("selected_scene_ids")
    if not isinstance(identifiers, list) or not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Diagnostic summary needs unique non-empty selected_scene_ids")
    return set(identifiers), file_hash(diagnostic_summary)


def _summary(records, cem_config, trajectory_config, manifest_sha256, group, diagnostic_sha256):
    tiers = cem_config["budget_tiers"]
    result = {
        "status": ("bounded CEM reachability diagnostic; no success is not a global impossibility proof, "
                   "and collisions are not a real-world crash-rate estimate"),
        "group": group, "scene_count": len(records),
        "recording_count": len({record["recording_id"] for record in records}),
        "cem_config": cem_config, "trajectory_config": trajectory_config,
        "manifest_sha256": manifest_sha256, "diagnostic_summary_sha256": diagnostic_sha256,
        "budget_tiers": {},
    }
    for tier in tiers:
        snapshots = [next(snapshot for snapshot in record["snapshots"] if snapshot["budget"] == tier)
                     for record in records]
        selected = [snapshot["selected"] for snapshot in snapshots if snapshot["selected"] is not None]
        firsts = [snapshot["first_feasible_target_collision_evaluation"] for snapshot in snapshots
                  if snapshot["first_feasible_target_collision_evaluation"] is not None]
        result["budget_tiers"][str(tier)] = {
            "attempted_scenes": len(snapshots),
            "scenes_with_feasible_candidate": len(selected),
            "target_future_collision_scenes": sum(
                item["target_future_collision"] for item in selected),
            "near_miss_scenes": sum(item["near_miss"] for item in selected),
            "scenes_with_any_feasible_target_collision_candidate": sum(
                snapshot["feasible_target_future_collision_candidates"] > 0 for snapshot in snapshots),
            "scenes_with_any_feasible_near_miss_candidate": sum(
                snapshot["feasible_near_miss_candidates"] > 0 for snapshot in snapshots),
            "no_feasible_collision_found_within_budget_scenes": sum(
                snapshot["feasible_target_future_collision_candidates"] == 0 for snapshot in snapshots),
            "feasible_candidates": sum(snapshot["feasible_candidates"] for snapshot in snapshots),
            "candidate_evaluations": sum(snapshot["evaluations"] for snapshot in snapshots),
            "mean_selected_target_future_clearance_m": (float(np.mean(
                [item["target_future_min_clearance_m"] for item in selected])) if selected else None),
            "mean_selected_controlled_actor_ade_m": (float(np.mean(
                [item["controlled_actor_ade_m"] for item in selected])) if selected else None),
            "mean_first_feasible_target_collision_evaluation": float(np.mean(firsts)) if firsts else None,
            "violation_counts": dict(sum((Counter(snapshot["violation_counts"]) for snapshot in snapshots), Counter())),
        }
    return result


def run_reachability(manifest_path, output, trajectory_config, cem_config, split, group="all",
                     diagnostic_summary=None):
    if split not in ALLOWED_SPLITS:
        raise ValueError("Reachability diagnostics only allow train and validation; test remains locked")
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    validate_benchmark_config(trajectory_config)
    validate_cem_config(cem_config)
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_highd_config(manifest["config"])
    manifest_sha256 = file_hash(manifest_path)
    selected_ids, diagnostic_sha256 = None, None
    if group == "high_conflict":
        if diagnostic_summary is None:
            raise ValueError("high_conflict group requires diagnostic_summary")
        selected_ids, diagnostic_sha256 = _load_selected_ids(diagnostic_summary, manifest_sha256)
    items = [item for item in manifest["scenes"] if item["split"] == split]
    if group == "high_conflict":
        items = [item for item in items if item["scene_id"] in selected_ids]
    if not items:
        raise ValueError("No scenes match requested split and group")
    records = []
    for item in items:
        scene = _load_scene(manifest_path, manifest, item, split)
        result = cem_reachability(scene, trajectory_config, cem_config,
                                  scene_seed(scene["scene_id"], cem_config["seed"]))
        records.append({"scene_id": scene["scene_id"], "recording_id": scene["recording_id"],
                        "location_id": scene["location_id"], "split": split,
                        "interaction_stratum": scene["interaction"]["stratum"],
                        "snapshots": result["snapshots"]})
        print(f"{len(records)}/{len(items)}: {scene['scene_id']}", flush=True)
    summary = _summary(records, cem_config, trajectory_config, manifest_sha256, group, diagnostic_sha256)
    write_json(output / "results.json", records)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trajectory_config", default="configs/trajectory_baselines.json")
    parser.add_argument("--cem_config", default="configs/reachability_cem.json")
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--group", choices=GROUPS, default="all")
    parser.add_argument("--diagnostic_summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_reachability(
        args.manifest, args.output,
        json.loads(Path(args.trajectory_config).read_text(encoding="utf-8")),
        json.loads(Path(args.cem_config).read_text(encoding="utf-8")), args.split,
        args.group, args.diagnostic_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
