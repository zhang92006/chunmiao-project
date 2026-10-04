"""Collect fresh CAV-labeled episodes from the frozen dual-BV research target.

Preserves all failures/censoring; exports no made-up safe/crash labels or replay
sampling weights. Critical-sequence training is a separate next stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path

from .highd_dual_ndd_bundle import DualNDDBundle
from .highd_dual_ndd_pilot import run
from .highd_pair_data_inventory import digest, read_json, write_new


def frozen_config(path):
    contract = read_json(path)
    for name, expected in contract["pins"].items():
        if digest(name) != expected:
            raise ValueError(f"Frozen reference changed: {name}; version it explicitly before collecting")
    # Follow existing artifact provenance rather than merely trusting a status.
    DualNDDBundle(contract["bundle_manifest"])
    config = read_json(contract["pilot_config"])
    if Path(config["bundle_manifest"]).resolve() != Path(contract["bundle_manifest"]).resolve():
        raise ValueError("Pilot points to another bundle")
    if config["maximum_pairs"] != 1 or config["cav_id"] != 100:
        raise ValueError("Pilot selection/CAV contract changed")
    return config


def collect(config_path, output, repeats=3, seed=7, scenario_ids=None, modes=("natural", "behavior")):
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    spec = read_json(config_path)
    base = frozen_config(spec["freeze_contract"])
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh OutputRoot; existing episodes are preserved")
    selected = [s for s in spec["scenarios"] if scenario_ids is None or s["id"] in scenario_ids]
    if not selected or (scenario_ids is not None and set(scenario_ids) != {s["id"] for s in selected}):
        raise ValueError("Unknown or empty scenario selection")
    if not modes or not set(modes).issubset({"natural", "behavior"}):
        raise ValueError("Unknown behavior mode")
    write_new(output / "collection_protocol.json", {
        "config": spec, "config_sha256": digest(config_path), "freeze_sha256": digest(spec["freeze_contract"]),
        "freeze_contract": read_json(spec["freeze_contract"]),
        "collector_code_sha256": digest(__file__), "repeats": repeats, "seed": seed,
        "scenario_ids": [s["id"] for s in selected], "modes": list(modes),
        "criticality_code_sha256": digest(Path(__file__).with_name("highd_dual_ndd_criticality.py")),
        "proposal_code_sha256": digest(Path(__file__).with_name("highd_dual_ndd_proposal.py")),
        "scope": "Configured diagnostic scene mixture only; no formal risk or D2RL improvement claim"})
    runs = []
    for scene in selected:
        for repetition in range(repeats):
            for mode in modes:
                episode_config = copy.deepcopy(base)
                episode_config.update(actors=scene["actors"], duration_s=spec["duration_s"],
                                      scenario_id=scene["id"])
                folder = output / scene["id"] / f"seed{seed + repetition}_{mode}"
                result = run(folder, episode_config, seed=seed + repetition,
                    proposal_epsilon=spec["epsilon_natural_mass"] if mode == "behavior" else None,
                    proposal_mode="cav_criticality_v1", criticality_config=spec["criticality"])
                runs.append({"scenario_id": scene["id"], "mode": mode, "seed": seed + repetition,
                    "episode_path": str(folder / "episode.json"), "summary_path": str(folder / "pilot_summary.json"),
                    "termination": result["termination"], "raw_cav_collision": result["raw_cav_collision"],
                    "d2rl_replay_path": result["d2rl_replay_path"],
                    "critical_steps": result["conditional_proposal_decisions"],
                    "max_probability_error": result["max_probability_reconstruction_error"]})
    summaries = {}
    for mode in modes:
        subset = [r for r in runs if r["mode"] == mode]
        summaries[mode] = {
            "attempted": len(subset), "raw_cav_crashes": sum(r["raw_cav_collision"] for r in subset),
            "bv_only_collision_censored": sum(r["termination"] == "collision" and not r["raw_cav_collision"] for r in subset),
            "termination_counts": dict(Counter(r["termination"] for r in subset)),
            "eligible_replay_crashes": sum(bool(r["d2rl_replay_path"]) and r["raw_cav_collision"] for r in subset),
            "eligible_replay_safe": sum(bool(r["d2rl_replay_path"]) and r["termination"] == "duration_reached" for r in subset),
            "critical_steps": sum(r["critical_steps"] for r in subset),
            "max_probability_error": max(r["max_probability_error"] for r in subset)}
    result = {"schema_version": 1, "by_mode": summaries, "runs": runs,
              "formal_training_ready": False,
              "next": "Inspect coverage, CAV outcomes and censoring; implement full critical-sequence replay before formal PPO"}
    write_new(output / "collection_summary.json", result)
    return result


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/highd_dual_ndd_behavior_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", choices=("natural", "behavior"), default=["natural", "behavior"])
    args = parser.parse_args()
    result = collect(args.config, args.output, args.repeats, args.seed, args.scenarios, args.modes)
    print(json.dumps(result["by_mode"], indent=2), flush=True)


if __name__ == "__main__":
    main()
