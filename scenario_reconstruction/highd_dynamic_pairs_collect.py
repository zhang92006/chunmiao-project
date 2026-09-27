"""Portable, opt-in collection for the dynamic multi-pair research target."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path

from .highd_portable import PortableAssets, ROOT, check_sumo, read, write_new, digest


def collect(config_path, output, repeats=1, seed=7, scenario_ids=None, modes=("natural", "behavior")):
    # Consumers must be imported AFTER PortableAssets.install rebinds paths.
    from .highd_dual_ndd_behavior import frozen_config
    from .highd_dynamic_pairs import PARTITION_VERSION, INTERVENTION_VERSION, partition
    from .highd_dynamic_pairs_pilot import run
    from .highd_pair_data_inventory import read_json
    extension = read(config_path)
    if (extension["partition_version"] != PARTITION_VERSION or
            extension["intervention_version"] != INTERVENTION_VERSION or
            extension["maximum_intervened_pairs"] != 1):
        raise ValueError("Unknown dynamic partition/intervention contract")
    partition([], maximum_pairs=extension["maximum_pairs"])
    if type(repeats) is not int or repeats < 1 or not modes or not set(modes).issubset({"natural", "behavior"}):
        raise ValueError("Invalid repeats or modes")
    if len(set(modes)) != len(modes):
        raise ValueError("Duplicate modes")
    spec = read_json(extension["base_collection_config"])
    base = frozen_config(spec["freeze_contract"])
    base["maximum_pairs"] = extension["maximum_pairs"]
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output; existing data is preserved")
    selected = [s for s in spec["scenarios"] if scenario_ids is None or s["id"] in scenario_ids]
    if not selected or (scenario_ids is not None and set(scenario_ids) != {s["id"] for s in selected}):
        raise ValueError("Unknown/empty scene selection")
    write_new(output / "collection_protocol.json", {
        "config": spec, "config_sha256": digest(extension["base_collection_config"]),
        "freeze_contract": read_json(spec["freeze_contract"]), "freeze_sha256": digest(spec["freeze_contract"]),
        "extension": extension, "extension_sha256": digest(config_path),
        "collector_code_sha256": digest(__file__), "repeats": repeats, "seed": seed,
        "scenario_ids": [s["id"] for s in selected], "modes": list(modes),
        "scope": "New dynamic natural target; do not merge old single-pair trajectories or weights"})
    runs = []
    for scenario in selected:
        for repeat in range(repeats):
            for mode in modes:
                config = copy.deepcopy(base)
                config.update(actors=scenario["actors"], duration_s=spec["duration_s"], scenario_id=scenario["id"])
                folder = output / scenario["id"] / f"seed{seed + repeat}_{mode}"
                result = run(folder, config, seed=seed + repeat,
                             proposal_epsilon=spec["epsilon_natural_mass"] if mode == "behavior" else None,
                             criticality_config=spec["criticality"])
                runs.append({"scenario_id": scenario["id"], "seed": seed + repeat, "mode": mode,
                             "episode_path": str(folder / "episode.json"), "summary_path": str(folder / "pilot_summary.json"),
                             "termination": result["termination"], "raw_cav_collision": result["raw_cav_collision"],
                             "d2rl_replay_path": result["d2rl_replay_path"],
                             "critical_steps": result["conditional_proposal_decisions"],
                             "max_probability_error": result["max_probability_reconstruction_error"],
                             "natural_target_sha256": result["natural_target_sha256"],
                             "pair_count_histogram": result["pair_count_histogram"],
                             "intervention_count_histogram": result["intervention_count_histogram"],
                             "reconfiguration_decisions": result["reconfiguration_decisions"]})
    targets = {r["natural_target_sha256"] for r in runs}
    if len(targets) != 1:
        raise ValueError("Collection mixes natural targets")
    summary = {"schema_version": 1, "natural_target_sha256": targets.pop(), "runs": runs,
               "by_mode": {}, "formal_training_ready": False,
               "scope": "Technical dynamic-pair rollout evidence, not realism or policy efficiency"}
    for mode in modes:
        subset = [r for r in runs if r["mode"] == mode]
        pairs, interventions = Counter(), Counter()
        for row in subset:
            pairs.update(row["pair_count_histogram"])
            interventions.update(row["intervention_count_histogram"])
        summary["by_mode"][mode] = {
            "attempted": len(subset), "raw_cav_crashes": sum(r["raw_cav_collision"] for r in subset),
            "termination_counts": dict(Counter(r["termination"] for r in subset)),
            "pair_count_histogram": dict(pairs), "intervention_count_histogram": dict(interventions),
            "reconfiguration_decisions": sum(r["reconfiguration_decisions"] for r in subset),
            "max_probability_error": max(r["max_probability_error"] for r in subset)}
    write_new(output / "collection_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets")
    parser.add_argument("--config", default="configs/highd_dynamic_pairs_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("natural", "behavior"), default=["natural", "behavior"])
    parser.add_argument("--prepare", action="store_true", help="Audit and export fresh CAV-first sequences")
    args = parser.parse_args()
    # Resolve caller-relative paths before moving to the project root.
    output = Path(args.output).resolve()
    assets = PortableAssets(assets=args.assets)
    os.chdir(ROOT)
    assets.verify()
    assets.install()
    check_sumo()
    summary = collect(args.config, output, args.repeats, args.seed, args.scenarios, args.modes)
    if args.prepare:
        from .highd_critical_sequences import prepare_first_collision
        from .highd_dynamic_pairs import audit_dynamic_decision, pair_key
        for run in summary["runs"]:
            previous = set()
            for decision in read(run["episode_path"])["decisions"]:
                if set(map(tuple, decision["previous_pairs"])) != previous:
                    raise ValueError("Pair history was reset or changed inside an episode")
                audit_dynamic_decision(decision)
                previous = {pair_key(c) for c in decision["selected_pairs"]}
        summary["sequence_audit"] = prepare_first_collision(output, output / "sequence_audit")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
