"""Frozen dynamic-pair handoff: verify, collect, train and online evaluate.

Run in a dedicated process: path rebinding and the online provider replacement
are process-local. The frozen P, matching and executor files are never patched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

from .highd_portable import (ROOT, PortableAssets, RestoredPolicy, check_sumo,
                             digest, read, write_new, policy_provider_class)


class DynamicAssets(PortableAssets):
    def __init__(self, root=ROOT, assets=None, dynamic_assets=None):
        super().__init__(root, assets)
        self.dynamic_root = Path(dynamic_assets).resolve() if dynamic_assets else self.root / "dynamic_assets"
        lock = read(self.root / "configs/highd_dynamic_assets.lock.json")
        manifest_path = self.dynamic_root / "manifest.json"
        if digest(manifest_path) != lock["private_manifest_sha256"]:
            raise ValueError("Dynamic assets manifest differs from its lock")
        self.dynamic = read(manifest_path)
        self.dynamic_original = self.dynamic["source_root"].replace("\\", "/").rstrip("/")
        # Only default replay/checkpoint selection changes; base asset file maps
        # and frozen hashes remain intact and are verified separately.
        self.manifest = dict(self.manifest, sequence_manifest=self.dynamic["sequence_manifest"],
                             training_root=self.dynamic["training_root"])

    def resolve(self, value):
        text = str(value).replace("\\", "/")
        if hasattr(self, "dynamic"):
            if text.lower().startswith(self.dynamic_original.lower() + "/"):
                text = text[len(self.dynamic_original) + 1:]
            if text in self.dynamic["files"]:
                return str(self.dynamic_root / "files" / PurePosixPath(text))
            # Rebind public paths recorded on the source machine as well.
            if str(value).replace("\\", "/").lower().startswith(self.dynamic_original.lower() + "/"):
                return super().resolve(text)
        return super().resolve(value)

    def translate(self, obj):
        if isinstance(obj, dict):
            return {k: self.translate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.translate(v) for v in obj]
        if isinstance(obj, str):
            text = obj.replace("\\", "/")
            if text in self.dynamic["files"] or text.lower().startswith(self.dynamic_original.lower() + "/"):
                return self.resolve(obj)
        return super().translate(obj)

    def verify(self):
        result = super().verify()
        for rel, info in self.dynamic["files"].items():
            if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts or ":" in rel:
                raise ValueError("Unsafe dynamic asset path")
            if digest(self.resolve(rel)) != info["sha256"]:
                raise ValueError("Dynamic asset checksum mismatch: " + rel)
        contract = read(self.root / "configs/highd_dynamic_pairs_freeze_v1.json")
        for rel, expected in contract["pins"].items():
            if digest(self.root / rel) != expected:
                raise ValueError("Dynamic reference changed; create a new target: " + rel)
        if self.dynamic["natural_target_sha256"] != contract["natural_target_sha256"]:
            raise ValueError("Dynamic training pool belongs to another natural target")
        result.update(dynamic_assets_verified=len(self.dynamic["files"]),
                      natural_target_sha256=contract["natural_target_sha256"])
        return result


def target_identity(assets, extension_path, scenario_ids=None):
    from .highd_dual_ndd_behavior import frozen_config
    from .highd_dynamic_pairs import PARTITION_VERSION, INTERVENTION_VERSION
    from d2rl_training.highd_event_contract import FIRST_EVENT_DEFINITION, target_digest
    extension = read(extension_path)
    spec = assets.read(extension["base_collection_config"])
    base = frozen_config(spec["freeze_contract"])
    target = {"bundle_sha256": digest(base["bundle_manifest"]),
              "runtime_code_sha256": digest(ROOT / "scenario_reconstruction/highd_dual_ndd_runtime.py"),
              "pilot_code_sha256": digest(ROOT / "scenario_reconstruction/highd_dynamic_pairs_pilot.py"),
              "partition_contract": {"version": PARTITION_VERSION, "maximum_pairs": extension["maximum_pairs"],
                                     "intervention_version": INTERVENTION_VERSION},
              "dynamic_runtime_sha256": digest(ROOT / "scenario_reconstruction/highd_dynamic_pairs.py"),
              "map_sha256": digest(ROOT / base["sumo_net"]),
              "frozen_executor_sha256": digest(ROOT / "scenario_reconstruction/highd_dual_ndd_pilot.py")}
    target.update({k: base.get(k) for k in ("relation_mode", "lateral_acceleration_mps2", "use_motion_model", "lateral_duration_diagnostic_s")})
    natural = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
    scenes = sorted((s for s in spec["scenarios"] if scenario_ids is None or s["id"] in scenario_ids), key=lambda s: s["id"])
    if not scenes or (scenario_ids is not None and set(scenario_ids) != {s["id"] for s in scenes}):
        raise ValueError("Unknown/empty scenario selection")
    experiment = {"natural_target_sha256": natural, "event_definition": FIRST_EVENT_DEFINITION,
                  "initial_distribution": {"kind": "uniform_configured_scenes", "scenarios": scenes}}
    if spec["duration_s"] != FIRST_EVENT_DEFINITION["horizon_s"]:
        raise ValueError("Changed first-collision horizon")
    contract = read(ROOT / "configs/highd_dynamic_pairs_freeze_v1.json")
    if natural != contract["natural_target_sha256"]:
        raise ValueError("Current runtime differs from frozen dynamic target")
    return natural, target_digest(experiment)


def validate_policy_target(protocol, natural, experiment, smoke=False):
    if protocol["natural_target_sha256"] != natural:
        raise ValueError("Checkpoint natural target differs; old single-pair policies cannot be evaluated here")
    same = protocol["experiment_target_sha256"] == experiment
    if not same and not smoke:
        raise ValueError("Initial scene mixture differs; use all scenes or explicitly label --smoke")
    return same


def audit_collection(collection, output, assets):
    from .highd_dynamic_pairs import audit_dynamic_decision, pair_key
    from .highd_critical_sequences import prepare_first_collision
    collection = Path(collection)
    summary = assets.read(collection / "collection_summary.json")
    for run in summary["runs"]:
        previous = set()
        for decision in assets.read(run["episode_path"])["decisions"]:
            if set(map(tuple, decision["previous_pairs"])) != previous:
                raise ValueError("Pair history changed within an episode")
            audit_dynamic_decision(decision)
            previous = {pair_key(c) for c in decision["selected_pairs"]}
    return prepare_first_collision(collection, output)


def evaluate(assets, args):
    from . import highd_dual_ndd_criticality as criticality
    from .highd_dynamic_pairs_collect import collect
    import numpy as np
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Use a fresh evaluation output")
    natural, experiment = target_identity(assets, args.config, args.scenarios)
    folder = Path(args.training_root).resolve() if args.training_root else Path(assets.resolve(assets.manifest["training_root"] + "/training_protocol.json")).parent
    protocol = read(folder / "training_protocol.json")
    same = validate_policy_target(protocol, natural, experiment, args.smoke)
    # Check target before importing Ray or running any simulation.
    policy = RestoredPolicy(assets, args.training_root)
    original = criticality.CAVCriticalityProvider
    try:
        metadata = {"mode": "dynamic_pairs_deterministic_ppo", "natural_target_sha256": natural,
                    "experiment_target_sha256": experiment, "same_initial_mixture": same,
                    "smoke_only": args.smoke, "adapter_sha256": digest(__file__),
                    "checkpoint_sha256": digest(policy.checkpoint), "training_protocol_sha256": digest(folder / "training_protocol.json"),
                    "formal_performance_evidence": False}
        write_new(output / "policy_protocol.json", metadata)
        # Frozen matching scores depend on P and challenge, never policy epsilon.
        # This replacement changes only Q and is restored even on exceptions.
        criticality.CAVCriticalityProvider = policy_provider_class(policy)
        collection = collect(args.config, output / "collection", args.repeats, args.seed, args.scenarios, ("behavior",))
        criticality.CAVCriticalityProvider = original
        if collection["natural_target_sha256"] != natural:
            raise ValueError("Actual rollout natural target differs from preflight")
        summary = audit_collection(output / "collection", output / "audit", assets)
        checked, error = 0, 0.
        for run in collection["runs"]:
            for decision in read(run["episode_path"])["decisions"]:
                for unit in decision["units"]:
                    if "proposal_components" in unit:
                        predicted = policy(unit["training_observation"]["joint"])
                        error = max(error, float(np.max(np.abs(predicted - unit["proposal_components"]["epsilon"]))))
                        checked += 1
        if error > 1e-7:
            raise ValueError("Logged epsilon differs from restored policy")
        result = {**metadata, "summary": summary["by_mode"]["behavior"],
                  "checked_policy_decisions": checked, "max_policy_reconstruction_error": error,
                  "probability_audit_passed": summary["audit_passed"]}
        write_new(output / "closed_loop_summary.json", result)
        return result
    finally:
        criticality.CAVCriticalityProvider = original
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets")
    parser.add_argument("--dynamic-assets")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--sumo", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--output", required=True)
    train.add_argument("--iterations", type=int, default=2)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--sequence-manifest")
    train.add_argument("--config", default="configs/highd_dual_bv_meanprecision_ppo_v1.json")
    for name in ("collect", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--config", default="configs/highd_dynamic_pairs_v1.json")
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--seed", type=int, default=2000)
        p.add_argument("--scenarios", nargs="+")
        if name == "evaluate":
            p.add_argument("--training-root")
            p.add_argument("--smoke", action="store_true")
        else:
            p.add_argument("--modes", nargs="+", choices=("natural", "behavior"), default=["natural", "behavior"])
    prep = sub.add_parser("prepare")
    prep.add_argument("--collection", required=True)
    prep.add_argument("--output", required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    # All relative operation paths are relative to the delivered project root.
    assets = DynamicAssets(assets=args.assets, dynamic_assets=args.dynamic_assets)
    os.chdir(ROOT)
    result = assets.verify()
    assets.install()
    target_identity(assets, "configs/highd_dynamic_pairs_v1.json")
    if args.command == "verify":
        if args.sumo:
            result.update(check_sumo())
        result.update(sequence_manifest=str(assets.sequence_manifest()), research_reference_only=True)
    elif args.command == "train":
        from .highd_critical_sequence_train import main as train_main
        manifest = args.sequence_manifest or str(assets.sequence_manifest())
        spec = read(manifest)
        data = read(spec["data_path"])
        if data["natural_target_sha256"] != result["natural_target_sha256"]:
            raise ValueError("Training manifest belongs to another NDD target")
        sys.argv = [sys.argv[0], "--sequence_manifest", manifest, "--output", args.output,
                    "--iterations", str(args.iterations), "--seed", str(args.seed), "--config", args.config]
        train_main()
        return
    elif args.command == "prepare":
        result = audit_collection(args.collection, args.output, assets)
    else:
        check_sumo()
        if args.command == "evaluate":
            result = evaluate(assets, args)
        else:
            from .highd_dynamic_pairs_collect import collect
            result = collect(args.config, args.output, args.repeats, args.seed, args.scenarios, args.modes)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
