"""Portable entry for the frozen two-lane research reference (not NDD acceptance).

Only JSON path resolution is adapted; pinned probability/executor source and
model bytes remain unchanged. Use a fresh process, not concurrent in-process runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_sumo():
    import sumolib
    binary = sumolib.checkBinary("sumo")
    version = subprocess.check_output([binary, "--version"], text=True).splitlines()[0]
    if " 1.27.0" not in version:
        raise ValueError("This reference uses SUMO 1.27.0; do not silently mix simulator versions: " + version)
    return {"sumo_binary": binary, "sumo_version": version}


class PortableAssets:
    def __init__(self, root=ROOT, assets=None):
        self.root = Path(root).resolve()
        self.assets = Path(assets).resolve() if assets else self.root / "portable_assets"
        self.lock = read(self.root / "configs/highd_portable_assets.lock.json")
        manifest_path = self.assets / "manifest.json"
        if digest(manifest_path) != self.lock["private_manifest_sha256"]:
            raise ValueError("Private assets manifest differs from the published lock")
        self.manifest = read(manifest_path)
        self.original_root = self.manifest["source_root"].replace("\\", "/").rstrip("/")
        self.files = self.manifest["files"]

    def resolve(self, value):
        text = str(value).replace("\\", "/")
        if text.lower().startswith(self.original_root.lower() + "/"):
            text = text[len(self.original_root) + 1:]
        if text in self.files:
            return str(self.assets / "files" / PurePosixPath(text))
        if not Path(text).is_absolute() and ":" not in text:
            return str(self.root / text)
        return str(value)

    def translate(self, obj):
        if isinstance(obj, dict):
            return {k: self.translate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.translate(v) for v in obj]
        if isinstance(obj, str):
            text = obj.replace("\\", "/")
            if text in self.files or text.lower().startswith(self.original_root.lower() + "/"):
                return self.resolve(obj)
        return obj

    def read(self, path):
        return self.translate(read(self.resolve(path)))

    def verify(self):
        for rel, spec in self.files.items():
            if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts or ":" in rel:
                raise ValueError("Unsafe asset path")
            if digest(self.resolve(rel)) != spec["sha256"]:
                raise ValueError("Asset checksum mismatch: " + rel)
        for rel, expected in self.manifest["frozen_source_hashes"].items():
            if digest(self.root / rel) != expected:
                raise ValueError("Frozen source bytes changed: " + rel)
        return {"assets_verified": len(self.files),
                "source_files_verified": len(self.manifest["frozen_source_hashes"])}

    def install(self):
        # Import helpers before frozen consumers bind 'from ... import read_json'.
        for name in ("highd_pair_data_inventory", "highd_sequence_io"):
            module = importlib.import_module("scenario_reconstruction." + name)
            module.read_json = self.read
            module.digest = lambda p: digest(self.resolve(p))

    def sequence_manifest(self):
        original = self.manifest["sequence_manifest"]
        local = self.read(original)
        if digest(local["data_path"]) != local["sha256"]:
            raise ValueError("Sequence contents changed")
        target = self.root / "outputs/portable_metadata" / (local["sha256"][:16] + "_sequence_manifest.json")
        if target.exists():
            if read(target) != local:
                raise ValueError("Existing portable manifest differs")
        else:
            write_new(target, local)
        return target


class RestoredPolicy:
    """Trusted RLlib checkpoint restoration with explicit local path rebinding."""
    def __init__(self, assets, training_root=None):
        if sys.version_info[:2] != (3, 9):
            raise RuntimeError("The shared Ray 1.11 checkpoint requires Python 3.9")
        import ray
        from ray.rllib.agents.ppo import PPOTrainer
        from ray.tune.registry import register_env
        from d2rl_training.highd_critical_sequence_env import HighDCriticalSequenceEnv
        from .highd_mean_precision_policy import register_mean_precision_policy, register_legacy_action_distribution

        self.ray, self.trainer = ray, None
        folder = Path(training_root).resolve() if training_root else Path(assets.resolve(assets.manifest["training_root"] + "/training_protocol.json")).parent
        self.protocol = read(folder / "training_protocol.json")
        self.summary = read(folder / "training_summary.json")
        original_manifest = assets.resolve(self.protocol["ppo_config"]["env_config"]["sequence_manifest"])
        if digest(original_manifest) != self.protocol["sequence_manifest_sha256"]:
            raise ValueError("Checkpoint's source manifest changed")
        config = copy.deepcopy(self.protocol["ppo_config"])
        if training_root:
            config["env_config"]["sequence_manifest"] = original_manifest
        else:
            config["env_config"]["sequence_manifest"] = str(assets.sequence_manifest())
        self.env = HighDCriticalSequenceEnv(config["env_config"])
        if self.env.dataset["natural_target_sha256"] != self.protocol["natural_target_sha256"]:
            raise ValueError("Checkpoint natural target differs from replay data")
        policy_file = Path(__file__).with_name("highd_mean_precision_policy.py")
        self.policy_bytes_match = digest(policy_file) == self.protocol["policy_code_sha256"]
        if not self.policy_bytes_match:
            compatibility = read(ROOT / "configs/highd_checkpoint_compatibility_v1.json")
            normalized_hash = hashlib.sha256(policy_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if (self.protocol["policy_code_sha256"] != compatibility["source_policy_sha256"]
                    or normalized_hash != compatibility["packaged_policy_lf_sha256"]
                    or config["model"]["custom_action_dist"] != "highd_mean_precision_beta_v1"):
                raise ValueError("Unknown policy/checkpoint migration; do not bypass version checks")
        register_mean_precision_policy()
        register_legacy_action_distribution()
        register_env(config["env"], lambda cfg: HighDCriticalSequenceEnv(cfg))
        config.update(num_workers=0, num_gpus=0)
        checkpoint = self.summary["checkpoint"]
        # For a moved training directory prefer its own checkpoint subtree.
        suffix = checkpoint.replace("\\", "/").split("/checkpoint/", 1)
        self.checkpoint = str(folder / "checkpoint" / suffix[1]) if len(suffix) == 2 else assets.resolve(checkpoint)
        ray.init(num_cpus=1, num_gpus=0, include_dashboard=False, ignore_reinit_error=True)
        try:
            self.trainer = PPOTrainer(config=config)
            self.trainer.restore(self.checkpoint)
            import numpy as np
            restored = self.env.empirical_objective(self)
            expected = self.summary["trained_deterministic_empirical_scaled_second_moment"]
            if not np.isclose(restored, expected, rtol=1e-6, atol=1e-10):
                raise ValueError("Checkpoint migration failed the recorded full-pool objective check")
        except BaseException:
            self.close()
            raise

    def __call__(self, obs):
        import numpy as np
        value = np.asarray(self.trainer.compute_single_action(obs, explore=False), dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all() or (value < .05).any() or (value > 1.).any():
            raise ValueError("Invalid online epsilon; never clip silently")
        return value

    def close(self):
        if self.trainer is not None:
            self.trainer.stop()
        self.ray.shutdown()


def policy_provider_class(policy):
    from .highd_dual_ndd_criticality import CAVCriticalityProvider
    from .highd_dual_ndd_proposal import ConditionalChainProposal

    class OnlineProvider(CAVCriticalityProvider):
        def __call__(self, scene, candidate, distribution):
            # Compute exactly the frozen risk gate/H. No action has been sampled.
            baseline = super().__call__(scene, candidate, distribution)
            if baseline is None:
                return None
            obs = baseline.training_observation
            epsilon = policy(obs["joint"])
            parts = baseline.components()
            import numpy as np
            h = np.asarray(parts["critical_first"])[:, None] * np.asarray(parts["critical_second_given_first"])
            proposal = ConditionalChainProposal(distribution.natural, h, epsilon)
            proposal.training_observation = obs
            proposal.criticality = baseline.criticality
            self.records[-1].update(policy_epsilon=epsilon.tolist(), policy_observation=obs["joint"])
            return proposal
    return OnlineProvider


def evaluate(assets, args):
    from . import highd_dual_ndd_criticality as criticality
    from .highd_dual_ndd_behavior import collect, frozen_config
    from .highd_critical_sequences import prepare_first_collision
    from d2rl_training.highd_event_contract import FIRST_EVENT_DEFINITION, target_digest
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Use a fresh evaluation output")
    spec = assets.read(args.config)
    base = frozen_config(spec["freeze_contract"])
    freeze = assets.read(spec["freeze_contract"])
    natural = {"bundle_sha256": digest(assets.resolve(freeze["bundle_manifest"])),
               "runtime_code_sha256": digest(ROOT / "scenario_reconstruction/highd_dual_ndd_runtime.py"),
               "pilot_code_sha256": digest(ROOT / "scenario_reconstruction/highd_dual_ndd_pilot.py")}
    natural.update({k: base.get(k) for k in ("relation_mode", "lateral_acceleration_mps2", "use_motion_model", "lateral_duration_diagnostic_s")})
    natural_hash = hashlib.sha256(json.dumps(natural, sort_keys=True).encode()).hexdigest()
    scenes = sorted([s for s in spec["scenarios"] if not args.scenarios or s["id"] in args.scenarios], key=lambda s: s["id"])
    experiment = {"natural_target_sha256": natural_hash, "event_definition": FIRST_EVENT_DEFINITION,
                  "initial_distribution": {"kind": "uniform_configured_scenes", "scenarios": scenes}}
    policy = RestoredPolicy(assets, args.training_root)
    original = criticality.CAVCriticalityProvider
    try:
        if policy.protocol["natural_target_sha256"] != natural_hash or spec["duration_s"] != 8:
            raise ValueError("Policy and simulator target/horizon differ")
        # Subsets are allowed only as explicitly labeled technical smoke runs.
        same_target = policy.protocol["experiment_target_sha256"] == target_digest(experiment)
        if not same_target and not args.smoke:
            raise ValueError("Initial mixture differs from training; use all scenes, or label --smoke")
        root_metadata = {"mode": "deterministic_ppo_closed_loop", "checkpoint_sha256": digest(policy.checkpoint),
                         "training_protocol_sha256": digest(Path(args.training_root).resolve() / "training_protocol.json") if args.training_root else assets.files[assets.manifest["training_root"] + "/training_protocol.json"]["sha256"],
                         "natural_target_sha256": natural_hash, "same_initial_mixture": same_target,
                         "smoke_only": args.smoke, "adapter_sha256": digest(__file__),
                         "event_definition": FIRST_EVENT_DEFINITION, "formal_performance_evidence": False}
        write_new(output / "policy_protocol.json", root_metadata)
        criticality.CAVCriticalityProvider = policy_provider_class(policy)
        collect(args.config, output / "collection", args.repeats, args.seed, args.scenarios, ("behavior",))
        criticality.CAVCriticalityProvider = original
        summary = prepare_first_collision(output / "collection", output / "audit")
        # Replay policy inference from logged online observations, independently
        # of the logged P/Q reconstruction in prepare_first_collision.
        checked, error = 0, 0.
        import numpy as np
        for record in read(output / "collection/collection_summary.json")["runs"]:
            episode = read(record["episode_path"])
            for decision in episode["decisions"]:
                for unit in decision["units"]:
                    if "proposal_components" not in unit:
                        continue
                    predicted = policy(unit["training_observation"]["joint"])
                    error = max(error, float(np.max(np.abs(predicted - unit["proposal_components"]["epsilon"]))))
                    checked += 1
        if error > 1e-7:
            raise ValueError("Online epsilon differs from deterministic checkpoint output")
        result = {**root_metadata, "policy_code_bytes_match_training": policy.policy_bytes_match,
                  "summary": summary["by_mode"]["behavior"],
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
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--sumo", action="store_true")
    collect = sub.add_parser("collect")
    evaluate_parser = sub.add_parser("evaluate")
    for p in (collect, evaluate_parser):
        p.add_argument("--config", default="configs/highd_dual_ndd_behavior_v1.json")
        p.add_argument("--output", required=True)
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--scenarios", nargs="+")
    evaluate_parser.add_argument("--training-root")
    evaluate_parser.add_argument("--smoke", action="store_true")
    collect.add_argument("--modes", nargs="+", choices=("natural", "behavior"), default=["natural", "behavior"])
    prep = sub.add_parser("prepare")
    prep.add_argument("--collection", required=True)
    prep.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--output", required=True)
    train.add_argument("--iterations", type=int, default=2)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--config", default="configs/highd_dual_bv_meanprecision_ppo_v1.json")
    train.add_argument("--sequence-manifest")
    diagnostic = sub.add_parser("diagnose")
    diagnostic.add_argument("--training-root")
    diagnostic.add_argument("--output", required=True)
    args = parser.parse_args()
    # Avoid BLAS oversubscription in this intentionally single-worker harness.
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.chdir(ROOT)
    assets = PortableAssets(assets=args.assets)
    verification = assets.verify()
    assets.install()
    if args.command in ("collect", "evaluate"):
        check_sumo()
    if args.command == "verify":
        from .highd_dual_ndd_behavior import frozen_config
        frozen_config("configs/highd_dual_ndd_research_freeze_v1.json")
        verification.update(frozen_ndd_loaded=True, python=platform.python_version(), platform=platform.platform(),
                            sequence_manifest=str(assets.sequence_manifest()), macos_tested=False)
        if args.sumo:
            verification.update(check_sumo())
        result = verification
    elif args.command == "collect":
        from .highd_dual_ndd_behavior import collect
        result = collect(args.config, args.output, args.repeats, args.seed, args.scenarios, args.modes)["by_mode"]
    elif args.command == "prepare":
        from .highd_critical_sequences import prepare_first_collision
        result = prepare_first_collision(args.collection, args.output)
    elif args.command == "train":
        from .highd_critical_sequence_train import main as train_main
        manifest = args.sequence_manifest or str(assets.sequence_manifest())
        sys.argv = [sys.argv[0], "--sequence_manifest", manifest, "--output", args.output,
                    "--iterations", str(args.iterations), "--seed", str(args.seed), "--config", args.config]
        train_main()
        return
    elif args.command == "evaluate":
        result = evaluate(assets, args)
    else:
        import numpy as np
        output = Path(args.output)
        if output.exists():
            raise ValueError("Use a fresh diagnostic output")
        policy = RestoredPolicy(assets, args.training_root)
        try:
            value = policy.env.empirical_objective(policy)
            expected = policy.summary["trained_deterministic_empirical_scaled_second_moment"]
            if not np.isclose(value, expected, rtol=1e-6, atol=1e-10):
                raise ValueError("Restored model does not reproduce its recorded objective")
            result = {"restored_empirical_scaled_second_moment": value, "expected": expected,
                      "checkpoint_sha256": digest(policy.checkpoint),
                      "policy_code_bytes_match_training": policy.policy_bytes_match,
                      "explicit_packaging_migration": not policy.policy_bytes_match, "passed": True}
            write_new(output / "portable_policy_diagnostic.json", result)
        finally:
            policy.close()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
