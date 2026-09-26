"""Bounded PPO on the complete critical-sequence replay interface.

Does not fit NDD, change the simulator, or validate policy generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="configs/highd_dual_bv_meanprecision_ppo_v1.json")
    args = parser.parse_args()
    if sys.version_info >= (3, 10):
        raise RuntimeError("Use the existing Python 3.9 / Ray 1.11 training environment")
    output = Path(args.output).resolve()
    if output.exists() or args.iterations < 1:
        raise ValueError("Use a fresh output and positive iteration count")
    from .highd_sequence_io import digest, read_json, write_new
    config_path = Path(args.config).resolve()
    experiment_config = read_json(config_path)
    if experiment_config.get("schema_version") != 1:
        raise ValueError("Unsupported PPO experiment config schema")
    policy_parameterization = experiment_config.get("policy_parameterization", "mean_precision")
    if policy_parameterization not in ("legacy_beta", "mean_precision"):
        raise ValueError("Unsupported policy parameterization")
    ppo_overrides = experiment_config.get("ppo", {})
    ppo_keys = {"gamma", "lambda", "train_batch_size", "sgd_minibatch_size", "num_sgd_iter",
                "lr", "entropy_coeff", "vf_clip_param", "vf_loss_coeff", "clip_param", "grad_clip",
                "kl_coeff", "kl_target"}
    model_overrides = experiment_config.get("model", {})
    model_keys = {"fcnet_hiddens", "fcnet_activation", "vf_share_layers"}
    unsupported_model = sorted(set(model_overrides) - model_keys)
    if unsupported_model:
        raise ValueError(f"Unsupported model config fields: {unsupported_model}")
    if "fcnet_hiddens" in model_overrides and (not model_overrides["fcnet_hiddens"] or
            any(type(width) is not int or width < 1 for width in model_overrides["fcnet_hiddens"])):
        raise ValueError("model.fcnet_hiddens must be a nonempty list of positive integers")
    unsupported = sorted(set(ppo_overrides) - ppo_keys)
    if unsupported:
        raise ValueError(f"Unsupported PPO config fields: {unsupported}")
    if "fcnet_hiddens" in ppo_overrides and (not ppo_overrides["fcnet_hiddens"] or
            any(type(width) is not int or width < 1 for width in ppo_overrides["fcnet_hiddens"])):
        raise ValueError("fcnet_hiddens must be a nonempty list of positive integers")
    from d2rl_training.highd_critical_sequence_env import HighDCriticalSequenceEnv
    env_config = {"sequence_manifest": str(Path(args.sequence_manifest).resolve()), "seed": args.seed, "epsilon_min": .05}
    check_env = HighDCriticalSequenceEnv(env_config)
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env
    from .highd_mean_precision_policy import register_legacy_action_distribution

    register_env("highd_full_critical_sequence", lambda cfg: HighDCriticalSequenceEnv(cfg))
    register_legacy_action_distribution()
    model_config = {"custom_action_dist": "d2rl_bounded_beta", "fcnet_hiddens": [64, 64]}
    policy_code = Path(__file__).with_name("d2rl_bounded_action_dist.py")
    if policy_parameterization == "mean_precision":
        from .highd_mean_precision_policy import register_mean_precision_policy, policy_model_config
        register_mean_precision_policy()
        model_config = policy_model_config(check_env.action_space.low, check_env.action_space.high,
                                           experiment_config.get("initial_epsilon", .1),
                                           experiment_config.get("initial_precision", 80.))
        policy_code = Path(__file__).with_name("highd_mean_precision_policy.py")
    output.mkdir(parents=True)
    trainer = None
    ray.init(num_cpus=1, num_gpus=0, include_dashboard=False, ignore_reinit_error=True)
    try:
        ppo_config = {
            "env": "highd_full_critical_sequence", "env_config": env_config,
            "framework": "torch", "num_workers": 0, "num_gpus": 0, "seed": args.seed,
            "gamma": 1., "lambda": 1., "batch_mode": "complete_episodes",
            "rollout_fragment_length": 128, "train_batch_size": 512,
            "sgd_minibatch_size": 128, "num_sgd_iter": 5,
            "lr": .0003, "vf_clip_param": 100., "normalize_actions": True,
            "model": model_config,
        }
        ppo_config.update(ppo_overrides)
        ppo_config["model"].update(model_overrides)
        if policy_parameterization == "legacy_beta":
            ppo_config["model"]["custom_action_dist"] = "d2rl_bounded_beta"
            ppo_config["model"].pop("custom_model", None)
            ppo_config["model"].pop("custom_model_config", None)
        trainer = PPOTrainer(config=ppo_config)
        provenance = {"sequence_manifest_sha256": digest(args.sequence_manifest),
            "dataset_contract": check_env.dataset["contract"],
            "event_definition": check_env.dataset.get("event_definition"),
            "experiment_target_sha256": check_env.dataset.get("experiment_target_sha256"),
            "natural_target_sha256": check_env.dataset["natural_target_sha256"],
            "training_env_sha256": digest(Path(__file__).parents[1] / "d2rl_training/highd_critical_sequence_env.py"),
            "event_contract_code_sha256": digest(Path(__file__).parents[1] / "d2rl_training/highd_event_contract.py"),
            "trainer_code_sha256": digest(__file__), "training_seed": args.seed,
            "policy_parameterization": policy_parameterization, "policy_code_sha256": digest(policy_code),
            "experiment_config_sha256": digest(config_path),
            "behavior_episode_count": len(check_env.sequences), "nonempty_sequence_count": len(check_env.eligible)}
        write_new(output / "training_config.json", dict(experiment_config,
                  config_path=str(config_path), config_sha256=digest(config_path),
                  requested_iterations=args.iterations, seed=args.seed))
        write_new(output / "training_protocol.json", dict(provenance, ppo_config=ppo_config, requested_iterations=args.iterations))
        baseline = check_env.empirical_objective(lambda obs: [.1, .1])
        untrained = check_env.empirical_objective(lambda obs: trainer.compute_single_action(obs, explore=False))
        initialization = {"fixed_behavior_empirical_scaled_second_moment": baseline,
                          "untrained_deterministic_empirical_scaled_second_moment": untrained}
        if policy_parameterization == "mean_precision":
            import numpy as np
            observed_actions = []
            def initial_policy(obs):
                action = trainer.compute_single_action(obs, explore=False)
                observed_actions.append(action)
                return action
            check_env.empirical_objective(initial_policy)
            expected_epsilon = float(experiment_config.get("initial_epsilon", .1))
            error = float(np.max(np.abs(np.asarray(observed_actions) - expected_epsilon)))
            initialization["initial_epsilon_target"] = expected_epsilon
            initialization["maximum_initial_epsilon_error"] = error
            initialization["baseline_initialization_passed"] = bool(error < 1e-6 and
                (expected_epsilon != .1 or np.isclose(untrained, baseline, rtol=1e-6, atol=1e-12)))
            if not initialization["baseline_initialization_passed"]:
                raise RuntimeError("Untrained deterministic policy does not reproduce the configured initialization epsilon")
        write_new(output / "initialization_check.json", initialization)
        print(json.dumps({"initialization": initialization}), flush=True)
        records = []
        for iteration in range(1, args.iterations + 1):
            result = trainer.train()
            row = {"iteration": iteration, "timesteps_total": result["timesteps_total"],
                   "episode_len_mean": result["episode_len_mean"], "episode_reward_mean": result["episode_reward_mean"]}
            learner = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
            row["learner_stats"] = {key: float(learner[key]) for key in
                                    ("policy_loss", "vf_loss", "entropy", "kl") if key in learner}
            import math
            if not all(math.isfinite(value) for value in [row["episode_reward_mean"], *row["learner_stats"].values()]):
                raise RuntimeError("Non-finite PPO training metric; output preserved for diagnosis")
            if iteration % 5 == 0 or iteration == args.iterations:
                row["deterministic_empirical_scaled_second_moment"] = check_env.empirical_objective(
                    lambda obs: trainer.compute_single_action(obs, explore=False))
            records.append(row)
            write_new(output / f"iteration_{iteration:04d}.json", row)
            print(json.dumps(row), flush=True)
        checkpoint = trainer.save(str(output / "checkpoint"))
        trained = records[-1]["deterministic_empirical_scaled_second_moment"]
        summary = {**provenance, "interface_smoke_passed": True, "iterations": records,
            "initialization_check": initialization,
            "objective": "negative_offpolicy_path_second_moment_fixed_positive_scaling_no_reward_clip",
            "log_reward_scale": check_env.log_scale, "nonempty_sampling_correction": check_env.sampling_correction,
            "fixed_behavior_empirical_scaled_second_moment": baseline,
            "untrained_deterministic_empirical_scaled_second_moment": untrained,
            "trained_deterministic_empirical_scaled_second_moment": trained,
            "trained_to_fixed_behavior_second_moment_ratio": trained / baseline,
            "checkpoint": checkpoint, "formal_performance_evidence": False,
            "scope": "Resubstitution on tiny diagnostic collection; not held-out or online risk/efficiency evidence"}
        write_new(output / "training_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if trainer is not None:
            trainer.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
