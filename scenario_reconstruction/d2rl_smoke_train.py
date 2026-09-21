"""Run a bounded K-BV PPO interface smoke test on prepared episodes.

This deliberately lives in the scenario-reconstruction repository instead of
changing the outer legacy training entrypoint.  It validates only that RLlib
can construct a policy with the configured joint observation/action spaces and
consume the reviewed episode pool; it is not a final training protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def build_rllib_config(config: dict, env_name: str = "shrp2_multibv_smoke") -> dict:
    """Build the shared legacy RLlib configuration used by train and eval."""
    rllib_config = {
        "env": env_name,
        "num_gpus": 0,
        "num_workers": int(config.get("num_workers", 1)),
        "num_envs_per_worker": 1,
        "gamma": 1.0,
        "rollout_fragment_length": 600,
        "vf_clip_param": config["clip_reward_threshold"],
        "framework": "torch",
        "ignore_worker_failures": False,
        "seed": int(config.get("seed", 7)),
    }
    if config.get("action_distribution") == "bounded_beta":
        rllib_config["model"] = {"custom_action_dist": "d2rl_bounded_beta"}
    return rllib_config


def run_direct_training(config: dict, iterations: int) -> None:
    """Train in the driver process to reduce memory use on 16 GiB hosts."""
    from ray.rllib.agents.ppo import PPOTrainer

    output_dir = (
        Path(config["local_dir"]).resolve()
        / config["experiment_name"]
        / "direct"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = PPOTrainer(config=build_rllib_config(config))
    try:
        for iteration in range(1, iterations + 1):
            result = trainer.train()
            checkpoint_path = trainer.save(str(output_dir))
            summary = {
                "training_iteration": iteration,
                "timesteps_total": result.get("timesteps_total"),
                "episode_reward_mean": result.get("episode_reward_mean"),
                "episode_reward_min": result.get("episode_reward_min"),
                "episode_reward_max": result.get("episode_reward_max"),
                "episode_len_mean": result.get("episode_len_mean"),
                "checkpoint_path": checkpoint_path,
            }
            print(json.dumps(summary, ensure_ascii=False))
    finally:
        trainer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded PPO smoke training for K-BV D2RL data.")
    parser.add_argument("--yaml_conf", required=True, help="K-BV smoke YAML configuration.")
    parser.add_argument("--stop_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the YAML random seed.")
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Override the YAML Ray result directory name.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Train PPO in the driver process instead of a Tune trial actor.",
    )
    args = parser.parse_args()

    if sys.version_info >= (3, 10):
        raise RuntimeError(
            "This legacy PPO smoke path needs Python 3.9 on Windows: Ray 1.11 "
            "does not publish a wheel for the current Python 3.10+ runtime."
        )

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "K=2 PPO smoke training requires PyYAML and ray[rllib]==1.11.0; "
            "see requirements_d2rl_train.txt"
        ) from exc

    with Path(args.yaml_conf).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    iterations = args.stop_iterations or int(config.get("training_iterations", 2))
    if iterations < 1:
        raise ValueError("stop_iterations must be positive")
    if not config.get("multi_bv_training"):
        raise ValueError("The smoke runner requires multi_bv_training: true")

    try:
        import ray
        from ray import tune
        from ray.tune.registry import register_env
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "K=2 PPO smoke training requires ray[rllib]==1.11.0; see requirements_d2rl_train.txt"
        ) from exc

    from d2rl_training.d2rl_training_env import D2RLTrainingEnv

    if config.get("action_distribution") == "bounded_beta":
        from scenario_reconstruction.d2rl_bounded_action_dist import (
            register_bounded_action_distribution,
        )
        register_bounded_action_distribution()

    def env_creator(_env_config):
        return D2RLTrainingEnv(config)

    register_env("shrp2_multibv_smoke", env_creator)
    # Ray 1.11 probes ``nvidia-smi`` when GPU resources are unspecified.  The
    # K=2 interface smoke test is intentionally CPU-only and must also run on
    # Windows hosts without NVIDIA tooling installed.
    ray.init(num_gpus=0, include_dashboard=False, ignore_reinit_error=True)
    try:
        if args.direct or config.get("direct_training", False):
            run_direct_training(config, iterations)
        else:
            tune.run(
                "PPO",
                stop={"training_iteration": iterations},
                config=build_rllib_config(config),
                checkpoint_freq=1,
                local_dir=config["local_dir"],
                name=config["experiment_name"],
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
