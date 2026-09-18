"""Run a bounded K-BV PPO interface smoke test on prepared episodes.

This deliberately lives in the scenario-reconstruction repository instead of
changing the outer legacy training entrypoint.  It validates only that RLlib
can construct a policy with the configured joint observation/action spaces and
consume the reviewed episode pool; it is not a final training protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded PPO smoke training for K-BV D2RL data.")
    parser.add_argument("--yaml_conf", required=True, help="K-BV smoke YAML configuration.")
    parser.add_argument("--stop_iterations", type=int, default=None)
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

    def env_creator(_env_config):
        return D2RLTrainingEnv(config)

    register_env("shrp2_multibv_smoke", env_creator)
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    try:
        tune.run(
            "PPO",
            stop={"training_iteration": iterations},
            config={
                "env": "shrp2_multibv_smoke",
                "num_gpus": 0,
                "num_workers": int(config.get("num_workers", 1)),
                "num_envs_per_worker": 1,
                "gamma": 1.0,
                "rollout_fragment_length": 600,
                "vf_clip_param": config["clip_reward_threshold"],
                "framework": "torch",
                "ignore_worker_failures": False,
                "seed": int(config.get("seed", 7)),
            },
            checkpoint_freq=1,
            local_dir=config["local_dir"],
            name=config["experiment_name"],
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
