from __future__ import annotations

import argparse

import numpy as np

from d2rl_training.d2rl_training_env import D2RLTrainingEnv


def validate_training_env(experiment_path: str) -> None:
    yaml_conf = {
        "root_folder": "",
        "data_folders": [experiment_path],
        "data_folder_weights": [1],
        "clip_reward_threshold": 100,
    }
    env = D2RLTrainingEnv(yaml_conf)
    obs = env.reset()
    if len(obs) != 10:
        raise ValueError(f"Expected 10-dim observation, got {len(obs)}")

    done = False
    reward = 0
    step_count = 0
    while not done:
        obs, reward, done, _ = env.step(np.array([0.99], dtype=np.float32))
        step_count += 1
        if step_count > 1000:
            raise RuntimeError("Training env validation exceeded 1000 steps.")

    print("D2RLTrainingEnv validation OK")
    print(f"steps={step_count}")
    print(f"final_reward={reward}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate reconstructed data with D2RLTrainingEnv."
    )
    parser.add_argument(
        "experiment_path",
        help="Experiment directory that contains crash_weight_dict.json.",
    )
    args = parser.parse_args()
    validate_training_env(args.experiment_path)


if __name__ == "__main__":
    main()
