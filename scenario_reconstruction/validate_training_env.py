from __future__ import annotations

import argparse

import numpy as np

from d2rl_training.d2rl_training_env import D2RLTrainingEnv


def validate_training_env(
    experiment_path: str,
    multi_bv_training: bool = False,
    multi_bv_num: int = 2,
) -> None:
    yaml_conf = {
        "root_folder": "",
        "data_folders": [experiment_path],
        "data_folder_weights": [1],
        "clip_reward_threshold": 100,
        "multi_bv_training": multi_bv_training,
        "multi_bv_num": multi_bv_num,
    }
    env = D2RLTrainingEnv(yaml_conf)
    obs = env.reset()
    expected_dim = 6 + 4 * multi_bv_num if multi_bv_training else 10
    if len(obs) != expected_dim:
        raise ValueError(f"Expected {expected_dim}-dim observation, got {len(obs)}")

    done = False
    reward = 0
    step_count = 0
    while not done:
        action = np.full(env.action_space.shape, 0.99, dtype=np.float32)
        obs, reward, done, _ = env.step(action)
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
    parser.add_argument(
        "--multi_bv_training",
        action="store_true",
        help="Validate centralized K-BV observation/action records.",
    )
    parser.add_argument(
        "--multi_bv_num",
        type=int,
        default=2,
        help="Number of jointly controlled BVs when --multi_bv_training is set.",
    )
    args = parser.parse_args()
    validate_training_env(
        args.experiment_path,
        multi_bv_training=args.multi_bv_training,
        multi_bv_num=args.multi_bv_num,
    )


if __name__ == "__main__":
    main()
