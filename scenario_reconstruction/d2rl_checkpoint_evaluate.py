"""Deterministically evaluate a legacy RLlib D2RL checkpoint on held-out crashes."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import statistics
import sys
from typing import Any


def discover_crash_episodes(pool: str | Path, expected_split: str) -> list[Path]:
    """Return event-disjoint crash episodes and reject split contamination."""
    crash_dir = Path(pool) / "crash"
    paths = sorted(crash_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No crash episode JSON files found in {crash_dir}")
    mismatches = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        actual = episode.get("scenario_metadata", {}).get("source_split")
        if actual != expected_split:
            mismatches.append(f"{path.name}:{actual!r}")
    if mismatches:
        raise ValueError(
            f"Expected only {expected_split!r} episodes; split mismatches: "
            + ", ".join(mismatches[:5])
        )
    return paths


def summarize_records(
    records: list[dict[str, Any]],
    *,
    expected_split: str,
    checkpoint: str,
    clip_reward_threshold: float,
) -> dict[str, Any]:
    accepted = [item for item in records if item["status"] == "evaluated"]
    rejected = [item for item in records if item["status"] != "evaluated"]
    rewards = [float(item["reward"]) for item in accepted]
    action_count = max((len(item["action"]) for item in accepted), default=0)
    action_summary = []
    for index in range(action_count):
        values = [float(item["action"][index]) for item in accepted]
        action_summary.append({
            "agent_index": index,
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        })
    by_source_event = {}
    for source_event_id in sorted({item["source_event_id"] for item in accepted}, key=str):
        source_records = [
            item for item in accepted if item["source_event_id"] == source_event_id
        ]
        source_rewards = [float(item["reward"]) for item in source_records]
        by_source_event[str(source_event_id)] = {
            "episode_count": len(source_records),
            "reward_mean": statistics.fmean(source_rewards),
            "reward_min": min(source_rewards),
            "reward_max": max(source_rewards),
            "lower_clipped_count": sum(
                value <= -clip_reward_threshold for value in source_rewards
            ),
        }
    summary = {
        "schema_version": 1,
        "evaluation_mode": "deterministic_explore_false",
        "expected_split": expected_split,
        "checkpoint": checkpoint,
        "episode_count": len(records),
        "evaluated_count": len(accepted),
        "rejected_count": len(rejected),
        "rejected_by_reason": dict(Counter(item.get("reason", "unknown") for item in rejected)),
        "unique_source_event_count": len({item["source_event_id"] for item in accepted}),
        "reward": None,
        "action_by_agent": action_summary,
        "by_source_event": by_source_event,
        "records": records,
    }
    if rewards:
        summary["reward"] = {
            "mean": statistics.fmean(rewards),
            "min": min(rewards),
            "max": max(rewards),
            "lower_clipped_count": sum(
                value <= -clip_reward_threshold for value in rewards
            ),
            "lower_clipped_fraction": sum(
                value <= -clip_reward_threshold for value in rewards
            ) / len(rewards),
        }
    return summary


def _checkpoint_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    matches = sorted(candidate.glob("checkpoint-*"))
    matches = [item for item in matches if not item.name.endswith(".tune_metadata")]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint-* file in {candidate}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_conf", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode_pool", required=True)
    parser.add_argument("--expected_split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if sys.version_info >= (3, 10):
        raise RuntimeError("Checkpoint evaluation requires the Python 3.9 Ray 1.11 environment")
    if args.expected_split == "test":
        raise ValueError(
            "The test split remains locked during model development; evaluate validation first"
        )

    import numpy as np
    import yaml
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env

    from d2rl_training.d2rl_training_env import D2RLTrainingEnv
    from scenario_reconstruction.d2rl_smoke_train import build_rllib_config

    with Path(args.yaml_conf).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    episode_paths = discover_crash_episodes(args.episode_pool, args.expected_split)
    checkpoint = _checkpoint_file(args.checkpoint).resolve()
    env_name = "shrp2_multibv_checkpoint_eval"

    def env_creator(_env_config):
        return D2RLTrainingEnv(config)

    register_env(env_name, env_creator)
    ray.init(num_gpus=0, include_dashboard=False, ignore_reinit_error=True)
    trainer = None
    records = []
    try:
        trainer_config = build_rllib_config(config, env_name=env_name)
        trainer_config["num_workers"] = 0
        trainer_config["explore"] = False
        trainer = PPOTrainer(config=trainer_config, env=env_name)
        trainer.restore(str(checkpoint))
        env = D2RLTrainingEnv(config)
        for path in episode_paths:
            with path.open("r", encoding="utf-8") as stream:
                raw_episode = json.load(stream)
            metadata = raw_episode.get("scenario_metadata", {})
            record = {
                "episode_path": str(path),
                "source_event_id": metadata.get("source_event_id"),
                "source_split": metadata.get("source_split"),
            }
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    observation = env.reset(str(path))
                    if hasattr(trainer, "compute_single_action"):
                        action = trainer.compute_single_action(observation, explore=False)
                    else:
                        action = trainer.compute_action(observation, explore=False)
                    _, reward, done, info = env.step(action)
                selection = info.get("multi_bv_decision_selection") or {}
                record.update({
                    "status": "evaluated",
                    "action": np.asarray(action, dtype=float).reshape(-1).tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "selected_timestep": selection.get("selected_timestep"),
                    "selected_criticality": selection.get("selected_criticality"),
                    "reference_q_amplifier": selection.get("reference_q_amplifier"),
                })
            except (KeyError, TypeError, ValueError) as exc:
                record.update({"status": "rejected", "reason": str(exc)})
            records.append(record)
    finally:
        if trainer is not None:
            trainer.stop()
        ray.shutdown()

    summary = summarize_records(
        records,
        expected_split=args.expected_split,
        checkpoint=str(checkpoint),
        clip_reward_threshold=float(config["clip_reward_threshold"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "expected_split", "episode_count", "evaluated_count", "rejected_count",
        "unique_source_event_count", "reward", "action_by_agent",
        "by_source_event",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
