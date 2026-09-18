import json
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.d2rl_checkpoint_evaluate import (
    discover_crash_episodes,
    summarize_records,
)
from scenario_reconstruction.d2rl_smoke_train import build_rllib_config


class D2RLCheckpointEvaluationTests(unittest.TestCase):
    def test_discover_crashes_enforces_declared_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            crash = Path(temp_dir) / "crash"
            crash.mkdir()
            (crash / "one.json").write_text(json.dumps({
                "scenario_metadata": {
                    "source_split": "validation",
                    "source_event_id": 1,
                }
            }), encoding="utf-8")

            paths = discover_crash_episodes(temp_dir, "validation")
            self.assertEqual([path.name for path in paths], ["one.json"])
            with self.assertRaisesRegex(ValueError, "split mismatches"):
                discover_crash_episodes(temp_dir, "train")

    def test_summary_reports_clipping_sources_and_actions(self):
        records = [
            {
                "status": "evaluated", "source_event_id": 1,
                "action": [0.2, 0.4], "reward": 80.0,
            },
            {
                "status": "evaluated", "source_event_id": 2,
                "action": [0.4, 0.8], "reward": -100.0,
            },
            {
                "status": "rejected", "source_event_id": 3,
                "reason": "no trainable candidate",
            },
        ]

        summary = summarize_records(
            records,
            expected_split="validation",
            checkpoint="checkpoint-20",
            clip_reward_threshold=100,
        )

        self.assertEqual(summary["evaluated_count"], 2)
        self.assertEqual(summary["rejected_count"], 1)
        self.assertEqual(summary["unique_source_event_count"], 2)
        self.assertEqual(summary["reward"]["lower_clipped_count"], 1)
        self.assertEqual(summary["reward"]["lower_clipped_fraction"], 0.5)
        self.assertAlmostEqual(summary["action_by_agent"][0]["mean"], 0.3)
        self.assertEqual(summary["by_source_event"]["1"]["reward_mean"], 80.0)
        self.assertEqual(summary["by_source_event"]["2"]["lower_clipped_count"], 1)

    def test_train_and_eval_share_rllib_shape_configuration(self):
        config = {"num_workers": 3, "clip_reward_threshold": 100, "seed": 11}
        result = build_rllib_config(config, env_name="held_out")
        self.assertEqual(result["env"], "held_out")
        self.assertEqual(result["num_workers"], 3)
        self.assertEqual(result["seed"], 11)
        self.assertEqual(result["vf_clip_param"], 100)


if __name__ == "__main__":
    unittest.main()
