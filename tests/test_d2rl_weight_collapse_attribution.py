import json
import math
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.d2rl_weight_collapse_attribution import (
    analyze_experiment,
    summarize_experiments,
)


class WeightCollapseAttributionTests(unittest.TestCase):
    def _write_pool(self, root: Path, episodes: list[dict], audit_passed: bool = True):
        (root / "crash").mkdir(parents=True)
        for index, episode in enumerate(episodes):
            (root / "crash" / f"{index}.json").write_text(json.dumps(episode))
        (root / "closed_loop_audit.json").write_text(json.dumps({
            "audit_passed": audit_passed,
            "raw_cav_crashes": len(episodes),
        }))

    @staticmethod
    def _episode(log_weight: float, source: int, actor: str = "BV_primary") -> dict:
        return {
            "collision_result": 1,
            "collision_id": ["CAV", actor],
            "log_importance_weight": log_weight,
            "scenario_metadata": {"source_event_id": source},
            "log_probability_step_info": {
                "0.1": {"log_importance_weight": log_weight},
            },
            "online_policy_step_info": {
                "0.1": {
                    "status": "inferred",
                    "sampled_terms": {
                        actor: {
                            "p": math.exp(log_weight), "q": 1.0,
                        },
                    },
                },
            },
        }

    def test_attributes_normalized_contributions_sources_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_pool(root, [
                self._episode(0.0, 10),
                self._episode(math.log(0.5), 20, "BV_context"),
            ])
            result = analyze_experiment("seed", root)
            self.assertAlmostEqual(result["crash_contribution_ess"], 1.8)
            self.assertAlmostEqual(result["largest_normalized_crash_contribution"], 2 / 3)
            self.assertEqual(result["episodes_by_contribution"][0]["episode_id"], 0)
            self.assertAlmostEqual(
                result["source_contributions"]["20"]["global_normalized_contribution"],
                1 / 3,
            )
            self.assertEqual(
                result["most_negative_probability_steps"][0]["episode_id"], 1
            )
            self.assertAlmostEqual(
                result["actor_log_importance_weight_totals_across_crashes"]["BV_context"],
                math.log(0.5),
            )
            self.assertAlmostEqual(
                result["actor_negative_log_penalty_share"]["BV_context"], 1.0
            )

    def test_summarizes_common_and_union_episode_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            self._write_pool(first, [self._episode(0, 1), self._episode(-1, 1)])
            self._write_pool(second, [self._episode(-2, 2)])
            (second / "crash" / "0.json").rename(second / "crash" / "1.json")
            result = summarize_experiments({"a": first, "b": second})
            self.assertEqual(result["common_crash_episode_ids"], [1])
            self.assertEqual(result["union_crash_episode_ids"], [0, 1])

    def test_rejects_failed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_pool(root, [self._episode(0, 1)], audit_passed=False)
            with self.assertRaisesRegex(ValueError, "did not pass"):
                analyze_experiment("bad", root)


if __name__ == "__main__":
    unittest.main()
