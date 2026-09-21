import json
import math
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.d2rl_stratified_replication_compare import summarize


class StratifiedReplicationCompareTests(unittest.TestCase):
    def _experiment(self, root: Path, weights: list[float]) -> None:
        for folder in ("crash", "tested_and_safe", "rejected"):
            (root / folder).mkdir(parents=True)
        templates = ["a.json", "b.json"]
        results = []
        for episode_id, weight in enumerate(weights):
            template = templates[episode_id % 2]
            episode = {
                "collision_result": int(weight > 0.0),
                "collision_id": ["CAV", "BV_primary"] if weight > 0.0 else None,
                "log_importance_weight": math.log(weight) if weight > 0.0 else 0.0,
                "scenario_metadata": {"source_event_id": template[0]},
            }
            folder = "crash" if weight > 0.0 else "tested_and_safe"
            (root / folder / f"{episode_id}.json").write_text(json.dumps(episode))
            results.append({
                "episode": episode_id,
                "status": "ok",
                "template": template,
            })
        weighted_mean = sum(weights) / len(weights)
        (root / "manifest_run_summary.json").write_text(json.dumps({
            "results": results,
            "stratified_allocation": {
                "rollouts_by_template": {"a.json": 2, "b.json": 2}
            },
        }))
        (root / "closed_loop_audit.json").write_text(json.dumps({
            "audit_passed": True,
            "attempted": 4,
            "raw_cav_crashes": sum(value > 0.0 for value in weights),
            "stratified_estimation": {
                "weighted_cav_collision_mean": weighted_mean,
                "estimated_95pct_relative_half_width": 1.0,
                "crash_contribution_ess": 2.0,
            },
        }))

    def test_summarizes_repeated_arms_and_normalized_contributions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, weights in enumerate(([2.0, 0.0, 1.0, 0.0],
                                             [1.0, 0.0, 1.0, 0.0],
                                             [1.0, 1.0, 1.0, 1.0])):
                path = root / str(index)
                self._experiment(path, list(weights))
                paths.append(path)
            result = summarize({"uniform": paths[:2], "stratified": paths[1:]})

        self.assertEqual(result["arms"]["uniform"]["batch_count"], 2)
        self.assertEqual(result["arms"]["stratified"]["total_rollouts"], 8)
        self.assertAlmostEqual(
            result["arms"]["uniform"]["batches"][0]["largest_normalized_contribution"],
            2.0 / 3.0,
        )
        self.assertEqual(
            result["arms"]["uniform"]["batches"][0]
            ["effective_contributing_source_count_at_1pct"],
            1,
        )
        self.assertIsNotNone(
            result["arms"]["uniform"]["weighted_mean_between_batch_relative_std"]
        )
        self.assertEqual(result["decision"]["status"], "inconclusive_or_reject")
        self.assertFalse(
            result["decision"]["checks"]["at_least_three_batches_per_arm"]
        )


if __name__ == "__main__":
    unittest.main()
