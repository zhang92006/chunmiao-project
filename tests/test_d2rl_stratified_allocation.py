import json
import math
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.d2rl_stratified_allocation import build_allocation


class StratifiedAllocationTests(unittest.TestCase):
    def _write_experiment(self, root: Path, contributions: dict[str, list[float]]):
        (root / "crash").mkdir(parents=True)
        (root / "tested_and_safe").mkdir()
        results = []
        episode_id = 0
        crash_count = 0
        for template, values in contributions.items():
            for value in values:
                results.append({
                    "episode": episode_id,
                    "status": "ok",
                    "template": template,
                })
                episode = {
                    "collision_result": int(value > 0),
                    "collision_id": ["CAV", "BV_primary"] if value > 0 else None,
                    "log_importance_weight": 0.0 if value <= 0 else math.log(value),
                    "scenario_metadata": {"source_event_id": template},
                }
                folder = "crash" if value > 0 else "tested_and_safe"
                (root / folder / f"{episode_id}.json").write_text(json.dumps(episode))
                crash_count += int(value > 0)
                episode_id += 1
        (root / "manifest_run_summary.json").write_text(json.dumps({"results": results}))
        (root / "closed_loop_audit.json").write_text(json.dumps({
            "audit_passed": True,
            "complete_episode_count": episode_id,
            "raw_cav_crashes": crash_count,
        }))

    def test_allocates_floor_and_remainder_to_high_variance_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_experiment(root, {
                "high.json": [0.0, 1.0, 0.0, 1.0],
                "zero.json": [0.0, 0.0, 0.0, 0.0],
            })
            result = build_allocation(
                {"pilot": root}, additional_budget=10,
                minimum_per_template=2, variance_floor_fraction=.01,
            )
            self.assertEqual(result["allocation_total"], 10)
            self.assertGreater(
                result["templates"]["high.json"]["additional_rollouts"],
                result["templates"]["zero.json"]["additional_rollouts"],
            )
            self.assertGreaterEqual(
                result["templates"]["zero.json"]["additional_rollouts"], 2
            )

    def test_rejects_incompatible_template_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a", Path(tmp) / "b"
            self._write_experiment(first, {"one.json": [0.0, 1.0]})
            self._write_experiment(second, {"two.json": [0.0, 1.0]})
            with self.assertRaisesRegex(ValueError, "same template set"):
                build_allocation(
                    {"a": first, "b": second}, additional_budget=4,
                    minimum_per_template=1,
                )


if __name__ == "__main__":
    unittest.main()
