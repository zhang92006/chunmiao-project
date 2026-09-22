import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scenario_reconstruction.training_pool_ledger_audit import audit_training_pool


class TrainingPoolLedgerAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, *, episode_log_weight: float = 0.0) -> Path:
        episode = root / "experiment" / "crash" / "0.json"
        episode.parent.mkdir(parents=True)
        episode.write_text(json.dumps({
            "scenario_metadata": {"source_event_id": 101},
            "log_importance_weight": episode_log_weight,
            "log_probability_step_info": {
                "0.0": {"log_importance_weight": 0.2},
                "0.1": {"log_importance_weight": -0.2},
            },
            "weight_step_info": {
                "0.0": {"joint": 1.2},
                "0.1": {"joint": 1 / 1.2},
            },
            "weight_episode": 1.0 if episode_log_weight == 0.0 else 2.718281828459045,
        }), encoding="utf-8")
        index = root / "pool" / "crash_log_weight_dict.json"
        index.parent.mkdir()
        index.write_text(json.dumps({
            "log_weights": {episode.relative_to(root).as_posix(): 0.0}
        }), encoding="utf-8")
        return index

    def test_accepts_consistent_three_way_ledger(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = self._fixture(root)

            result = audit_training_pool(index, workspace_root=root)

            self.assertTrue(result["audit_passed"])
            self.assertEqual(result["consistent_episode_count"], 1)
            self.assertAlmostEqual(
                result["importance_weight_diagnostics"]["effective_sample_size"],
                1.0,
            )

    def test_rejects_episode_total_that_disagrees_with_ledgers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = self._fixture(root, episode_log_weight=1.0)

            result = audit_training_pool(index, workspace_root=root)

            self.assertFalse(result["audit_passed"])
            self.assertEqual(result["consistent_episode_count"], 0)
            self.assertTrue(any(
                "ledger_vs_episode" in failure for failure in result["failures"]
            ))

    def test_rejects_zero_support_step_omitted_from_log_ledger(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = self._fixture(root)
            episode_path = root / "experiment" / "crash" / "0.json"
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["weight_step_info"]["0.2"] = {"joint": 0.0}
            episode["weight_episode"] = 0.0
            episode_path.write_text(json.dumps(episode), encoding="utf-8")

            result = audit_training_pool(index, workspace_root=root)

            self.assertFalse(result["audit_passed"])
            self.assertTrue(any(
                "positive and finite" in failure for failure in result["failures"]
            ))


if __name__ == "__main__":
    unittest.main()
