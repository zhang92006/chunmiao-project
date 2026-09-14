import json
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.reachability_cem import (cem_reachability, run_reachability,
                                                       validate_cem_config)
from tests.test_trajectory_benchmark import synthetic_scene


class ReachabilityCemTests(unittest.TestCase):
    def setUp(self):
        self.trajectory_config = json.loads(Path("configs/trajectory_baselines.json").read_text())
        self.cem_config = json.loads(Path("configs/reachability_cem.json").read_text())
        self.cem_config.update(budget_tiers=[4, 9], population_size=4)

    def test_cem_is_reproducible_and_snapshots_are_exact(self):
        first = cem_reachability(synthetic_scene(), self.trajectory_config, self.cem_config, 7)
        second = cem_reachability(synthetic_scene(), self.trajectory_config, self.cem_config, 7)
        self.assertEqual(first["snapshots"], second["snapshots"])
        self.assertEqual([snapshot["evaluations"] for snapshot in first["snapshots"]], [4, 9])
        self.assertTrue(all(snapshot["selected"]["feasible"] for snapshot in first["snapshots"]))

    def test_invalid_config_and_test_split_are_rejected_before_file_access(self):
        bad = dict(self.cem_config, budget_tiers=[9, 4])
        with self.assertRaisesRegex(ValueError, "increasing"):
            validate_cem_config(bad)
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "test remains locked"):
                run_reachability("missing.json", Path(output) / "new", self.trajectory_config,
                                 self.cem_config, "test")


if __name__ == "__main__":
    unittest.main()
