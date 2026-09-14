import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scenario_reconstruction.conflict_diagnostics import conflict_features, run_diagnostics, validate_config
from tests.test_trajectory_benchmark import synthetic_scene


class ConflictDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/conflict_diagnostics.json").read_text())

    def test_future_overlap_features_and_label_are_auditable(self):
        scene = synthetic_scene()
        scene.update(event_time=2.0, split="train", recording_id="01", location_id=1,
                     interaction={"net_gap_m": 11.0, "time_headway_s": 0.5,
                                  "closing_speed_mps": 2.0, "ttc_s": 5.5,
                                  "stratum": "short_closing"})
        result = conflict_features(scene, self.config)
        self.assertTrue(result["has_future_lateral_overlap"])
        self.assertGreaterEqual(result["intervention_lead_to_overlap_s"], 0.2)
        self.assertTrue(result["high_conflict_potential"])
        self.assertEqual(result["exclusion_reasons"], [])

    def test_opening_scene_records_exclusion_reason(self):
        scene = synthetic_scene()
        scene.update(event_time=2.0, split="validation", recording_id="10", location_id=1,
                     interaction={"net_gap_m": 11.0, "time_headway_s": 0.5,
                                  "closing_speed_mps": -1.0, "ttc_s": None,
                                  "stratum": "short_opening"})
        result = conflict_features(scene, self.config)
        self.assertFalse(result["high_conflict_potential"])
        self.assertIn("event_closing_speed_above_minimum", result["exclusion_reasons"])

    def test_bad_config_and_test_split_are_rejected_before_file_access(self):
        bad = dict(self.config, min_intervention_lead_s=np.nan)
        with self.assertRaisesRegex(ValueError, "lead"):
            validate_config(bad)
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "test remains locked"):
                run_diagnostics("missing.json", Path(output) / "new", self.config, ["test"])


if __name__ == "__main__":
    unittest.main()
