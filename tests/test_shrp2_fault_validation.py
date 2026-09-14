import json
import tempfile
import unittest
from pathlib import Path

from scenario_reconstruction.shrp2_fault_validation import build_fault_validation_template


class SHRP2FaultValidationTests(unittest.TestCase):
    def test_build_replaces_action_override_with_control_delay(self):
        config = {
            "schema_version": 1, "source_split": "train", "gap_offset_m": 2.0,
            "bv_braking_acceleration_mps2": -2.0, "bv_braking_start_time_s": 0.0,
            "bv_braking_duration_s": 5.0, "control_delay_start_time_s": 0.0,
            "control_delay_duration_s": 4.0, "control_delay_s": 4.0,
            "initial_action": {"lateral": "central", "longitudinal": 0.0},
            "target_collision_time_s": 4.0, "time_tolerance_s": 0.3,
        }
        source = {
            "template_id": "seed", "description": "seed", "map": "2Lane", "route": "route_0", "duration": 6.0,
            "tags": ["shrp2", "rear_end", "high"],
            "ego": {"id": "CAV", "lane_index": 1, "position": 400.0, "speed": 3.0},
            "actors": [{"id": "BV_primary", "lane_index": 1, "position": 416.0, "speed": 0.5}],
            "events": [], "perturbations": [],
            "bridge_metadata": {"source_quality": "high"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest = build_fault_validation_template(source_path, root / "output", config)
            template = json.loads(Path(manifest["template_path"]).read_text(encoding="utf-8"))
            self.assertEqual([event["type"] for event in template["events"]], ["forced_bv_action", "control_delay"])
            self.assertNotIn("calibration_cav_action", [event["type"] for event in template["events"]])
            self.assertTrue(template["events"][1]["params"]["not_for_d2rl_training"])


if __name__ == "__main__":
    unittest.main()
