import json
import tempfile
import unittest
from pathlib import Path

from scenario_reconstruction.shrp2_collision_calibration import (
    _outcome,
    generate_calibration_candidates,
)
from scenario_reconstruction.templates import ScenarioTemplate


class SHRP2CollisionCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema_version": 1,
            "source_split": "train",
            "gap_offsets_m": [-8.0, 0.0],
            "braking_accelerations_mps2": [-2.0],
            "braking_start_times_s": [0.0],
            "braking_duration_s": 3.0,
            "minimum_initial_gap_m": 7.0,
            "target_collision_time_s": 4.0,
            "time_tolerance_s": 0.3,
            "max_candidates": 2,
        }

    def test_generates_safe_grid_with_declared_intervention(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_path = temporary_path / "source.json"
            source_path.write_text(json.dumps(self._source_template()), encoding="utf-8")
            manifest = generate_calibration_candidates(source_path, temporary_path / "output", self.config)
            generated = [record for record in manifest["records"] if record["status"] == "generated"]
            self.assertEqual(len(generated), 2)
            candidate = json.loads(Path(generated[0]["template_path"]).read_text(encoding="utf-8"))
            self.assertTrue(candidate["events"][0]["params"]["calibration_only"])
            self.assertTrue(candidate["calibration_metadata"]["not_for_d2rl_training"])

    def test_selection_requires_target_collision_and_timing(self):
        record = {"initial_gap_m": 8.0}
        manifest = {
            "minimum_initial_gap_m": 7.0,
            "target_collision_time_s": 4.0,
            "time_tolerance_s": 0.3,
        }
        episode = {
            "collision_result": 1,
            "collision_id": ["CAV", "BV_primary"],
            "episode_info": {"end_time": 4.1},
            "ttc_step_info": {"4.0": 0.1},
            "distance_step_info": {"4.0": 0.0},
        }
        outcome = _outcome(record, Path("crash/0.json"), episode, manifest)
        self.assertEqual(outcome["selection_status"], "selected")
        episode["collision_id"] = ["CAV", "BV_context"]
        self.assertEqual(
            _outcome(record, Path("crash/0.json"), episode, manifest)["selection_status"],
            "not_selected",
        )

    def test_v2_adds_auditable_cav_reachability_action(self):
        config = {
            **self.config,
            "schema_version": 2,
            "cav_override_accelerations_mps2": [0.0],
            "cav_override_start_time_s": 0.0,
            "cav_override_durations_s": [4.0],
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_path = temporary_path / "source.json"
            source_path.write_text(json.dumps(self._source_template()), encoding="utf-8")
            manifest = generate_calibration_candidates(
                source_path, temporary_path / "output", config
            )
            record = next(
                item for item in manifest["records"] if item["status"] == "generated"
            )
            candidate = json.loads(
                Path(record["template_path"]).read_text(encoding="utf-8")
            )
            cav_event = candidate["events"][1]
            self.assertEqual(cav_event["type"], "calibration_cav_action")
            self.assertTrue(cav_event["params"]["not_for_d2rl_training"])
            self.assertEqual(
                candidate["calibration_metadata"]["cav_override_semantics"],
                "calibration reachability intervention; not a delay model",
            )

    def test_cav_calibration_action_cannot_be_used_as_training_event(self):
        source = self._source_template()
        source["events"] = [{
            "type": "calibration_cav_action",
            "actor": "CAV",
            "start_time": 0.0,
            "duration": 1.0,
            "params": {"calibration_only": True},
        }]
        with self.assertRaisesRegex(ValueError, "not_for_d2rl_training"):
            ScenarioTemplate.from_dict(source)

    @staticmethod
    def _source_template():
        return {
            "template_id": "sumo_seed", "description": "seed", "map": "2Lane", "route": "route_0",
            "duration": 6.0, "tags": ["shrp2", "rear_end", "high"],
            "ego": {"id": "CAV", "lane_index": 1, "position": 400.0, "speed": 8.0},
            "actors": [
                {"id": "BV_primary", "lane_index": 1, "position": 416.0, "speed": 2.0},
                {"id": "BV_context", "lane_index": 0, "position": 388.0, "speed": 2.0},
            ],
            "events": [], "perturbations": [],
            "bridge_metadata": {"source_quality": "high", "source_split": "train"},
        }


if __name__ == "__main__":
    unittest.main()
