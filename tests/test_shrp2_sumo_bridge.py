import json
import tempfile
import unittest
from pathlib import Path

from scenario_reconstruction.shrp2_sumo_bridge import (
    bridge_seed_directory,
    rear_end_template_from_seed,
)
from scenario_reconstruction.templates import ScenarioTemplate


class SHRP2SUMOBridgeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema_version": 1,
            "allowed_quality": ["high", "medium"],
            "mappable_types": ["rear_end"],
            "duration_s": 6.0,
            "cav_position_m": 400.0,
            "context_gap_m": 12.0,
            "context_speed_mps": 2.0,
        }

    def test_rear_end_seed_becomes_valid_2lane_template(self):
        template = rear_end_template_from_seed(self._seed("rear_end", "high"), self.config)
        loaded = ScenarioTemplate.from_dict(template)
        self.assertEqual(loaded.map, "2Lane")
        self.assertEqual(loaded.ego.lane_index, loaded.actors[0].lane_index)
        self.assertGreater(loaded.actors[0].position, loaded.ego.position)
        self.assertEqual(loaded.events, [])

    def test_bridge_reports_blocked_and_excluded_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "seeds"
            scenario_dir = root / "scenarios"
            scenario_dir.mkdir(parents=True)
            (root / "summary.json").write_text(json.dumps({"dataset_doi": "doi:test"}), encoding="utf-8")
            for index, (kind, quality) in enumerate((
                ("rear_end", "high"), ("crossing", "medium"), ("lateral_sideswipe", "low"),
            )):
                (scenario_dir / f"seed_{index}.json").write_text(
                    json.dumps(self._seed(kind, quality)), encoding="utf-8"
                )
            summary = bridge_seed_directory(root, Path(temporary) / "bridge", self.config)
        self.assertEqual(summary["template_count"], 1)
        self.assertEqual(summary["blocked_count"], 1)
        self.assertEqual(summary["excluded_count"], 1)

    @staticmethod
    def _seed(simulation_type, quality):
        return {
            "scenario_id": f"seed_{simulation_type}_{quality}",
            "simulation_type": simulation_type,
            "source": {"event_id": 1, "associated_target_id": 2},
            "impact_conditioning": {"quality": quality, "requested_impact_time_s": 4.0},
            "actors": [
                {"id": "CAV", "speed_mps": 8.0, "xy_m": [[0.0, 0.0]]},
                {"id": "BV_primary", "speed_mps": 3.0, "xy_m": [[18.0, 0.0]]},
            ],
        }


if __name__ == "__main__":
    unittest.main()
