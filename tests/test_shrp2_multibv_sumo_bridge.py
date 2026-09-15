import json
from pathlib import Path
import unittest

from scenario_reconstruction.shrp2_multibv_sumo_bridge import multibv_template_from_seed
from scenario_reconstruction.templates import ScenarioTemplate


class MultiBVSumoBridgeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            Path("configs/shrp2_multibv_sumo_bridge.json").read_text(encoding="utf-8")
        )
        self.seed = {
            "record_type": "shrp2_measured_multibv_seed_v1",
            "time_s": [4.0],
            "states": [[[0.0, 0.0, 20.0, 0.0], [15.0, 0.2, 18.0, 0.0], [-12.0, 3.5, 19.0, 0.0]]],
            "actors": [
                {"id": "CAV", "role": "CAV"},
                {"id": "BV_primary", "source_target_id": 2},
                {"id": "BV_context", "source_target_id": 3},
            ],
            "source": {
                "event_id": 1,
                "category": "NearCrash",
                "split": "train",
                "conflict": "leading",
                "context_target_ids": [3],
            },
        }

    def test_maps_two_bvs_without_forced_actions(self):
        template = multibv_template_from_seed(self.seed, self.config)
        ScenarioTemplate.from_dict(template)
        self.assertEqual(len(template["actors"]), 2)
        self.assertEqual(template["actors"][0]["lane_index"], 1)
        self.assertEqual(template["actors"][1]["lane_index"], 0)
        self.assertEqual(template["events"], [])
        self.assertFalse(template["bridge_metadata"]["drl_training_ready"])

    def test_blocks_unsupported_topology(self):
        seed = json.loads(json.dumps(self.seed))
        seed["source"]["conflict"] = "pedestrian"
        with self.assertRaisesRegex(ValueError, "requires another SUMO topology"):
            multibv_template_from_seed(seed, self.config)

    def test_blocks_initial_primary_overlap(self):
        seed = json.loads(json.dumps(self.seed))
        seed["states"][0][1][0] = 2.0
        with self.assertRaisesRegex(ValueError, "primary_gap_out_of_range"):
            multibv_template_from_seed(seed, self.config)


if __name__ == "__main__":
    unittest.main()
