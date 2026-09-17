import json
from pathlib import Path
import unittest

from scenario_reconstruction.shrp2_multibv_sumo_bridge import (
    _reason_code,
    multibv_template_from_seed,
)
from scenario_reconstruction.templates import ScenarioTemplate


class MultiBVSumoBridgeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            Path("configs/shrp2_multibv_sumo_bridge.json").read_text(encoding="utf-8")
        )
        self.seed = {
            "record_type": "shrp2_measured_multibv_seed_v1",
            "time_s": [4.0],
            "states": [[[0.0, 0.0, 20.0, 0.0], [15.0, 0.2, 21.0, 0.0], [-12.0, 3.5, 22.0, 0.0]]],
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
        self.seed["condition"] = {
            "critical_time_s": 4.0,
            "initialization_offset_before_critical_s": 2.0,
        }
        template = multibv_template_from_seed(self.seed, self.config)
        ScenarioTemplate.from_dict(template)
        self.assertEqual(len(template["actors"]), 2)
        self.assertEqual(template["actors"][0]["lane_index"], 1)
        self.assertEqual(template["actors"][1]["lane_index"], 0)
        self.assertEqual(template["events"], [])
        self.assertFalse(template["bridge_metadata"]["drl_training_ready"])
        self.assertEqual(template["bridge_metadata"]["source_critical_time_s"], 4.0)
        self.assertEqual(
            template["bridge_metadata"][
                "source_initialization_offset_before_critical_s"
            ],
            2.0,
        )

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

    def test_blocks_context_overlap_after_lane_projection(self):
        seed = json.loads(json.dumps(self.seed))
        seed["states"][0][2][0] = 1.0
        seed["states"][0][2][1] = 0.1
        with self.assertRaisesRegex(ValueError, "same_lane_gap_out_of_range"):
            multibv_template_from_seed(seed, self.config)

    def test_blocks_speed_above_d2rl_domain(self):
        seed = json.loads(json.dumps(self.seed))
        seed["states"][0][1][2] = 43.51
        with self.assertRaisesRegex(ValueError, "primary_speed_exceeds_d2rl_domain"):
            multibv_template_from_seed(seed, self.config)

    def test_blocks_speed_below_d2rl_domain(self):
        seed = json.loads(json.dumps(self.seed))
        seed["states"][0][2][2] = 4.5
        with self.assertRaisesRegex(ValueError, "context_speed_below_d2rl_domain"):
            multibv_template_from_seed(seed, self.config)

    def test_reason_code_omits_numeric_detail(self):
        self.assertEqual(
            _reason_code("primary_speed_below_d2rl_domain:4.500<20.000"),
            "primary_speed_below_d2rl_domain",
        )


if __name__ == "__main__":
    unittest.main()
