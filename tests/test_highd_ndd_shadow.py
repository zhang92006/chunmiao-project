import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scenario_reconstruction.highd_ndd_shadow import HighDShadowNDD


class HighDShadowNDDTests(unittest.TestCase):
    def _model(self, root: Path):
        longitudinal = root / "long.npz"
        np.savez_compressed(
            longitudinal,
            speed_axis=np.array([20.0, 21.0]),
            gap_axis=np.array([0.0, 1.0]),
            range_rate_axis=np.array([-1.0, 0.0, 1.0]),
            acceleration_axis=np.linspace(-4, 2, 31),
            car_following_counts=np.ones((2, 3, 2, 31), dtype=np.uint32),
            free_flow_counts=np.ones((2, 31), dtype=np.uint32),
            car_following_probability=np.full((2, 3, 2, 31), 1 / 31),
            free_flow_probability=np.full((2, 31), 1 / 31),
        )
        config = {
            "speed_range_mps": [20, 40], "relative_speed_range_mps": [-10, 8],
            "maximum_gap_m": 115, "speed_boundaries_mps": [25, 30, 35],
            "require_current_leader_for_lateral": True,
            "current_gap_boundaries_m": [15, 30, 60],
            "target_gap_boundaries_m": [15, 30, 60],
            "relative_speed_boundaries_mps": [-2, 2],
            "maximum_total_lane_change_probability": 0.5,
        }
        config_path = root / "context.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        context_shape_size = 4 * 4 * 3 * 5 * 4 * 5 * 4 * 2
        context = root / "context.npz"
        np.savez_compressed(
            context,
            base_total=np.ones(4 * 4 * 3),
            base_positive=np.zeros(4 * 4 * 3),
            context_total=np.ones(context_shape_size),
            context_positive=np.zeros(context_shape_size),
            selected_base_concentration=100,
            selected_context_concentration=1000,
            probability_scale=1.0,
        )
        return HighDShadowNDD(longitudinal, context, config_path)

    def test_falls_back_outside_longitudinal_domain(self):
        with TemporaryDirectory() as temporary:
            model = self._model(Path(temporary))
            obs = {
                "Ego": {
                    "veh_id": "BV", "velocity": 10,
                    "could_drive_adjacent_lane_left": False,
                    "could_drive_adjacent_lane_right": False,
                },
                "Lead": None, "LeftLead": None, "LeftFoll": None,
                "RightLead": None, "RightFoll": None,
            }
            original = np.full(33, 1 / 33)

            result = model.compare(obs, original)

            self.assertTrue(result["fallback"])
            np.testing.assert_allclose(result["highd_shadow_pdf"], original)

    def test_candidate_pdf_is_normalised_and_does_not_change_input(self):
        with TemporaryDirectory() as temporary:
            model = self._model(Path(temporary))
            obs = {
                "Ego": {
                    "veh_id": "BV", "velocity": 20.5,
                    "could_drive_adjacent_lane_left": True,
                    "could_drive_adjacent_lane_right": False,
                },
                "Lead": None, "LeftLead": None, "LeftFoll": None,
                "RightLead": None, "RightFoll": None,
            }
            original = np.full(33, 1 / 33)
            before = original.copy()

            result = model.compare(obs, original)

            self.assertFalse(result["fallback"])
            self.assertAlmostEqual(sum(result["highd_shadow_pdf"]), 1.0)
            self.assertEqual(result["highd_lane_change_probability"], 0.0)
            self.assertEqual(
                result["lateral_source"], "original_structure_no_current_leader"
            )
            np.testing.assert_array_equal(original, before)


if __name__ == "__main__":
    unittest.main()
