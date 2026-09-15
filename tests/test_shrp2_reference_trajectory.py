import unittest

import numpy as np

from scenario_reconstruction.shrp2_reference_trajectory import (
    _actor_track,
    _interp_angle,
    _quality_summary,
    validate_config,
)


class SHRP2ReferenceTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema_version": 1,
            "history_s": 4.0,
            "sample_hz": 10,
            "heading_speed_threshold_mps": 0.5,
            "maximum_speed_path_rmse_mps": 1.0,
            "maximum_heading_path_mae_rad": 0.35,
        }

    def test_angle_interpolation_crosses_pi_without_discontinuity(self):
        result = _interp_angle(
            np.array([0.5]),
            np.array([0.0, 1.0]),
            np.array([np.pi - 0.1, -np.pi + 0.1]),
        )
        self.assertAlmostEqual(abs(result[0]), np.pi, places=6)

    def test_straight_track_passes_consistency_rules(self):
        times = np.array([0.0, 1.0, 2.0])
        track = _actor_track(
            times,
            np.column_stack((2.0 * times, np.zeros(3))),
            np.full(3, 2.0),
            np.zeros(3),
            np.zeros(3),
            self.config,
        )
        result = _quality_summary(track, self.config)
        self.assertTrue(result["position_speed_consistent"])
        self.assertTrue(result["heading_usable"])
        self.assertAlmostEqual(result["reported_vs_path_speed_rmse_mps"], 0.0)

    def test_config_rejects_nonpositive_history(self):
        invalid = {**self.config, "history_s": 0.0}
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_alternate_front_bumper_path_is_reported_separately(self):
        times = np.array([0.0, 1.0, 2.0])
        track = _actor_track(
            times,
            np.column_stack((4.0 * times, np.zeros(3))),
            np.full(3, 2.0),
            np.zeros(3),
            None,
            self.config,
        )
        track["source_front_bumper_xy_m"] = np.column_stack(
            (2.0 * times, np.zeros(3))
        ).tolist()
        result = _quality_summary(
            track, self.config, alternate_position_key="source_front_bumper_xy_m"
        )
        self.assertFalse(result["position_speed_consistent"])
        self.assertAlmostEqual(
            result["source_front_bumper_diagnostics"][
                "reported_vs_path_speed_rmse_mps"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
