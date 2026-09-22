import unittest

import numpy as np

from scenario_reconstruction.shrp2_trajectory_metrics import _track_metrics


class SHRP2TrajectoryMetricsTests(unittest.TestCase):
    def test_track_metrics_reports_zero_for_identical_tracks(self):
        times = np.array([0.0, 1.0, 2.0])
        track = {
            "times": times,
            "position": np.column_stack((times, np.zeros(3))),
            "speed": np.ones(3),
            "heading": np.zeros(3),
        }
        result = _track_metrics(track, track, times)
        self.assertAlmostEqual(result["ade_m"], 0.0)
        self.assertAlmostEqual(result["position_rmse_m"], 0.0)
        self.assertAlmostEqual(result["speed_mae_mps"], 0.0)
        self.assertEqual(result["samples"], 3)

    def test_track_metrics_resamples_and_wraps_heading(self):
        actual = {
            "times": np.array([0.0, 2.0]),
            "position": np.array([[0.0, 0.0], [2.0, 0.0]]),
            "speed": np.array([1.0, 1.0]),
            "heading": np.array([np.pi, np.pi]),
        }
        reference = {
            "times": np.array([0.0, 2.0]),
            "position": np.array([[0.0, 1.0], [2.0, 1.0]]),
            "speed": np.array([2.0, 2.0]),
            "heading": np.array([-np.pi, -np.pi]),
        }
        result = _track_metrics(actual, reference, np.array([0.0, 1.0, 2.0]))
        self.assertAlmostEqual(result["ade_m"], 1.0)
        self.assertAlmostEqual(result["speed_mae_mps"], 1.0)
        self.assertAlmostEqual(result["heading_mae_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
