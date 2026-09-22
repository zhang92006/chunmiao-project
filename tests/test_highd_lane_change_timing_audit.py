import unittest

import numpy as np

from scenario_reconstruction.highd_lane_change_timing_audit import (
    _onset_to_crossing_seconds,
    _quantiles,
)


class HighDLaneChangeTimingAuditTests(unittest.TestCase):
    def test_detects_signed_lateral_velocity_onset(self):
        frames = np.arange(20)
        lanes = np.array([2] * 15 + [3] * 5)
        velocity = np.array([0.0] * 10 + [0.4] * 10)

        duration = _onset_to_crossing_seconds(
            frames,
            lanes,
            velocity,
            15,
            source_hz=10,
            threshold_mps=0.3,
            smoothing_window_s=0.1,
            maximum_lookback_s=2.0,
            endpoint_tolerance_s=0.2,
        )

        self.assertAlmostEqual(duration, 0.5)

    def test_reverses_velocity_for_negative_lane_delta(self):
        frames = np.arange(20)
        lanes = np.array([3] * 15 + [2] * 5)
        velocity = np.array([0.0] * 10 + [-0.4] * 10)

        duration = _onset_to_crossing_seconds(
            frames,
            lanes,
            velocity,
            15,
            source_hz=10,
            threshold_mps=0.3,
            smoothing_window_s=0.1,
            maximum_lookback_s=2.0,
            endpoint_tolerance_s=0.2,
        )

        self.assertAlmostEqual(duration, 0.5)

    def test_quantiles_report_median(self):
        result = _quantiles([1, 2, 3, 4, 5])
        self.assertEqual(result["p50"], 3)


if __name__ == "__main__":
    unittest.main()
