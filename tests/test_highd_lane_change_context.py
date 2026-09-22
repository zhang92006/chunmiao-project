import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_lane_change_context import (
    _bin,
    _candidate_probabilities,
    _neighbor_features,
    hierarchical_probability,
)


class HighDLaneChangeContextTests(unittest.TestCase):
    def test_bins_values_on_declared_boundaries(self):
        values = np.array([0, 14.9, 15, 30, 100])
        np.testing.assert_array_equal(_bin(values, [15, 30, 60]), [0, 0, 1, 2, 3])

    def test_unseen_context_uses_base_probability(self):
        positive = np.array([0.0, 2.0])
        total = np.array([0.0, 10.0])
        index = np.array([0, 1])
        base = np.array([0.01, 0.02])

        probability = hierarchical_probability(
            positive, total, index, base, concentration=100
        )

        self.assertAlmostEqual(probability[0], 0.01)
        self.assertGreater(probability[1], 0.02)

    def test_rejects_nonpositive_concentration(self):
        with self.assertRaises(ValueError):
            hierarchical_probability(
                np.zeros(1), np.zeros(1), np.zeros(1, dtype=int),
                np.zeros(1), concentration=0,
            )

    def test_rejects_nonpositive_probability_scale(self):
        data = {
            "sides": {
                side: {
                    "available": np.ones(1, dtype=bool),
                    "base_index": np.zeros(1, dtype=int),
                    "context_index": np.zeros(1, dtype=int),
                }
                for side in ("left", "right")
            }
        }
        counts = {
            "base_total": np.ones(1), "base_positive": np.zeros(1),
            "context_total": np.ones(1), "context_positive": np.zeros(1),
        }
        with self.assertRaises(ValueError):
            _candidate_probabilities(data, counts, 1, 1, 0.5, probability_scale=0)

    def test_neighbor_geometry_uses_bumper_gap(self):
        rows = pd.DataFrame({
            "frame": [1], "id": [1], "x": [0.0], "width": [5.0],
            "leftPrecedingId": [2], "leftFollowingId": [3],
            "leftAlongsideId": [0],
        })
        lookup = pd.DataFrame({
            "frame": [1, 1], "id": [2, 3], "x": [15.0, -15.0],
            "width": [5.0, 5.0], "xVelocity": [31.0, 29.0],
        })

        features = _neighbor_features(
            rows, lookup, "left", np.array([1.0]), np.array([30.0]),
            [15.0, 30.0, 60.0], [-2.0, 2.0],
        )

        self.assertEqual([int(values[0]) for values in features], [1, 2, 1, 2, 0])


if __name__ == "__main__":
    unittest.main()
