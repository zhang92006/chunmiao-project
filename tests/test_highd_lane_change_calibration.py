import unittest
import numpy as np
from scenario_reconstruction.highd_lane_change_calibration import (
    empirical_prior, shrunk_probability, score_counts, choose_candidate,
)


class CalibrationTests(unittest.TestCase):
    def test_empty_state_uses_rare_event_prior(self):
        counts = np.array([[1, 998, 1], [0, 0, 0]])
        prior = empirical_prior(counts)
        p = shrunk_probability(counts, prior, 10)
        np.testing.assert_allclose(p[1], prior)
        self.assertLess(p[1, 0] + p[1, 2], .01)
        np.testing.assert_allclose(p.sum(-1), 1)

    def test_scores_measure_rate_inflation(self):
        counts = np.array([[1, 998, 1]])
        calibrated = score_counts(counts, np.array([[.001, .998, .001]]))
        uniform = score_counts(counts, np.full((1, 3), 1/3))
        self.assertAlmostEqual(calibrated['expected_to_observed_ratio'], 1)
        self.assertGreater(uniform['expected_to_observed_ratio'], 300)
        self.assertEqual(choose_candidate({'calibrated': calibrated, 'uniform': uniform}), 'calibrated')

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            score_counts(np.ones((1, 3)), np.ones((1, 3)))
        with self.assertRaises(ValueError):
            shrunk_probability(np.ones((1, 3)), np.ones(3)/3, 0)
