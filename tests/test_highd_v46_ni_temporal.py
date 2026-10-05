"""Focused statistical/lag-convention checks; no simulations or fitting."""
import unittest
import numpy as np
from scenario_reconstruction.highd_v46_noninferiority import paired_test
from scenario_reconstruction.highd_v46_temporal_diagnostic import (
    accumulate, arrays_from_window, curves)
from scenario_reconstruction.highd_v46_matched_report import rho


class DiagnosticTests(unittest.TestCase):
    def test_noninferiority_is_not_equivalence(self):
        result = paired_test(np.linspace(-.022, -.018, 100))
        self.assertTrue(result['noninferior_at_alpha_0_05'])
        self.assertFalse(result['equivalent_at_alpha_0_05'])
        result = paired_test(np.linspace(-.003, .003, 100))
        self.assertTrue(result['equivalent_at_alpha_0_05'])
        self.assertGreater(result['ci90'][0], -.01)
        self.assertLess(result['ci90'][1], .01)
        self.assertFalse(paired_test(np.full(100, .01))['noninferior_at_alpha_0_05'])

    def test_positive_lag_means_rear_later(self):
        rng = np.random.default_rng(52)
        front = rng.normal(size=(1000, 13))
        data = np.stack([front[:, 2:], front[:, :-2]], axis=-1)
        blocks = accumulate(data, data, [-1, 0, 1])
        correlation = [rho(blocks[0, 2, k])[0] for k in range(3)]
        self.assertAlmostEqual(correlation[2], 1.)
        self.assertLess(abs(correlation[0]), .1)
        self.assertLess(abs(correlation[1]), .1)

    def test_half_frame_interpolation(self):
        time = np.arange(151)/25
        acceleration = np.array([.25, -.5])
        speed = 30 + time[:, None]*acceleration
        interval, instant = arrays_from_window(speed, np.tile(acceleration, (151, 1)), 25)
        np.testing.assert_allclose(interval, np.tile(acceleration, (11, 1)))
        np.testing.assert_allclose(instant, interval)

    def test_identical_arms_have_zero_paired_bootstrap_difference(self):
        rng = np.random.default_rng(3)
        blocks = []
        for _ in range(5):
            data = rng.normal(size=(40, 11, 2))
            blocks.append(accumulate(data, data, [-1, 0, 1]))
        values, differences = curves(dict(highd=blocks, single=blocks, dual=blocks),
            dict(lags_s=[-1, 0, 1], bootstrap_replicates=50, bootstrap_seed=8))
        self.assertEqual(len(values['dual']), 27)
        for row in differences:
            np.testing.assert_allclose(row['dual_minus_single_ci95'], [0, 0])
            np.testing.assert_allclose(row['dual_minus_single_absolute_error_vs_highd_ci95'], [0, 0])


if __name__ == '__main__':
    unittest.main()
