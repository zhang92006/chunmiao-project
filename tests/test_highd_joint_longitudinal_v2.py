import unittest

import numpy as np

from scenario_reconstruction.highd_joint_longitudinal_v2 import (
    factorized_probability, lift_dependence_to_33, shrink_joint_probability,
)


class JointLongitudinalV2Tests(unittest.TestCase):
    def test_shrinkage_is_normalized_and_empty_state_is_factorized(self):
        counts = np.zeros((2, 3, 3), dtype=np.uint32)
        counts[1, 0, 0] = 8
        counts[1, 2, 2] = 2
        factorized = factorized_probability(counts, .5)
        candidate = shrink_joint_probability(counts, 10, .5)
        np.testing.assert_allclose(candidate.sum(axis=(-2, -1)), 1)
        np.testing.assert_allclose(candidate[0], factorized[0])
        self.assertGreater(candidate[1, 0, 0], factorized[1, 0, 0])

    def test_lift_preserves_both_33_action_marginals(self):
        first = np.arange(1, 34, dtype=float); first /= first.sum()
        second = np.arange(33, 0, -1, dtype=float); second /= second.sum()
        coarse = np.array([[.22, .04, .03], [.05, .30, .06], [.02, .08, .20]])
        coarse /= coarse.sum()
        joint = lift_dependence_to_33(first, second, coarse,
                                      np.linspace(-4, 2, 31), [-.5, .5])
        self.assertAlmostEqual(float(joint.sum()), 1.0)
        np.testing.assert_allclose(joint.sum(axis=1), first, atol=1e-10)
        np.testing.assert_allclose(joint.sum(axis=0), second, atol=1e-10)
        self.assertGreater(np.abs(joint - np.outer(first, second)).sum(), .01)

    def test_independent_coarse_table_lifts_to_outer_product(self):
        first = np.full(33, 1 / 33)
        second = np.full(33, 1 / 33)
        a = np.array([.2, .5, .3]); b = np.array([.4, .4, .2])
        joint = lift_dependence_to_33(first, second, np.outer(a, b),
                                      np.linspace(-4, 2, 31), [-.5, .5])
        np.testing.assert_allclose(joint, np.outer(first, second), atol=1e-12)

    def test_lift_accepts_masked_lane_change_actions(self):
        first = np.r_[0.0, 0.0, np.full(31, 1 / 31)]
        second = np.r_[0.0, .05, np.full(31, .95 / 31)]
        coarse = np.array([[.20, .04, .03], [.05, .32, .06], [.02, .08, .20]])
        coarse /= coarse.sum()
        joint = lift_dependence_to_33(first, second, coarse,
                                      np.linspace(-4, 2, 31), [-.5, .5])
        np.testing.assert_allclose(joint.sum(axis=1), first, atol=1e-10)
        np.testing.assert_allclose(joint.sum(axis=0), second, atol=1e-10)
        self.assertEqual(float(joint[0].sum()), 0.0)

    def test_invalid_parameters_are_rejected(self):
        counts = np.zeros((1, 3, 3))
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                shrink_joint_probability(counts, value, .5)


if __name__ == "__main__":
    unittest.main()
