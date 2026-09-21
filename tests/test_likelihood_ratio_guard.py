import math
import unittest

from scenario_reconstruction.likelihood_ratio_guard import constrain_epsilon


class LikelihoodRatioGuardTests(unittest.TestCase):
    def test_increases_epsilon_to_satisfy_proposal_ratio_bound(self):
        epsilon, diagnostics = constrain_epsilon(
            0.1, naturalistic_pdf=[0.9, 0.1], critical_pdf=[0.0, 1.0],
            maximum_proposal_ratio=5.0,
        )
        self.assertAlmostEqual(epsilon, 5.0 / 9.0)
        self.assertAlmostEqual(diagnostics["applied_maximum_proposal_ratio"], 5.0)
        self.assertTrue(diagnostics["adjusted"])

    def test_keeps_policy_epsilon_when_already_bounded(self):
        epsilon, diagnostics = constrain_epsilon(
            0.8, naturalistic_pdf=[0.9, 0.1], critical_pdf=[0.0, 1.0],
            maximum_proposal_ratio=5.0,
        )
        self.assertEqual(epsilon, 0.8)
        self.assertFalse(diagnostics["adjusted"])

    def test_uses_naturalistic_distribution_for_unsupported_critical_mass(self):
        epsilon, diagnostics = constrain_epsilon(
            0.1, naturalistic_pdf=[1.0, 0.0], critical_pdf=[0.0, 1.0],
            maximum_proposal_ratio=10.0,
        )
        self.assertEqual(epsilon, 1.0)
        self.assertTrue(math.isinf(diagnostics["requested_maximum_proposal_ratio"]))
        self.assertEqual(diagnostics["applied_maximum_proposal_ratio"], 1.0)

    def test_rejects_invalid_limit(self):
        for value in (1.0, 0.0, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    constrain_epsilon(0.5, [1.0], [1.0], value)


if __name__ == "__main__":
    unittest.main()
