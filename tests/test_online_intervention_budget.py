import unittest

from scenario_reconstruction.run_template import (
    _validated_online_intervention_budget,
    _validated_online_likelihood_ratio,
)


class OnlineInterventionBudgetValidationTests(unittest.TestCase):
    def test_budget_requires_policy_and_positive_integer(self):
        policy = object()
        self.assertIsNone(_validated_online_intervention_budget(policy, None))
        self.assertEqual(_validated_online_intervention_budget(policy, 5), 5)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validated_online_intervention_budget(policy, invalid)
        with self.assertRaisesRegex(ValueError, 'requires an online policy'):
            _validated_online_intervention_budget(None, 1)

    def test_likelihood_ratio_requires_policy_and_finite_limit_above_one(self):
        policy = object()
        self.assertIsNone(_validated_online_likelihood_ratio(policy, None))
        self.assertEqual(_validated_online_likelihood_ratio(policy, 50), 50.0)
        for invalid in (0, 1, float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validated_online_likelihood_ratio(policy, invalid)
        with self.assertRaisesRegex(ValueError, 'requires an online policy'):
            _validated_online_likelihood_ratio(None, 50)


if __name__ == '__main__':
    unittest.main()
