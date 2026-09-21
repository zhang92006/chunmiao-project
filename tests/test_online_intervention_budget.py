import unittest

from scenario_reconstruction.run_template import _validated_online_intervention_budget


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


if __name__ == '__main__':
    unittest.main()
