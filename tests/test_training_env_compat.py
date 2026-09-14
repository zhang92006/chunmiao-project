import unittest

from d2rl_training.d2rl_training_env import D2RLTrainingEnv
from scenario_reconstruction.environment import _duration_reached


class MultiBVCompatibilityTests(unittest.TestCase):
    def test_joint_value_supports_legacy_and_multibv_records(self):
        self.assertEqual(D2RLTrainingEnv._joint_value(0.25), 0.25)
        self.assertEqual(
            D2RLTrainingEnv._joint_value({"joint": 0.125, "per_agent": [0.25, 0.5]}),
            0.125,
        )

    def test_primary_observation_projects_first_agent(self):
        primary = list(range(10))
        record = {
            "joint": list(range(14)),
            "per_agent": [primary, list(range(10, 20))],
        }
        self.assertEqual(D2RLTrainingEnv._primary_observation(record), primary)

    def test_scenario_duration_boundary_is_inclusive(self):
        self.assertFalse(_duration_reached(5.99, 6.0))
        self.assertTrue(_duration_reached(6.0, 6.0))
        self.assertTrue(_duration_reached(6.01, 6.0))


if __name__ == "__main__":
    unittest.main()
