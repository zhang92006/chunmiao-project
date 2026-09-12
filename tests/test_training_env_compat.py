import unittest

from d2rl_training.d2rl_training_env import D2RLTrainingEnv


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


if __name__ == "__main__":
    unittest.main()
