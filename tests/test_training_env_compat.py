import unittest

from d2rl_training.d2rl_training_env import D2RLTrainingEnv


class MultiBVCompatibilityTests(unittest.TestCase):
    def test_joint_value_supports_only_legacy_records(self):
        self.assertEqual(D2RLTrainingEnv._joint_value(0.25), 0.25)
        with self.assertRaisesRegex(ValueError, "MultiBV"):
            D2RLTrainingEnv._joint_value({"joint": 0.125, "per_agent": [0.25, 0.5]})

    def test_primary_observation_rejects_silent_projection(self):
        primary = list(range(10))
        record = {
            "joint": list(range(14)),
            "per_agent": [primary, list(range(10, 20))],
        }
        with self.assertRaisesRegex(ValueError, "MultiBV"):
            D2RLTrainingEnv._primary_observation(record)
        self.assertEqual(D2RLTrainingEnv._primary_observation(primary), primary)


if __name__ == "__main__":
    unittest.main()
