import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from d2rl_training.d2rl_training_env import D2RLTrainingEnv
from scenario_reconstruction.environment import ScenarioNADE, _duration_reached
from scenario_reconstruction.templates import EventSpec


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

    def test_low_speed_calibration_action_bypasses_vehicle_action_clamp(self):
        environment = ScenarioNADE.__new__(ScenarioNADE)
        environment.simulator = MagicMock(step_size=0.1)
        vehicle = SimpleNamespace(id="CAV", action_step_size=0.1)
        environment.vehicle_list = {"CAV": vehicle}
        event = EventSpec(
            type="calibration_cav_action",
            actor="CAV",
            start_time=0.0,
            duration=1.0,
            params={"longitudinal": 0.0},
        )

        environment._apply_calibration_longitudinal_action(event)

        environment.simulator.set_vehicle_speedmode.assert_called_once_with("CAV", 0)
        environment.simulator.change_vehicle_speed.assert_called_once_with(
            "CAV", 0.0, 0.1
        )


if __name__ == "__main__":
    unittest.main()
