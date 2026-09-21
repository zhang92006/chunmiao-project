import unittest

from scenario_reconstruction.cav_fault_model import CAVFaultModel
from scenario_reconstruction.templates import EventSpec, ScenarioTemplate


def _observation(distance):
    return {
        "Ego": {"veh_id": "CAV", "position": [0.0, 0.0]},
        "Lead": {
            "veh_id": "BV_primary",
            "distance": distance,
            "position": [distance, 0.0],
            "position3D": [distance, 0.0, 0.0],
        },
        "Foll": None, "LeftLead": None, "LeftFoll": None,
        "RightLead": None, "RightFoll": None,
    }


class CAVFaultModelTests(unittest.TestCase):
    def test_perception_delay_uses_a_historical_lead_observation(self):
        model = CAVFaultModel([EventSpec(
            type="perception_delay", actor="CAV", start_time=0.0, duration=2.0,
            params={"delay_s": 1.0, "target_vehicle": "BV_primary"},
        )])
        model.transform_observation(0.0, _observation(10.0))
        delayed = model.transform_observation(1.0, _observation(5.0))
        self.assertEqual(delayed["Lead"]["distance"], 10.0)
        self.assertEqual(model.last_audit["active_events"][0]["type"], "perception_delay")

    def test_dropout_and_position_bias_change_only_the_target(self):
        dropout = CAVFaultModel([EventSpec(
            type="perception_dropout", actor="CAV", start_time=0.0, duration=1.0,
            params={"target_vehicle": "BV_primary"},
        )])
        self.assertIsNone(dropout.transform_observation(0.0, _observation(5.0))["Lead"])
        bias = CAVFaultModel([EventSpec(
            type="perception_position_bias", actor="CAV", start_time=0.0, duration=1.0,
            params={"target_vehicle": "BV_primary", "offset_x_m": 2.0},
        )])
        biased = bias.transform_observation(0.0, _observation(5.0))
        self.assertEqual(biased["Lead"]["distance"], 7.0)
        self.assertEqual(biased["Lead"]["position"][0], 7.0)

    def test_control_delay_holds_declared_initial_action_then_replays_history(self):
        model = CAVFaultModel([EventSpec(
            type="control_delay", actor="CAV", start_time=0.0, duration=2.0,
            params={"delay_s": 1.0, "initial_action": {"lateral": "central", "longitudinal": 0.0}},
        )])
        self.assertEqual(model.delay_control(0.0, {"lateral": "central", "longitudinal": -3.0})["longitudinal"], 0.0)
        self.assertEqual(model.delay_control(1.0, {"lateral": "central", "longitudinal": -2.0})["longitudinal"], -3.0)

    def test_template_rejects_missing_delay_parameter(self):
        template = {
            "template_id": "fault", "description": "fault", "map": "2Lane", "route": "route_0", "duration": 2.0,
            "ego": {"id": "CAV", "lane_index": 1, "position": 400.0, "speed": 3.0},
            "actors": [],
            "events": [{"type": "control_delay", "actor": "CAV", "start_time": 0.0, "duration": 1.0, "params": {}}],
            "perturbations": [],
        }
        with self.assertRaisesRegex(ValueError, "delay_s"):
            ScenarioTemplate.from_dict(template)


if __name__ == "__main__":
    unittest.main()
