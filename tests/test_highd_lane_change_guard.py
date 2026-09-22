import copy
import unittest
import numpy as np

from scenario_reconstruction.highd_lane_change_guard import gap_envelope, blocked_sides
from scenario_reconstruction.highd_naturalistic_rollout import execution_pdf, _summary


class GuardTests(unittest.TestCase):
    config = {"mode": "constant_velocity_gap_envelope_v1", "duration_s": 1.0, "minimum_gap_m": 0.0}

    def obs(self):
        return {"Ego": {"velocity": 30, "could_drive_adjacent_lane_left": True,
                        "could_drive_adjacent_lane_right": False}, "Lead": None}

    def test_free_flow_without_current_leader_is_not_prohibited(self):
        obs = self.obs()
        p = np.ones(33)/33
        self.assertEqual(blocked_sides(obs, self.config), {})
        np.testing.assert_array_equal(execution_pdf(p, obs), execution_pdf(p, obs, self.config))
        self.assertGreater(execution_pdf(p, obs, self.config)[0], 0)

    def test_front_and_rear_collision_within_command_are_blocked(self):
        for relation, speed in (("Lead", 20), ("Foll", 40)):
            obs = self.obs()
            obs["Left"+relation] = {"distance": 4.62, "velocity": speed}
            self.assertIn("left", blocked_sides(obs, self.config))
            pdf = execution_pdf(np.ones(33)/33, obs, self.config)
            self.assertEqual(pdf[0], 0)
            self.assertAlmostEqual(pdf.sum(), 1)

    def test_does_not_use_arbitrary_large_ttc_cutoff(self):
        self.assertIsNone(gap_envelope(12, -10, self.config)["reason"])
        self.assertEqual(gap_envelope(-1, 20, self.config)["reason"], "initial_gap_not_clear")

    def test_fails_closed_for_nonfinite_geometry_and_zero_remaining_mass(self):
        with self.assertRaises(ValueError):
            gap_envelope(float("nan"), 0, self.config)
        obs = self.obs()
        obs["LeftLead"] = {"distance": -1, "velocity": 30}
        with self.assertRaises(ValueError):
            execution_pdf([1]+[0]*32, obs, self.config)

    def test_guarded_probability_and_diagnostics_are_reconstructed(self):
        obs = self.obs()
        obs["LeftFoll"] = {"distance": 2, "velocity": 40}
        raw = np.ones(33)/33
        pdf = execution_pdf(raw, obs, self.config)
        row = {"candidate_pdf": raw.tolist(), "executed_pdf": pdf.tolist(),
               "action_id": 2, "p_action": pdf[2], "q_action": pdf[2],
               "observation": obs, "applied": True, "fallback": False,
               "speed_boundary_adjusted": False, "longitudinal_source": "highd",
               "lane_change_guard": blocked_sides(obs, self.config)}
        self.assertTrue(_summary([row], [], self.config)["probability_audit_passed"])
        broken = copy.deepcopy(row)
        broken["p_action"] = raw[2]
        self.assertFalse(_summary([broken], [], self.config)["probability_audit_passed"])
        broken = copy.deepcopy(row)
        broken["lane_change_guard"] = {}
        self.assertFalse(_summary([broken], [], self.config)["probability_audit_passed"])
