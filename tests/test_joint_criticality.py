import unittest

import numpy as np

from conf import conf
from scenario_reconstruction.joint_criticality import (
    joint_pair_proposal,
    pairwise_joint_criticality_arrays,
    pairwise_joint_criticality_details,
    sample_joint_action_pair,
)


def _vehicle(vehicle_id, x, lane, speed):
    return {
        "veh_id": vehicle_id,
        "position": [x, 42.0 + 4.0 * lane],
        "velocity": speed,
        "lane_index": lane,
    }


class JointCriticalityTests(unittest.TestCase):
    def test_pair_marginals_capture_second_bv_lane_entry_risk(self):
        action_count = len(conf.BV_ACTIONS)
        first_pdf = np.zeros(action_count)
        second_pdf = np.zeros(action_count)
        first_pdf[2] = 1.0
        second_pdf[0] = 0.5  # lane 0 -> lane 1, the CAV lane
        second_pdf[2] = 0.5
        full_obs = {
            "CAV": _vehicle("CAV", 0.0, 1, 10.0),
            "BV_primary": _vehicle("BV_primary", 50.0, 1, 5.0),
            "BV_context": _vehicle("BV_context", 6.0, 0, 5.0),
        }

        first_array, second_array, debug = pairwise_joint_criticality_arrays(
            full_obs, "BV_primary", "BV_context", first_pdf, second_pdf
        )

        self.assertGreater(float(np.sum(first_array)), 0.0)
        self.assertGreater(float(np.sum(second_array)), 0.0)
        self.assertGreater(second_array[0], second_array[2])
        self.assertGreater(debug["max_challenge"], 0.0)

    def test_pair_marginals_stay_zero_without_cav_lane_conflict(self):
        action_count = len(conf.BV_ACTIONS)
        pdf = np.zeros(action_count)
        pdf[2] = 1.0
        full_obs = {
            "CAV": _vehicle("CAV", 0.0, 1, 10.0),
            "BV_primary": _vehicle("BV_primary", 200.0, 0, 10.0),
            "BV_context": _vehicle("BV_context", -200.0, 0, 10.0),
        }

        first_array, second_array, debug = pairwise_joint_criticality_arrays(
            full_obs, "BV_primary", "BV_context", pdf, pdf
        )

        self.assertEqual(float(np.sum(first_array)), 0.0)
        self.assertEqual(float(np.sum(second_array)), 0.0)
        self.assertEqual(debug["max_challenge"], 0.0)

    def test_leading_hard_brake_is_preferred_over_acceleration(self):
        action_count = len(conf.BV_ACTIONS)
        first_pdf = np.zeros(action_count)
        second_pdf = np.zeros(action_count)
        hard_brake, acceleration = 2, action_count - 1
        first_pdf[hard_brake] = 0.5
        first_pdf[acceleration] = 0.5
        second_pdf[hard_brake] = 1.0
        full_obs = {
            "CAV": _vehicle("CAV", 0.0, 1, 30.0),
            "BV_primary": _vehicle("BV_primary", 15.0, 1, 26.0),
            "BV_context": _vehicle("BV_context", 0.0, 0, 30.0),
        }

        first_array, _, debug = pairwise_joint_criticality_arrays(
            full_obs, "BV_primary", "BV_context", first_pdf, second_pdf
        )

        self.assertGreater(first_array[hard_brake], first_array[acceleration])
        self.assertEqual(debug["first_best_action"], hard_brake)
        self.assertGreater(debug["collision_action_pair_count"], 0)
        self.assertGreater(debug["escape_blocking_action_pair_count"], 0)

    def test_late_longitudinal_conflict_is_discounted_when_escape_lane_is_free(self):
        action_count = len(conf.BV_ACTIONS)
        pdf = np.zeros(action_count)
        pdf[2] = 1.0
        full_obs = {
            "CAV": _vehicle("CAV", 0.0, 1, 30.0),
            "BV_primary": _vehicle("BV_primary", 15.0, 1, 26.0),
            "BV_context": _vehicle("BV_context", -100.0, 0, 25.0),
        }

        first_array, second_array, debug = pairwise_joint_criticality_arrays(
            full_obs, "BV_primary", "BV_context", pdf, pdf
        )

        self.assertEqual(float(np.sum(first_array)), 0.0)
        self.assertEqual(float(np.sum(second_array)), 0.0)
        self.assertEqual(debug["collision_action_pair_count"], 0)

    def test_correlated_joint_proposal_preserves_pair_weight(self):
        details = {
            "naturalistic_pdf": np.full((2, 2), 0.25),
            "challenge": np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        }
        proposal = joint_pair_proposal(details, epsilon=0.001)
        self.assertIsNotNone(proposal)
        self.assertAlmostEqual(float(np.sum(proposal["proposal_pdf"])), 1.0)
        self.assertGreater(proposal["proposal_pdf"][0, 0], proposal["naturalistic_pdf"][0, 0])
        sampled = sample_joint_action_pair(proposal, rng=np.random.default_rng(5))
        self.assertAlmostEqual(
            sampled["importance_weight"],
            sampled["naturalistic_probability"] / sampled["proposal_probability"],
        )


if __name__ == "__main__":
    unittest.main()
