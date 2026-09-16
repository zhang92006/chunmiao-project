import unittest

import numpy as np

from conf import conf
from scenario_reconstruction.joint_criticality import pairwise_joint_criticality_arrays


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
            "BV_primary": _vehicle("BV_primary", 12.0, 1, 5.0),
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


if __name__ == "__main__":
    unittest.main()
