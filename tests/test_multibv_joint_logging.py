from types import SimpleNamespace
import unittest
import numpy as np

from controller.nadeglobalcontroller import NADEBVGlobalController


def _ego(vehicle_id, x, y, speed):
    return {
        "veh_id": vehicle_id,
        "position": [x, y],
        "velocity": speed,
        "lane_index": 1,
    }


class MultiBVJointLoggingTests(unittest.TestCase):
    def test_per_bv_epsilon_is_assigned_by_vehicle_id_not_candidate_order(self):
        controller = NADEBVGlobalController.__new__(NADEBVGlobalController)
        controller.control_log = {}
        candidates = [
            SimpleNamespace(id="BV_context"),
            SimpleNamespace(id="BV_primary"),
        ]

        by_index, values = controller._selected_epsilon_values(
            {"BV_primary": 0.001, "BV_context": 0.2},
            [0, 1],
            candidates,
        )

        self.assertEqual(by_index, {0: 0.2, 1: 0.001})
        self.assertEqual(values, [0.2, 0.001])
        self.assertEqual(
            controller.control_log["epsilon_by_bv_id"],
            {"BV_context": 0.2, "BV_primary": 0.001},
        )

    def test_joint_context_contains_k_agents_and_probability_products(self):
        controller = NADEBVGlobalController.__new__(NADEBVGlobalController)
        controller.joint_control_num = 2
        controller.control_log = {}
        controller.drl_epsilon_value = 0.7
        controller.real_epsilon_value = 0.7
        controller.env = SimpleNamespace(
            initial_weight=1.0,
            info_extractor=SimpleNamespace(episode_log={"weight_episode": 1.0}),
        )
        bvs = [SimpleNamespace(id="BV_primary"), SimpleNamespace(id="BV_context")]
        controller._joint_full_obs = {
            "CAV": _ego("CAV", 400.0, 46.0, 20.0),
            "BV_primary": _ego("BV_primary", 410.0, 46.0, 18.0),
            "BV_context": _ego("BV_context", 395.0, 42.0, 19.0),
        }

        controller._record_joint_training_context(
            bvs,
            [0.2, 0.3],
            [0.01, 0.02],
            [0.5, 0.4],
        )

        self.assertTrue(controller.control_log["joint_training"])
        self.assertEqual(controller.control_log["joint_controlled_bv_ids"], [
            "BV_primary", "BV_context"
        ])
        self.assertEqual(len(controller.control_log["drl_obs_joint"]), 14)
        self.assertEqual([len(obs) for obs in controller.control_log["drl_obs_per_agent"]], [10, 10])
        self.assertAlmostEqual(controller.control_log["weight_record"]["joint"], 0.06)
        self.assertAlmostEqual(controller.control_log["ndd_record"]["joint"], 0.0002)
        self.assertEqual(controller.control_log["probability_record"]["proposal_type"], "factorized")
        self.assertAlmostEqual(
            controller.control_log["probability_record"]["importance_weight"], 0.06
        )
        self.assertEqual(controller.drl_epsilon_value, [0.7, 0.7])

    def test_joint_pair_sampler_keeps_a_single_correlated_weight(self):
        controller = NADEBVGlobalController.__new__(NADEBVGlobalController)
        controller.joint_control_num = 2
        controller._joint_pair_details = {
            (0, 1): {
                "ids": ["BV_primary", "BV_context"],
                "details": {
                    "naturalistic_pdf": np.full((2, 2), 0.25),
                    "challenge": np.asarray([[1.0, 0.0], [0.0, 0.0]]),
                },
            }
        }

        record = controller._sample_selected_joint_pair([0, 1], {0: 0.001, 1: 0.001})

        self.assertIsNotNone(record)
        self.assertEqual(record["proposal_type"], "joint_pair")
        self.assertEqual(record["selected_bv_ids"], ["BV_primary", "BV_context"])
        self.assertAlmostEqual(
            record["importance_weight"],
            record["naturalistic_probability"] / record["proposal_probability"],
        )
        self.assertAlmostEqual(
            record["weight_record"]["joint"], record["importance_weight"]
        )


if __name__ == "__main__":
    unittest.main()
