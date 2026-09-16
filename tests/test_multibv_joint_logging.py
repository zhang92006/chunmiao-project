from types import SimpleNamespace
import unittest

from controller.nadeglobalcontroller import NADEBVGlobalController


def _ego(vehicle_id, x, y, speed):
    return {
        "veh_id": vehicle_id,
        "position": [x, y],
        "velocity": speed,
        "lane_index": 1,
    }


class MultiBVJointLoggingTests(unittest.TestCase):
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
        self.assertEqual(controller.drl_epsilon_value, [0.7, 0.7])


if __name__ == "__main__":
    unittest.main()
