import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_dual_ndd import PairActionDistribution
from scenario_reconstruction.highd_dual_ndd_criticality import challenge_grid, CAVCriticalityProvider
from scenario_reconstruction.highd_dual_ndd_runtime import source_rows
from d2rl_training.conditional_chain import replay_weight


def actor(i, x, lane=0, speed=30):
    return {"id": i, "center_x": x, "center_y": lane * 3.2,
            "lane": lane, "length": 5., "width": 1.8, "speed": speed,
            "acceleration": 0., "lateral_speed": 0.}


class CriticalityTests(unittest.TestCase):
    def test_far_scene_is_not_critical_and_bv_only_overlap_is_not_cav_crash(self):
        actors = [actor(100, 0), actor(1, 250), actor(2, 250)]
        score, collision, _ = challenge_grid(actors, 100, (1, 2), [0, 3.2])
        self.assertEqual(score.shape, (93, 93))
        self.assertFalse(collision.any())
        self.assertFalse(score.any())

    def test_rear_bv_acceleration_more_dangerous_than_braking(self):
        actors = [actor(100, 50), actor(1, 40, speed=32), actor(2, 250, 1)]
        score, collision, _ = challenge_grid(actors, 100, (1, 2), [0, 3.2])
        self.assertGreater(score[90, 60], score[0, 60])
        self.assertTrue(collision[90, 60])
        self.assertFalse(collision[0, 60])

    def test_lane_initiation_has_continuous_effect_not_instant_teleport(self):
        actors = [actor(100, 50), actor(1, 40, 1, speed=34), actor(2, 250, 1)]
        score, collision, _ = challenge_grid(actors, 100, (1, 2), [0, 3.2], {"horizon_s": .1})
        self.assertFalse(collision.any())
        self.assertFalse(score.any())
        score, _, _ = challenge_grid(actors, 100, (1, 2), [0, 3.2])
        self.assertGreater(score[62, 60], score[60, 60])

    def test_provider_support_weight_replay_and_no_change_to_p(self):
        actors = [actor(100, 50), actor(1, 40, speed=32), actor(2, 250, 1)]
        latest = pd.DataFrame(source_rows(actors, 0, [0, 3.2]))
        scene = SimpleNamespace(latest=latest, cav_id=100, centers={2: 0., 3: 3.2}, last_decision=0)
        scene._own = lambda i: latest[latest.id == i].reset_index(drop=True)
        p = np.zeros((93, 93))
        p[::3, ::3] = 1 / (31 * 31)
        dist = PairActionDistribution((1, 2), ("first", "second"), p, "fixture", structured=True)
        before = dist.natural.copy()
        provider = CAVCriticalityProvider([.1, .3])
        result = provider(scene, {}, dist)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(dist.natural, before)
        self.assertTrue((result.matrix[p == 0] == 0).all())
        self.assertTrue((result.matrix[p > 0] > 0).all())
        draw = result.sample(dist, np.random.default_rng(7))
        self.assertAlmostEqual(np.log(replay_weight(draw["weight_record"], [.1, .3], draw["ndd_record"])),
                               draw["log_importance_ratio"], places=12)
        self.assertEqual(provider.records[0]["status"], "critical_proposal")
        latest.loc[latest.id == 100, "x"] = -1000
        self.assertIsNone(provider(scene, {}, dist))
        self.assertEqual(provider.records[-1]["status"], "noncritical_natural")

    def test_original_33_action_contract_is_rejected(self):
        dist = PairActionDistribution((1, 2), ("a", "b"), np.ones((31, 31)) / 961, "fixture")
        with self.assertRaises(ValueError):
            CAVCriticalityProvider()(None, {}, dist)


if __name__ == "__main__":
    unittest.main()
