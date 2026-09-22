from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from controller.nadeglobalcontroller import NADEBVGlobalController
from scenario_reconstruction.multibv import build_multibv_joint_obs


class OnlinePolicyTests(unittest.TestCase):
    def controller(self):
        ctrl = NADEBVGlobalController.__new__(NADEBVGlobalController)
        ctrl.control_log = {}
        ctrl.online_intervention_decisions_used = 0
        ctrl.env = SimpleNamespace(
            online_epsilon_policy=Mock(compute_action=Mock(return_value=[.2, .8])),
            info_extractor=SimpleNamespace(episode_log={'weight_episode': 1.0}),
            simulator=SimpleNamespace(get_time=lambda: .1),
        )
        return ctrl

    def test_risk_order_does_not_swap_training_order_or_vehicle_actions(self):
        ctrl = self.controller()
        obs = {
            'CAV': {'position': [400, 46], 'velocity': 30},
            'BV_primary': {'position': [410, 46], 'velocity': 29},
            'BV_context': {'position': [395, 42], 'velocity': 28},
        }
        bvs = [SimpleNamespace(id='BV_primary'), SimpleNamespace(id='BV_context')]
        result = ctrl._online_epsilon_action(obs, bvs, [1, 0], [2, 1])
        self.assertEqual(result, {'BV_primary': .2, 'BV_context': .8})
        actual = ctrl.env.online_epsilon_policy.compute_action.call_args.args[0]
        np.testing.assert_equal(actual, build_multibv_joint_obs(obs, ['BV_primary', 'BV_context'], 1.0))
        ctrl.control_log = {}
        mapped, _ = ctrl._selected_epsilon_values(result, [1, 0], bvs)
        self.assertEqual(mapped, {1: .8, 0: .2})

    def test_missing_actor_and_noncritical_steps_are_audited_without_inference(self):
        ctrl = self.controller()
        bvs = [SimpleNamespace(id='BV_primary')]
        self.assertEqual(ctrl._online_epsilon_action({}, bvs, [0], [1]), {'BV_primary': 1.0})
        ctrl.env.online_epsilon_policy.compute_action.assert_not_called()
        self.assertIn('fallback', ctrl.control_log['online_policy']['status'])
        ctrl._online_epsilon_action({}, bvs, [0], [0])
        self.assertEqual(ctrl.control_log['online_policy']['status'], 'noncritical')

    def test_intervention_budget_counts_only_successful_critical_inference(self):
        ctrl = self.controller()
        ctrl.env.online_intervention_budget = 1
        bvs = [SimpleNamespace(id='BV_primary'), SimpleNamespace(id='BV_context')]
        ctrl._online_epsilon_action({}, bvs, [0, 1], [0, 0])
        self.assertEqual(ctrl.online_intervention_decisions_used, 0)
        first = ctrl._online_epsilon_action({
            'CAV': {'position': [400, 46], 'velocity': 30},
            'BV_primary': {'position': [410, 46], 'velocity': 29},
            'BV_context': {'position': [395, 42], 'velocity': 28},
        }, bvs, [0, 1], [1, 1])
        self.assertEqual(first, {'BV_primary': .2, 'BV_context': .8})
        second = ctrl._online_epsilon_action({}, bvs, [0, 1], [1, 1])
        self.assertEqual(second, {'BV_primary': 1.0, 'BV_context': 1.0})
        self.assertEqual(ctrl.online_intervention_decisions_used, 1)
        self.assertEqual(ctrl.control_log['online_policy']['status'], 'budget_exhausted_naturalistic')
        self.assertEqual(ctrl.env.online_epsilon_policy.compute_action.call_count, 1)

    def test_likelihood_ratio_guard_adjusts_only_the_unbounded_actor(self):
        ctrl = self.controller()
        ctrl.env.online_max_proposal_likelihood_ratio = 5.0
        candidates = [
            SimpleNamespace(
                id='BV_primary',
                controller=SimpleNamespace(get_NDD_possi=lambda: np.array([.9, .1])),
            ),
            SimpleNamespace(
                id='BV_context',
                controller=SimpleNamespace(get_NDD_possi=lambda: np.array([.5, .5])),
            ),
        ]
        applied, diagnostics = ctrl._apply_likelihood_ratio_guard(
            {0: .1, 1: .8}, [0, 1], candidates,
            [np.array([0., 1.]), np.array([.5, .5])],
        )
        self.assertAlmostEqual(applied[0], 5 / 9)
        self.assertEqual(applied[1], .8)
        self.assertEqual(diagnostics['adjusted_actor_ids'], ['BV_primary'])
        self.assertLessEqual(
            diagnostics['by_actor']['BV_primary']['applied_maximum_proposal_ratio'],
            5.0,
        )

    def test_likelihood_ratio_guard_can_target_context_actor_only(self):
        ctrl = self.controller()
        ctrl.env.online_max_proposal_likelihood_ratio = 5.0
        ctrl.env.online_likelihood_ratio_guard_actor_ids = ['BV_context']
        candidates = [
            SimpleNamespace(
                id='BV_primary',
                controller=SimpleNamespace(get_NDD_possi=lambda: np.array([.9, .1])),
            ),
            SimpleNamespace(
                id='BV_context',
                controller=SimpleNamespace(get_NDD_possi=lambda: np.array([.9, .1])),
            ),
        ]
        applied, diagnostics = ctrl._apply_likelihood_ratio_guard(
            {0: .1, 1: .1}, [0, 1], candidates,
            [np.array([0., 1.]), np.array([0., 1.])],
        )
        self.assertEqual(applied[0], .1)
        self.assertAlmostEqual(applied[1], 5 / 9)
        self.assertEqual(diagnostics['guarded_actor_ids'], ['BV_context'])
        self.assertEqual(set(diagnostics['by_actor']), {'BV_context'})


if __name__ == '__main__':
    unittest.main()
