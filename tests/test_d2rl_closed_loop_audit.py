import json
import math
from pathlib import Path
import tempfile
import unittest
from scenario_reconstruction.d2rl_closed_loop_audit import summarize


class ClosedLoopAuditTests(unittest.TestCase):
    def test_single_actor_sample_cannot_be_missing_even_if_raw_weight_underflows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = self.make_pool(root)
            (root / 'tested_and_safe').mkdir()
            (root / 'tested_and_safe' / '1.json').write_text(json.dumps(dict(
                episode, collision_result=0, collision_id=None)))
            episode.update(weight_episode=0.0, log_importance_weight=-1000.0)
            episode['log_probability_step_info'] = {'1': {'log_importance_weight': -1000.0}}
            episode['online_policy_step_info'] = {'0': {
                'status': 'inferred', 'actor_ids': ['a', 'b'],
                'epsilon_by_bv_id': {'a': .5, 'b': .5},
                'sampled_terms': {'a': {'epsilon': .5, 'p': .1, 'q': .2, 'c': .3, 'weight': .5}},
            }}
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            result = summarize(root)
            self.assertFalse(result['audit_passed'])
            self.assertTrue(any('missing from probability ledger' in x for x in result['failures']))
            episode['log_probability_step_info']['0'] = {'log_importance_weight': math.log(.5)}
            episode['log_importance_weight'] += math.log(.5)
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            self.assertTrue(summarize(root)['audit_passed'])

    def make_pool(self, root):
        (root / 'crash').mkdir()
        (root / 'manifest_run_summary.json').write_text(json.dumps({
            'attempted': 2, 'results': [{'episode': 0, 'status': 'ok'}, {'episode': 1, 'status': 'ok'}],
        }))
        episode = {
            'collision_result': 1, 'collision_id': ['CAV', 'BV_primary'],
            'weight_episode': 1, 'log_importance_weight': 0,
            'log_probability_step_info': {},
        }
        (root / 'crash' / '0.json').write_text(json.dumps(episode))
        return episode

    def test_missing_safe_outcome_invalidates_probability_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_pool(root)
            result = summarize(root)
            self.assertFalse(result['audit_passed'])
            self.assertIsNone(result['conditional_weighted_cav_collision_mean'])

    def test_complete_pool_includes_safe_outcome_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = self.make_pool(root)
            (root / 'tested_and_safe').mkdir()
            episode.update(collision_result=0, collision_id=None)
            (root / 'tested_and_safe' / '1.json').write_text(json.dumps(episode))
            result = summarize(root)
            self.assertTrue(result['audit_passed'])
            self.assertEqual(result['conditional_weighted_cav_collision_mean'], .5)
            self.assertEqual(result['crash_contribution_ess'], 1)
            episode['weight_episode'] = .5
            (root / 'tested_and_safe' / '1.json').write_text(json.dumps(episode))
            self.assertFalse(summarize(root)['audit_passed'])


if __name__ == '__main__':
    unittest.main()
