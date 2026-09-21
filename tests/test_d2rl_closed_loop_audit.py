import json
import math
from pathlib import Path
import tempfile
import unittest
from scenario_reconstruction.d2rl_closed_loop_audit import summarize


class ClosedLoopAuditTests(unittest.TestCase):
    def test_likelihood_ratio_guard_is_reconstructed_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = self.make_pool(root)
            run_path = root / 'manifest_run_summary.json'
            run = json.loads(run_path.read_text())
            run.update(
                online_policy={'model': 'test'},
                online_max_proposal_likelihood_ratio=5.0,
            )
            run_path.write_text(json.dumps(run))
            (root / 'tested_and_safe').mkdir()
            safe = dict(episode, collision_result=0, collision_id=None)
            (root / 'tested_and_safe' / '1.json').write_text(json.dumps(safe))
            log_weight = math.log(.2)
            episode.update(
                weight_episode=.2,
                log_importance_weight=log_weight,
                log_probability_step_info={'0': {'log_importance_weight': log_weight}},
                online_policy_step_info={'0': {
                    'status': 'inferred', 'actor_ids': ['a', 'b'],
                    'requested_epsilon_by_bv_id': {'a': .1, 'b': .8},
                    'epsilon_by_bv_id': {'a': .5, 'b': .8},
                    'likelihood_ratio_guard': {
                        'maximum_proposal_ratio': 5.0,
                        'adjusted_actor_ids': ['a'],
                        'by_actor': {
                            'a': {'applied_maximum_proposal_ratio': 5.0},
                            'b': {'applied_maximum_proposal_ratio': 2.0},
                        },
                    },
                    'sampled_terms': {
                        'a': {'epsilon': .5, 'p': .1, 'q': .5, 'c': .9, 'weight': .2},
                    },
                }},
            )
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            result = summarize(root)
            self.assertTrue(result['audit_passed'])
            self.assertEqual(result['likelihood_ratio_adjusted_steps'], 1)
            self.assertEqual(result['likelihood_ratio_adjusted_actor_counts'], {'a': 1})
            episode['online_policy_step_info']['0']['sampled_terms']['a']['q'] = .6
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            self.assertFalse(summarize(root)['audit_passed'])

    def test_budgeted_policy_records_one_inference_then_naturalistic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = self.make_pool(root)
            run_path = root / 'manifest_run_summary.json'
            run = json.loads(run_path.read_text())
            run.update(online_policy={'model': 'test'}, online_intervention_budget=1)
            run_path.write_text(json.dumps(run))
            (root / 'tested_and_safe').mkdir()
            episode['online_policy_step_info'] = {
                '0': {
                    'status': 'inferred', 'actor_ids': ['a', 'b'],
                    'intervention_budget': 1, 'decisions_used_before': 0,
                    'decisions_used_after': 1,
                    'epsilon_by_bv_id': {'a': .2, 'b': .8},
                },
                '1': {
                    'status': 'budget_exhausted_naturalistic', 'actor_ids': ['a', 'b'],
                    'intervention_budget': 1, 'decisions_used_before': 1,
                    'decisions_used_after': 1,
                    'epsilon_by_bv_id': {'a': 1, 'b': 1},
                },
            }
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            safe = dict(episode, collision_result=0, collision_id=None, online_policy_step_info={})
            (root / 'tested_and_safe' / '1.json').write_text(json.dumps(safe))
            result = summarize(root)
            self.assertTrue(result['audit_passed'])
            self.assertEqual(result['online_intervention_budget'], 1)
            self.assertEqual(result['online_step_status_counts'][
                'budget_exhausted_naturalistic'], 1)
            episode['online_policy_step_info']['1']['epsilon_by_bv_id']['a'] = .5
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            self.assertFalse(summarize(root)['audit_passed'])

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

    def test_subnormal_raw_weight_uses_authoritative_log_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = self.make_pool(root)
            (root / 'tested_and_safe').mkdir()
            (root / 'tested_and_safe' / '1.json').write_text(json.dumps(dict(
                episode, collision_result=0, collision_id=None)))
            log_weight = -743.2287963322533
            episode.update(
                weight_episode=math.exp(log_weight),
                log_importance_weight=log_weight,
                log_probability_step_info={'0': {'log_importance_weight': log_weight}},
            )
            (root / 'crash' / '0.json').write_text(json.dumps(episode))
            result = summarize(root)
            self.assertTrue(result['audit_passed'])
            self.assertEqual(result['raw_weight_status_counts']['subnormal_quantized'], 1)

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
