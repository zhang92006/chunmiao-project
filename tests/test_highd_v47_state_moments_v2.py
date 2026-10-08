from itertools import product

import numpy as np
import pytest
from scipy.stats import binom

from scenario_reconstruction.highd_v47_state_moments_v2 import (
    integer_budget, partition, planning_ranges, make_schedules,
    bernoulli_variance_range, paired_variance, compare_state_proposal, summarize,
)


def config():
    return {'draws_per_state_per_arm': 20, 'coefficient_singleton_fraction': .001,
            'historical_singleton_fraction': .000001, 'minimum_stratum_draws': 2,
            'planning_beta_prior': [.5, .5], 'planning_probability_level': .95,
            'arms': ['equal', 'uncertainty_allocated'], 'batches': 2,
            'confidence': .95, 'continuation_seed_starts': [100, 1000], 'action_schedule_seed': 61}


def test_integer_allocation_exactly_minimizes_convex_objective():
    a = [100., 1., .01]; n = integer_budget(a, 11)
    feasible = [ns for ns in product(range(2, 8), repeat=3) if sum(ns) == 11]
    objective = lambda ns: sum(x/k for x, k in zip(a, ns))
    assert np.isclose(objective(n), min(objective(ns) for ns in feasible))
    assert objective(n) <= objective([4, 4, 3])
    assert n[0] > n[1] >= n[2]


def test_vector_floors_protect_event_strata_and_bound_true_variance():
    scores = [100., .01, 1.]; equal = np.array([4, 4, 3])
    floors = [2, 4, 2]; n = np.array(integer_budget(scores, 11, floors=floors))
    assert n[1] >= equal[1] and np.all(n >= .5*equal)
    for true_coefficients in ([0, 100., 1.], [100., 0, 0], [0, 0, 100.]):
        assert np.sum(np.array(true_coefficients)/n) <= 2*np.sum(np.array(true_coefficients)/equal)
    feasible = [ns for ns in product(range(2, 8), repeat=3) if sum(ns) == 11 and np.all(np.array(ns) >= floors)]
    assert np.isclose(sum(a/k for a,k in zip(scores,n)), min(sum(a/k for a,k in zip(scores,ns)) for ns in feasible))


@pytest.mark.parametrize('scores,budget', [([], 10), ([1, -1], 10), ([1, np.nan], 10), ([1, 1], 3), ([1], 2.5)])
def test_invalid_budgets_fail(scores, budget):
    with pytest.raises(ValueError): integer_budget(scores, budget)


def test_matrix_coefficient_head_covers_important_zero_risk_action_without_event_labels():
    p = np.array([[.49, .49], [.019, .001]])
    q = np.full((2, 2), .25)
    strata = partition(p, q, np.zeros_like(p), [], config())
    heads = [h['cells'][0] for h in strata if h['kind'] == 'coefficient_action']
    assert 0 in heads and 1 in heads
    assert sorted(c for h in strata for c in h['cells']) == [0, 1, 2, 3]
    assert np.isclose(sum(h['coefficient_mass'] for h in strata), np.sum(p*p/q))


def test_tiny_historical_actions_merge_but_remain_supported():
    p = np.array([[.999, .001], [1e-15, 0.]])
    p /= p.sum(); q = np.where(p > 0, 1/3, 0.)
    strata = partition(p, q, np.zeros_like(p), [2], config())
    assert any(2 in h['cells'] for h in strata)
    assert not any(h['kind'] == 'historical_event_action' and h['cells'] == [2] for h in strata)


def test_zero_history_does_not_zero_planning_variance():
    p = q = np.array([[.5, .5]])
    strata = partition(p, q, np.zeros_like(p), [], config())
    ranges = planning_ranges(strata, np.array([20, 0]), np.array([0, 0]), config())
    assert all(r['variance_upper'] > 0 and r['objective_coefficient'] > 0 for r in ranges)
    assert ranges[1]['variance_upper'] == .25


def test_variance_range_contains_all_feasible_bernoulli_variances():
    for lo, hi in [(0., .1), (.8, 1.), (.1, .9), (.5, .5)]:
        lower, upper = bernoulli_variance_range(lo, hi)
        for p in np.linspace(lo, hi, 20):
            assert lower-1e-15 <= p*(1-p) <= upper+1e-15


def test_shared_schedule_is_nested_complete_and_batches_disjoint():
    c = config(); p = np.array([[.4, .6]])
    h = [{'stratum_id': 0, 'cells': [0, 1], 'measurement_probabilities': [.4, .6], 'coefficient_mass': 1.}]
    states = [{'state_id': 0, 'target_QA': p.tolist(), 'strata': h,
               'counts': {'equal': [4], 'uncertainty_allocated': [7]}}]
    ds = [{'P': p.tolist()}]
    s = make_schedules(states, ds, c)
    assert s == make_schedules(states, ds, c)
    assert len(s[0]['probes']) == 7
    assert sum('equal' in r['arms'] for r in s[0]['probes']) == 4
    assert len({r['native_seed'] for b in s for r in b['probes']}) == 14
    assert all(r['arms'] == c['arms'] for r in s[0]['probes'][:4])


def test_pair_variance_and_covariance_unbiased_by_exact_bernoulli_enumeration():
    strata = [{'stratum_id': 0, 'coefficient_mass': 2.}]
    counts = {'equal': [2], 'uncertainty_allocated': [4]}
    probability = .3; n = 4
    ve = vu = contrast = 0.
    for k in range(n+1):
        rows = [{'stratum_id': 0, 'event_result': j < k} for j in range(n)]
        r = paired_variance(strata, counts, rows, .05)
        mass = binom.pmf(k, n, probability)
        ve += mass*r['equal_variance_same_probes']; vu += mass*r['allocated_variance_same_probes']
        contrast += mass*r['paired_difference_variance']
    assert np.isclose(ve, 4*probability*(1-probability)/2)
    assert np.isclose(vu, 4*probability*(1-probability)/4)
    assert np.isclose(contrast, 4*probability*(1-probability)*(1/2+1/4-2/4))


def test_zero_events_cannot_certify_precision_or_variance_win():
    strata = [{'stratum_id': 0, 'coefficient_mass': 10.}]
    rows = [{'stratum_id': 0, 'event_result': False} for _ in range(4)]
    r = paired_variance(strata, {'equal': [2], 'uncertainty_allocated': [4]}, rows, .05)
    assert r['variance_ratio_point_only'] is None
    assert not r['certified_variance_improvement']
    assert r['variance_difference_interval'][1] == 0


def test_state_identity_comparison_exactly_zero_and_never_certifies():
    a = [.04, .013333333333333334, .9466666666666667]
    estimates = [{'J': 1., 'estimator_variance': .1, 'bank_simultaneous_interval': [.5, 2.]}]*3
    r = compare_state_proposal(a, a, estimates)
    assert r['M2_difference_interval'] == [0., 0.]
    assert r['difference_empirical_MCSE'] == 0
    assert not r['certified_improvement']


def test_two_batch_summary_preserves_all_zero_states_and_averages_independent_variances():
    c = config(); p = np.array([[1.]])
    plans = [{'source_seed': 1, 'target_mass': 1.}]
    strata = [{'stratum_id': 0, 'cells': [0], 'measurement_probabilities': [1.], 'coefficient_mass': 1., 'draws': 4}]
    state = {'state_id': 0, 'target_QA': p.tolist(), 'strata': strata, 'counts': {arm: [4] for arm in c['arms']}}
    schedules = make_schedules([state], [{'P': p.tolist()}], c)
    for b in schedules:
        for r in b['probes']: r['event_result'] = False
    result = summarize(c, plans, {'states': [state], 'state_proposals': {'target': [1.]}}, schedules)
    assert len(result['state_estimates']) == 4
    for arm in c['arms']:
        state = result['arms'][arm]['states'][0]
        assert state['J'] == 0 and state['relative_empirical_MCSE'] is None
        assert state['bank_simultaneous_interval'][1] > 0


def test_worker_uses_each_shared_physical_draw_once(monkeypatch):
    from scenario_reconstruction import highd_v47_state_moments_v2 as module
    calls = []
    monkeypatch.setattr(module, '_MODEL', object(), raising=False)
    def native(model, assets, physical, state, forced, seed, horizon, scale):
        calls.append((forced, seed))
        return {'cav_first_collision': True, 'clearance_risk': 1., 'bv_only_collision': False}
    monkeypatch.setattr(module, 'continue_state', native)
    schedule = {'batch': 0, 'state_id': 0, 'probes': [
        {'action_indices': [1, 2], 'native_seed': 123, 'arms': ['equal', 'uncertainty_allocated']}]}
    result = module._collect((schedule, {}, {'actor_ids': [7, 9]}, '.', {}, 4, 1.))
    assert calls == [({7: 1, 9: 2}, 123)]
    assert result['probes'][0]['event_result'] is True
    assert schedule['probes'][0].get('event_result') is None


def test_exclusive_run_publishes_plan_before_workers_and_retains_all_cases(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction import highd_v47_state_moments_v2 as module
    from scenario_reconstruction.highd_v46_assets import save, read
    cfg = config(); cfg.update(workers=2, horizon_s=4)
    plans = [{'source_seed': 1, 'actor_ids': [1, 2], 'target_mass': 1.}]
    strata = [{'stratum_id': 0, 'cells': [0], 'measurement_probabilities': [1.], 'coefficient_mass': 1., 'draws': 4}]
    states = [{'state_id': 0, 'plan': plans[0], 'target_QA': [[1.]], 'strata': strata,
               'counts': {arm: [4] for arm in cfg['arms']}}]
    designs = [{'P': [[1.]], 'state': {}}]
    design = {'states': states, 'schedules': make_schedules(states, designs, cfg),
              'state_proposals': {'target': [1.]}, 'logical_draws': 16, 'unique_native_draws': 8}
    expected = {'test_lock': 'fixed'}; lock = tmp_path/'lock.json'; save(lock, expected)
    out = tmp_path/'run'
    monkeypatch.setattr(module, 'inputs', lambda _: (cfg, {'proxy': {'clearance_scale_m': 1.}}, {}, tmp_path,
                                                   plans, designs, design, expected))
    monkeypatch.setattr(module, 'ProcessPoolExecutor', lambda **_: ThreadPoolExecutor(max_workers=2))
    def collect(task):
        assert read(out/'measurement_plan.json') == design
        return dict(task[0], probes=[dict(p, event_result=False, native={'cav_first_collision': False}) for p in task[0]['probes']])
    monkeypatch.setattr(module, '_collect', collect)
    result = module.run('fake', lock, out)
    assert result['unique_native_draws'] == 8 and result['logical_draws'] == 16
    assert len(result['state_estimates']) == 4 and len(result['artifact_hashes']) == 4
    assert result['complete'] and not result['production_changed']
    with pytest.raises(FileExistsError): module.run('fake', lock, out)
