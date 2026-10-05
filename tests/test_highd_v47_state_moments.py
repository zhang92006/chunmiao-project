from itertools import product

import numpy as np
import pytest
from scipy.stats import binom

from scenario_reconstruction.highd_v47_state_moments import (
    action_strata, binomial_bounds, estimate_state, moment_comparison,
)


def toy():
    p = np.array([[.1, .2], [.3, .4]])
    q = np.array([[.4, .1], [.1, .4]])
    f = np.array([[.8, .2], [.1, .6]])
    strata = action_strata(p, q, [[0, .4], [.6, .9]], [.5], [-1, 0], 20)
    return p, q, f, strata


def test_exact_enumeration_matches_J_and_removes_within_stratum_weight_noise():
    p, q, f, strata = toy()
    expectation = variance = 0.
    for h in strata:
        cells = h['cells']; g = np.array(h['measurement_probabilities']); c = h['coefficient_mass']
        fp = f.ravel()[cells]; probability = g@fp
        weights = p.ravel()[cells]**2/q.ravel()[cells]/g
        np.testing.assert_allclose(weights, c)
        expectation += c*probability
        variance += c*c*probability*(1-probability)/h['draws']
    assert np.isclose(expectation, np.sum(p*p*f/q))
    assert variance > 0
    assert sum(h['draws'] for h in strata) == 20
    assert sorted(cell for h in strata for cell in h['cells']) == list(range(4))


def test_bernoulli_variance_estimator_is_unbiased_not_variance_of_IS_mean():
    n, prob, mass = 8, .15, 3.7
    measured = []
    for k in range(n+1):
        rows = [{'stratum_id': 0, 'within_stratum_draw': j, 'event_result': j < k} for j in range(n)]
        estimate = estimate_state([{'stratum_id': 0, 'draws': n, 'coefficient_mass': mass}], rows, .95, 1)
        measured.append(estimate['estimator_variance'])
    assert np.isclose(binom.pmf(np.arange(n+1), n, prob)@measured, mass*mass*prob*(1-prob)/n)


def test_all_zero_retains_upper_uncertainty_and_cannot_certify_allocation():
    h = [{'stratum_id': 0, 'draws': 8, 'coefficient_mass': 10.}]
    rows = [{'stratum_id': 0, 'within_stratum_draw': j, 'event_result': False} for j in range(8)]
    e = estimate_state(h, rows, .95, 2)
    assert e['J'] == e['empirical_MCSE'] == 0
    assert e['state_interval'][1] > 0 and e['bank_simultaneous_interval'][1] >= e['state_interval'][1]
    assert not e['precision_supported']
    result = moment_comparison([.5, .5], [.9, .1], [e, e])
    assert result['simultaneous_M2_difference_interval'][1] > 0
    assert not result['certified_improvement']


def test_exact_binomial_interval_finite_sample_coverage_including_boundaries():
    n, error = 12, .05
    intervals = np.array([binomial_bounds(k, n, error) for k in range(n+1)])
    for probability in np.linspace(0, 1, 51):
        covered = (intervals[:, 0] <= probability) & (probability <= intervals[:, 1])
        assert binom.pmf(np.arange(n+1), n, probability)@covered >= 1-error-1e-12


def test_signed_interval_bound_holds_at_every_corner_and_can_certify():
    alpha = np.array([.5, .5]); q = np.array([.9, .1])
    estimates = [{'J': 100., 'estimator_variance': 1., 'bank_simultaneous_interval': [90., 110.]},
                 {'J': 1., 'estimator_variance': .01, 'bank_simultaneous_interval': [.5, 1.5]}]
    r = moment_comparison(alpha, q, estimates)
    lo, hi = r['simultaneous_M2_difference_interval']
    for j in product(*[e['bank_simultaneous_interval'] for e in estimates]):
        contrast = (alpha*alpha/q-alpha)@j
        assert lo-1e-12 <= contrast <= hi+1e-12
    assert r['certified_improvement']
    expected_var = np.sum((alpha*alpha/q-alpha)**2*np.array([1., .01]))
    assert np.isclose(r['difference_empirical_MCSE']**2, expected_var)


def test_risk_zero_actions_are_not_discarded():
    p = np.array([[.5, 0], [.2, .3]])
    h = action_strata(p, p, np.zeros((2, 2)), [.2], [-3, 0], 6)
    assert sorted(c for x in h for c in x['cells']) == [0, 2, 3]
    assert np.isclose(sum(x['coefficient_mass'] for x in h), 1)


def test_past_event_singletons_leave_full_complement_and_do_not_reuse_labels():
    p = np.array([[.1, .2], [.3, .4]])
    strata = action_strata(p, p, np.zeros((2, 2)), [.2], [-3, 0], 12, [1, 3])
    assert [h['cells'] for h in strata if h['kind'] == 'historical_event_action'] == [[1], [3]]
    assert [h['cells'] for h in strata if h['kind'] == 'remaining_support'] == [[0, 2]]
    rows = [{'stratum_id': h['stratum_id'], 'within_stratum_draw': j, 'event_result': False}
            for h in strata for j in range(h['draws'])]
    assert estimate_state(strata, rows, .95, len(strata))['J'] == 0
    with pytest.raises(ValueError):
        action_strata([[1., 0.]], [[1., 0.]], [[0., 0.]], [.2], [], 10, [1])


def test_small_supported_mass_is_not_lost_by_p_squared_underflow():
    p = np.array([[1., 1e-200]])
    h = action_strata(p, p, [[0., 0.]], [.2], [-3, 0], 4)
    assert h[0]['measurement_probabilities'][1] > 0


@pytest.mark.parametrize('budget', [0, 1, 2.5])
def test_reject_inadequate_stratum_budget(budget):
    with pytest.raises(ValueError):
        action_strata([[.5, .5]], [[.5, .5]], [[0., 1.]], [.5], [], budget)


def test_reject_changed_support_and_nonfinite():
    with pytest.raises(ValueError):
        action_strata([[.5, .5]], [[1., 0.]], [[0., 1.]], [.5], [], 4)
    with pytest.raises(ValueError):
        action_strata([[.5, .5]], [[.5, .5]], [[np.nan, 1.]], [.5], [], 4)


def test_incomplete_or_duplicate_measurement_rejected():
    h = [{'stratum_id': 0, 'draws': 2, 'coefficient_mass': 1.}]
    row = {'stratum_id': 0, 'within_stratum_draw': 0, 'event_result': False}
    for rows in ([row], [row, row]):
        with pytest.raises(ValueError):
            estimate_state(h, rows, .95, 1)


def test_same_probe_direct_variance_subtracts_unbiased_J_squared():
    from scenario_reconstruction.highd_v47_state_moments_audit import implied_direct_variance
    n, probability = 4, .2
    strata = [{'stratum_id': 0, 'draws': n, 'coefficient_mass': 1.}]
    values = []
    for k in range(n+1):
        phat = k/n
        probes = [{'stratum_id': 0, 'event_result': j < k, 'P_action': 1., 'target_QA_action': 1.} for j in range(n)]
        values.append(implied_direct_variance(strata, probes, phat, phat*(1-phat)/(n-1), n))
    assert np.isclose(binom.pmf(np.arange(n+1), n, probability)@values, probability*(1-probability)/n)


def test_two_state_run_saves_complete_fixed_plan_before_outcomes(tmp_path, monkeypatch):
    from scenario_reconstruction import highd_v47_state_moments as module
    from scenario_reconstruction.highd_v46_assets import save, read
    plans = [{'source_seed': i, 'target_mass': .5, 'actor_ids': [1, 2]} for i in range(2)]
    cfg = {'confidence': .95, 'draws_per_state_per_arm': 2, 'epsilon': [.1, .1],
           'horizon_s': 4, 'direct_seed_start': 100}
    measurements = []
    for i in range(2):
        measurements.append({'strata': [{'stratum_id': 0, 'draws': 2, 'coefficient_mass': 1.}],
                             'schedule': [{'stratum_id': 0, 'within_stratum_draw': j,
                                           'action_indices': [0, 0], 'native_seed': 200+2*i+j} for j in range(2)]})
    expected = {'mock_lock': 'test'}; lock_path = tmp_path/'lock.json'; save(lock_path, expected)
    out = tmp_path/'run'
    monkeypatch.setattr(module, 'inputs', lambda _: (cfg, {'proxy': {'clearance_scale_m': 1.}}, {}, tmp_path,
        plans, [{'state': {}, 'H': [[1.]]}]*2, measurements, {'target': [.5, .5]}, expected))
    monkeypatch.setattr(module.bank, 'PairModelQueries', lambda *a, **k: None)
    def native(*a, **k):
        assert (out/'measurement_plan.json').exists()
        return {'event_result': True, 'generation_log_weight': 0.}
    monkeypatch.setattr(module.bank, 'sample_continuation', native)
    monkeypatch.setattr(module, 'audit_sample', lambda *a: None)
    monkeypatch.setattr(module, 'continue_state', lambda *a: {'cav_first_collision': True})
    result = module.run('mock', lock_path, out)
    assert result['native_draws'] == 8 and len(result['artifact_hashes']) == 4
    assert read(out/'state_001.json')['state_id'] == 1
    assert all(e['J'] == 1 for e in result['state_estimates'])
    assert result['common_measurement_comparisons']['measurement_fitted_allocation']['difference_empirical_MCSE'] is None
    with pytest.raises(FileExistsError):
        module.run('mock', lock_path, out)
