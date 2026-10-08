import copy
from collections import Counter
from itertools import product

import numpy as np
import pytest
from scipy.optimize import minimize

from scenario_reconstruction.highd_v46_assets import read, save
from scenario_reconstruction.highd_v46_cross_unit import draw_intervention, proposal_factors
from scenario_reconstruction.highd_v46_pair_synergy import pair_marginal
from scenario_reconstruction.highd_v47_allocation import (
    CONTRACT, allocate, bin_indices, check_q, draw_allocation, factor_tv,
    fit_calibrator, natural_factors, second_moment, waterfill,
)
from scenario_reconstruction.highd_v47_probability_allocation import (
    DEFAULT_CONFIG, CANDIDATES, candidates_for_case, crossfit, run,
    summarize_offpolicy, validate_config,
)


def test_stochastic_first_action_optimum_is_square_root_not_event_probability():
    p = np.full((2, 2), .25)
    failure = np.array([[.01, .25], [.36, .81]])
    q1, q2 = allocate(p, failure, [.05, .05])
    expected = p*np.sqrt(failure); expected /= expected.sum()
    np.testing.assert_allclose(q1[:, None]*q2, expected, atol=1e-14)
    linear = p*failure; linear /= linear.sum()
    assert second_moment(p, failure, expected) < second_moment(p, failure, linear)
    assert second_moment(p, failure, expected) == pytest.approx((p*np.sqrt(failure)).sum()**2)
    # Bernoulli residual randomness remains; first-action optimization is not zero variance.
    assert second_moment(p, failure, expected) > (p*failure).sum()**2


@pytest.mark.parametrize('epsilon', [[.05, .05], [.1, .2], [1., .1], [.1, 1.], [1., 1.]])
def test_allocator_matches_independent_convex_joint_solver(epsilon):
    p = np.array([[.08, .02, .10], [.30, .45, .05]])
    f = np.array([[.00001, .8, .001], [.1, .03, .8]])
    q1, q2 = allocate(p, f, epsilon)
    q = q1[:, None]*q2
    p1, p2 = natural_factors(p)[1:]
    # Independent joint-coordinate oracle: row sums >= eps1*P1 and
    # Qij >= eps2*P(j|i)*sum_j Qij are linear constraints in Qij.
    rows = np.kron(np.eye(2), np.ones((1, 3)))
    conditional = np.eye(6)
    for i in range(6):
        conditional[i] -= epsilon[1]*p2.flat[i]*rows[i//3]
    coeff = (p*p*f).ravel()
    result = minimize(lambda x: np.sum(coeff/x), p.ravel(),
        jac=lambda x: -coeff/x**2, method='SLSQP',
        bounds=list(zip((epsilon[0]*epsilon[1]*p).ravel(), np.ones(6))),
        constraints=[{'type': 'eq', 'fun': lambda x: x.sum()-1, 'jac': lambda x: np.ones(6)},
            {'type': 'ineq', 'fun': lambda x: rows@x-epsilon[0]*p1, 'jac': lambda x: rows},
            {'type': 'ineq', 'fun': lambda x: conditional@x, 'jac': lambda x: conditional}],
        options={'ftol': 1e-12, 'maxiter': 1000})
    assert result.success, result.message
    assert second_moment(p, f, q) == pytest.approx(result.fun, rel=1e-7, abs=1e-10)


def test_floor_active_first_row_can_still_allocate_second_factor():
    p = np.full((2, 2), .25)
    f = np.array([[1e-10, 1e-6], [.9, .9]])
    q1, q2 = allocate(p, f)
    assert q1[0] == pytest.approx(.1*p[0].sum())
    assert q2[0, 1] > .9
    # This row cannot be represented by the old joint-H zero-marginal fallback.
    assert not np.allclose(q2[0], [.5, .5])
    first = allocate(p, f, first_only=True)
    assert second_moment(p, f, q1[:, None]*q2) <= second_moment(p, f, first[0][:, None]*first[1])


def test_zero_risk_and_zero_support_rows_return_valid_natural_fallback():
    p = np.array([[.2, .3, 0], [0, 0, 0], [.1, .4, 0]])
    q1, q2 = allocate(p, np.zeros_like(p))
    np.testing.assert_allclose(q1[:, None]*q2, p)
    assert factor_tv(p, q1, q2) == pytest.approx([0, 0])
    assert np.all(q2[1] == 0) and np.all(q2[:, 2] == 0)
    np.testing.assert_allclose(waterfill([.4, .6, 0], [1e-300, 4e-300, 1e300], .1), [.25, .75, 0])


@pytest.mark.parametrize('bad', ['negative_p', 'unnormalized', 'nan_f', 'f_above_one', 'shape', 'epsilon'])
def test_bad_allocation_inputs_rejected(bad):
    p = np.full((2, 2), .25); f = np.full((2, 2), .5); eps = [.1, .1]
    if bad == 'negative_p': p[0, 0] = -.1
    if bad == 'unnormalized': p *= 2
    if bad == 'nan_f': f[0, 0] = np.nan
    if bad == 'f_above_one': f[0, 0] = 1.1
    if bad == 'shape': f = np.ones((2, 3))
    if bad == 'epsilon': eps = [.01, .1]
    with pytest.raises(ValueError): allocate(p, f, eps)


def partition():
    return [{'ids': [1, 2], 'p': np.array([[.4, .1], [.2, .3]])},
            {'ids': [3, 4], 'p': np.array([[.1, .2], [.3, .4]])}]


@pytest.mark.parametrize('ids', [[1, 2], [2, 1], [1, 3], [2, 3]])
def test_full_four_actor_likelihood_and_actual_sampling(ids):
    units = partition(); p, _ = pair_marginal(units, ids)
    q1, q2 = allocate(p, np.array([[.001, .1], [.02, .9]]))
    expected = {}; natural_event = weighted_event = mean_w = 0.
    for vals in product(range(2), repeat=4):
        actions = dict(zip((1, 2, 3, 4), vals)); a, b = [actions[i] for i in ids]
        full_p = np.prod([u['p'][tuple(actions[i] for i in u['ids'])] for u in units])
        full_q = q1[a]*q2[a, b]*full_p/p[a, b]
        expected[vals] = full_q
        weight = p[a, b]/(q1[a]*q2[a, b]); event = actions[2] == 1 and actions[4] == 0
        mean_w += full_q*weight; natural_event += full_p*event; weighted_event += full_q*weight*event
    assert sum(expected.values()) == pytest.approx(1.) and mean_w == pytest.approx(1.)
    assert natural_event == pytest.approx(weighted_event)
    rng = np.random.default_rng(4701); counts = Counter()
    for _ in range(2400):
        actions, log = draw_allocation(units, ids, q1, q2, [.1, .1], rng)
        vals = tuple(actions[i] for i in (1, 2, 3, 4)); counts[vals] += 1
        full_p = np.prod([u['p'][tuple(actions[i] for i in u['ids'])] for u in units])
        assert log['contract'] == CONTRACT and 'h_factors' not in log
        assert log['full_log_q'] == pytest.approx(np.log(expected[vals]))
        assert log['log_weight'] == pytest.approx(np.log(full_p/expected[vals]))
    assert all(abs(counts[v]/2400-q) < .04 for v, q in expected.items())


def test_direct_sampler_exactly_replays_legacy_q_without_rng_change():
    units = partition(); ids = [2, 3]; p, _ = pair_marginal(units, ids)
    h = p*np.array([[1., 2.], [3., 9.]]); h /= h.sum()
    q1, q2 = proposal_factors(p, h, [.1, .1])[-2:]
    for seed in range(32):
        old, old_log = draw_intervention(units, ids, h, [.1, .1], np.random.default_rng(seed))
        new, new_log = draw_allocation(units, ids, q1, q2, [.1, .1], np.random.default_rng(seed))
        assert old == new
        assert old_log['log_weight'] == new_log['log_weight']
        assert old_log['full_log_q'] == new_log['full_log_q']


@pytest.mark.parametrize('bad', ['support', 'row_sum', 'floor', 'negative', 'nan'])
def test_invalid_direct_q_is_rejected(bad):
    p = np.array([[.5, 0], [.25, .25]]); q1, q2 = allocate(p, np.ones_like(p))
    if bad == 'support': q2[0] = [.5, .5]
    if bad == 'row_sum': q2[1] = [.3, .3]
    if bad == 'floor': q1[:] = [.01, .99]
    if bad == 'negative': q2[1] = [-.1, 1.1]
    if bad == 'nan': q1[0] = np.nan
    with pytest.raises(ValueError): check_q(p, q1, q2, [.1, .1])


def toy_rows():
    return [{'case_id': s-1, 'source_seed': s, 'component': 'g', 'seed': i,
             'action': [0, i], 'event': bool(s == 1 and i == 1),
             'bin': i, 'p': .5, 'behavior_q': .5, 'design_mass': 1/6}
            for s in (1, 2, 3) for i in range(2)]


def test_whole_source_exclusion_prevents_label_leakage():
    cfg = read(DEFAULT_CONFIG)['calibration']; rows = toy_rows()
    first = fit_calibrator(rows, cfg, excluded_source=1)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row['source_seed'] == 1:
            row.update(event=not row['event'], p=.1, behavior_q=.01, bin=9)
    assert fit_calibrator(changed, cfg, excluded_source=1) == first
    assert first['source_seeds'] == [2, 3] and first['event_sources'] == []
    assert np.all(np.asarray(first['probabilities']) >= cfg['probability_floor'])


def test_calibrator_corrects_behavior_and_counts_sources_not_rows():
    cfg = read(DEFAULT_CONFIG)['calibration']; cfg['regularization_mass'] = 1e-12
    # Q samples the event action three times as often; P assigns equal mass.
    rows = [{'source_seed': 1, 'design_mass': .25, 'p': .5,
             'behavior_q': .75 if y else .25, 'event': bool(y), 'bin': 0}
            for y in (1, 1, 1, 0)]
    model = fit_calibrator(rows, cfg)
    assert model['probabilities'][0] == pytest.approx(.5)
    assert model['bin_event_source_counts'][0] == 1
    assert model['global_probability'] == pytest.approx(.5)
    assert model['probabilities'][9] == pytest.approx(.5)  # empty-bin declared shrinkage


def test_bin_edges_and_interaction_threshold_are_predeclared():
    cfg = read(DEFAULT_CONFIG)['calibration']
    np.testing.assert_array_equal(bin_indices([0., .2, 1.], [0., .049, .05], cfg), [0, 2, 9])
    with pytest.raises(ValueError): bin_indices([1.1], [0.], cfg)


def test_offpolicy_m2_recovers_exact_expectation_for_distinct_behaviors():
    p = np.array([[.1, .2], [.3, .4]])
    f = np.array([[.01, .1], [.3, .8]])
    q1, q2 = allocate(p, f); q = q1[:, None]*q2
    records = [{'case_id': 0, 'source_seed': 1, 'target_mass': 1.}]
    candidates = {name: {'q': q, 'plugin_M2': second_moment(p, f, q), 'factor_tv': [0., 0.]}
                  for name in CANDIDATES}
    rows = []
    for index in range(7):
        g = p**(1+index/7); g /= g.sum()
        for a, b in np.ndindex(p.shape):
            for y in (False, True):
                # Enumerate the expectation under each g and Bernoulli outcome.
                rows.append({'case_id': 0, 'source_seed': 1, 'component': str(index),
                    'seed': 100*a+10*b+int(y), 'action': [a, b], 'event': y,
                    'design_mass': g[a, b]*(f[a, b] if y else 1-f[a, b])/7,
                    'p': p[a, b], 'behavior_q': g[a, b]})
    reports = summarize_offpolicy(records, rows, {0: {'proposals': candidates}})
    for report in reports.values():
        assert report['offpolicy_conditional_M2'] == pytest.approx(second_moment(p, f, q))
        assert report['behavior_IS_conditional_mean'] == pytest.approx((p*f).sum())
        assert report['new_Q_event_count'] is None
        assert all(x == pytest.approx(second_moment(p, f, q)) for x in report['by_behavior_component_M2'].values())


@pytest.mark.parametrize('field,value', [('split', 'validation'), ('new_native_continuations', 64),
    ('expected_cases', 30), ('minimum_event_sources_for_calibration_review', 2), ('epsilon', [.05, .1])])
def test_protocol_rejects_changed_scope_and_relaxed_guards(field, value):
    cfg = read(DEFAULT_CONFIG); cfg[field] = value
    with pytest.raises(ValueError): validate_config(cfg)


def test_freeze_mismatch_fails_before_creating_destination(tmp_path, monkeypatch):
    import scenario_reconstruction.highd_v47_probability_allocation as module
    save(tmp_path/'lock.json', {'implementation': 'old'})
    monkeypatch.setattr(module, 'load_inputs', lambda config: ({}, [], [], {'implementation': 'new'}))
    with pytest.raises(ValueError, match='changed after freeze'):
        run('unused', tmp_path/'lock.json', tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_crossfit_uses_same_source_target_for_all_candidates():
    cfg = read(DEFAULT_CONFIG); rows = toy_rows(); records = []
    for s in (1, 2, 3):
        p = np.array([[.5, .5]]); risk = np.array([[.1, .9]])
        records.append({'case_id': s-1, 'source_seed': s, 'target_mass': 1/3, 'p': p,
            'risk': risk, 'bins': np.array([[0, 1]]), 'proposals': {k: p for k in ('interaction', 'absolute_risk', 'mixed')}})
    models, prepared = crossfit(records, rows, cfg)
    assert all(source not in model['source_seeds'] for source, model in models.items())
    for item in prepared.values():
        values = [v['plugin_M2'] for v in item['proposals'].values()]
        assert item['proposals']['allocated_dual']['plugin_M2'] == pytest.approx(min(values))
