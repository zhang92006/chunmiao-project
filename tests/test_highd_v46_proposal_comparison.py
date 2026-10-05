import copy
from itertools import product

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import digest, read, save
from scenario_reconstruction.highd_v46_cross_unit import draw_intervention, proposal_factors
from scenario_reconstruction.highd_v46_pair_synergy import pair_marginal
from scenario_reconstruction.highd_v46_proposal_comparison import (
    DEFAULT_CONFIG, build_h, compare_reports, factor_tv, load_inputs, run, summarize,
)


def test_normalized_components_endpoints_and_reference_column_coverage():
    p = np.array([[.4, .1], [.2, .3]])
    risk = np.array([[.1, .1], [.9, 1.]])
    hi, meta = build_h(p, risk, 1.)
    hr, _ = build_h(p, risk, 0.)
    hm, _ = build_h(p, risk, .5)
    assert hi[1, 0] == 0 and hr[1, 0] > 0 and hm[1, 0] > 0
    np.testing.assert_allclose(hm, (hi+hr)/2)
    np.testing.assert_allclose(hr, p*risk/(p*risk).sum())
    assert not meta['component_natural_fallback']['interaction']
    _, _, _, h2, _, _ = proposal_factors(p, hm, [.1, .1])
    assert h2[1, 0] > 0


def test_zero_component_mass_declared_natural_fallback():
    p = np.array([[.4, .1], [.2, .3]])
    for lam in (0., .5, 1.):
        h, meta = build_h(p, np.zeros_like(p), lam)
        np.testing.assert_array_equal(h, p)
        assert all(meta['component_natural_fallback'].values())
    h, meta = build_h(p, np.array([[.1, .1], [.8, .8]]), .5)
    assert meta['component_natural_fallback']['interaction']
    np.testing.assert_allclose(h, .5*p+.5*(p*np.array([[.1, .1], [.8, .8]]))/(p*np.array([[.1, .1], [.8, .8]])).sum())


@pytest.mark.parametrize('lam', [0., .5, 1.])
def test_no_support_expansion_and_exact_enumeration(lam):
    p = np.array([[.6, 0.], [.1, .3]])
    h, _ = build_h(p, np.array([[.1, .2], [.8, 1.]]), lam)
    assert h[0, 1] == 0
    p1, p2, _, _, q1, q2 = proposal_factors(p, h, [.05, .1])
    q = q1[:, None]*q2
    supported = p > 0
    assert q.sum() == pytest.approx(1.)
    assert np.all(q[supported] >= .005*p[supported])
    w = p[supported]/q[supported]
    assert np.sum(q[supported]*w) == pytest.approx(1.)
    event = np.array([[0., 0.], [1., 1.]])
    assert np.sum(q[supported]*w*event[supported]) == pytest.approx(np.sum(p*event))


@pytest.mark.parametrize('ids', [[1, 2], [2, 1], [1, 3], [2, 3]])
@pytest.mark.parametrize('lam', [0., .5, 1.])
def test_new_scores_full_correlated_partner_likelihood(ids, lam):
    units = [{'ids': [1, 2], 'p': np.array([[.4, .1], [.2, .3]])},
             {'ids': [3, 4], 'p': np.array([[.1, .2], [.3, .4]])}]
    p, _ = pair_marginal(units, ids)
    h, _ = build_h(p, np.array([[.1, .2], [.3, 1.]]), lam)
    q1, q2 = proposal_factors(p, h, [.1, .2])[-2:]
    total = mean_weight = event_is = event_p = 0.
    for outcome in product(range(2), repeat=4):
        actions = dict(zip((1, 2, 3, 4), outcome)); a, b = [actions[i] for i in ids]
        natural = np.prod([u['p'][tuple(actions[i] for i in u['ids'])] for u in units])
        # Independent oracle: P(rest|controlled)=P(full)/P(controlled).
        q = q1[a]*q2[a, b]*natural/p[a, b]
        weight = p[a, b]/(q1[a]*q2[a, b])
        total += q; mean_weight += q*weight
        event = actions[2] == 1 and actions[4] == 0
        event_is += q*weight*event; event_p += natural*event
    assert total == pytest.approx(1.) and mean_weight == pytest.approx(1.)
    assert event_is == pytest.approx(event_p)
    for seed in range(8):
        _, log = draw_intervention(units, ids, h, [.1, .2], np.random.default_rng(seed))
        assert log['full_log_p']-log['full_log_q'] == pytest.approx(log['log_weight'])


@pytest.mark.parametrize('bad', ['negative_risk', 'nan', 'unnormalized', 'shape', 'lambda'])
def test_invalid_score_inputs_rejected(bad):
    p = np.ones((2, 2))/4; risk = np.ones((2, 2)); lam = .5
    if bad == 'negative_risk': risk[0, 0] = -1
    if bad == 'nan': p[0, 0] = np.nan
    if bad == 'unnormalized': p *= 2
    if bad == 'shape': risk = np.ones((2, 3))
    if bad == 'lambda': lam = 1.1
    with pytest.raises(ValueError): build_h(p, risk, lam)


def record(source, events):
    # Natural factor record with weight one. The supplied TV is independent
    # metadata here; native integration checks actual TV and factor records.
    seqs = [{'seed': j, 'event_result': event, 'generation_log_weight': 0.,
             'initial_likelihood_audit': {'p_factors': [.5, .5], 'h_factors': [.5, .5],
                'q_factors': [.5, .5], 'generation_epsilon': [.1, .1], 'full_log_p': 0., 'full_log_q': 0.}}
            for j, event in enumerate(events)]
    return {'source_seed': source, 'epsilon': [.1, .1], 'factor_tv': [.3, .4], 'sequences': seqs}


def test_source_then_state_weights_zeros_and_stratified_variance():
    records = [record(1, [True, False]), record(1, [False, False]), record(2, [True, True])]
    stats = read(DEFAULT_CONFIG)['statistics']
    report = summarize(records, stats)
    assert report['conditional_IS_mean'] == pytest.approx(.625)
    assert report['conditional_M2'] == pytest.approx(.625)
    assert report['fixed_state_stratified_estimator_variance'] == pytest.approx(.25**2*.5/2)
    assert report['maximum_source_M2_share'] == pytest.approx(.8)
    assert report['maximum_single_event_M2_share'] == pytest.approx(.4)
    assert report['event_H_categories']['H1_positive_H2_positive']['M2_share'] == pytest.approx(1.)
    assert report['by_source'][0]['source_M2'] == pytest.approx(.25)
    assert not report['event_interval_guard']
    zero = summarize([record(1, [False, False])], stats)
    assert zero['conditional_M2'] == 0
    assert zero['maximum_source_M2_share'] is None
    assert zero['maximum_single_event_M2_share'] is None


def test_factor_ablations_and_sparse_source_bootstrap():
    p = np.array([[.4, .1], [.2, .3]]); h, _ = build_h(p, np.array([[.1, .2], [.8, 1.]]), .5)
    assert factor_tv(p, h, [1., 1.]) == pytest.approx([0., 0.])
    assert factor_tv(p, h, [.1, 1.])[1] == 0
    assert factor_tv(p, h, [1., .1])[0] == 0
    stats = read(DEFAULT_CONFIG)['statistics']; stats['bootstrap_repeats'] = 100
    report = summarize([record(1, [True, False]), record(2, [False, False])], stats)
    result = compare_reports({'interaction': report, 'mixed': copy.deepcopy(report)}, stats)
    comp = result['comparisons']['mixed']
    assert comp['M2_ratio_to_interaction'] == pytest.approx(1.)
    assert comp['M2_ratio_source_interval']['interval'] is None
    assert comp['M2_ratio_source_interval']['censored_replicates'] == 100
    bad = copy.deepcopy(report); bad['by_source'].reverse()
    with pytest.raises(ValueError): compare_reports({'interaction': report, 'mixed': bad}, stats)


def test_supported_paired_bootstrap_keeps_whole_sources():
    stats = read(DEFAULT_CONFIG)['statistics']; stats['bootstrap_repeats'] = 100
    report = summarize([record(i, [True, False]*4) for i in range(12)], stats)
    result = compare_reports({'interaction': report, 'mixed': copy.deepcopy(report)}, stats)
    assert result['comparisons']['mixed']['M2_ratio_source_interval']['interval'] == pytest.approx([1., 1.])


def test_loader_rejects_validation_before_opening_any_source(tmp_path):
    cfg = read(DEFAULT_CONFIG); cfg['split'] = 'validation'
    path = tmp_path/'config.json'; save(path, cfg)
    with pytest.raises(ValueError, match='training'): load_inputs(path)


def test_loader_rejects_changed_source_checksum(tmp_path):
    # Config passes contract/environment but no corrupted manifest is accepted.
    from scenario_reconstruction.highd_v46_training_expansion import load_calibration
    save(tmp_path/'summary.json', {})
    with pytest.raises(ValueError, match='checksum'):
        load_calibration(tmp_path, {}, 'invalid')


def test_two_case_exclusive_output_and_preoutcome_plan(monkeypatch, tmp_path):
    from scenario_reconstruction import highd_v46_proposal_comparison as module
    cfg = read(DEFAULT_CONFIG); cfg['replicates_per_case_candidate'] = 2
    cfg['statistics']['bootstrap_repeats'] = 10
    p = np.array([[.4, .1], [.2, .3]]); risk = np.array([[.1, .2], [.3, 1.]])
    proposals = {name: {'ids': [1, 2], 'h': build_h(p, risk, lam)[0],
                       'factor_tv': factor_tv(p, build_h(p, risk, lam)[0], cfg['epsilon'])}
                 for name, lam in cfg['candidates'].items()}
    cases = [{'state': {'seed': i}} for i in (1, 2)]
    plans = [{'source_seed': i, 'continuation_seed_start': 100+i*2} for i in (1, 2)]
    lock = {'version': 1}
    monkeypatch.setattr(module, 'load_inputs', lambda path: (cfg, cases, plans, {'proxy': {'clearance_scale_m': 2}}, {}, tmp_path, lock))
    monkeypatch.setattr(module, 'PairModelQueries', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'prepare', lambda *args: (p, risk, proposals))
    monkeypatch.setattr(module, 'replay_event_coverage', lambda *args: [])
    def sample(model, assets, physical, state, chosen, epsilon, seed, config, scale):
        destination = tmp_path/'run'
        assert (destination/'case_plan.json').exists()
        assert all((destination/f'case_{i:02d}_proposals.json').exists() for i in (0, 1))
        _, log = draw_intervention([{'ids': [1, 2], 'p': p}], [1, 2], chosen['h'], epsilon, np.random.default_rng(seed))
        return {'seed': seed, 'event_result': bool(seed % 2), 'generation_log_weight': log['log_weight'],
                'initial_likelihood_audit': log}
    monkeypatch.setattr(module, 'sample_continuation', sample)
    lock_path = tmp_path/'lock.json'; save(lock_path, lock)
    result = run('unused', lock_path, tmp_path/'run')
    assert result['complete'] and not result['validation_sources_opened'] and not result['ppo_trained']
    assert len(result['artifact_hashes']) == 11
    for name, sha in result['artifact_hashes'].items(): assert digest(tmp_path/'run'/name) == sha
    with pytest.raises(FileExistsError): run('unused', lock_path, tmp_path/'run')
    bad_lock = tmp_path/'changed_lock.json'; save(bad_lock, {'version': 2})
    with pytest.raises(ValueError, match='freeze'): run('unused', bad_lock, tmp_path/'not_opened')
    assert not (tmp_path/'not_opened').exists()
