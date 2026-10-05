import copy
import math

import pytest

from scenario_reconstruction.highd_v46_assets import digest, save
from scenario_reconstruction.highd_v46_collision_coverage import (
    CATEGORIES, audit_sample, run, summarize, support_category,
)


def sample(p=(.2, .5), h=(.5, 0.), epsilon=(.1, .1), event=True, seed=0):
    q = [e*pi+(1-e)*hi for pi, hi, e in zip(p, h, epsilon)]
    lw = sum(math.log(pi/qi) for pi, qi in zip(p, q))
    return {'seed': seed, 'event_result': event, 'generation_log_weight': lw,
            'initial_likelihood_audit': {'p_factors': list(p), 'h_factors': list(h),
                'q_factors': q, 'generation_epsilon': list(epsilon),
                'full_log_p': lw-20, 'full_log_q': -20}}


def case(source, weights_and_categories):
    events = [{'source_seed': source, 'case_id': source, 'seed': i, 'weight': w,
               'category': category, 'factor_lr': [w, 1.]}
              for i, (w, category) in enumerate(weights_and_categories)]
    counts = dict.fromkeys(CATEGORIES, 0)
    for e in events:
        counts[e['category']] += 1
    counts[CATEGORIES[0]] += 64-len(events)
    return {'source_seed': source, 'trials': 64, 'events': events, 'category_samples': counts}


def test_factor_tail_is_product_not_second_factor_alone():
    x = sample(); r = audit_sample(x, [.1, .1])
    assert r['category'] == 'H1_positive_H2_zero'
    assert r['factor_lr'] == pytest.approx([.2/.47, 10.])
    assert r['weight'] == pytest.approx((.2/.47)*10.)


def test_zero_first_row_has_natural_second_conditional():
    r = audit_sample(sample(h=(0., .5)), [.1, .1])
    assert r['category'] == 'H1_zero_H2_positive'
    assert r['factor_lr'] == pytest.approx([10., 1.])
    with pytest.raises(ValueError, match='natural H2 fallback'):
        audit_sample(sample(h=(0., 0.)), [.1, .1])


def test_exact_zero_and_disabled_factor_are_not_confused():
    assert support_category([1e-300, 1e-300]) == CATEGORIES[0]
    assert support_category([0., 0.]) == CATEGORIES[3]
    assert audit_sample(sample(epsilon=(.1, 1.)), [.1, 1.])['factor_lr'][1] == 1.
    assert audit_sample(sample(epsilon=(1., 1.)), [1., 1.])['weight'] == 1.


def test_high_absolute_risk_can_be_removed_by_baseline_interaction_contrast():
    import numpy as np
    from scenario_reconstruction.highd_v46_cross_unit import positive_interaction, proposal_factors
    p = np.array([[.6, .1], [.2, .1]])
    risk = np.array([[.04, .08], [.89, 1.]])
    interaction = positive_interaction(p, risk)
    assert risk[1, 0] > .8 and interaction[1, 0] == 0.
    assert np.array_equal(interaction[0], [0., 0.])
    h = p*interaction/(p*interaction).sum()
    _, p2, h1, h2, _, q2 = proposal_factors(p, h, [.1, .1])
    assert h1[1] > 0 and h2[1, 0] == 0.
    assert p2[1, 0]/q2[1, 0] == pytest.approx(10.)


@pytest.mark.parametrize('field', ['q_factors', 'generation_epsilon', 'full_log_q'])
def test_rejects_corrupt_mixture_or_likelihood(field):
    x = sample(); log = x['initial_likelihood_audit']
    if field == 'full_log_q': log[field] += 1.
    else: log[field][0] += .01
    with pytest.raises(ValueError): audit_sample(x, [.1, .1])


def test_natural_fallback_is_not_a_zero_H_pair():
    x = {'event_result': True, 'generation_log_weight': 0.,
         'initial_likelihood_audit': {'abstained': True, 'full_log_p': -2., 'full_log_q': -2.}}
    assert audit_sample(x, [.1, .1])['category'] == 'natural_fallback'
    x['generation_log_weight'] = 1.
    with pytest.raises(ValueError): audit_sample(x, [.1, .1])


def test_moment_categories_keep_zero_sources_and_sensitivity_is_separate():
    data = [case(1, [(10., CATEGORIES[2]), (1., CATEGORIES[0])]),
            case(2, [(1., CATEGORIES[0])]), case(3, [])]
    before = copy.deepcopy(data); r = summarize(data)
    assert r['collision_events'] == 3 and r['continuations'] == 192
    assert r['sum_Iw2'] == 102. and r['conditional_second_moment'] == 102/192
    assert r['categories'][CATEGORIES[2]]['second_moment_share'] == 100/102
    assert r['maximum_source_share'] == 101/102
    assert r['sensitivity_only']['remaining_maximum_source_share'] == .5
    assert r['sensitivity_only']['remaining_sum_Iw2'] == 2.
    assert data == before
    with pytest.raises(ValueError): summarize([data[0], data[0]])


def test_no_observed_events_has_undefined_shares():
    r = summarize([case(1, [])])
    assert r['conditional_second_moment'] == 0.
    assert r['maximum_source_share'] is None and r['largest_event'] is None
    assert all(c['second_moment_share'] is None for c in r['categories'].values())


def test_locked_collection_shared_cases_and_artifact_tampering(tmp_path):
    root = tmp_path/'collection'; root.mkdir(); eps = [.1, .1]
    cases = [{'case_id': 0, 'source_seed': 240000, 'selected_for': ['A', 'B'], 'natural_fallback': False},
             {'case_id': 1, 'source_seed': 240001, 'selected_for': ['A'], 'natural_fallback': False}]
    save(root/'case_plan.json', {'cases': cases, 'source_seeds': [240000, 240001], 'selection_uses_native_outcomes': False})
    for c in cases:
        ss = [sample(p=(.2, .5), h=(.2, 0.), event=c['case_id']==0 and i==0,
                     seed=2900000+c['case_id']*2+i) for i in range(2)]
        save(root/f'case_{c["case_id"]:03d}_dual.json', {'plan': c, 'epsilon': eps, 'sequences': ss})
    native = lambda n: {'native': {'cav_events': 1, 'continuations': n, 'conditional_second_moment': 100/n,
                                  'by_source': [{'source_seed': 240000, 'cav_events': 1}]}}
    summary = {'complete': True, 'split': 'validation', 'case_count': 2, 'config_sha256': 'source-config',
        'natural_target_sha256': 'P',
        'config': {'environment': {}, 'arms': {'dual': eps}, 'replicates_per_case_arm': 2, 'continuation_seed_start': 2900000},
        'selectors': {'A': {'arms': {'dual': native(4)}}, 'B': {'arms': {'dual': native(2)}}},
        'artifacts': {p.name: digest(p) for p in root.iterdir()}}
    save(root/'summary.json', summary)
    cfg = {'collection': str(root), 'summary_sha256': digest(root/'summary.json'), 'source_seed_range': [240000, 240001],
           'scope': 'test', 'environment': {}, 'pool_arms': False, 'fit_on_validation': False,
           'replay_largest_dual_proxy': False}
    save(tmp_path/'cfg.json', cfg)
    result = run(tmp_path/'cfg.json', tmp_path/'audit.json')
    assert result['unique_physical_sequences_checked'] == 4
    assert result['selectors']['A']['dual']['conditional_second_moment'] == pytest.approx(25.)
    assert result['selectors']['B']['dual']['conditional_second_moment'] == pytest.approx(50.)
    assert result['selectors']['B']['event_source_union_across_arms'] == [240000]
    with pytest.raises(FileExistsError): run(tmp_path/'cfg.json', tmp_path/'audit.json')
    (root/'case_000_dual.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact differs'): run(tmp_path/'cfg.json', tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()
