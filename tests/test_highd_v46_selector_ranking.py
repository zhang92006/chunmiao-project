import numpy as np
import pytest

from scenario_reconstruction.highd_v46_selector_ranking import (
    METHODS, key, score, select, evaluate, native_metrics, tv_summary,
)
from scenario_reconstruction.highd_v46_conditional_baseline import conditional_observation


def row(seed, time=10., actors=(1, 2), interaction=.1, risk=.5, strict=True):
    return {'seed': seed, 'time_s': time, 'actor_ids': list(actors), 'mechanism': 'bracket',
            'metrics': {'positive_interaction_mass': interaction}, 'ranker_features': [1., .1, risk, 0.],
            'views': {'production_strict': strict}, 'actual_H_tv': [.6, .7]}


def labelled(r, events, weight=1.):
    state = {'cav_id': 0, 'vehicles': [{'id': i, 'x': float(12*i), 'v': 20., 'lane': 0, 'length': 5.} for i in range(3)]}
    case = {'source_seed': r['seed'], 'time_s': r['time_s'], 'row': r,
            'observation': conditional_observation(state, r)}
    audit = {'p_factors': [.5, .5], 'h_factors': [.5, .5], 'q_factors': [.5, .5], 'generation_epsilon': [.1, .1]}
    raw = [{'seed': i, 'event_result': event, 'bv_only_collision': False,
            'initial_likelihood_audit': audit, 'generation_log_weight': float(np.log(weight))} for i,event in enumerate(events)]
    return {'case': case, 'raw': raw, 'record': {'q_second_tv': .6}}


def test_fixed_three_scores_and_gate():
    a = row(1, interaction=.9, risk=.2)
    b = row(2, interaction=.1, risk=1.)
    c = row(3, interaction=.5, risk=.8)
    assert [select([a,b,c], m, 1)[0]['seed'] for m in METHODS] == [1,2,3]
    rejected = row(4, interaction=1., risk=1., strict=False)
    assert rejected not in select([rejected,a,b,c], METHODS[2], 4)


@pytest.mark.parametrize('method', METHODS)
def test_selection_ignores_native_labels_and_is_order_invariant(method):
    rows = [row(2), row(1), row(3)]
    expected = [key(r) for r in select(rows, method, 2)]
    for i,r in enumerate(rows):
        r.update(cav_events=100*i, native_risk=i, event_result=True)
    assert [key(r) for r in select(list(reversed(rows)), method, 2)] == expected


def test_pair_and_source_caps_ties_and_duplicate_rejection():
    rows = [row(1, 10., (1,2)), row(1,10.,(2,3)), row(1,11.), row(1,12.), row(2)]
    chosen = select(rows, METHODS[0], 5)
    assert [key(r) for r in chosen] == [key(rows[0]),key(rows[2]),key(rows[4])]
    with pytest.raises(ValueError, match='Duplicate'): select([rows[0],rows[0]], METHODS[0], 3)
    with pytest.raises(ValueError): select(rows, METHODS[0], 0)


@pytest.mark.parametrize('field,value', [('risk',float('nan')), ('risk',1.1), ('risk',-.1), ('interaction',-.1)])
def test_nonfinite_or_invalid_scores_rejected(field,value):
    r = row(1, **{field:value})
    with pytest.raises(ValueError): score(r, METHODS[0])


def test_unknown_native_rows_never_zero_filled():
    a,b = row(1),row(2)
    labels = {key(a):labelled(a,[True,False])}
    result = evaluate([a,b], labels, {1}, {key(a):.5})
    assert result['unknown_native_pairs'] == 1
    assert result['native'] is None
    assert result['known_collision_source_recall'] == 1.
    assert result['h_second_factor_tv']['minimum'] == .7


def test_zero_observed_events_are_not_missing_labels():
    a = row(1); labels = {key(a):labelled(a,[False,False])}
    result = evaluate([a], labels, set(), {key(a):0.})
    assert result['native']['continuation_collision_rate'] == 0.
    assert result['native']['concentration']['top_source_fraction'] is None
    assert result['known_collision_source_recall'] is None
    assert result['reference_law_retained_observed_second_moment_fraction'] is None


def test_same_source_different_state_does_not_count_as_witness_recall():
    a,b = row(1),row(1,11.)
    labels = {key(a):labelled(a,[True,False]), key(b):labelled(b,[False,False])}
    result = evaluate([b], labels, {1}, {key(a):.25,key(b):0.})
    assert result['source_episodes'] == 1 and result['known_collision_source_recall'] == 0.
    assert result['native']['event_source_episodes'] == 0


def test_source_weighted_concentration_and_fixed_reference_are_distinct():
    a,b,c = row(1),row(1,11.),row(2)
    labels = {key(a):labelled(a,[True,False],2.), key(b):labelled(b,[False,False]),
              key(c):labelled(c,[True,False])}
    result = evaluate([a,b,c],labels,{1,2},{key(a):.5,key(b):0.,key(c):.25})
    native = result['native']
    assert native['continuation_collision_rate'] == pytest.approx(2/6)
    assert native['source_state_weighted_collision_rate'] == pytest.approx(.375)
    assert native['concentration']['weighted_event_second_moment'] == pytest.approx(.75)
    assert native['concentration']['top_source_fraction'] == pytest.approx(2/3)
    reduced = evaluate([a,c],labels,{1,2},{key(a):.5,key(b):0.,key(c):.25})
    assert reduced['known_collision_source_recall'] == 1.
    assert reduced['native']['concentration']['top_source_fraction'] == pytest.approx(.8)
    assert reduced['reference_law_retained_observed_second_moment_fraction'] == 1.
    assert reduced['native']['q_second_factor_tv']['minimum'] == .6


def test_empty_selection_has_undefined_native_metrics():
    assert select([],METHODS[0],16) == []
    assert native_metrics([], {}) is None
    assert tv_summary([],[]) is None


def test_ranking_run_rejects_validation_source(tmp_path, monkeypatch):
    import scenario_reconstruction.highd_v46_selector_ranking as module
    from scenario_reconstruction.highd_v46_selector_ranking import run
    from scenario_reconstruction.highd_v46_assets import read, save, digest
    cfg=read('configs/highd_v46_selector_ranking_v1.json')
    source=tmp_path/'validation'
    save(source/'summary.json', {'complete':True, 'split':'validation',
         'config':read(cfg['sources'][0]['config'])})
    cfg['sources'][0].update(path=str(source),summary_sha256=digest(source/'summary.json'))
    monkeypatch.setattr(module,'ensure_assets',lambda:(None,{'natural_target_sha256':'P'}))
    save(tmp_path/'bad.json',cfg)
    with pytest.raises(ValueError, match='training calibrations'):
        run(tmp_path/'bad.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()
