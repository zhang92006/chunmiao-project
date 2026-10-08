import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import ROOT, read, save
from scenario_reconstruction.highd_v46_native_calibration import (
    aggregate, case_masses, fit_binomial, predict, ranker_report, select_cases, validate_config,
)


def config():
    return read(ROOT / 'configs/highd_v46_native_calibration_v1.json')


def test_calibration_is_training_only_and_rejects_consumed_sources():
    cfg = config()
    dc = read(ROOT / cfg['discovery_config'])
    validate_config(cfg, dc)
    for start in (100000, 160000, 170000, 180000, 190000):
        bad = copy.deepcopy(cfg)
        bad['source']['seed_start'] = start
        with pytest.raises(ValueError):
            validate_config(bad, dc)
    bad = copy.deepcopy(cfg)
    bad['source']['split'] = 'validation'
    with pytest.raises(ValueError):
        validate_config(bad, dc)
    dc['pairs']['ranked'] = 28
    with pytest.raises(ValueError):
        validate_config(cfg, dc)


def test_case_budget_diversity_and_no_native_label_selection():
    bank = [{'seed': seed, 'time_s': t} for seed in range(6) for t in range(3)]
    rows = [{'state_index': i, 'seed': s['seed'], 'mechanism': 'lateral' if i % 3 == 2 else 'bracket',
             'views': {'production_strict': s['seed'] < 4},
             'metrics': {'positive_interaction_mass': 3-i % 3}} for i, s in enumerate(bank)]
    cases = select_cases(bank, rows, config()['cases'])
    assert len(cases) == 10
    eligible = [c for c in cases if c['role'] == 'eligible']
    assert len(eligible) == 8
    for first, second in zip(eligible[::2], eligible[1::2]):
        assert first['state_index'] != second['state_index']
        assert rows[first['row_index']]['mechanism'] == 'bracket'
        assert rows[second['row_index']]['mechanism'] == 'lateral'
    for row in rows:
        row['native_events'] = 100-row['state_index']
    assert cases == select_cases(bank, rows, config()['cases'])
    assert all(c['row_index'] is None for c in cases[-2:])


def test_source_mass_and_precision_guard_do_not_double_count_states():
    records = [{'source_seed': s} for s in (1, 1, 2)]
    np.testing.assert_array_equal(case_masses(records), [.25, .25, .5])
    values = {a: [np.ones(64), np.ones(64), np.zeros(64)] for a in config()['arms']}
    result = aggregate(records, values, config()['statistics'])
    for arm in result['arms'].values():
        assert arm['conditional_event_mean'] == .5
        assert arm['weighted_event_second_moment'] == .5
        assert arm['event_source_episodes'] == 1
        assert not arm['evidence_guard_passed']
    assert not result['dual_comparisons']['first_factor_only']['lower_dual_second_moment_supported']
    values = {a: [np.array([0., 1.]), np.array([0., 2.]), np.array([0., 4.])] for a in config()['arms']}
    result = aggregate(records, values, config()['statistics'])
    expected = (.25**2*.5 + .25**2*2 + .5**2*8) / 2
    assert result['arms']['dual']['stratified_standard_error']**2 == pytest.approx(expected)
    assert aggregate([], {}, config()['statistics'])['precision_status'] == 'no_eligible_cases'


def test_regularized_binomial_model_and_zero_labels_guard():
    cfg = config()['ranker']
    x = [[1., .1, .2, 0.], [1., .1, .2, 0.], [1., .8, .9, 1.], [1., .8, .9, 1.]]
    model = fit_binomial(x, [0, 1, 20, 24], [64]*4, [1, 2, 3, 4], cfg)
    assert model['converged'] and model['deployment_status'] == 'training_only_requires_fresh_validation'
    p = predict(model, x)
    assert 0 < p[0] < p[2] < 1
    with pytest.raises(ValueError):
        fit_binomial(x, [0]*4, [64]*4, [1, 2, 3, 4], cfg)
    with pytest.raises(ValueError):
        fit_binomial(x, [65]*4, [64]*4, [1, 2, 3, 4], cfg)


def labeled_records():
    return [{'source_seed': seed, 'state_index': seed*10+t, 'role': 'eligible',
             'ranker_features': [1., .1 if seed >= 4 else .8, .9, float(seed < 4)],
             'arms': {'dual': {'positive_episodes': (8+t if seed < 4 else 0), 'episodes': 64}}}
            for seed in range(1, 8) for t in (0, 1)]


def test_ranker_requires_source_diversity_and_leave_episode_out_has_no_label_leakage():
    cfg = config()['ranker']
    records = labeled_records()
    result = ranker_report(records, cfg)
    assert result['status'] == 'fitted_training_only'
    predictions = result['leave_source_out_predictions']
    assert len(predictions) == len(records)
    assert all(p['source_seed'] not in p['training_source_seeds'] for p in predictions)
    changed = copy.deepcopy(records)
    for record in changed:
        if record['source_seed'] == 1:
            record['arms']['dual']['positive_episodes'] = 63
    other = ranker_report(changed, cfg)['leave_source_out_predictions']
    assert [p['predicted_dual_event_rate'] for p in predictions if p['source_seed'] == 1] == [p['predicted_dual_event_rate'] for p in other if p['source_seed'] == 1]
    two = [r for r in records if r['source_seed'] != 3]
    result = ranker_report(two, cfg)
    assert result['status'] == 'insufficient_native_training_labels' and result['model'] is None


@pytest.mark.parametrize('positive', [False, True])
def test_multi_case_smoke_retains_negatives_and_never_exports_ppo(tmp_path, monkeypatch, positive):
    from scenario_reconstruction import highd_v46_native_calibration as module
    monkeypatch.chdir(ROOT)
    cfg = config()
    cfg['source']['episodes'] = 1
    cfg['replicates_per_case_arm'] = 2
    save(tmp_path / 'config.json', cfg)
    monkeypatch.setattr(module, 'environment', lambda: cfg['environment'])
    monkeypatch.setattr(module, 'ensure_assets', lambda: (tmp_path, {'natural_target_sha256': 'P'}))
    monkeypatch.setattr(module, 'experiment_target', lambda *args: {'target': 'P'})
    monkeypatch.setattr(module, 'PairModelQueries', lambda *args, **kwargs: None)

    class Road:
        def __init__(self, *args):
            self.time, self.crashes, self.cav = 0., [], SimpleNamespace(id=0)

        def step_policy(self, mode, epsilon):
            self.choose([], [], {}, mode)
            self.time += 1
            return self.time < 7

    monkeypatch.setattr(module, 'NativeExperiment', Road)
    monkeypatch.setattr(module, 'screen_pairs', lambda *args: [
        {'actor_ids': ids, 'score': 1., 'mechanism': 'bracket'} for ids in ([1, 2], [1, 3], [2, 3])])
    monkeypatch.setattr(module, 'snapshot', lambda road, units, pmfs, seed, features:
                        {'seed': seed, 'time_s': road.time, 'cav_id': 0, 'vehicles': [{'id': i} for i in range(4)],
                         'units': [{'ids': [i], 'p': [.5, .5]} for i in (1, 2, 3)], 'pmfs': {str(i): [.5, .5] for i in range(4)}})
    risk = np.array([[.1, .1], [.1, 1.]]) if positive else np.zeros((2, 2))
    monkeypatch.setattr(module, 'response_risk', lambda *args, **kwargs: risk)
    monkeypatch.setattr(module, 'prepare_case', lambda state, row, *args: None if row is None else
                        {'ids': row['actor_ids'], 'mass': .225, 'factor_tv': [.5, .5], 'q_second_tv': .45, 'mechanism': 'bracket'})
    monkeypatch.setattr(module, 'sample_continuation', lambda *args:
                        {'event_result': False, 'generation_log_weight': 0., 'bv_only_collision': False,
                         'initial_likelihood_audit': {'full_log_p': -1., 'full_log_q': -1.},
                         'clearance_risk': .1, 'cpu_s': .01, 'wall_s': .01})
    result = module.run(tmp_path / 'config.json', tmp_path / 'output')
    assert result['complete'] and result['split'] == 'train'
    assert result['discovery']['bank_states'] == 5
    assert result['discovery']['evaluated_pairs'] == 15
    assert len(result['cases']) == (2 if positive else 1)
    assert result['ranker']['model'] is None and not result['fresh_validation_performed']
    assert not result['ppo_trained'] and result['population_acceleration'] is None
    assert not (tmp_path / 'output/sequence_manifest.json').exists()
    assert not (tmp_path / 'output/candidate_ranker.json').exists()
    assert read(tmp_path / 'output/case_plan.json')['selection_uses_native_outcomes'] is False
