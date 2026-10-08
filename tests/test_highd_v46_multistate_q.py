import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import ROOT, read, save
from scenario_reconstruction.highd_v46_cross_unit import proposal_factors
from scenario_reconstruction.highd_v46_d2rl import identity
from scenario_reconstruction.highd_v46_multistate_q import (
    check_protocol, compare_strata, mixture_summary, select_cases,
)


def config():
    return read(ROOT / 'configs/highd_v46_multistate_q_v1.json')


def test_fresh_sources_completed_training_and_code_lock():
    cfg = config()
    lock = {'complete': True, 'split': 'train', 'config_identity': identity(cfg),
            'implementation_sha256': 'code', 'dependencies': {'proxy': 'hash'}, 'environment': cfg['environment']}
    check_protocol(cfg, 'validation', 'code', {'proxy': 'hash'}, lock)
    for changed_field in ('implementation_sha256', 'dependencies', 'environment', 'config_identity', 'complete'):
        bad = copy.deepcopy(lock)
        bad[changed_field] = False
        with pytest.raises(ValueError):
            check_protocol(cfg, 'validation', 'code', {'proxy': 'hash'}, bad)
    with pytest.raises(ValueError):
        check_protocol(cfg, 'validation', 'code', {'proxy': 'hash'})
    for start in (100000, 160000, 170000, 180000):
        bad = copy.deepcopy(cfg)
        bad['validation']['seed_start'] = start
        with pytest.raises(ValueError):
            check_protocol(bad, 'train', 'code', {})
    bad = copy.deepcopy(cfg)
    bad['validation']['continuation_seed_start'] = bad['train']['continuation_seed_start']
    with pytest.raises(ValueError):
        check_protocol(bad, 'train', 'code', {})


def test_case_selection_is_episode_disjoint_and_never_uses_native_labels():
    bank = [{'seed': i, 'time_s': t} for i in range(1, 6) for t in (1, 2)]
    rows = [{'state_index': j, 'seed': b['seed'], 'views': {'production_strict': b['seed'] < 4},
             'metrics': {'positive_interaction_mass': j + 1}} for j, b in enumerate(bank)]
    selected = select_cases(bank, rows, config()['cases'])
    assert len(selected) == 5
    assert len({bank[c['state_index']]['seed'] for c in selected}) == 5
    assert [c['state_index'] for c in selected[:3]] == [1, 3, 5]
    assert all(c['row_index'] is None for c in selected[3:])
    for row in rows:
        row['native_cav_events'] = 1000 - row['state_index']
    assert select_cases(bank, rows, config()['cases']) == selected
    for row in rows:
        row['views']['production_strict'] = False
    assert len(select_cases(bank, rows, config()['cases'])) == 2


def test_stratified_variance_excludes_between_state_risk_difference():
    stats = config()['statistics']
    result = mixture_summary([np.ones(4), np.zeros(4)], stats)
    assert result['conditional_event_mean'] == .5
    assert result['weighted_event_second_moment'] == .5
    assert result['stratified_estimator_variance'] == 0
    assert np.concatenate([np.ones(4), np.zeros(4)]).var(ddof=1) > 0
    assert not result['evidence_guard_passed']  # One event source is not multi-state evidence.
    result = mixture_summary([np.array([0., 1., 2., 3.]), np.array([0., 0., 0., 4.])], stats)
    expected = (np.var([0, 1, 2, 3], ddof=1) + np.var([0, 0, 0, 4], ddof=1)) / (4 * 4)
    assert result['stratified_estimator_variance'] == pytest.approx(expected)
    with pytest.raises(ValueError):
        mixture_summary([np.ones(4), np.ones(3)], stats)


def test_zero_events_cannot_be_reported_as_a_variance_winner():
    stats = config()['statistics']
    values = {arm: [np.zeros(8)] for arm in config()['arms']}
    values['dual'] = [np.array([.01, .01, 0, 0, 0, 0, 0, 0])]
    reports = {arm: mixture_summary(x, stats) for arm, x in values.items()}
    comparisons = compare_strata(values, reports, stats)
    assert reports['natural']['relative_half_width'] is None
    assert reports['natural']['precision_status'] == 'insufficient_multistate_events'
    for comparison in comparisons.values():
        assert comparison['dual_over_reference_second_moment_descriptive'] is None
        assert comparison['second_moment_ratio_interval'] is None
        assert not comparison['lower_dual_second_moment_supported']
        assert comparison['zero_reference_second_moment_repeats'] == stats['bootstrap_repeats']
    assert mixture_summary([], stats)['precision_status'] == 'no_eligible_states'


def test_exact_q_event_mean_and_second_moment_objective():
    p = np.array([[.85, .05], [.05, .05]])
    h = np.array([[0., 0.], [0., 1.]])
    event = np.array([[0., 0.], [0., 1.]])
    moments = {}
    for arm, epsilon in config()['arms'].items():
        q1, q2 = proposal_factors(p, h, epsilon)[-2:]
        q = q1[:, None] * q2
        w = p / q
        assert np.sum(q*w) == pytest.approx(1.)
        assert np.sum(q*w*event) == pytest.approx(.05)
        moments[arm] = np.sum(q*(w*event)**2)
    assert moments['dual'] < moments['first_factor_only'] < moments['natural']
    assert moments['dual'] < moments['second_factor_only'] < moments['natural']


@pytest.mark.parametrize('chosen', [None, {'ids': [1, 2], 'h': np.array([[0., 0.], [0., 1.]])}])
def test_initial_draw_is_real_q_and_later_steps_are_natural(monkeypatch, chosen):
    from scenario_reconstruction import highd_v46_multistate_q as module
    initial_calls, later_calls = [], []

    class Road:
        def __init__(self, seed):
            self.time, self.crashes, self.cav = 0., [], SimpleNamespace(id=0)
            self.rng, self.minimum_gap = np.random.default_rng(seed), 2.

        def step_policy(self, mode, epsilon):
            assert mode == 'natural' and epsilon == 1.
            if 'sample' in self.__dict__:
                initial_calls.append(self.sample(self.action_pmfs(), mode, epsilon))
            else:
                later_calls.append(self.time)
            self.time += 1

    monkeypatch.setattr(module, 'restore_prepared_state', lambda m, a, p, s, seed: Road(seed))
    state = {'time_s': 0., 'units': [{'ids': [1, 2], 'p': [[.4, .1], [.1, .4]]}], 'pmfs': {'0': [1.]}}
    first = module.sample_continuation(None, None, None, state, chosen, [.1, .1], 700, config(), 2.)
    second = module.sample_continuation(None, None, None, state, chosen, [.1, .1], 701, config(), 2.)
    assert len(initial_calls) == 2 and len(later_calls) == 6
    for result in (first, second):
        logged = result['initial_likelihood_audit']
        assert logged['full_log_p'] - logged['full_log_q'] == pytest.approx(logged['log_weight'])
        if chosen is None:
            assert logged['abstained'] and result['generation_log_weight'] == 0.
        else:
            assert logged['generation_epsilon'] == [.1, .1]


def test_complete_train_validation_smoke_and_exclusive_artifacts(tmp_path, monkeypatch):
    from scenario_reconstruction import highd_v46_multistate_q as module
    monkeypatch.chdir(ROOT)
    cfg = config()
    cfg['train']['replicates_per_state_arm'] = 2
    cfg['validation']['replicates_per_state_arm'] = 2
    save(tmp_path / 'config.json', cfg)
    monkeypatch.setattr(module, 'environment', lambda: cfg['environment'])
    monkeypatch.setattr(module, 'ensure_assets', lambda: (tmp_path, {'natural_target_sha256': 'P'}))
    monkeypatch.setattr(module, 'experiment_target', lambda *args: {'P': 'P'})
    monkeypatch.setattr(module, 'PairModelQueries', lambda *args, **kwargs: None)
    bank = [{'seed': i, 'time_s': 1.} for i in (1, 2, 3)]
    rows = [{'state_index': j, 'seed': b['seed'], 'views': {'production_strict': j == 0},
             'metrics': {'positive_interaction_mass': .1}} for j, b in enumerate(bank)]
    monkeypatch.setattr(module, 'training_source', lambda *args: (bank, rows, {}))
    monkeypatch.setattr(module, 'discover', lambda *args: (bank, rows, {}))
    monkeypatch.setattr(module, 'prepare_case', lambda state, row, *args: None if row is None else
                        {'ids': [1, 2], 'mass': .1, 'factor_tv': [.2, .3], 'q_second_tv': .27, 'mechanism': 'lateral'})
    monkeypatch.setattr(module, 'sample_continuation', lambda *args:
                        {'event_result': False, 'generation_log_weight': 0., 'bv_only_collision': False,
                         'initial_likelihood_audit': {'full_log_p': -1., 'full_log_q': -1.},
                         'clearance_risk': .1, 'cpu_s': .01, 'wall_s': .01})
    result = module.run(tmp_path / 'config.json', 'train', tmp_path / 'train')
    assert result['complete'] and len(result['cases']) == 3
    assert result['groups']['eligible_only_primary']['arms']['dual']['states'] == 1
    assert result['groups']['all_cases_secondary']['arms']['dual']['states'] == 3
    assert not result['checkpoint_exported'] and result['population_acceleration'] is None
    result = module.run(tmp_path / 'config.json', 'validation', tmp_path / 'val', tmp_path / 'train/summary.json')
    assert result['training_lock_sha256']
    assert not (tmp_path / 'val/sequence_manifest.json').exists()
    with pytest.raises(FileExistsError):
        module.run(tmp_path / 'config.json', 'train', tmp_path / 'train')
