import copy

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import ROOT, read
from scenario_reconstruction.highd_v46_discovery_coverage import (
    check_source, choose_omitted, state_key, summarize,
)


def test_uniform_omitted_states_exclude_selected_and_respect_horizon():
    states = [{'seed': 1, 'time_s': t, 'screen_score': t} for t in range(60)]
    selected = {state_key(s) for s in states[:2]}
    a = choose_omitted(states, selected, 100, 60, 4, np.random.default_rng(61))
    assert a == list(range(2, 57))
    a = choose_omitted(states, selected, 2, 60, 4, np.random.default_rng(61))
    for s in states:
        s['screen_score'] = -s['screen_score']
        s['future_collision'] = True
    assert a == choose_omitted(states, selected, 2, 60, 4, np.random.default_rng(61))
    assert choose_omitted([], set(), 2, 60, 4, np.random.default_rng(61)) == []


def test_coverage_rejects_validation_wrong_source_and_incomplete_horizon():
    cfg = read(ROOT / 'configs/highd_v46_discovery_coverage_v1.json')
    source = {'split': 'train', 'config': read(ROOT / 'configs/highd_v46_state_discovery_v1.json')}
    check_source(source, cfg, cfg['source_summary_sha256'])
    with pytest.raises(ValueError):
        check_source(dict(source, split='validation'), cfg, cfg['source_summary_sha256'])
    with pytest.raises(ValueError):
        check_source(source, cfg, 'different')
    with pytest.raises(ValueError):
        check_source(source, dict(cfg, minimum_remaining_s=3), cfg['source_summary_sha256'])
    bad = copy.deepcopy(source)
    bad['config']['train']['seed_start'] = 100000
    with pytest.raises(ValueError):
        check_source(bad, cfg, cfg['source_summary_sha256'])


def test_pair_and_state_misses_are_distinct_and_zero_denominator_is_undefined():
    def row(state, rank, prior, group='selected_bank', positive=True):
        return {'state_index': state, 'seed': state // 2, 'rank': rank, 'group': group,
                'was_v1_audited': prior, 'views': {'strict': positive},
                'mechanism': 'bracket', 'first_failed_gate': 'passed' if positive else 'leverage'}
    rows = [row(0, 1, True), row(0, 4, False), row(1, 5, False),
            row(2, 2, False, 'omitted_uniform', False)]
    states = [{'group': 'selected_bank'}, {'group': 'selected_bank'}, {'group': 'omitted_uniform'}]
    report = summarize(rows, states, ['strict'])
    strict = report['selected_bank']['views']['strict']
    assert strict['pairs'] == 3 and strict['states'] == 2 and strict['source_episodes'] == 1
    assert strict['new_positive_pairs'] == 2
    assert strict['states_without_any_v1_positive_pair'] == 1
    assert strict['within_audited_universe_v1_pair_recall'] == 1 / 3
    assert strict['top_k_positive_state_hits']['3'] == 1
    assert strict['top_k_positive_state_hits']['5'] == 2
    assert report['omitted_uniform']['views']['strict']['within_audited_universe_v1_pair_recall'] is None


def test_new_witnesses_exclude_previously_audited_pairs():
    from scenario_reconstruction.highd_v46_discovery_witness import select_witnesses
    cfg = read(ROOT / 'configs/highd_v46_coverage_witness_v1.json')
    rows = [{'seed': i, 'time_s': 20, 'actor_ids': [1, 2], 'selection': 'exhaustive',
             'methods': ['selected_bank'] + ([] if i == 0 else ['missed_by_v1']),
             'views': {'production_strict': True}, 'metrics': {'positive_interaction_mass': 3 - i}}
            for i in range(3)]
    assert [r['seed'] for r in select_witnesses(rows, cfg)] == [1, 2]


def test_two_state_run_writes_distinct_incremental_artifacts(tmp_path, monkeypatch):
    from scenario_reconstruction import highd_v46_discovery_coverage as module
    from scenario_reconstruction.highd_v46_assets import digest, save

    dc = read(ROOT / 'configs/highd_v46_state_discovery_v1.json')
    candidate = {'actor_ids': [1, 2], 'mechanism': 'bracket', 'score': 1.}

    def fake_snapshot(road, units, pmfs, seed, features):
        return {'seed': seed, 'time_s': road.time, 'cav_id': 0,
                'vehicles': [{'id': i} for i in range(3)],
                'units': [{'ids': [i], 'p': [.5, .5]} for i in (1, 2)],
                'pmfs': {str(i): [.5, .5] for i in range(3)}, 'features': features}

    class FakeRoad:
        def __init__(self, *args):
            self.time = 0.

        def step_policy(self, mode, epsilon):
            self.choose([], [], {}, mode)
            self.time += 1
            return self.time < 2

    source = tmp_path / 'source'
    source.mkdir()
    original = fake_snapshot(FakeRoad(), [], {}, 160000, {'discovery_only': True})
    original.update(screen_pairs=[candidate], screen_score=1., methods=['geometry_1s'])
    save(source / 'state_bank.json', {'states': [original]})
    save(source / 'pair_audit.json', {'rows': []})
    summary = {'split': 'train', 'config': dc, 'natural_target_sha256': 'fake-target',
               'source_episodes': [{'seed': 160000, 'states': 2}],
               'implementation_sha256': digest(ROOT / 'scenario_reconstruction/highd_v46_state_discovery.py'),
               'dependencies': {'physical_config': digest(ROOT / dc['physical_config'])},
               'state_bank_sha256': digest(source / 'state_bank.json'),
               'pair_audit_sha256': digest(source / 'pair_audit.json')}
    save(source / 'summary.json', summary)
    cfg = read(ROOT / 'configs/highd_v46_discovery_coverage_v1.json')
    cfg['source_summary_sha256'] = digest(source / 'summary.json')
    save(tmp_path / 'config.json', cfg)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(module, 'NativeExperiment', FakeRoad)
    monkeypatch.setattr(module, 'snapshot', fake_snapshot)
    monkeypatch.setattr(module, 'screen_pairs', lambda *args: [candidate])
    monkeypatch.setattr(module, 'response_risk', lambda *args, **kwargs: np.zeros((2, 2)))
    monkeypatch.setattr(module, 'PairModelQueries', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'ensure_assets', lambda: (tmp_path, {'natural_target_sha256': 'fake-target'}))
    output = tmp_path / 'audit'
    result = module.run(source, tmp_path / 'config.json', output)
    assert result['groups']['selected_bank']['states'] == 1
    assert result['groups']['omitted_uniform']['states'] == 1
    assert len(list(output.glob('audit_state_*.json'))) == 2
    assert len(read(output / 'pair_audit.json')['rows']) == 2
    assert result['pair_audit_sha256'] == digest(output / 'pair_audit.json')
