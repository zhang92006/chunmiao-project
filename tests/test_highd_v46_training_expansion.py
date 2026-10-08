import copy
from pathlib import Path

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import digest, read, save
from scenario_reconstruction.highd_v46_d2rl import identity
from scenario_reconstruction.highd_v46_native_calibration import validate_config
from scenario_reconstruction.highd_v46_conditional_baseline import make_sequences, conditional_observation
from scenario_reconstruction.highd_v46_training_expansion import concentration, load_calibration
from scenario_reconstruction.highd_v46_multistate_q import dependencies


def test_expansion_is_fixed_fresh_budget():
    cfg = read('configs/highd_v46_training_expansion_v1.json')
    dc = read(cfg['discovery_config'])
    validate_config(cfg, dc)
    assert cfg['source'] == {'split': 'train', 'seed_start': 230000, 'episodes': 64}
    assert cfg['cases']['maximum_source_episodes'] == 64
    assert cfg['replicates_per_case_arm'] == 64
    for seed in (100000, 170000, 180000, 190000, 200000, 210000, 220000):
        bad = copy.deepcopy(cfg); bad['source']['seed_start'] = seed
        with pytest.raises(ValueError):
            validate_config(bad, dc)


def fixture(tmp_path):
    cfg = read('configs/highd_v46_training_expansion_v1.json')
    cfg['source']['episodes'] = 1; cfg['replicates_per_case_arm'] = 2
    state = {'seed': 230000, 'time_s': 10., 'cav_id': 0,
             'vehicles': [{'id': i, 'x': i*12., 'v': 20., 'lane': 0, 'length': 5.} for i in range(3)]}
    row = {'seed': 230000, 'state_index': 0, 'actor_ids': [1, 2], 'mechanism': 'bracket',
           'views': {'production_strict': True}, 'metrics': {'positive_interaction_mass': .1}}
    plan = {'row_index': 0, 'state_index': 0, 'role': 'eligible', 'selection': 'best_mass',
            'source_seed': 230000, 'time_s': 10., 'state_sha256': identity(state), 'selected_pair': [1, 2]}
    save(tmp_path/'state_bank.json', {'states': [state], 'split': 'train'})
    save(tmp_path/'pair_audit.json', {'rows': [row]})
    save(tmp_path/'case_plan.json', {'cases': [plan], 'selection_uses_native_outcomes': False})
    save(tmp_path/'native_calibration.json', {'records': []})
    for arm, eps in cfg['arms'].items():
        p = np.asarray([.5, .5]); h = np.asarray([.8, 0.]); q = np.asarray(eps)*p+(1-np.asarray(eps))*h
        w = float(np.log(p/q).sum())
        audit = {'p_factors': p.tolist(), 'h_factors': h.tolist(), 'q_factors': q.tolist(),
                 'generation_epsilon': eps, 'full_log_p': w, 'full_log_q': 0.}
        seqs = [{'seed': 2800000+j, 'event_result': j == 0, 'bv_only_collision': False,
                 'initial_likelihood_audit': audit, 'generation_log_weight': w} for j in range(2)]
        save(tmp_path/f'case_00_{arm}.json', {'conditional_case': plan, 'epsilon': eps, 'sequences': seqs})
    dc = read(cfg['discovery_config']); base = read(dc['base_discovery_config'])
    deps = dependencies(cfg, dc, base)
    deps['highd_v46_multistate_q.py'] = digest('scenario_reconstruction/highd_v46_multistate_q.py')
    summary = {'complete': True, 'split': 'train', 'config': cfg, 'config_identity': identity(cfg),
               'dependencies': deps, 'environment': cfg['environment'],
               'implementation_sha256': digest('scenario_reconstruction/highd_v46_native_calibration.py'),
               'discovery': {'episodes': [{'seed': 230000}]},
               'artifacts': {p.name: digest(p) for p in tmp_path.iterdir()}}
    save(tmp_path/'summary.json', summary)
    return cfg


def rewrite(path, data):
    # Synthetic fixture mutations only; production exporters are exclusive-create.
    import json
    path.write_text(json.dumps(data))


def test_locked_loader_keeps_negative_continuations(tmp_path):
    cfg = fixture(tmp_path)
    summary, seeds, cases, raw, records, values = load_calibration(tmp_path, cfg, digest(tmp_path/'summary.json'))
    assert seeds == {230000} and len(cases) == 1
    assert [s['event_result'] for s in raw[0]] == [True, False]
    assert all(len(strata[0]) == 2 for strata in values.values())


@pytest.mark.parametrize('mutation', ['summary_lock', 'raw_lock', 'validation', 'implementation',
    'missing_artifact', 'source_episode', 'schedule', 'selection', 'native_selection', 'weight', 'behavior'])
def test_loader_rejects_corruption_and_leakage(tmp_path, mutation):
    cfg = fixture(tmp_path); summary_path = tmp_path/'summary.json'; summary = read(summary_path)
    expected = digest(summary_path)
    if mutation == 'summary_lock': expected = 'wrong'
    elif mutation == 'raw_lock':
        path = tmp_path/'case_00_dual.json'; data = read(path); data['sequences'].pop(); rewrite(path, data)
    elif mutation == 'validation': summary['split'] = 'validation'
    elif mutation == 'implementation': summary['implementation_sha256'] = 'wrong'
    elif mutation == 'missing_artifact': del summary['artifacts']['case_00_natural.json']
    elif mutation == 'source_episode': summary['discovery']['episodes'] = []
    else:
        path = tmp_path/('case_plan.json' if mutation in ('selection', 'native_selection') else 'case_00_dual.json')
        data = read(path)
        if mutation == 'schedule': data['sequences'].pop()
        elif mutation == 'selection': data['cases'][0]['selection'] = 'best_event'
        elif mutation == 'native_selection': data['selection_uses_native_outcomes'] = True
        elif mutation == 'weight': data['sequences'][0]['generation_log_weight'] += 1
        elif mutation == 'behavior': data['sequences'][0]['initial_likelihood_audit']['q_factors'][0] = .9
        rewrite(path, data); summary['artifacts'][path.name] = digest(path)
    if mutation not in ('summary_lock', 'raw_lock'):
        rewrite(summary_path, summary); expected = digest(summary_path)
    with pytest.raises(ValueError): load_calibration(tmp_path, cfg, expected)


def test_union_reweights_sources_and_preserves_h_zero_events(tmp_path):
    cfg = fixture(tmp_path)
    _, _, cases, raw, _, _ = load_calibration(tmp_path, cfg, digest(tmp_path/'summary.json'))
    other = copy.deepcopy(cases[0]); other['source_seed'] += 1
    negative = copy.deepcopy(raw[0])
    for s in negative: s['event_result'] = False
    seqs = make_sequences(cases+[other, copy.deepcopy(other)], raw+[negative, negative])
    assert [s['target_mass'] for s in seqs] == [.25, .25, .125, .125, .125, .125]
    report = concentration(seqs)
    assert report['source_episodes'] == 2 and report['event_source_episodes'] == 1
    assert report['top_source_fraction'] == 1.
    assert report['by_source'][230000]['h2_zero_events'] == 1
    assert report['by_source'][230001]['events'] == 0
    assert report['weighted_event_second_moment'] == pytest.approx(.25*np.exp(2*raw[0][0]['generation_log_weight']))
    for s in seqs: s['event_result'] = False
    assert concentration(seqs)['top_source_fraction'] is None


def test_export_union_replay_and_exclusive_destination(tmp_path, monkeypatch):
    import scenario_reconstruction.highd_v46_training_expansion as module
    root = tmp_path/'source'; root.mkdir(); cfg = fixture(root)
    first = list(load_calibration(root, cfg, digest(root/'summary.json')))
    first[0].update(natural_target_sha256='P', experiment_target_sha256='physical')
    second = copy.deepcopy(first); second[1] = {230001}
    second[2][0]['source_seed'] = 230001; second[4][0]['source_seed'] = 230001
    second[2][0]['state_sha256'] = 'other_state'
    for s in second[3][0]: s['seed'] += 10
    monkeypatch.setattr(module, 'load_calibration', lambda path, cfg, sha: first if str(path).endswith('v1_train') else second)
    monkeypatch.setattr(module, 'ensure_assets', lambda: (None, {'natural_target_sha256': 'P'}))
    report = module.export('configs/highd_v46_training_expansion_v1.json', 'new_source', 'new_sha', tmp_path/'union')
    assert report['sequences'] == 4 and report['after']['event_source_episodes'] == 2
    assert report['after']['top_source_fraction'] == pytest.approx(.5)
    assert report['ppo_trained'] is False and report['checkpoint_selection_ready'] is False
    data = read(tmp_path/'union/sequences.json')
    assert sum(s['target_mass'] for s in data['sequences']) == 1.
    with pytest.raises(FileExistsError):
        module.export('configs/highd_v46_training_expansion_v1.json', 'new_source', 'new_sha', tmp_path/'union')
    second[1] = first[1]
    with pytest.raises(ValueError, match='cohorts overlap'):
        module.export('configs/highd_v46_training_expansion_v1.json', 'new_source', 'new_sha', tmp_path/'bad')
    second[1] = {230001}
    second[3][0][0]['seed'] = first[3][0][0]['seed']
    with pytest.raises(ValueError, match='continuation seeds overlap'):
        module.export('configs/highd_v46_training_expansion_v1.json', 'new_source', 'new_sha', tmp_path/'bad')
