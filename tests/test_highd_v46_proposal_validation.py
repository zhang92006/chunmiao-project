import copy

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import read, save
from scenario_reconstruction.highd_v46_proposal_validation import DEFAULT_CONFIG, make_plan, validate_cohort


def row(seed, risk=.8, strict=True, state_index=0, time=5., ids=(1, 2)):
    return {'seed': seed, 'state_index': state_index, 'time_s': time, 'actor_ids': list(ids),
        'ranker_features': [1., .1, risk, 0.], 'views': {'production_strict': strict},
        'proxy_cpu_s': .1,
        'metrics': {'baseline_actions': [0, 0], 'max_interaction_actions': [1, 1], 'max_positive_interaction': .1}}


@pytest.mark.parametrize('seed', [1000, 100000, 160000, 200000, 230000, 240000])
def test_previously_consumed_or_reserved_sources_rejected(seed):
    cfg = read(DEFAULT_CONFIG); cfg['source']['seed_start'] = seed
    with pytest.raises(ValueError, match='fresh'): validate_cohort(cfg)


@pytest.mark.parametrize('change', ['training_split', 'budget', 'scores', 'epsilon', 'seed_overlap', 'confirmation'])
def test_new_protocol_rejects_unreviewed_contract_changes(change):
    cfg = read(DEFAULT_CONFIG)
    if change == 'training_split': cfg['source']['split'] = 'train'
    if change == 'budget': cfg['replicates_per_case_candidate'] = 128
    if change == 'scores': cfg['candidates']['mixed'] = .75
    if change == 'epsilon': cfg['epsilon'] = [.05, .05]
    if change == 'seed_overlap': cfg['confirmation']['continuation_seed_start'] = cfg['continuation_seed_start']
    if change == 'confirmation': cfg['confirmation']['maximum_source_episodes'] = 32
    with pytest.raises(ValueError): validate_cohort(cfg)


def test_outcome_blind_common_selection_and_eligible_probe_coverage():
    cfg = read(DEFAULT_CONFIG); validate_cohort(cfg)
    seeds = [250000+i*5 for i in range(20)]
    bank = [{'seed': seed, 'time_s': 5.} for seed in seeds]
    rows = [row(seed, state_index=i) for i, seed in enumerate(seeds)]
    plans, all_sources, probes = make_plan(bank, rows, cfg)
    assert len(plans) == 20 and len(all_sources) == 256 and len(probes) == 16
    assert set(probes) <= set(seeds)
    assert sum(p['confirmation_planned'] for p in plans) == 16
    assert plans[1]['continuation_seed_start'] == cfg['continuation_seed_start']+5*64
    before = copy.deepcopy(plans)
    for r in rows: r['native_events'] = 1000
    assert make_plan(bank, rows, cfg)[0] == before
    assert len(set(p['continuation_seed_start'] for p in plans)) == len(plans)


def test_strict_selector_preserved_even_when_non_strict_risk_higher():
    cfg = read(DEFAULT_CONFIG)
    bank = [{'seed': 250000, 'time_s': 5.}, {'seed': 250000, 'time_s': 6.}]
    rows = [row(250000, strict=False, risk=.9), row(250000, risk=.7, state_index=1, time=6.)]
    plans, _, probes = make_plan(bank, rows, cfg)
    assert len(plans) == 1 and plans[0]['row_index'] == 1 and probes == [250000]
    empty, _, probes = make_plan(bank, rows[:1], cfg)
    assert empty == [] and probes == []


def test_freeze_requires_complete_training_evidence_before_any_fresh_source(monkeypatch, tmp_path):
    from scenario_reconstruction import highd_v46_proposal_validation as module
    cfg = read(DEFAULT_CONFIG); cfg['training_comparison'] = {'path': str(tmp_path), 'summary_sha256': 'invalid'}
    path = tmp_path/'config.json'; save(path, cfg); save(tmp_path/'summary.json', {})
    monkeypatch.setattr(module, 'load_inputs', lambda path: ({'statistics': cfg['statistics']}, [], [], {}, {}, tmp_path, {}))
    monkeypatch.setattr(module, 'environment', lambda: cfg['environment'])
    monkeypatch.setattr(module, 'discover', lambda *args: pytest.fail('Freeze opened fresh sources'))
    with pytest.raises(ValueError, match='checksum'): module.freeze(path, tmp_path/'lock.json')
    assert not (tmp_path/'lock.json').exists()


def test_two_source_native_adapter_and_separate_confirmations(monkeypatch, tmp_path):
    from scenario_reconstruction import highd_v46_proposal_validation as module
    from scenario_reconstruction.highd_v46_cross_unit import draw_intervention
    from scenario_reconstruction.highd_v46_proposal_comparison import build_h, factor_tv
    cfg = read(DEFAULT_CONFIG); cfg['source']['episodes'] = 2
    cfg['replicates_per_case_candidate'] = 2; cfg['statistics']['bootstrap_repeats'] = 10
    bank = [{'seed': 250000+i, 'time_s': 5.} for i in (0, 1)]
    rows = [row(250000+i, state_index=i) for i in (0, 1)]
    lock = {'version': 1}; lock_path = tmp_path/'lock.json'; save(lock_path, lock)
    p = np.array([[.4, .1], [.2, .3]]); risk = np.array([[.1, .2], [.3, 1.]])
    proposals = {name: {'ids': [1, 2], 'h': build_h(p, risk, lam)[0],
                       'factor_tv': factor_tv(p, build_h(p, risk, lam)[0], cfg['epsilon'])}
                 for name, lam in cfg['candidates'].items()}
    monkeypatch.setattr(module, 'inputs', lambda _: (cfg, {}, {'proxy': {'clearance_scale_m': 2}}, {}, tmp_path, lock))
    monkeypatch.setattr(module, 'PairModelQueries', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'discover', lambda *args: (bank, rows, {'episodes': [{'seed': s['seed']} for s in bank]}))
    monkeypatch.setattr(module, 'prepare', lambda *args: (p, risk, proposals))
    def sample(model, assets, physical, state, chosen, eps, seed, config, scale):
        assert (tmp_path/'run'/'case_plan.json').exists()
        _, log = draw_intervention([{'ids': [1, 2], 'p': p}], [1, 2], chosen['h'], eps, np.random.default_rng(seed))
        return {'seed': seed, 'event_result': bool(seed % 2), 'generation_log_weight': log['log_weight'],
                'initial_likelihood_audit': log, 'cpu_s': .1}
    monkeypatch.setattr(module, 'sample_continuation', sample)
    monkeypatch.setattr(module, 'mechanism', lambda model, assets, physical, state, plan, cfg, scale: {
        'status': 'measured', 'source_seed': plan['source_seed'], 'native_interaction_mean': .1, 'fixed_state_interaction_mcse': .01})
    result = module.run('unused', lock_path, tmp_path/'run')
    assert result['complete'] and result['selected_sources'] == 2 and not result['training_exported']
    assert result['native_confirmation']['measured_source_episodes'] == 2
    assert all(r['events'] == 2 for r in result['reports'].values())
    assert all(r['observed_event_source_yield_all_sources'] == 1. for r in result['reports'].values())
    with pytest.raises(FileExistsError): module.run('unused', lock_path, tmp_path/'run')
