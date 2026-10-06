import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_source_transfer as parent
from scenario_reconstruction import highd_v47_source_transfer_validation as m
from scenario_reconstruction.highd_v46_assets import digest, read, save
from scenario_reconstruction.highd_v47_state_features import FEATURE_NAMES


def example(n=24):
    predictions = np.arange(1, n+1, dtype=float)**2
    alpha = np.full(n, 1/n); q = parent.waterfill(alpha, predictions, 1/1.1)
    design = dict(states=[dict(plan=dict(source_seed=i)) for i in range(n)],
        target_masses=alpha.tolist(), candidate_q=q.tolist(), predicted_J=predictions.tolist())
    rows = [dict(source_seed=i, J=float(predictions[i]), variance=.01, positive_roots=1) for i in range(n)]
    cfg = dict(read(parent.DEFAULT)['statistics'], bootstrap_repeats=100)
    return design, rows, cfg


def test_multi_source_bootstrap_catches_old_cap_floor_confusion_and_uses_reciprocal(monkeypatch):
    design, rows, cfg = example()
    # This prospective synthetic test exercises the path sparse fixtures missed.
    with pytest.raises(ValueError, match='water-filling'):
        parent.source_comparison(design, rows, list(range(24)), cfg)
    original = parent.waterfill; observed = []
    def checked(a, j, floor):
        observed.append(floor); q = original(a, j, floor)
        assert q.sum() == pytest.approx(1.) and np.all(q >= a/1.1-1e-14)
        return q
    monkeypatch.setattr(parent, 'waterfill', checked)
    result = m.source_comparison(design, rows, list(range(24)), cfg)
    assert observed and all(f == pytest.approx(1/1.1) for f in observed)
    assert result['valid_replicates'] == 100 and result['censored_replicates'] == 0
    assert result['paired_source_bootstrap_ratio_interval'][1] < 1
    assert result['source_generalization_supported']
    assert result['difference_empirical_MCSE'] > 0


def test_bootstrap_matches_independent_manual_normalization_and_keeps_abstentions():
    design, rows, cfg = example()
    all_sources = list(range(32)); rng = np.random.default_rng(cfg['bootstrap_seed']); ratios = []
    js = np.array([r['J'] for r in rows]); pred = np.array(design['predicted_J'])
    for _ in range(cfg['bootstrap_repeats']):
        draw = rng.choice(all_sources, len(all_sources), replace=True)
        selected = draw[draw < len(rows)]
        if len(set(selected)) < cfg['minimum_positive_sources']: continue
        alpha = np.full(len(selected), 1/len(selected))
        q = parent.waterfill(alpha, pred[selected], 1/1.1)
        ratios.append(float(np.sum(alpha**2*js[selected]/q)/np.mean(js[selected])))
    expected = np.quantile(ratios, [.025, .975])
    actual = m.source_comparison(design, rows, all_sources, cfg)
    assert actual['all_source_episodes'] == 32 and actual['selected_sources'] == 24
    np.testing.assert_allclose(actual['paired_source_bootstrap_ratio_interval'], expected, rtol=1e-14, atol=0)
    assert actual == m.source_comparison(design, rows, all_sources, cfg)


def test_sparse_sources_remain_censored_not_repaired_by_copies():
    design, rows, cfg = example(3)
    result = m.source_comparison(design, rows, list(range(32)), cfg)
    assert result['censored_replicates'] == 100 and result['paired_source_bootstrap_ratio_interval'] is None
    assert not result['source_generalization_supported']


def test_v2_protocol_preserves_exact_parent_and_adds_validation_locks(monkeypatch):
    monkeypatch.setattr(parent, 'protocol', lambda _: ({'parent': True}, None, None, None, None, {'implementation_sha256': 'old'}))
    cfg, *_, lock = m.protocol(m.DEFAULT)
    assert cfg == {'parent': True} and lock['implementation_sha256'] == 'old'
    assert lock['validation_protocol_sha256'] == digest(m.DEFAULT)
    assert lock['validation_implementation_sha256'] == digest(m.__file__)
    assert lock['shared_audit_sha256'] == digest(Path(m.__file__).with_name('highd_v47_source_transfer_audit.py'))


@pytest.mark.parametrize('change', [{'version': 1}, {'parent_config_sha256': 'bad'}, {'training_unchanged': False}])
def test_v2_mutations_reject_before_parent_or_source_opening(tmp_path, monkeypatch, change):
    cfg = read(m.DEFAULT); cfg.update(change); save(tmp_path/'cfg.json', cfg)
    monkeypatch.setattr(parent, 'protocol', lambda _: pytest.fail('must reject first'))
    with pytest.raises(ValueError, match='version'): m.protocol(tmp_path/'cfg.json')


def test_parent_loader_is_not_replaced_or_its_lock_relaxed(monkeypatch):
    original = {'config_sha256': 'original', 'implementation_sha256': 'old'}
    monkeypatch.setattr(parent, 'protocol', lambda _: ({'parent': True}, None, None, None, None, original.copy()))
    seen = []
    monkeypatch.setattr(parent, 'completed_training', lambda path, cfg, lock: seen.append((path, cfg, lock)))
    m.completed_training('train', {'parent': True}, dict(original, validation_version=2))
    assert seen == [('train', {'parent': True}, original)]
    with pytest.raises(ValueError, match='identity'):
        m.completed_training('train', {'parent': True}, dict(original, implementation_sha256='changed'))


def test_v2_run_checks_lock_before_any_source_opens(tmp_path, monkeypatch):
    save(tmp_path/'lock.json', dict(split='validation', sources_opened=False, validation_version=2, fixture=1))
    monkeypatch.setattr(m, 'protocol', lambda _: ({}, None, None, None, None, {'fixture': 2}))
    monkeypatch.setattr(parent, 'discover', lambda *args: pytest.fail('must not open source'))
    with pytest.raises(ValueError, match='after freeze'): m.run('unused', tmp_path/'lock.json', tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_training_guards_and_independent_audit_still_required(monkeypatch):
    cfg = read(parent.DEFAULT)
    fitted = dict(measured_positive_sources=4, selected_OOF_ratio_point=.9)
    monkeypatch.setattr(m, 'completed_training', lambda *args: ('root', {}, fitted))
    monkeypatch.setattr(parent, 'check_training_audit', lambda *args: pytest.fail('review rejects before audit'))
    with pytest.raises(ValueError, match='review failed'): m.checked_training('train', 'audit', cfg, {})
    fitted.update(measured_positive_sources=5, selected_OOF_ratio_point=1.01)
    with pytest.raises(ValueError, match='review failed'): m.checked_training('train', 'audit', cfg, {})
    fitted['selected_OOF_ratio_point'] = .99
    called = []; monkeypatch.setattr(parent, 'check_training_audit', lambda *args: called.append(args))
    m.checked_training('train', 'audit', cfg, {})
    assert called == [('audit', 'root')]


def test_versioned_audit_injects_only_the_declared_backend_and_pins_shared_engine(tmp_path, monkeypatch):
    from scenario_reconstruction import highd_v47_source_transfer_validation_audit as audit
    out = tmp_path/'validation'; out.mkdir(); config = read(m.DEFAULT)
    record = dict(split='validation', validation_protocol=config,
        lock=dict(validation_version=2, training_path='train', training_audit_path=str(tmp_path/'training_audit.json')))
    save(tmp_path/'training_audit.json', {}); record['lock']['training_audit_sha256'] = digest(tmp_path/'training_audit.json')
    save(out/'summary.json', record); save(out/'opened_protocol.json', dict(validation_protocol=config))
    monkeypatch.setattr(m, 'protocol', lambda _: ({}, None, None, None, None, {}))
    monkeypatch.setattr(m, 'checked_training', lambda *args: None)
    seen = []
    def reconstruct(config, directory, replay, native):
        seen.append((config, directory, replay, native))
        return dict(verified=True, audit_implementation_sha256='shared')
    monkeypatch.setattr(audit, 'reconstruct', reconstruct)
    result = audit.audit(m.DEFAULT, out, True)
    assert seen == [(m.DEFAULT, out, True, m)]
    assert result['shared_audit_implementation_sha256'] == 'shared'
    assert result['audit_implementation_sha256'] == digest(audit.__file__)
    assert result['validation_implementation_sha256'] == digest(m.__file__)
    record['validation_protocol']['version'] = 1; save(out/'changed.json', record)
    (out/'summary.json').rename(out/'retained_summary.json')
    save(out/'summary.json', record)
    with pytest.raises(ValueError, match='Versioned'): audit.audit(m.DEFAULT, out)


def test_complete_v2_synthetic_run_never_refits_runs_iid_even_when_certificate_fails(tmp_path, monkeypatch):
    cfg = read(parent.DEFAULT); cfg['workers'] = 1
    cs = [dict(plan=dict(case_id=i, source_seed=270000+i, state_sha256='fixture', actor_ids=[1,2],
        target_mass=.5, state_index=0), features=dict(names=list(FEATURE_NAMES), values=[0.]*len(FEATURE_NAMES), coefficient_mass=1.),
        strata=[dict(stratum_id=0, cells=[0,1], measurement_probabilities=[.3,.7], coefficient_mass=1., draws=4)],
        state={}, P=[], H=[]) for i in range(2)]
    training = tmp_path/'training'; training.mkdir()
    save(training/'summary.json', {}); save(training/'model.json', {})
    audit = tmp_path/'audit.json'; save(audit, {})
    fitted = dict(model={}, benefit_bet={'rate':1000.})
    lock = dict(split='validation', sources_opened=False, validation_version=2, fixture=1, training_path=str(training),
        training_summary_sha256=digest(training/'summary.json'), model_sha256=digest(training/'model.json'),
        training_audit_path=str(audit), training_audit_sha256=digest(audit), benefit_bet_rate=1000.)
    save(tmp_path/'lock.json', lock)
    monkeypatch.setattr(m, 'protocol', lambda _: (cfg, {}, {}, {}, tmp_path, {'fixture':1}))
    monkeypatch.setattr(m, 'checked_training', lambda *args: (training, {}, fitted))
    monkeypatch.setattr(parent.features, 'predict', lambda *args: np.array([100.,1.]))
    monkeypatch.setattr(m, 'PairModelQueries', lambda *args, **kwargs: None)
    monkeypatch.setattr(parent, 'discover', lambda *args: ([], [], dict(episodes=[dict(seed=i) for i in range(270000,270256)])))
    monkeypatch.setattr(m, 'prepare_cases', lambda *args: (copy.deepcopy(cs), {270000:0,270001:1}))
    monkeypatch.setattr(m, 'ProcessPoolExecutor', lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(parent, 'fit_model', lambda *args: pytest.fail('must never refit'))
    out = tmp_path/'validation'
    def collect(task):
        schedule = task[0]
        assert (out/'measurement_plan.json').exists() and (out/'case_plan.json').exists()
        return dict(schedule, rows=[dict(r, measurement_value=0.) for r in schedule['rows']])
    monkeypatch.setattr(parent, '_collect', collect)
    monkeypatch.setattr(parent, 'iid_comparison', lambda *args: ({'fixture':{}}, {'ran':True}))
    result = m.run(m.DEFAULT, tmp_path/'lock.json', out)
    summary = read(out/'summary.json')
    assert not result['transfer_review_passed'] and not summary['ppo_ready']
    assert summary['iid_smoke']['ran'] and summary['validation_protocol']['version'] == 2
    assert not (out/'model.json').exists()
    with pytest.raises(FileExistsError): m.run(m.DEFAULT, tmp_path/'lock.json', out)
