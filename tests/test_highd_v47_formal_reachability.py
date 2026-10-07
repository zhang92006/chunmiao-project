"""Synthetic checks only: these tests open no fresh traffic sources."""
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_formal_reachability as r
from scenario_reconstruction.highd_v46_assets import read, save, digest
from scenario_reconstruction.highd_v46_d2rl import identity


def test_peak_support_and_declared_tie_break():
    p = np.array([[0., .1], [.5, .4]])
    assert r.peak(p, [[20., 1.], [1., 1.]]) == (1, 0)
    assert r.peak([.5, .5], [1., 1.]) == (0,)
    assert r.peak([1e-100, 1.-1e-100], [1., .9]) == (0,)


@pytest.mark.parametrize('p,score', [([0., 0.], [1., 1.]), ([1.], [1., 2.]),
                                    ([1., 0.], [np.nan, 1.]), ([-1., 2.], [1., 2.])])
def test_invalid_support_rejected(p, score):
    with pytest.raises(ValueError):
        r.peak(p, score)


def test_persistence_is_explicit_not_silent_target_selector_change():
    assert r.controlled_ids('initial_peak', [1], [2], 0) == [1]
    assert r.controlled_ids('initial_peak', [1], [2], 1) == []
    assert r.controlled_ids('reselected_peak', [1], [2], 1) == [2]
    assert r.controlled_ids('reselected_peak', [1], [], 1) == []
    assert r.controlled_ids('persistent_peak', [1, 2], [], 1) == [1, 2]
    with pytest.raises(ValueError):
        r.controlled_ids('retry_until_positive', [1], [2], 1)


def test_full_bank_fixed_budget_and_case_crn():
    cfg = read(r.DEFAULT)
    bank = dict(cases=[dict(case_id=str(i), source_seed=7+i//2) for i in range(5)])
    tasks = r.task_plan(cfg, bank)
    assert len(tasks) == 30
    for i in range(5):
        part = [t for t in tasks if t['case_index'] == i]
        assert len(part) == 6 and {t['seed'] for t in part} == {1600000000+i}
    assert len({t['seed'] for t in tasks}) == 5
    assert not cfg['training_export'] and not cfg['validation_opened']


def test_locked_pair_uses_current_joint_or_cross_unit_law():
    joint = np.array([[.1, .4], [.2, .3]])
    assert np.array_equal(r.marginal([dict(ids=[1, 2], p=joint)], [1, 2]), joint)
    assert np.array_equal(r.marginal([dict(ids=[1, 2], p=joint)], [2, 1]), joint.T)
    units = [dict(ids=[1], p=np.array([.2, .8])), dict(ids=[2], p=np.array([.4, .6]))]
    assert r.marginal(units, [1, 2]) == pytest.approx(np.outer([.2, .8], [.4, .6]))


@pytest.mark.parametrize('arm,expected', [('initial_peak', [[1], [], [], []]),
    ('reselected_peak', [[1], [2], [2], [2]]), ('persistent_peak', [[1], [1], [1], [1]])])
def test_synthetic_probe_has_exact_partners_no_unsupported_forcing(monkeypatch, arm, expected):
    cars = [SimpleNamespace(id=i, lane=0, x=10.*i) for i in range(3)]
    road = SimpleNamespace(vehicles=cars, cav=cars[0], time=8., crashes=[],
        rng=np.random.default_rng(7), cav_acceleration=0., minimum_gap=9.)
    units = [dict(ids=[1, 2], p=np.array([[.5, 0.], [0., .5]]))]
    road.action_pmfs = lambda: {i: np.array([.5, .5]) for i in (1, 2)}
    road.prepare = lambda pmfs: (units, {c.id: c for c in cars})
    monkeypatch.setattr(r.native, 'restore_prepared_state', lambda *a: road)
    monkeypatch.setattr(r.native, 'common_choice', lambda *a: dict(ids=[1 if road.time == 8 else 2]))
    monkeypatch.setattr(r, 'challenge', lambda *a: np.array([.1, .9]))
    def advance(road, pmfs, actions):
        assert actions[1] == actions[2]  # exact diagonal conditional partner
        road.time += 1
    monkeypatch.setattr(r.native, 'advance_prepared', advance)
    state = dict(last_acceleration={}, cav_acceleration=0., pmfs={'1': [.5, .5], '2': [.5, .5]},
                 units=[dict(ids=[1, 2], p=[[.5, 0.], [0., .5]])])
    case = dict(state=state, case_id='0', source_seed=1, horizon_s=4)
    result = r.probe(None, None, {'criticality': {}}, case, 'single', arm, 7)
    assert [t['forced_actor_ids'] for t in result['decisions']] == expected
    assert result['elapsed_s'] == 4 and not result['event_result']
    assert result['natural_path_log_probability'] == pytest.approx(4*np.log(.5))
    assert 'generation_log_weight' not in result  # forced data cannot impersonate real-Q data
    for t in result['decisions']:
        assert t['conditional_partner_log_probability'] == pytest.approx(0. if t['forced'] else np.log(.5))


def test_summary_retains_negatives_and_does_not_promote_forced_events():
    rows = [dict(mode=m, arm=a, case_id=str(i), source_seed=i, seed=99+i,
                 event_result=i == 1, event_type='cav_first_collision' if i == 1 else 'horizon_no_collision',
                 natural_path_log_probability=-100.)
            for m in ('single', 'dual') for a in r.ARMS for i in range(3)]
    result = r.summarize(rows)
    assert not result['PPO_started'] and not result['training_export'] and not result['validation_opened']
    assert all(g['probes'] == 3 and g['CAV_events'] == 1 and g['event_sources'] == [1]
               for g in result['groups'].values())


@pytest.mark.parametrize('pairs,event', [([[1, 2]], False), ([[0, 1]], True)])
def test_first_collision_stops_probe_and_never_relabels_BV(monkeypatch, pairs, event):
    cars = [SimpleNamespace(id=i, lane=0, x=10.*i) for i in range(3)]
    road = SimpleNamespace(vehicles=cars, cav=cars[0], time=8., crashes=[],
        rng=np.random.default_rng(7), cav_acceleration=0., minimum_gap=9.)
    monkeypatch.setattr(r.native, 'restore_prepared_state', lambda *a: road)
    monkeypatch.setattr(r.native, 'common_choice', lambda *a: dict(ids=[1]))
    monkeypatch.setattr(r, 'challenge', lambda *a: np.array([.1, .9]))
    def advance(road, pmfs, actions):
        road.time += 1/15
        road.crashes = pairs
    monkeypatch.setattr(r.native, 'advance_prepared', advance)
    state = dict(last_acceleration={}, cav_acceleration=0., pmfs={'1': [.5, .5], '2': [.5, .5]},
                 units=[dict(ids=[1, 2], p=[[.5, 0.], [0., .5]])])
    case = dict(state=state, case_id='0', source_seed=1, horizon_s=4)
    row = r.probe(None, None, {'criticality': {}}, case, 'single', 'persistent_peak', 7)
    assert row['event_result'] == event and len(row['decisions']) == 1
    assert row['elapsed_s'] == pytest.approx(1/15)
    assert row['collision_pairs'] == pairs


def test_freeze_no_rollout_and_run_drift_fail_closed(tmp_path, monkeypatch):
    cfg = read(r.DEFAULT)
    bank = dict(cases=[dict(case_id='0', source_seed=7, horizon_s=4)], state_law=[dict(alpha=1.)])
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.recovery, 'relative', lambda p: str(p.relative_to(tmp_path)))
    monkeypatch.setattr(r, 'implementation', lambda c: (cfg, {'implementation': 'same'}))
    monkeypatch.setattr(r, 'evidence', lambda *a: ({}, None, bank, {'CAV_events': 0}))
    monkeypatch.setattr(r, 'probe', lambda *a: pytest.fail('freeze opened native outcomes'))
    for root, names in ((tmp_path/'recovery', ['lock.json', 'shared_schedule.json']),
            (tmp_path/'data', ['summary.json', 'audit.json', 'shared_schedule.json',
                              'single/sequences.json', 'single/sequence_manifest.json',
                              'dual/sequences.json', 'dual/sequence_manifest.json'])):
        for name in names:
            save(root/name, {})
    out = tmp_path/'frozen'
    result = r.freeze('cfg', tmp_path/'recovery', tmp_path/'data', out)
    assert result['probes'] == 6 and not result['new_outcomes_opened']
    assert len(r.checked('cfg', out)[4]) == 6
    with pytest.raises(FileExistsError):
        r.freeze('cfg', tmp_path/'recovery', tmp_path/'data', out)
    monkeypatch.setattr(r, 'implementation', lambda c: (cfg, {'implementation': 'changed'}))
    with pytest.raises(ValueError, match='changed after freeze'):
        r.checked('cfg', out)


def test_audit_requires_exact_complete_schedule(tmp_path, monkeypatch):
    cfg = read(r.DEFAULT)
    lock = {'test': True}
    task = dict(case_index=0, case_id='0', source_seed=7, mode='single', arm='initial_peak', seed=1)
    row = dict(task, event_result=False, cav_id=0, collision_pairs=[],
               event_type='horizon_no_collision', natural_path_log_probability=-1.)
    monkeypatch.setattr(r, 'checked', lambda *a: (cfg, {}, None, {}, [task], lock))
    save(tmp_path/'opened_protocol.json', lock)
    save(tmp_path/'plan.json', [task])
    save(tmp_path/'probe_00000.json', row)
    summary = dict(complete=True, lock_sha256=identity(lock),
                   artifacts={'probe_00000.json': digest(tmp_path/'probe_00000.json')}, **r.summarize([row]))
    save(tmp_path/'summary.json', summary)
    assert r.audit('c', 'f', tmp_path, tmp_path/'audit.json')['verified']
    monkeypatch.setattr(r, 'checked', lambda *a: (cfg, {}, None, {}, [task, task], lock))
    with pytest.raises(ValueError, match='Incomplete'):
        r.audit('c', 'f', tmp_path, tmp_path/'bad.json')
