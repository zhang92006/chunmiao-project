import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import ROOT, read, save
from scenario_reconstruction.highd_v46_d2rl import identity
from scenario_reconstruction.highd_v46_discovery_v2 import (
    calibration_candidates, check_lock, choose_pairs, choose_states, describe,
)


def config():
    return read(ROOT / 'configs/highd_v46_discovery_v2.json')


def states():
    return [{'time_s': t, 'screen_score': 100 - t,
             'screen_pairs': [{'mechanism': 'lateral', 'score': 1. if t >= 40 else 0.}]}
            for t in range(60)]


def test_mixed_budget_unique_horizon_and_lateral_slot_without_outcome_labels():
    cfg = config()['bank']
    s = states()
    chosen = choose_states(s, 'mixed5_top2', cfg, 60, 4, np.random.default_rng(2))
    assert len(chosen) == 5 and max(chosen) <= 56
    assert list(chosen.values()).count('geometry') == 2
    assert list(chosen.values()).count('uniform') == 2
    assert list(chosen.values()).count('lateral_phase') == 1
    assert next(i for i, role in chosen.items() if role == 'lateral_phase') >= 40
    for state in s:
        state['collision_outcome'] = True
    assert chosen == choose_states(s, 'mixed5_top2', cfg, 60, 4, np.random.default_rng(2))
    assert len(choose_states(s, 'geometry5_top2', cfg, 60, 4, np.random.default_rng(2))) == 5


def test_phase_fallback_and_short_episodes_do_not_duplicate_states():
    s = states()[:3]
    selected = choose_states(s, 'mixed5_top2', config()['bank'], 60, 4, np.random.default_rng(2))
    assert len(selected) == 3
    s = states()
    for state in s:
        state['screen_pairs'] = []
    selected = choose_states(s, 'mixed5_top2', config()['bank'], 60, 4, np.random.default_rng(2))
    assert 'phase_fallback' in selected.values()


def test_rank_two_is_not_displaced_and_pair_budget_is_three():
    rows = [{'actor_ids': [i, i + 10], 'mechanism': 'bracket' if i < 2 else 'lateral'} for i in range(28)]
    chosen = choose_pairs(rows, 'mixed5_top2', np.random.default_rng(1))
    assert [r['actor_ids'] for r in chosen[:2]] == [[0, 10], [1, 11]]
    assert len(chosen) == len({tuple(r['actor_ids']) for r in chosen}) == 3
    assert choose_pairs(rows[:1], 'mixed5_top2', np.random.default_rng(1))[0]['actor_ids'] == [0, 10]
    assert len(choose_pairs([], 'mixed5_top2', np.random.default_rng(1))) == 0


def test_fresh_validation_lock_and_consumed_seed_guards():
    cfg = config()
    lock = {'complete': True, 'split': 'train', 'config_identity': identity(cfg),
            'implementation_sha256': 'code', 'dependencies': {'proxy': 'hash'}}
    check_lock(cfg, 'validation', 'code', {'proxy': 'hash'}, lock)
    with pytest.raises(ValueError):
        check_lock(cfg, 'validation', 'changed', {'proxy': 'hash'}, lock)
    with pytest.raises(ValueError):
        check_lock(cfg, 'validation', 'code', {'proxy': 'changed'}, lock)
    with pytest.raises(ValueError):
        check_lock(cfg, 'validation', 'code', {'proxy': 'hash'})
    for start in (100000, 170000, 160000):
        changed = copy.deepcopy(cfg)
        changed['validation']['seed_start'] = start
        with pytest.raises(ValueError):
            check_lock(changed, 'train', 'code', {})


def test_calibration_is_episode_stratified_and_excludes_unsupported_zero_argmax():
    rows = [{'seed': seed, 'methods': ['mixed5_top2'], 'views': {'production_strict': strict},
             'metrics': {'positive_interaction_mass': mass, 'max_positive_interaction': interaction}}
            for seed, strict, mass, interaction in [(1, True, .1, .2), (1, True, .2, .3), (2, False, 0., 0.), (3, False, .1, .01)]]
    selected = dict(calibration_candidates(rows, ['mixed5_top2'], config()['calibration']))
    assert selected == {1: ['mixed5_top2'], 3: ['negative_control']}
    assert describe([])['mean'] is None


@pytest.mark.parametrize('positive', [False, True])
def test_smoke_run_multiple_states_artifacts_and_three_pair_budget(tmp_path, monkeypatch, positive):
    from scenario_reconstruction import highd_v46_discovery_v2 as module
    cfg = config()
    cfg['train'] = {'seed_start': 160000, 'episodes': 1}
    save(tmp_path / 'config.json', cfg)

    class FakeRoad:
        def __init__(self, *args):
            self.time, self.crashes, self.cav = 0., [], SimpleNamespace(id=0)

        def step_policy(self, mode, epsilon):
            self.choose([], [], {}, mode)
            self.time += 1
            return self.time < 7

    def fake_snapshot(road, units, pmfs, seed, features):
        return {'seed': seed, 'time_s': road.time, 'cav_id': 0,
                'vehicles': [{'id': i} for i in range(4)],
                'units': [{'ids': [i], 'p': [.5, .5]} for i in (1, 2, 3)],
                'pmfs': {str(i): [.5, .5] for i in range(4)}, 'features': features}

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(module, 'NativeExperiment', FakeRoad)
    monkeypatch.setattr(module, 'snapshot', fake_snapshot)
    monkeypatch.setattr(module, 'screen_pairs', lambda *args: [
        {'actor_ids': ids, 'mechanism': 'bracket', 'score': 1.} for ids in ([1, 2], [1, 3], [2, 3])])
    risk = np.array([[.1, .1], [.1, 1.]]) if positive else np.zeros((2, 2))
    monkeypatch.setattr(module, 'response_risk', lambda *args, **kwargs: risk)
    monkeypatch.setattr(module, 'continue_state', lambda model, assets, physical, state, forced, *args:
                        {'clearance_risk': .9 if all(forced.values()) else .1, 'cav_first_collision': False})
    monkeypatch.setattr(module, 'PairModelQueries', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'ensure_assets', lambda: (tmp_path, {'natural_target_sha256': 'fake'}))
    result = module.run(tmp_path / 'config.json', 'train', tmp_path / 'output')
    assert result['complete'] and result['natural_states'] == 7
    assert result['methods']['mixed5_top2']['candidate_states'] == 5
    assert result['methods']['mixed5_top2']['evaluated_pairs'] == 15
    assert result['methods']['mixed5_top2']['online_intervention_rate'] is None
    assert result['methods']['mixed5_top2']['second_factor_H_vs_P_tv_chosen']['n'] == (5 if positive else 0)
    assert result['methods']['mixed5_top2']['shadow_intervention_count'] == (5 if positive else 0)
    assert result['maximum_shadow_likelihood_error'] < 1e-10
    if positive:
        assert result['methods']['mixed5_top2']['native_interaction_ge_threshold_cases'] == 1
