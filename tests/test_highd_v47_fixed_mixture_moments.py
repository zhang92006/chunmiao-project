from itertools import product

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_fixed_mixture_moments as m


def fixture_design():
    h = dict(stratum_id=0, draws=5, coefficient_mass=1., cells=[0, 1],
             measurement_probabilities=[.7, .3])
    return dict(states=[dict(state_id=0, strata=[h], plan=dict(source_seed=1))],
                target_masses=[1.], state_proposals={'target': [1.]})


def test_exact_fixed_mis_and_removed_variance_not_just_a_simulation():
    # Four P roots / four T roots, exact enumeration of all 2^8 paths.
    p = np.array([.98, .02]); t = np.array([.4, .6]); g = (p+t)/2
    z = np.array([0., p[1]/g[1]])
    mp, mt = p@z, t@z; sp = p@(z*z)-mp*mp; st = t@(z*z)-mt*mt
    expected_fixed_var = (sp+st)/16
    iid_var = (g@(z*z)-(g@z)**2)/8
    assert iid_var-expected_fixed_var == pytest.approx((mp-mt)**2/32)
    e_mean = e_var = removed = inferred = 0.
    for actions in product((0, 1), repeat=8):
        probability = float(np.prod(p[list(actions[:4])])*np.prod(t[list(actions[4:])]))
        a = z[list(actions[:4])]; b = z[list(actions[4:])]
        result = m.fixed_statistics(a, b, .05)
        e_mean += probability*result['mean']
        e_var += probability*result['estimator_variance']
        removed += probability*result['removed_between_component_variance_unbiased']
        inferred += probability*result['iid_variance_inferred_same_draws_unbiased']
    assert e_mean == pytest.approx(.02)
    assert e_var == pytest.approx(expected_fixed_var)
    assert removed == pytest.approx(iid_var-expected_fixed_var)
    assert inferred == pytest.approx(iid_var)


def test_component_means_are_not_individually_clipped_as_probabilities():
    e = m.fixed_statistics([1.5]*4, [.1]*4, .05)
    assert e['mean'] == pytest.approx(.8)
    assert e['component_means'][0] == 1.5
    assert e['component_mean_intervals'][0][1] == 2
    assert e['estimator_variance'] == 0
    assert e['removed_between_component_variance_unbiased'] == pytest.approx(1.4**2/32)


def test_fixed_variance_excludes_between_component_sample_dispersion():
    e = m.fixed_statistics([0., 0.], [1., 1.], .05)
    assert e['mean'] == .5 and e['estimator_variance'] == 0
    assert np.var([0., 0., 1., 1.], ddof=1)/4 > 0
    e = m.fixed_statistics([0., 1.], [0., 1.], .05)
    assert e['removed_between_component_variance_unbiased'] < 0
    assert e['removed_between_component_variance_interval'][0] == 0


def test_zero_observations_retain_component_uncertainty():
    e = m.fixed_statistics([0., 0.], [0., 0.], .001)
    assert e['mean'] == e['estimator_variance'] == 0
    assert e['probability_interval'] == [0., 1.]
    assert e['removed_between_component_variance_interval'][1] > 0


@pytest.mark.parametrize('number', [1, 25, 52, 63, 64, 65, 90])
def test_equal_even_budget_keeps_every_stratum_and_variance_estimable(number):
    counts = m.even_counts(number)
    assert len(counts) == number
    assert all(n >= 4 and n % 2 == 0 for n in counts)
    assert sum(counts) == max(256, 4*number)
    assert max(counts)-min(counts) <= 2


@pytest.mark.parametrize('a,b', [([0.], [0.]), ([0., 0.], [0.]), ([-.1, 0.], [0., 0.]),
                              ([0., 3.], [0., 0.]), ([0., np.nan], [0., 0.])])
def test_reject_invalid_component_observations(a, b):
    with pytest.raises(ValueError): m.fixed_statistics(a, b, .05)


def test_schedule_fixed_half_counts_and_iid_coins_independent_and_reproducible():
    cfg = dict(batches=2, minimum_draws_per_state_per_arm=256,
               continuation_seed_starts=[6000000, 6100000], schedule_seed=600017)
    d = m.make_design(fixture_design(), [{'P': [[.7, .3]]}], cfg)
    assert d == m.make_design(fixture_design(), [{'P': [[.7, .3]]}], cfg)
    assert d['logical_draws'] == 1536
    seeds = []
    for s in d['schedules']:
        assert len(s['rows']) == 256
        assert sum(r['fixed_tilted_component'] for r in s['rows']) == 128
        # Random counts can differ, but fixed paths use the same action/seed per root.
        assert any(r['iid_tilted_component'] != r['fixed_tilted_component'] for r in s['rows'])
        seeds.extend(r['native_seed'] for r in s['rows'])
    assert len(set(seeds)) == 512


def test_correct_fixed_J_variance_and_completeness():
    hs = [dict(stratum_id=0, draws=4, coefficient_mass=3.)]
    rows = [dict(stratum_id=0, within_stratum_draw=i, fixed_tilted_component=i >= 2,
                 fixed_defensive=dict(weighted_event=x)) for i, x in enumerate([0., .2, 1., 1.5])]
    e = m.estimate(hs, rows, 'fixed_defensive', .01)
    assert e['J'] == pytest.approx(3*np.mean([0., .2, 1., 1.5]))
    assert e['estimator_variance'] == pytest.approx(9*.25*(np.var([0., .2], ddof=1)/2
                                                                       +np.var([1., 1.5], ddof=1)/2))
    with pytest.raises(ValueError): m.estimate(hs, rows[:-1], 'fixed_defensive', .01)
    with pytest.raises(ValueError): m.estimate(hs, rows[:-1]+[rows[0]], 'fixed_defensive', .01)


def test_worker_deduplicates_shared_physical_paths_without_fake_replication(monkeypatch):
    monkeypatch.setattr(m.parent, '_MODEL', object(), raising=False); calls = []
    def continuation(*args, tilted):
        calls.append((args[6], tilted))
        return dict(steps=[{}], weighted_event=float(tilted), event_result=tilted)
    monkeypatch.setattr(m.parent, 'continue_path', continuation)
    rows = [dict(action_indices=[0, 1], native_seed=10+i, iid_tilted_component=a,
                 fixed_tilted_component=b) for i, (a, b) in enumerate([(False, False), (True, False), (False, True), (True, True)])]
    r = m._collect((dict(batch=0, state_id=0, rows=rows), {}, dict(actor_ids=[1, 2]), '.', {},
                    dict(horizon_s=4, continuation={})))
    assert len(calls) == 7 and sum(tilt for _, tilt in calls) == 3
    assert r['rows'][0]['natural'] is r['rows'][0]['fixed_defensive'] is r['rows'][0]['iid_defensive']
    assert r['rows'][3]['iid_defensive'] is r['rows'][3]['fixed_defensive']


def test_first_step_endpoint_common_law_is_simulated_once(monkeypatch):
    monkeypatch.setattr(m.parent, '_MODEL', object(), raising=False); calls = []
    def continuation(*args, tilted):
        calls.append(tilted); return dict(steps=[], weighted_event=1., event_result=True)
    monkeypatch.setattr(m.parent, 'continue_path', continuation)
    row = dict(action_indices=[0, 1], native_seed=1, iid_tilted_component=True, fixed_tilted_component=True)
    r = m._collect((dict(batch=0, state_id=0, rows=[row]), {}, dict(actor_ids=[1, 2]), '.', {},
                    dict(horizon_s=4, continuation={})))
    assert calls == [False] and not r['rows'][0]['simulated_tilted_path']


def test_complete_run_writes_outcome_blind_plan_and_refuses_overwrite(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction.highd_v46_assets import save, read
    cfg = dict(batches=2, confidence=.95, workers=2)
    plan = dict(source_seed=1); strata = [dict(stratum_id=0, draws=4, coefficient_mass=1.)]
    schedules = [dict(batch=b, state_id=0, rows=[dict(stratum_id=0, within_stratum_draw=i,
                    fixed_tilted_component=i>=2, iid_tilted_component=i%2==1) for i in range(4)]) for b in range(2)]
    design = dict(states=[dict(plan=plan, strata=strata)], schedules=schedules, target_masses=[1.],
                  state_proposals={'target': [1.]}, logical_draws=24, planned_native_paths_upper_bound=14)
    expected = {'fixture': 'locked'}; lock = tmp_path/'lock.json'; save(lock, expected); out = tmp_path/'run'
    monkeypatch.setattr(m, 'inputs', lambda _: (cfg, {}, tmp_path, [plan], [{'state': {}}], design, expected))
    monkeypatch.setattr(m, 'ProcessPoolExecutor', lambda **_: ThreadPoolExecutor(max_workers=2))
    def collect(task):
        assert read(out/'measurement_plan.json') == design
        root = dict(event_result=False, weighted_event=0.)
        return dict(task[0], rows=[dict(r, natural=root, fixed_defensive=root, iid_defensive=root,
                                       simulated_tilted_path=False) for r in task[0]['rows']])
    monkeypatch.setattr(m, '_collect', collect)
    m.run('fixture', lock, out); s = read(out/'summary.json')
    assert s['complete'] and s['native_paths_simulated'] == 8
    assert s['arms']['fixed_defensive']['states'][0]['bank_simultaneous_interval'][1] > 0
    assert not s['production_changed'] and not s['ppo_ready']
    assert s['same_draw_structural_comparison']['removed_between_component_variance_unbiased'] == 0
    assert s['same_draw_structural_comparison']['removed_between_component_variance_interval'][1] > 0
    with pytest.raises(FileExistsError): m.run('fixture', lock, out)
