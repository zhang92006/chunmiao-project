from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_continuation_moments as m


def test_stopped_adaptive_path_mixture_is_exact_and_weight_bounded():
    # T depends on the preceding action. First action 1 is a stopped event;
    # action 0 proceeds to another decision. Enumerate the stopped path law.
    paths = [((1,), .03, .6), ((0, 1), .97*.02, .4*.7),
             ((0, 0), .97*.98, .4*.3)]
    assert sum(p for _, p, _ in paths) == pytest.approx(1.)
    assert sum(t for _, _, t in paths) == pytest.approx(1.)
    mass = event = second = squared_wrong = 0.
    for actions, p, t in paths:
        g = .5*p+.5*t; w = m.path_weight(np.log(t/p), .5)
        assert w == pytest.approx(p/g) and 0 < w <= 2
        y = actions[-1] == 1
        mass += g*w; event += g*y*w; second += g*y*w*w
        squared_wrong += g*y*w*w
    assert mass == pytest.approx(1.)
    assert event == pytest.approx(.03+.97*.02)
    assert second <= 2*event
    assert squared_wrong != pytest.approx(event)


@pytest.mark.parametrize('ratio', [-1000., -10., 0., 10., 700.])
def test_log_domain_mixture_on_natural_and_tilted_paths(ratio):
    assert 0 <= m.path_weight(ratio, .5) <= 2
    assert m.path_weight(ratio, 1.) == 1
    assert m.path_weight(0., .5) == pytest.approx(1.)


@pytest.mark.parametrize('ratio,eta', [(np.nan, .5), (np.inf, .5), (0, 0), (0, 1.1)])
def test_invalid_mixture(ratio, eta):
    with pytest.raises(ValueError): m.path_weight(ratio, eta)


def test_likelihood_underflow_is_not_silently_zeroed():
    with pytest.raises(FloatingPointError): m.path_weight(1000., .5)


def test_zero_weighted_events_keep_positive_uncertainty():
    mean, var, interval = m.bounded_interval([0.]*64, 2., .05)
    assert mean == var == 0 and interval[1] > 0
    x = [0., .2, .3, 1.2]
    mean, var, _ = m.bounded_interval(x, 2., .05)
    assert mean == pytest.approx(np.mean(x))
    assert var == pytest.approx(np.var(x, ddof=1)/len(x))


def test_j_uses_future_weight_once_and_correct_stratum_variance():
    strata = [dict(stratum_id=0, draws=3, coefficient_mass=.7),
              dict(stratum_id=1, draws=2, coefficient_mass=2.)]
    rows = []
    for sid, zs in [(0, [0., .4, .8]), (1, [0., 1.5])]:
        for j, z in enumerate(zs):
            rows.append(dict(stratum_id=sid, within_stratum_draw=j,
                             natural=dict(event_result=z>0), defensive=dict(weighted_event=z)))
    e = m.estimate(strata, rows, 'defensive', .01)
    assert e['J'] == pytest.approx(.7*.4+2*.75)
    assert e['estimator_variance'] == pytest.approx(.7**2*np.var([0., .4, .8], ddof=1)/3
                                                       +4*np.var([0., 1.5], ddof=1)/2)
    with pytest.raises(ValueError): m.estimate(strata, rows[:-1], 'defensive', .01)


def test_two_batches_and_paired_crn_are_not_independent_arm_replicates():
    strata = [dict(stratum_id=0, draws=2, coefficient_mass=1.)]
    design = dict(states=[dict(strata=strata, plan=dict(source_seed=12))],
                  target_masses=[1.], state_proposals={'target': [1.]})
    records = []
    for batch in range(2):
        records.append(dict(batch=batch, state_id=0, rows=[dict(stratum_id=0,
                     within_stratum_draw=j, natural=dict(event_result=bool(j)),
                     defensive=dict(event_result=bool(j), weighted_event=float(j))) for j in range(2)]))
    s = m.summarize(dict(batches=2, arms=['natural', 'defensive'], confidence=.95), design, records)
    for arm in s['arms'].values():
        assert arm['states'][0]['J'] == .5
        assert arm['states'][0]['estimator_variance'] == .125
        assert not arm['states'][0]['precision_supported']
    assert all(x['difference_variance'] == 0 for x in s['paired_differences'])


def test_initial_correlated_partners_and_each_future_prepare_once(monkeypatch):
    p = np.array([[.4, .1], [.2, .3]]); h = np.array([[.05, .15], [.2, .6]])
    *_, q1, q2 = m.proposal_factors(p, h, [.1, .1])
    units = [dict(ids=[1, 2], p=p)]
    state = dict(time_s=0., pmfs={'1': [.5, .5], '2': [.6, .4]}, units=units)
    roads = []
    class Road:
        def __init__(self, seed):
            self.time = 0.; self.rng = np.random.default_rng(seed)
            self.cav = SimpleNamespace(id=0); self.crashes = []; self.prepares = 0
        def action_pmfs(self): return {1: np.array([.5, .5]), 2: np.array([.6, .4])}
        def prepare(self, pmfs): self.prepares += 1; return units, {}
        def step_policy(self, *args):
            actions = self.sample(self.action_pmfs(), *args)
            self.time += 1
            if self.time == 2 and actions[1] == actions[2] == 1:
                self.crashes = [[0, 1]]
    def restore(model, assets, physical, state, seed):
        road = Road(seed); roads.append(road); return road
    monkeypatch.setattr(m, 'restore_prepared_state', restore)
    monkeypatch.setattr(m, 'pair_choice', lambda *args: ([1, 2], p, h, q1, q2))
    cfg = dict(natural_fraction=.5, epsilon=[.1, .1])
    for tilted in (False, True):
        r = m.continue_path(None, None, {}, state, [1, 2], [0, 1], 8, 4, cfg, tilted)
        assert roads[-1].prepares == len(r['steps'])
        assert r['initial_cancelled_log_conditional'] == pytest.approx(0.)
        assert r['log_t_over_p'] == pytest.approx(r['full_log_t']-r['full_log_p'])
        assert r['future_weight'] <= 2
        for x in r['steps']:
            assert x['log_t_over_p'] == pytest.approx(np.log(np.array(x['t_factors'])/x['p_factors']).sum())


def test_worker_shares_only_identical_natural_component_paths(monkeypatch):
    monkeypatch.setattr(m, '_MODEL', object(), raising=False); calls = []
    def path(*args, tilted):
        calls.append((args[6], tilted))
        return dict(event_result=False, weighted_event=0., future_weight=1., steps=[{}])
    monkeypatch.setattr(m, 'continue_path', path)
    schedule = dict(batch=0, state_id=0, rows=[dict(action_indices=[1, 2], native_seed=17,
                    tilted_component=False), dict(action_indices=[1, 2], native_seed=18, tilted_component=True)])
    r = m._collect((schedule, {}, dict(actor_ids=[3, 4]), '.', {}, dict(horizon_s=4, continuation={})))
    assert calls == [(17, False), (18, False), (18, True)]
    assert r['rows'][0]['shared_physical_path'] and not r['rows'][1]['shared_physical_path']


def test_first_step_endpoints_do_not_duplicate_or_tilt_initial_partners(monkeypatch):
    monkeypatch.setattr(m, '_MODEL', object(), raising=False); calls = []
    def path(*args, tilted):
        calls.append(tilted)
        return dict(event_result=True, weighted_event=1., future_weight=1., steps=[])
    monkeypatch.setattr(m, 'continue_path', path)
    r = m._collect((dict(batch=0, state_id=0, rows=[dict(action_indices=[1, 2], native_seed=19,
                  tilted_component=True)]), {}, dict(actor_ids=[3, 4]), '.', {}, dict(horizon_s=4, continuation={})))
    assert calls == [False] and r['rows'][0]['shared_physical_path']


def test_pair_choice_is_pre_draw_supported_and_outcome_blind(monkeypatch):
    p = np.array([[.2, .3], [.1, .4]])
    cars = [SimpleNamespace(id=i, x=x) for i, x in [(0, 0), (1, 1), (2, -1), (3, 500)]]
    road = SimpleNamespace(vehicles=cars, cav=cars[0])
    units = [dict(ids=[1, 2], p=p), dict(ids=[3], p=np.array([.5, .5]))]
    monkeypatch.setattr(m, 'challenge', lambda *args: np.array([[0., .2], [.8, 1.]]))
    cfg = dict(radius_m=115, proxy_step_s=1/3, clearance_scale_m=10., risk_power=4, epsilon=[.1, .1])
    ids, pp, hh, q1, q2 = m.pair_choice(road, units, {c.id: c for c in cars}, {}, 2., cfg)
    assert ids == [2, 1]  # rear-to-front ordering also transposes the natural table
    np.testing.assert_array_equal(pp, p.T)
    assert hh.sum() == pytest.approx(1.)
    assert np.all(q1[:, None]*q2 >= .01*pp-1e-15)


def test_complete_exclusive_run_locks_schedule_before_workers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction.highd_v46_assets import save, read
    cfg = dict(batches=2, arms=['natural', 'defensive'], confidence=.95, workers=2)
    plan = dict(actor_ids=[1, 2], source_seed=1)
    strata = [dict(stratum_id=0, coefficient_mass=1., draws=2)]
    schedules = [dict(batch=b, state_id=0, rows=[dict(stratum_id=0, within_stratum_draw=j,
                 native_seed=10+2*b+j, action_indices=[1, 2], tilted_component=False) for j in range(2)])
                 for b in range(2)]
    design = dict(states=[dict(plan=plan, strata=strata)], schedules=schedules, target_masses=[1.],
                  state_proposals={'target': [1.]}, logical_draws=8, planned_native_paths_upper_bound=4)
    expected = {'fixture': 'locked'}; lock = tmp_path/'lock.json'; save(lock, expected); out = tmp_path/'run'
    monkeypatch.setattr(m, 'inputs', lambda _: (cfg, {}, tmp_path, [plan], [{'state': {}}], design, expected))
    monkeypatch.setattr(m, 'ProcessPoolExecutor', lambda **_: ThreadPoolExecutor(max_workers=2))
    def collect(task):
        assert read(out/'measurement_plan.json') == design
        root = dict(event_result=False, weighted_event=0.)
        return dict(task[0], rows=[dict(r, natural=root, defensive=root, shared_physical_path=True)
                                  for r in task[0]['rows']])
    monkeypatch.setattr(m, '_collect', collect)
    m.run('fixture', lock, out)
    result = read(out/'summary.json')
    assert result['complete'] and len(result['artifact_hashes']) == 4
    assert result['native_paths_simulated'] == 4
    assert not result['production_changed'] and not result['ppo_ready']
    assert result['arms']['defensive']['states'][0]['bank_simultaneous_interval'][1] > 0
    with pytest.raises(FileExistsError): m.run('fixture', lock, out)
