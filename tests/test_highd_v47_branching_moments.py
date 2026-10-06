from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_branching_moments as m


def test_exact_informative_prefix_posterior_and_root_covariance():
    # P/T assign different probabilities to the prefix and to each suffix.
    # Enumerate all two P/two T branches; Y=1 on suffix action 1.
    pp, tp = np.array([.9, .1]), np.array([.3, .7])
    ps, ts = [np.array([.99, .01]), np.array([.8, .2])], [np.array([.4, .6]), np.array([.2, .8])]
    mean = wrong_half = second = inferred_removed = single_second = 0.
    conditional_true = 0.
    for x in (0, 1):
        gp = .5*(pp[x]+tp[x]); eta = pp[x]/(pp[x]+tp[x])
        z = np.array([0., 2*pp[x]*ps[x][1]/(pp[x]*ps[x][1]+tp[x]*ts[x][1])])
        mp, mt = ps[x]@z, ts[x]@z
        vp, vt = ps[x]@(z*z)-mp**2, ts[x]@(z*z)-mt**2
        true_removed = eta*vp+(1-eta)*vt+eta*(1-eta)*(mp-mt)**2-(eta**2*vp+(1-eta)**2*vt)/2
        conditional_true += gp*true_removed
        for a in product((0, 1), repeat=4):
            probability = gp*float(np.prod(ps[x][list(a[:2])])*np.prod(ts[x][list(a[2:])]))
            r = m.branch_statistics(z[list(a[:2])], z[list(a[2:])], eta)
            assert 0 <= r['value'] <= 2
            mean += probability*r['value']; second += probability*r['value']**2
            single_second += probability*(eta*z[a[0]]**2+(1-eta)*z[a[2]]**2)
            wrong_half += probability*.5*sum(z[list(a)].reshape(2, 2).mean(1))
            inferred_removed += probability*r['removed_conditional_variance_unbiased']
    truth = pp[0]*ps[0][1]+pp[1]*ps[1][1]
    assert mean == pytest.approx(truth)
    assert wrong_half != pytest.approx(truth)
    assert single_second-second == pytest.approx(conditional_true)
    assert inferred_removed == pytest.approx(conditional_true)
    assert second-mean**2 < single_second-mean**2


def test_clone_copies_all_dynamic_state_and_removes_parent_callbacks():
    cars = [SimpleNamespace(id=0, x=1.), SimpleNamespace(id=1, x=2.)]
    road = SimpleNamespace(model=object(), config={'a': 1}, vehicles=cars, cav=cars[0], time=2.,
        natural=SimpleNamespace(histories={1: [(0, 1), 3]}, previous={(1, 2)}),
        rng=np.random.default_rng(4), minimum_gap=3., minimum_ttc=4., crashes=[],
        last_acceleration={1: -.1}, diagnostics={'x': [1]}, sample=lambda _: 'parent')
    before = m.dynamic_identity(road); clone = m.clone_native(road, 5)
    assert m.dynamic_identity(clone) == before and clone.model is road.model
    assert clone.cav is clone.vehicles[0] and clone.cav is not road.cav
    assert not hasattr(clone, 'sample')
    clone.natural.histories[1][1] = 99; clone.vehicles[0].x = 5; clone.diagnostics['x'].append(2)
    assert road.natural.histories[1][1] == 3 and road.cav.x == 1 and road.diagnostics == {'x': [1]}


def test_protected_allocation_bound_for_every_uncertainty_box_vertex():
    a = np.array([.2, .3, .5]); lo = np.array([0., .1, 0.]); hi = np.array([.1, 2., .4])
    r = m.protected_allocation(a, lo, hi, 1.1); q = np.array(r['upper_envelope_candidate'])
    assert q.sum() == pytest.approx(1.) and np.all(q >= a/1.1-1e-14)
    assert r['state_LR_max'] <= 1.1+1e-14
    for j in product(*zip(lo, hi)):
        j = np.array(j)
        assert (a*a/q)@j <= 1.1*(a@j)+1e-14
        assert ((a*a/q-a)@j) <= r['worst_case_M2_difference_upper']+1e-14
    # The unconditional bound also holds OUTSIDE the fitted box.
    for j in np.eye(3)*1000:
        assert (a*a/q)@j <= 1.1*(a@j)+1e-10


def test_all_zero_lower_bounds_do_not_fake_a_strict_improvement():
    a = [.5, .5]; r = m.protected_allocation(a, [0., 0.], [1., 100.])
    assert not r['strict_improvement_certified'] and r['recommended_q'] == a
    assert r['upper_envelope_candidate'] != a
    assert r['worst_case_M2_difference_upper'] > 0
    sharp = m.protected_allocation(a, [1., 100.], [1., 100.])
    assert sharp['strict_improvement_certified'] and sharp['recommended_q'] != a
    assert m.protected_allocation(a, [0., 0.], [0., 0.])['recommended_q'] == a


@pytest.mark.parametrize('lo,hi,cap', [([-1, 0], [1, 1], 1.1), ([1, 0], [0, 1], 1.1),
                                    ([0, 0], [1, np.nan], 1.1), ([0, 0], [1, 1], .9)])
def test_invalid_uncertainty_sets(lo, hi, cap):
    with pytest.raises(ValueError): m.protected_allocation([.5, .5], lo, hi, cap)


def test_minimax_root_budget_retains_zeros_and_reallocates_to_large_coefficients():
    hs = [dict(stratum_id=i, coefficient_mass=c, draws=4, cells=[i], measurement_probabilities=[1.])
          for i, c in enumerate([.9, .05, 1e-20])]
    old = dict(states=[dict(state_id=0, strata=hs, plan={'source_seed': 1})],
               target_masses=[1.], state_proposals={'target': [1.]})
    cfg = dict(roots_per_state=256, seed_starts=[6200000, 6300000], schedule_seed=620017)
    d = m.design_for(old, [{'P': [[.9, .05, .05]]}], cfg)
    counts = [h['draws'] for h in d['states'][0]['strata']]
    assert sum(counts) == 256 and min(counts) >= 2 and counts[0] > 200
    assert d == m.design_for(old, [{'P': [[.9, .05, .05]]}], cfg)
    seeds = [v for s in d['schedules'] for r in s['rows'] for v in [r['prefix_seed']]+r['suffix_seeds']]
    assert len(seeds) == len(set(seeds))


def test_measurement_variance_uses_roots_not_correlated_leaves():
    hs = [dict(stratum_id=0, draws=3, coefficient_mass=.8)]
    rows = [dict(stratum_id=0, within_stratum_draw=i, branching_value=z,
                 removed_conditional_variance_unbiased=.01, branches=[{}, {}, {}, {}])
            for i, z in enumerate([0., .4, .8])]
    e = m.estimate(hs, rows, 'branching', .01)
    assert e['J'] == pytest.approx(.8*.4)
    assert e['estimator_variance'] == pytest.approx(.8**2*np.var([0., .4, .8], ddof=1)/3)
    with pytest.raises(ValueError): m.estimate(hs, rows[:-1], 'branching', .01)


def test_native_branch_clone_and_posterior_can_be_replayed_with_toy_road(monkeypatch):
    p = np.array([[.9, 0.], [0., .1]]); t = np.array([[.3, 0.], [0., .7]])
    units = [dict(ids=[1, 2], p=p)]
    state = dict(time_s=0., pmfs={'1': [.9, .1], '2': [.9, .1]}, units=units)
    class Road:
        def __init__(self, seed):
            self.time=0.; self.model=object(); self.config={}; self.crashes=[]; self.cav=SimpleNamespace(id=0, x=0.)
            self.vehicles=[self.cav, SimpleNamespace(id=1, x=1.), SimpleNamespace(id=2, x=2.)]
            self.natural=SimpleNamespace(histories={}, previous=set()); self.rng=np.random.default_rng(seed)
            self.minimum_gap=1.; self.minimum_ttc=1.; self.last_acceleration={}
        def action_pmfs(self): return {1: np.array([.9, .1]), 2: np.array([.9, .1])}
        def prepare(self, pmfs): return units, {}
        def step_policy(self, *args):
            a=self.sample(self.action_pmfs(), *args); self.time+=1
            if self.time==4 and a[1]==1:self.crashes=[[0, 1]]
    monkeypatch.setattr(m, 'restore_prepared_state', lambda *args: Road(args[-1]))
    # Existing sampler with epsilon .1 can realize t via an h with support.
    h=(t-.01*p)/.99
    *_, q1, q2=m.proposal_factors(p, h, [.1, .1])
    monkeypatch.setattr(m.parent.parent, 'pair_choice', lambda *args: ([1,2], p, h, q1, q2))
    cfg=dict(horizon_s=4, fork_s=2, continuation=dict(epsilon=[.1,.1]))
    s=dict(action_indices=[0,0], prefix_seed=1, prefix_tilted_component=True,
           suffix_seeds=[2,3,4,5], control_uniform=.9)
    a=m.root_continuation(None,None,{},state,[1,2],s,cfg)
    assert a==m.root_continuation(None,None,{},state,[1,2],s,cfg)
    assert len(a['branches'])==4 and a['future_decisions']==9
    assert a['posterior_natural_fraction'] == pytest.approx(a['prefix']['future_weight']/2)
    assert a['selected_branch']==(0 if .9<a['posterior_natural_fraction'] else 2)
    assert 0<=a['branching_value']<=2


def test_exclusive_two_stage_run_saves_allocation_before_evaluation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction.highd_v46_assets import read, save
    cfg=dict(confidence=.95, allocation_loss_cap=1.1, workers=1)
    hs=[dict(stratum_id=0, coefficient_mass=1., draws=2)]
    design=dict(states=[dict(strata=hs, plan=dict(source_seed=1))], target_masses=[1.],
                independent_roots=4, schedules=[dict(batch=b,state_id=0,rows=[dict(stratum_id=0,within_stratum_draw=i) for i in (0,1)]) for b in (0,1)])
    expected={'fixture':'locked'}; lock=tmp_path/'lock'; save(lock,expected); out=tmp_path/'run'
    monkeypatch.setattr(m,'inputs',lambda _:(cfg,{},tmp_path,[{'actor_ids':[1,2]}],[{'state':{}}],design,expected))
    monkeypatch.setattr(m,'ProcessPoolExecutor',lambda **_:ThreadPoolExecutor(max_workers=1))
    def collect(task):
        if task[0]['batch']==1: assert read(out/'batch0_allocation_before_batch1.json')['recommended_q']==[1.]
        assert read(out/'measurement_plan.json')==design
        return dict(task[0],rows=[dict(r,single_value=0.,branching_value=0.,branches=[],future_decisions=0,
                           removed_conditional_variance_unbiased=0.) for r in task[0]['rows']])
    monkeypatch.setattr(m,'_collect',collect)
    m.run('fixture',lock,out); s=read(out/'summary.json')
    assert s['complete'] and s['independent_roots']==4 and not s['ppo_ready']
    assert not s['allocation_improvement_validated']
    with pytest.raises(FileExistsError):m.run('fixture',lock,out)
