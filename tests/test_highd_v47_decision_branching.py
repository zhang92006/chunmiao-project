from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_decision_branching as m


def test_exact_recursive_variance_removal_includes_descendant_noise():
    eta=.37; dp,dt=.002,.003
    p=[dict(branching_value=b,single_value=b+s*np.sqrt(dp),removed_conditional_variance_unbiased=dp)
       for b in (.1,.3) for s in (-1,1)]
    t=[dict(branching_value=b,single_value=b+s*np.sqrt(dt),removed_conditional_variance_unbiased=dt)
       for b in (.8,1.2) for s in (-1,1)]
    mean=second=single_mean=single_second=removed=0.
    for c in product(p,p,t,t):
        r=m.combine_children(c,eta,0.); prob=1/256
        mean+=prob*r['branching_value'];second+=prob*r['branching_value']**2
        single_mean+=prob*(eta*c[0]['single_value']+(1-eta)*c[2]['single_value'])
        single_second+=prob*(eta*c[0]['single_value']**2+(1-eta)*c[2]['single_value']**2)
        removed+=prob*r['removed_conditional_variance_unbiased']
        assert r['selected_child']==0
        assert m.combine_children(c,eta,.99)['selected_child']==2
    assert mean==pytest.approx(eta*.2+(1-eta)*1.)
    assert single_mean==pytest.approx(mean)
    assert second-mean**2==pytest.approx((eta**2*.01+(1-eta)**2*.04)/2)
    assert removed==pytest.approx(single_second-second)
    assert removed>0


def test_terminal_integration_preserves_correlated_natural_partners(monkeypatch):
    joint=np.zeros((33,33));joint[1:3,1:3]=[[.4,.1],[.2,.3]]
    units=[dict(ids=[1,2],p=joint)]
    road=SimpleNamespace(cav=SimpleNamespace(id=0,x=0),vehicles=[SimpleNamespace(id=i,x=i) for i in (0,1,2)],
                         action_pmfs=lambda:{},prepare=lambda _:(units,{}))
    class Uniforms:
        def __init__(self,u):self.u=u
        def random(self):return self.u
    monkeypatch.setattr(m.np.random,'default_rng',lambda seed:Uniforms({1:.2,2:.6,3:.9}[seed]))
    def terminal(road,codes):
        event=codes[:,1]==2
        return dict(event=event,competing=np.zeros(len(codes),bool),elapsed_s=np.ones(len(codes)),
                    collision_pairs=[[[0,2]] if y else [] for y in event])
    monkeypatch.setattr(m,'terminal_decisions',terminal)
    es=[m.integrate_terminal(road,dict(future_weight=1.7),seed,.1) for seed in (1,2,3)]
    # U intervals have lengths .4,.4,.2. Integration recovers P(second=2)=.4.
    weights=np.array([.4,.4,.2])
    assert weights@[e['conditional_event_value'] for e in es]==pytest.approx(.4)
    assert weights@[e['branching_value'] for e in es]==pytest.approx(1.7*.4)
    assert weights@[e['removed_conditional_variance_unbiased'] for e in es]==pytest.approx(1.7**2*.1)
    assert all(e['probabilities']==[.5,.5] for e in es)
    assert es[1]['event_results']==[False,True]
    # No squared future LR or fictitious independence of integrated cells.
    assert es[1]['branching_value']==pytest.approx(.85)


def test_new_design_has_fresh_initial_actions_and_disjoint_tree_seed_blocks():
    strata=[dict(stratum_id=0,draws=3,cells=[0,34],measurement_probabilities=[0.,1.])]
    old=dict(states=[dict(strata=strata)],target_masses=[1.],independent_roots=6,
             schedules=[dict(batch=b,state_id=0,rows=[dict(action_indices=[0,0])]*3) for b in (0,1)])
    cfg=dict(seed_starts=[8000000,9000000],schedule_seed=800017,roots_per_state=3)
    d=m.design_for(old,cfg); assert d==m.design_for(old,cfg)
    seeds=[]
    for s in d['schedules']:
        for r in s['rows']:
            assert r['action_indices']==[1,1]  # Re-draw, don't reuse old outcome schedule.
            assert len(r['control_uniforms'])==21
            seeds.extend([r['initial_seed']]+r['step_seeds']+r['terminal_seeds'])
    assert len(seeds)==37*6 and len(seeds)==len(set(seeds))


def test_whole_decision_tree_replay_and_stopped_prefix(monkeypatch):
    p=np.zeros((33,33));p[1,1]=.95;p[2,2]=.05
    h=np.zeros_like(p);h[1,1]=h[2,2]=.5
    units=[dict(ids=[1,2],p=p)]
    class Road:
        def __init__(self,seed):
            self.time=0.;self.crashes=[];self.model=object();self.config={};self.mark=False
            self.cav=SimpleNamespace(id=0,x=0.);self.vehicles=[self.cav,SimpleNamespace(id=1,x=1.),SimpleNamespace(id=2,x=2.)]
            self.natural=SimpleNamespace(histories={},previous=set());self.rng=np.random.default_rng(seed)
            self.minimum_gap=self.minimum_ttc=1.;self.last_acceleration={}
        def action_pmfs(self):return {1:p.sum(1),2:p.sum(0)}
        def prepare(self,_):return units,{}
        def step_policy(self,*args):
            actions=self.sample(self.action_pmfs(),*args);self.time+=1
            if self.time==2:self.mark=actions[1]==2
    monkeypatch.setattr(m,'restore_prepared_state',lambda *args:Road(args[-1]))
    *_,q1,q2=m.parent.proposal_factors(p,h,[.1,.1])
    monkeypatch.setattr(m.parent.parent.parent,'pair_choice',lambda *args:([1,2],p,h,q1,q2))
    def terminal(road,codes):
        y=(codes[:,0]==2)&road.mark
        return dict(event=y,competing=np.zeros(len(codes),bool),elapsed_s=np.ones(len(codes)),collision_pairs=[[[0,1]] if a else [] for a in y])
    monkeypatch.setattr(m,'terminal_decisions',terminal)
    state=dict(time_s=0.,pmfs={'1':p.sum(1).tolist(),'2':p.sum(0).tolist()},units=units)
    s=dict(initial_seed=8000000,action_indices=[1,1],step_seeds=list(range(8000001,8000021)),
           terminal_seeds=list(range(8000021,8000037)),control_uniforms=[.3]*21)
    cfg=dict(continuation=dict(epsilon=[.1,.1]))
    a=m.root_continuation(None,None,{},state,[1,2],s,cfg)
    assert a==m.root_continuation(None,None,{},state,[1,2],s,cfg)
    assert m.tree_counts(a['tree'])==(20,32)
    assert 0<=a['branching_value']<=2
    assert len(a['tree']['children'])==4
    # An already stopped initial prefix is retained and never branched.
    original_step=Road.step_policy
    def early(self,*args):
        original_step(self,*args);self.crashes=[[1,2]]
    monkeypatch.setattr(Road,'step_policy',early)
    stopped=m.root_continuation(None,None,{},state,[1,2],s,cfg)
    assert stopped['tree']['status']=='stopped' and stopped['branching_value']==0.
    assert m.tree_counts(stopped['tree'])==(0,0)


def test_new_run_saves_choice_before_second_batch_and_refuses_overwrite(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction.highd_v46_assets import read,save
    cfg=dict(confidence=.95,allocation_loss_cap=1.1,workers=1)
    hs=[dict(stratum_id=0,coefficient_mass=1.,draws=2)]
    d=dict(states=[dict(strata=hs,plan=dict(source_seed=1))],target_masses=[1.],independent_roots=4,
           schedules=[dict(batch=b,state_id=0,rows=[dict(stratum_id=0,within_stratum_draw=i) for i in (0,1)]) for b in (0,1)])
    lock=tmp_path/'lock';save(lock,{'fixture':1});out=tmp_path/'run'
    monkeypatch.setattr(m,'inputs',lambda _:(cfg,{},tmp_path,[dict(actor_ids=[1,2])],[dict(state={})],d,{'fixture':1}))
    monkeypatch.setattr(m,'ProcessPoolExecutor',lambda **_:ThreadPoolExecutor(max_workers=1))
    def collect(task):
        if task[0]['batch']==1:assert read(out/'batch0_allocation_before_batch1.json')['recommended_q']==[1.]
        assert read(out/'measurement_plan.json')==d
        return dict(task[0],rows=[dict(r,single_value=0.,branching_value=0.,removed_conditional_variance_unbiased=0.,
                   tree=dict(status='stopped',children=[])) for r in task[0]['rows']])
    monkeypatch.setattr(m,'_collect',collect)
    m.run('fixture',lock,out)
    s=read(out/'summary.json');assert s['complete'] and not s['ppo_ready'] and s['independent_roots']==4
    assert s['terminal_integrated_decisions']==0
    with pytest.raises(FileExistsError):m.run('fixture',lock,out)
