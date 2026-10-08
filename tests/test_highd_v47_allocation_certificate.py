from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_allocation_certificate as m


def fixture():
    hs=[dict(stratum_id=0,coefficient_mass=1.,draws=4,cells=[34],measurement_probabilities=[1.])]
    old=dict(target_masses=[.5,.5],states=[dict(plan=dict(source_seed=i),strata=hs) for i in (0,1)])
    pilot={(b,i):dict(rows=[dict(stratum_id=0,branching_value=.2 if i==0 else .001)]*4)
           for b in (0,1) for i in (0,1)}
    cfg=dict(batches=2,allocation_loss_cap=1.1,minimum_roots_per_stratum=2,
        maximum_loss_root_bound=.001,maximum_natural_roots_per_batch=60000,
        confidence=.95,loss_bet_support_fraction=.8,benefit_bet_grid=[1,9,81],
        seed_starts=[11000000,13000000],natural_seed_offset=1000000,
        schedule_seed=110017,iid_draws_per_arm=4,iid_state_rng_seed=150017)
    return old,pilot,cfg


def test_candidate_counts_bets_and_all_schedules_are_outcome_blind():
    old,pilot,cfg=fixture();d=m.fit_and_design(old,pilot,cfg)
    assert d==m.fit_and_design(old,pilot,cfg)
    q=np.array(d['candidate_q']);assert np.max(np.array(old['target_masses'])/q)<=1.1+1e-14
    assert [s['side'] for s in d['states']]==['benefit','loss']
    assert d['maximum_loss_root_bound']<=cfg['maximum_loss_root_bound']
    assert d['branching_roots_per_batch']==4
    # Loss counts are set by uncertainty contribution, not observed negatives.
    assert d['natural_roots_per_batch']>=50
    seeds=[]
    for s in d['schedules']:
        for r in s['rows']:
            assert r['action_indices']==[1,1]
            seeds.extend([r['initial_seed']]+r['step_seeds']+r['terminal_seeds']
                         if s['side']=='benefit' else [r['native_seed']])
    assert len(seeds)==len(set(seeds))
    assert d['error_per_side_per_batch']==pytest.approx(.0125)
    assert all(len(s)==4 for s in d['iid_schedules'].values())


def test_natural_uncertainty_budget_cannot_be_silently_extended():
    old,pilot,cfg=fixture();cfg['maximum_natural_roots_per_batch']=2
    with pytest.raises(ValueError,match='budget exceeds'):m.fit_and_design(old,pilot,cfg)


def test_zero_pilot_keeps_entire_bank_and_target():
    old,pilot,cfg=fixture()
    for r in pilot.values():
        for row in r['rows']:row['branching_value']=0
    d=m.fit_and_design(old,pilot,cfg)
    assert d['candidate_q']==old['target_masses'] and len(d['states'])==2
    assert all(s['side']=='neutral' for s in d['states'])
    assert d['natural_roots_per_batch']==4


def test_negative_or_incomplete_pilot_is_rejected():
    old,pilot,cfg=fixture();pilot[(0,0)]['rows']=[]
    with pytest.raises(ValueError):m.fit_and_design(old,pilot,cfg)
    old,pilot,cfg=fixture();pilot[(0,0)]['rows'][0]['branching_value']=-1
    with pytest.raises(ValueError):m.fit_and_design(old,pilot,cfg)


def test_objective_certificate_does_not_require_per_state_relative_precision():
    old,pilot,cfg=fixture();d=m.fit_and_design(old,pilot,cfg);records={}
    # Increase independent benefit roots in this synthetic law for power.
    d['states'][0]['strata'][0]['draws']=1000;d['benefit_bet']['rate']=1000
    for b in (0,1):
        for s in d['states']:
            h=s['strata'][0]
            rows=[dict(stratum_id=0,measurement_value=.4 if s['side']=='benefit' else 0.) for _ in range(h['draws'])]
            records[(b,s['state_id'])]=dict(rows=rows)
    r=m.summarize(d,records)
    assert r['allocation_improvement_certified'] and r['family_confidence']==pytest.approx(.95)
    assert r['recommended_q']==d['candidate_q']
    assert all(b['loss']['upper']>0 for b in r['batches'])
    assert all(b['M2_difference_upper']<0 for b in r['batches'])
    # A whole independent batch without benefit must keep alpha.
    for row in records[(1,0)]['rows']:row['measurement_value']=0
    r=m.summarize(d,records)
    assert not r['allocation_improvement_certified'] and r['recommended_q']==d['target_masses']


def test_natural_root_matches_supported_partners_and_prepared_initial_step(monkeypatch):
    p=np.zeros((33,33));p[1,1]=.9;p[2,2]=.1
    units=[dict(ids=[1,2],p=p)];calls=[]
    class Road:
        def __init__(self,seed):
            self.time=0.;self.rng=np.random.default_rng(seed);self.crashes=[];self.cav=SimpleNamespace(id=0)
        def action_pmfs(self):return {}
        def prepare(self,_):calls.append(self.time);return units,{}
        def step_policy(self,*args):
            a=self.sample(self.action_pmfs(),*args);self.time+=1
            assert a[1]==a[2]
            if self.time==2:self.crashes=[[1,2]]  # competing endpoint, retained
    monkeypatch.setattr(m,'restore_prepared_state',lambda *args:Road(args[-1]))
    s=dict(time_s=0.,pmfs={},units=units)
    r=m.natural_root(None,None,{},s,[1,2],dict(native_seed=12,action_indices=[2,2]))
    assert calls==[1.] and r['elapsed_s']==2.
    assert not r['event_result'] and r['competing_result'] and r['measurement_value']==0.
    assert r['future_weight']==1. and len(r['steps'])==2
    assert r['steps'][0]['units'][0]['probability']==.1
    assert r['initial_cancelled_log_conditional']==0


def test_iid_smoke_uses_full_state_action_weight_and_shared_paths(monkeypatch,tmp_path):
    calls=[]
    def draw(*args):
        calls.append(args[6])
        return dict(seed=args[6],event_result=True,generation_log_weight=0.,initial_likelihood_audit=dict(
            full_log_p=0.,full_log_q=0.,log_weight=0.,abstained=True))
    monkeypatch.setattr(m.state_sampler,'sample_continuation',draw)
    monkeypatch.setattr(m.parent.parent.parent.parent,'_MODEL',None,raising=False)
    plans=[dict(source_seed=i,target_mass=.5,actor_ids=[1,2]) for i in (0,1)]
    ads=[dict(state={},H=[]) for _ in plans]
    d=dict(target_masses=[.5,.5],candidate_q=[.6,.4],
           iid_schedules=dict(target_iid=[0,1,0,1],allocated_iid=[0,0,0,1]))
    rows,r=m.iid_comparison(dict(iid_continuation_seed_start=15000000),tmp_path,{},plans,ads,d)
    assert len(calls)==r['physical_paths']==5  # CRN identical physical paths reused.
    assert r['arms']['allocated_iid']['mean']==pytest.approx((3*.5/.6+.5/.4)/4)
    assert rows['allocated_iid']['sequences'][0]['state_log_weight']==pytest.approx(np.log(.5/.6))
    assert rows['target_iid']['sequences'][0]['state_log_weight']==0.


def test_new_run_records_full_plan_before_any_native_outcome_and_refuses_overwrite(monkeypatch,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction.highd_v46_assets import read,save
    old,pilot,cfg=fixture();cfg['workers']=1
    design=m.fit_and_design(old,pilot,cfg);lock=tmp_path/'lock';out=tmp_path/'run';save(lock,{'fixture':1})
    monkeypatch.setattr(m,'inputs',lambda _:(cfg,{},tmp_path,[dict(actor_ids=[1,2])]*2,
        [dict(state={})]*2,design,{'fixture':1}))
    monkeypatch.setattr(m,'ProcessPoolExecutor',lambda **_:ThreadPoolExecutor(max_workers=1))
    def collect(task):
        assert read(out/'measurement_plan.json')==design
        return dict(task[0],rows=[dict(r,measurement_value=0.) for r in task[0]['rows']])
    monkeypatch.setattr(m,'_collect',collect)
    monkeypatch.setattr(m.parent.parent.parent.parent,'_init_worker',lambda _:None)
    monkeypatch.setattr(m,'iid_comparison',lambda *args:({'fixture':{}},{'still_ran_after_failure':True}))
    m.run('fixture',lock,out);r=read(out/'summary.json')
    assert r['complete'] and not r['allocation_improvement_certified'] and not r['ppo_ready']
    assert r['iid_smoke']['still_ran_after_failure'] and len(r['batches'])==2
    with pytest.raises(FileExistsError):m.run('fixture',lock,out)
