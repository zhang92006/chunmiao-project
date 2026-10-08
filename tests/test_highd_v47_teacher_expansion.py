import copy
from pathlib import Path

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_teacher_expansion as expansion
from scenario_reconstruction import highd_v47_teacher_restore as restore
from scenario_reconstruction.highd_v46_assets import read, save
from scenario_reconstruction.highd_v46_d2rl import identity


def row(seed=1, state=0, strict=False, mass=0., risk=.8, actors=(1,2), mechanism='following'):
    return dict(seed=seed, state_index=state, time_s=state+1, actor_ids=list(actors),
                views=dict(production_strict=strict), metrics=dict(positive_interaction_mass=mass),
                ranker_features=[1,0,risk,0], mechanism=mechanism)


def test_recover_cross_unit_and_global_excess_rejected_candidate():
    rows = [row(strict=True,mass=.2,risk=.6),
            row(state=1,strict=True,mass=.1,risk=.7,mechanism='lateral'),
            row(state=2,strict=False,risk=1,actors=(5,7))]
    chosen,funnel = expansion.selected_rows(rows,[1,2])
    assert chosen == [0,1,2]
    assert funnel[1]['selected_rows'] == []
    assert funnel[0]['maximum_risk'] == 1


def test_selection_ignores_endpoint_labels_and_keeps_negative_cases():
    rows = [row(strict=True,mass=.2),row(state=1,strict=True,mass=.1),row(state=2,risk=1)]
    expected = expansion.selected_rows(rows,[1])
    for r in rows: r['event_result'] = False
    assert expansion.selected_rows(rows,[1]) == expected
    for r in rows: r['event_result'] = True
    assert expansion.selected_rows(rows,[1]) == expected


def test_deduplicate_and_secondary_prefers_other_mechanism():
    rows = [row(strict=True,mass=.9,risk=1),row(state=1,strict=True,mass=.8,risk=.4),
            row(state=2,strict=True,mass=.1,risk=.4,mechanism='lateral')]
    assert expansion.selected_rows(rows,[1])[0] == [0,2]


def test_below_threshold_abstains_and_undeclared_source_rejected():
    assert expansion.selected_rows([row(risk=.499)],[1])[0] == []
    assert expansion.selected_rows([row(risk=.5)],[1])[0] == [0]
    with pytest.raises(ValueError): expansion.selected_rows([row(seed=2)],[1])


def small_plan():
    cfg = read(expansion.DEFAULT)
    cfg['source'] = dict(seed_start=1,episodes=3)
    chunk = dict(funnel=[dict(source_seed=s) for s in (1,2,3)], designs=[
        dict(state=dict(seed=seed,time_s=t,last_acceleration={12:.5,2:0.}),row=dict(actor_ids=[1,2]))
        for seed,t in [(1,1),(1,2),(2,1)]])
    chunk = expansion.json_value(chunk)
    return cfg,chunk,expansion.build_plan([('chunk_000.json',chunk)],cfg)


def test_uniform_source_then_case_not_uniform_cases():
    cfg,chunk,plan = small_plan()
    assert [c['alpha'] for c in plan['cases']] == [.25,.25,.5]
    assert [c['stratum_replay_correction'] for c in plan['cases']] == [.75,.75,1.5]
    assert all(c['state_likelihood_ratio']==1 for c in plan['cases'])
    assert plan['source_episodes']==3 and plan['selected_sources']==2
    assert plan['continuation_draws']==192 and len(plan['sources'])==3
    assert not plan['selection_uses_native_outcomes']
    assert len({c['continuation_seed_start']+j for c in plan['cases'] for j in range(64)})==192


def test_serialized_state_identity_stable(tmp_path):
    cfg,chunk,plan = small_plan()
    save(tmp_path/'chunk.json',chunk)
    assert expansion.build_plan([('chunk_000.json',read(tmp_path/'chunk.json'))],cfg)==plan
    assert identity(read(tmp_path/'chunk.json')['designs'][0]['state']) == plan['cases'][0]['state_sha256']


def test_missing_sources_duplicate_cases_or_excess_budget_rejected():
    cfg,chunk,_ = small_plan()
    bad=copy.deepcopy(chunk);bad['funnel'].pop()
    with pytest.raises(ValueError): expansion.build_plan([('x',bad)],cfg)
    bad=copy.deepcopy(chunk);bad['designs'].append(bad['designs'][0])
    with pytest.raises(ValueError): expansion.build_plan([('x',bad)],cfg)
    cfg['maximum_cases_per_source']=1
    with pytest.raises(ValueError): expansion.build_plan([('x',chunk)],cfg)


def test_case_statistics_and_stratified_target():
    cfg,chunk,plan=small_plan()
    records=[]
    for i,c in enumerate(plan['cases']):
        samples=[dict(event_result=bool(i==2 and j==0),generation_log_weight=np.log(.2),
                      bv_only_collision=False) for j in range(64)]
        records.append(expansion.statistics(c,samples))
    report=expansion.aggregate(records,cfg)
    assert report['events']==1 and report['collision_sources']==1
    assert report['conditional_IS_mean']==pytest.approx(.5*.2/64)
    assert report['M2']==pytest.approx(.5*.04/64)
    assert report['stratified_estimator_variance']==pytest.approx(.5**2*records[2]['within_variance']/64)
    assert report['maximum_source_M2_share']==1
    assert not report['collision_count_gate_passed'] and not report['ppo_ready']


def test_many_events_not_many_independent_sources_and_count_gate_not_ppo():
    cfg=read(expansion.DEFAULT)
    def record(i,source,events):
        return dict(case_id=i,source_seed=source,alpha=.01,draws=64,events=events,
            IS_mean=.01,M2=.01,within_variance=.1,largest_event_M2_contribution=.00001)
    same=[record(i,1,64) for i in range(100)]
    assert not expansion.aggregate(same,cfg)['collision_count_gate_passed']
    many=[record(i,i,10) for i in range(50)]
    result=expansion.aggregate(many,cfg)
    assert result['collision_count_gate_passed'] and not result['ppo_ready']
    assert not result['desired_100_collision_sources_reached']


def test_zero_events_undefined_concentration_not_safe():
    cfg=read(expansion.DEFAULT)
    r=expansion.aggregate([],cfg)
    assert r['maximum_source_M2_share'] is None and r['effective_M2_sources'] is None
    assert not r['collision_count_gate_passed'] and not r['ppo_ready']


def test_reproduction_excludes_only_execution_timing():
    x=dict(event_result=True,cpu_s=1,wall_s=2,initial_likelihood_audit=dict(p_factors=[.1,.2]),collision_pairs=[(1,2)])
    y=copy.deepcopy(x);y['cpu_s']=5;y['wall_s']=6;y['collision_pairs']=[[1,2]]
    assert restore.physical_record(x)==restore.physical_record(y)
    y['event_result']=False
    assert restore.physical_record(x)!=restore.physical_record(y)


def test_reproduction_likelihood_and_seed_changes_not_ignored():
    x=dict(event_result=True,seed=4600000,generation_log_weight=.1)
    for key,value in [('seed',4600001),('generation_log_weight',.2)]:
        y=dict(x);y[key]=value
        assert restore.physical_record(x)!=restore.physical_record(y)


def test_unverified_reproduction_does_not_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(expansion,'protocol',lambda _: (None,{}))
    save(tmp_path/'audit.json',dict(verified=True,full_reproduction_verified=False))
    with pytest.raises(ValueError,match='First reproduce'):
        expansion.freeze('unused',tmp_path/'audit.json',tmp_path/'lock.json')
    assert not (tmp_path/'lock.json').exists()


def test_forbidden_budget_changes_rejected_before_assets(tmp_path,monkeypatch):
    cfg=read(expansion.DEFAULT);cfg['source']['seed_start']=240000
    save(tmp_path/'bad.json',cfg)
    monkeypatch.setattr(expansion,'environment',lambda:cfg['environment'])
    with pytest.raises(ValueError,match='Version changed'):
        expansion.protocol(tmp_path/'bad.json')


def test_exclusive_output_does_not_overwrite(tmp_path):
    save(tmp_path/'old.json',dict(events=67))
    with pytest.raises(FileExistsError): save(tmp_path/'old.json',dict(events=0))
    assert read(tmp_path/'old.json')['events']==67


def test_synthetic_complete_pipeline_and_tamper_rejection(tmp_path,monkeypatch):
    """All stage wiring, no natural source or native simulator is opened."""
    from concurrent.futures import ThreadPoolExecutor
    from scenario_reconstruction.highd_v46_cross_unit import draw_intervention
    cfg=read(expansion.DEFAULT)
    cfg['source']=dict(seed_start=1,episodes=3,split='train');cfg['source_chunk_size']=1
    base={'proxy':{'clearance_scale_m':10}};lock={'synthetic':True}
    args=(cfg,{},base,{},tmp_path,lock)
    monkeypatch.setattr(expansion,'checked',lambda *_:args)
    monkeypatch.setattr(expansion,'pool',lambda _:ThreadPoolExecutor(1))
    p=np.array([[.3,.2],[.1,.4]]);risk=np.array([[.1,.9],[.6,.2]])
    h,_=expansion.teacher.exploration_h(p,risk,cfg['risk_edges'])
    monkeypatch.setattr(expansion,'_RUNTIME',(None,tmp_path,{},base,{},cfg))
    def fake_discover(cohort,*_):
        seed=cohort['source']['seed_start']
        state=dict(seed=seed,time_s=1,cav_id=99,units=[dict(ids=[1,2],p=p.tolist())])
        rows=[row(seed=seed,strict=True)] if seed!=3 else []
        return [state],rows,dict(episodes=[dict(seed=seed)])
    monkeypatch.setattr(expansion,'natural_discover',fake_discover)
    monkeypatch.setattr(expansion.teacher,'prepare',lambda *_:(p,risk,{}))
    def fake_sample(model,assets,physical,state,chosen,eps,seed,cfg,scale):
        _,log=draw_intervention(state['units'],chosen['ids'],chosen['h'],eps,np.random.default_rng(seed))
        event=seed%10==0
        return dict(seed=seed,event_result=event,bv_only_collision=False,generation_log_weight=log['log_weight'],
                    initial_likelihood_audit=log,collision_pairs=[[99,1]] if event else [],cpu_s=0.,wall_s=0.)
    monkeypatch.setattr(expansion,'sample_continuation',fake_sample)
    bank=tmp_path/'bank';data=tmp_path/'data'
    assert expansion.discover('unused','unused',bank)['selected_sources']==2
    assert expansion.collect('unused','unused',bank,data)['draws']==128
    result=expansion.audit('unused','unused',bank,data,tmp_path/'audit.json')
    assert result['verified'] and result['collision_sources']==2 and not result['ppo_ready']
    with pytest.raises(FileExistsError): expansion.collect('unused','unused',bank,data)
    (data/'case_00000.json').write_text('{}')
    with pytest.raises(ValueError,match='Artifact changed'):
        expansion.audit('unused','unused',bank,data,tmp_path/'bad_audit.json')
