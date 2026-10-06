import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_source_transfer as m
from scenario_reconstruction import highd_v47_state_features as features
from scenario_reconstruction.highd_v46_assets import read,save


def config():
    return read(m.DEFAULT)


def cases():
    return [dict(plan=dict(case_id=i,source_seed=260000+i,actor_ids=[1,2],target_mass=.5),
        features=dict(names=list(features.FEATURE_NAMES),values=[0.]*len(features.FEATURE_NAMES),coefficient_mass=1.),
        strata=[dict(stratum_id=0,cells=[0,1],measurement_probabilities=[.3,.7],coefficient_mass=1.,draws=4)],
        state={},P=[],H=[]) for i in range(2)]


def measured(design,value=.1):
    return [dict(s,rows=[dict(r,measurement_value=value) for r in s['rows']]) for s in design['schedules']]


def test_protocol_checks_never_open_source_and_lock_excludes_consumed(monkeypatch):
    monkeypatch.setattr(m,'ensure_assets',lambda: (Path('unused'),dict(natural_target_sha256='fixture')))
    monkeypatch.setattr(m,'experiment_target',lambda *args: {})
    cfg,*_,lock=m.protocol(m.DEFAULT)
    assert set(range(260000,260256)).isdisjoint(range(270000,270256))
    assert len(lock['forbidden_source_ranges'])>=14 and cfg['roots_per_state']==128
    assert 'highd_v47_state_features.py' in lock['dependencies']


@pytest.mark.parametrize('change',[
    {'train':{'seed_start':200000,'episodes':256}}, {'validation':{'seed_start':260000,'episodes':256}},
    {'epsilon':[.2,.1]}, {'roots_per_state':64}, {'maximum_natural_roots_per_batch':200000},
    {'measurement_seed_starts':{'train':[50000000,60000000],'validation':[50000000,60000000]}},
    {'selector':'outcome_top_k'}, {'coefficient_singletons':13},
])
def test_version_and_source_mutations_fail_before_assets(change,monkeypatch,tmp_path):
    cfg=config();cfg.update(change);path=tmp_path/'cfg.json';save(path,cfg)
    monkeypatch.setattr(m,'ensure_assets',lambda: pytest.fail('should reject before asset/source opening'))
    with pytest.raises(ValueError): m.protocol(path)


def test_training_design_retains_all_strata_and_independent_seed_blocks():
    cs=cases();cfg=config();d=m.measurement_design(cs,cfg,'train')
    assert d['roots']==16 and d['target_masses']==[.5,.5] and d['candidate_q']==[.5,.5]
    seeds=set()
    for s in d['schedules']:
        assert s['side']=='training'
        for r in s['rows']:
            ss=[r['initial_seed']]+r['step_seeds']+r['terminal_seeds']
            assert len(ss)==37 and not seeds.intersection(ss);seeds.update(ss)
            assert r['action_indices'] in ([0,0],[0,1])
    assert d==m.measurement_design(cs,cfg,'train')


def test_validation_prediction_and_budget_fixed_before_labels(monkeypatch):
    cs=cases();cfg=config();monkeypatch.setattr(features,'predict',lambda *args:np.array([100.,1.]))
    fitted=dict(model={},benefit_bet={'rate':1000.})
    d=m.measurement_design(cs,cfg,'validation',fitted)
    assert d['states'][0]['side']=='benefit' and d['states'][1]['side']=='loss'
    assert np.max(np.array(d['target_masses'])/d['candidate_q'])<=1.1+1e-14
    assert d['natural_roots_per_batch']>0
    donor=d['states'][1];h=donor['strata'][0]
    assert donor['signed_objective_coefficient']*h['coefficient_mass']/h['draws']<=cfg['maximum_loss_root_bound']
    assert len(d['iid_schedules']['target_iid'])==2112
    c2=dict(cfg,maximum_natural_roots_per_batch=2)
    with pytest.raises(ValueError,match='cap'):m.measurement_design(cs,c2,'validation',fitted)


def test_estimates_use_independent_roots_not_branch_leaves_and_retain_negatives():
    d=m.measurement_design(cases(),config(),'train');rs=measured(d,.2)
    est=m.estimate(d,rs)
    assert est[0]['J']==pytest.approx(.2) and est[0]['variance']==0
    assert est[0]['effective_measurement_roots']==8
    for r in rs:
        if r['state_id']==1:
            for z in r['rows']:z['measurement_value']=0
    assert len(m.estimate(d,rs))==2 and m.estimate(d,rs)[1]['J']==0
    rs[0]['rows'].append(rs[0]['rows'][0])
    with pytest.raises(ValueError,match='duplicate'):m.estimate(d,rs)


def test_objective_bounds_are_on_same_target_and_zero_loss_is_not_zero_uncertainty(monkeypatch):
    monkeypatch.setattr(features,'predict',lambda *args:np.array([100.,1.]))
    d=m.measurement_design(cases(),config(),'validation',dict(model={},benefit_bet={'rate':1000.}))
    rs=measured(d,0.)
    for r in rs:
        if r['side']=='benefit':
            for z in r['rows']:z['measurement_value']=.5
    reports=m.objective_reports(d,rs)
    for r in reports:
        assert r['loss']['upper']>0 and r['benefit']['lower']>0
        assert r['M2_difference_point']<0 and r['improvement_certified']
    for r in rs:
        for z in r['rows']:z['measurement_value']=0
    assert not any(r['improvement_certified'] for r in m.objective_reports(d,rs))


def test_source_bootstrap_preserves_abstentions_recomputes_q_and_censors_sparse_sources():
    cfg=config()['statistics']; cfg=dict(cfg,bootstrap_repeats=100)
    d=dict(states=[dict(plan=dict(source_seed=1)),dict(plan=dict(source_seed=2))],
        target_masses=[.5,.5],candidate_q=[.55,.45],predicted_J=[100.,1.])
    rows=[dict(source_seed=i,J=j,variance=0.,positive_roots=1) for i,j in [(1,1.),(2,.01)]]
    r=m.source_comparison(d,rows,[1,2,3,4],cfg)
    assert r['all_source_episodes']==4 and r['selected_sources']==2
    assert r['M2_ratio_point']<1 and r['censored_replicates']==100
    assert r['paired_source_bootstrap_ratio_interval'] is None and not r['source_generalization_supported']


def test_validation_freeze_requires_complete_new_training_and_review(monkeypatch,tmp_path):
    cfg=config();lock={'fixture':1}
    monkeypatch.setattr(m,'protocol',lambda _:(cfg,None,None,None,None,lock.copy()))
    with pytest.raises(ValueError,match='complete training'):m.freeze('fixture','validation',tmp_path/'lock')
    monkeypatch.setattr(m,'completed_training',lambda *args:(tmp_path,{},dict(measured_positive_sources=2,selected_OOF_ratio_point=.9)))
    with pytest.raises(ValueError,match='review failed'):m.freeze('fixture','validation',tmp_path/'lock',tmp_path)
    assert not (tmp_path/'lock').exists()


def test_audit_requirement_rejects_stale_unreplayed_or_wrong_split(tmp_path):
    from scenario_reconstruction.highd_v46_assets import digest
    root=tmp_path/'train';root.mkdir();save(root/'summary.json',dict(selected_sources=12))
    valid=dict(verified=True,split='train',summary_sha256=digest(root/'summary.json'),
        audit_implementation_sha256=digest(Path(m.__file__).with_name('highd_v47_source_transfer_audit.py')),
        predetermined_native_root_replays=24)
    for i,change in enumerate([{'verified':False},{'split':'validation'},{'summary_sha256':'stale'},
            {'predetermined_native_root_replays':0},{'audit_implementation_sha256':'old'}]):
        path=tmp_path/f'audit_{i}.json';save(path,dict(valid,**change))
        with pytest.raises(ValueError,match='audit'):m.check_training_audit(path,root)
    path=tmp_path/'good.json';save(path,valid);m.check_training_audit(path,root)


def test_model_freeze_cannot_open_validation_without_independent_audit(monkeypatch,tmp_path):
    cfg=config();lock={'fixture':1}
    monkeypatch.setattr(m,'protocol',lambda _:(cfg,None,None,None,None,lock.copy()))
    monkeypatch.setattr(m,'completed_training',lambda *args:(tmp_path,{},dict(measured_positive_sources=10,selected_OOF_ratio_point=.9)))
    with pytest.raises(ValueError,match='audit required'):
        m.freeze('fixture','validation',tmp_path/'lock',tmp_path)


def test_rule_lock_failure_occurs_before_opening_any_source(monkeypatch,tmp_path):
    cfg=config();save(tmp_path/'lock',dict(split='train',sources_opened=False,fixture=1))
    monkeypatch.setattr(m,'protocol',lambda _:(cfg,None,None,None,None,{'fixture':2}))
    monkeypatch.setattr(m,'discover',lambda *args:pytest.fail('must not open source'))
    with pytest.raises(ValueError,match='after freeze'):m.run('fixture','train',tmp_path/'lock',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_run_saves_outcome_blind_plan_before_native_and_refuses_overwrite(monkeypatch,tmp_path):
    cfg=config(); cfg['workers']=1; cs=cases()
    for c in cs: c['plan'].update(state_sha256='fixture',state_index=0)
    lock={'fixture':1,'split':'train','sources_opened':False};save(tmp_path/'lock',lock)
    monkeypatch.setattr(m,'protocol',lambda _:(cfg,{},dict(proxy={}),{},tmp_path,{'fixture':1}))
    monkeypatch.setattr(m,'PairModelQueries',lambda *args,**kwargs:None)
    monkeypatch.setattr(m,'discover',lambda *args:([],[],dict(episodes=[dict(seed=i) for i in range(260000,260256)])))
    monkeypatch.setattr(m,'prepare_cases',lambda *args:(cs,{260000:0,260001:1}))
    monkeypatch.setattr(m,'ProcessPoolExecutor',lambda **kwargs:ThreadPoolExecutor(max_workers=1))
    out=tmp_path/'run'
    def collect(task):
        plan=read(out/'measurement_plan.json');assert plan['target_masses']==[.5,.5]
        assert len(plan['schedules'])==4 and (out/'case_plan.json').exists()
        return dict(task[0],rows=[dict(r,measurement_value=0.) for r in task[0]['rows']])
    monkeypatch.setattr(m,'_collect',collect)
    monkeypatch.setattr(m,'fit_model',lambda *args:{'synthetic':True})
    r=m.run('fixture','train',tmp_path/'lock',out)
    assert r['selected_sources']==2 and not r['ppo_ready']
    assert read(out/'summary.json')['complete']
    with pytest.raises(FileExistsError):m.run('fixture','train',tmp_path/'lock',out)


def test_validation_never_fits_and_iid_runs_even_after_failed_objective(monkeypatch,tmp_path):
    from scenario_reconstruction.highd_v46_assets import digest
    cfg=config();cfg['workers']=1;cs=cases()
    for i,c in enumerate(cs):
        c['plan'].update(source_seed=270000+i,state_sha256='fixture',state_index=0)
    training=tmp_path/'training';training.mkdir();save(training/'summary.json',{});save(training/'model.json',{})
    audit=tmp_path/'audit.json';save(audit,{})
    fitted=dict(model={},benefit_bet={'rate':1000.},measured_positive_sources=10,selected_OOF_ratio_point=.9)
    lock=dict(fixture=1,split='validation',sources_opened=False,training_path=str(training),
        training_summary_sha256=digest(training/'summary.json'),model_sha256=digest(training/'model.json'),
        benefit_bet_rate=1000.,training_audit_path=str(audit),training_audit_sha256=digest(audit))
    save(tmp_path/'lock',lock);monkeypatch.setattr(m,'protocol',lambda _:(cfg,{},dict(proxy={}),{},tmp_path,{'fixture':1}))
    monkeypatch.setattr(m,'completed_training',lambda *args:(training,{},fitted))
    monkeypatch.setattr(m,'check_training_audit',lambda *args:None)
    monkeypatch.setattr(features,'predict',lambda *args:np.array([100.,1.]))
    monkeypatch.setattr(m,'PairModelQueries',lambda *args,**kwargs:None)
    monkeypatch.setattr(m,'discover',lambda *args:([],[],dict(episodes=[dict(seed=i) for i in range(270000,270256)])))
    monkeypatch.setattr(m,'prepare_cases',lambda *args:(cs,{270000:0,270001:1}))
    monkeypatch.setattr(m,'ProcessPoolExecutor',lambda **kwargs:ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(m,'fit_model',lambda *args:pytest.fail('validation must never fit'))
    out=tmp_path/'validation'
    def collect(task):
        plan=read(out/'measurement_plan.json')
        assert plan['candidate_q']!=plan['target_masses'] and len(plan['iid_schedules'])==2
        return dict(task[0],rows=[dict(r,measurement_value=0.) for r in task[0]['rows']])
    monkeypatch.setattr(m,'_collect',collect)
    monkeypatch.setattr(m,'iid_comparison',lambda *args:({'fixture':{}},{'ran_even_when_certificate_failed':True}))
    m.run('fixture','validation',tmp_path/'lock',out);s=read(out/'summary.json')
    assert not s['transfer_review_passed'] and not s['fixed_new_bank_improvement_certified']
    assert s['iid_smoke']['ran_even_when_certificate_failed'] and not (out/'model.json').exists()


def test_default_protocol_json_has_no_duplicate_keys():
    import json
    def pairs(items):
        assert len({k for k,v in items})==len(items)
        return dict(items)
    json.loads(Path(m.DEFAULT).read_text(),object_pairs_hook=pairs)


def test_complete_training_model_round_trip_and_loader_refuse_label_tamper(monkeypatch,tmp_path):
    cfg=config();cfg['workers']=1;cs=[]
    for i in range(12):
        c=copy.deepcopy(cases()[0]);c['plan'].update(case_id=i,source_seed=260000+i,state_sha256='fixture',state_index=0,target_mass=1/12)
        c['features']['values'][1]=i/12;cs.append(c)
    lock=dict(fixture=1,split='train',sources_opened=False);save(tmp_path/'lock',lock)
    monkeypatch.setattr(m,'protocol',lambda _:(cfg,{},dict(proxy={}),{},tmp_path,{'fixture':1}))
    monkeypatch.setattr(m,'PairModelQueries',lambda *args,**kwargs:None)
    monkeypatch.setattr(m,'discover',lambda *args:([],[],dict(episodes=[dict(seed=i) for i in range(260000,260256)])))
    monkeypatch.setattr(m,'prepare_cases',lambda *args:(cs,{260000+i:i for i in range(12)}))
    monkeypatch.setattr(m,'ProcessPoolExecutor',lambda **kwargs:ThreadPoolExecutor(max_workers=1))
    def collect(task):
        s=task[0];return dict(s,rows=[dict(r,measurement_value=.01*(s['state_id']+1)) for r in s['rows']])
    monkeypatch.setattr(m,'_collect',collect);out=tmp_path/'train';m.run('fixture','train',tmp_path/'lock',out)
    root,summary,fitted=m.completed_training(out,cfg,{'fixture':1})
    assert fitted['measured_positive_sources']==12 and summary['fitting_status']=='fitted_training_only_model'
    assert fitted==m.fit_model(read(out/'measurement_plan.json'),m.read_records(out))
    # Corruption fails the checksum BEFORE any model refit or validation source.
    target=out/'batch_0_state_000.json';target.rename(out/'retained_original.json')
    changed=read(out/'retained_original.json');changed['rows'][0]['measurement_value']=1.
    save(target,changed)
    with pytest.raises(ValueError,match='artifact changed'):m.completed_training(out,cfg,{'fixture':1})
