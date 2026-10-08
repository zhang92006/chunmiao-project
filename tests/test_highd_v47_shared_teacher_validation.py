import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest

from scenario_reconstruction import highd_v47_shared_teacher_validation as runner
from scenario_reconstruction import highd_v47_shared_validation_native as native
from scenario_reconstruction.highd_v46_assets import read,save,digest
from scenario_reconstruction.highd_v46_d2rl import identity
from scenario_reconstruction.highd_v46_contract import enrich_observation
from scenario_reconstruction.highd_v47_formal_comparison import canonical_path
from scenario_reconstruction.highd_v47_fixed_alpha_contract import state_schedule


def cfg():return read(runner.DEFAULT)


def description(mode,pair=False):
    p=np.array([[.15,.25],[.1,.5]]) if pair else np.array([.4,.6])
    h=np.array([[.4,.1],[.2,.3]]) if pair else np.array([.8,.2])
    dim=1 if mode=='single' else 2
    return dict(P=p,H=h,actor_ids=[1,2] if pair else [1],dimension=dim,time_s=1.,source_time_s=34.,
        factor_mask=[1,1] if pair else [1] if dim==1 else [1,0],
        observation=enrich_observation([0.]*14,.1,.2,0.,1.,4,[1,int(pair)]))


@pytest.mark.parametrize('mode,pair',[('single',False),('dual',False),('dual',True)])
def test_native_factors_match_training_view_and_exact_partner_law(monkeypatch,mode,pair):
    d=description(mode,pair);units=[dict(ids=[1,2],p=np.array([[.15,.25],[.1,.5]]))]
    road=SimpleNamespace(rng=np.random.default_rng(7),time=34.,log_weight=2.)
    # Single marginal must match the correlated natural joint.
    if not pair:d['P']=units[0]['p'].sum(1)
    monkeypatch.setattr(native,'common_choice',lambda *args:{'ids':d['actor_ids']})
    monkeypatch.setattr(native.target,'describe',lambda *args:d)
    seen=[]
    def policy(obs):seen.append(obs);return [.1]*d['dimension']
    actions,t,a=native.act(road,units,{}, {},mode,policy,33.)
    view=native.target.target_step(d,actions,t['full_log_p'],2.)
    assert t['observation']==view['observation']==seen[0]
    assert t['p_factors']==view['p_factors'] and t['h_factors']==view['h_factors']
    assert t['log_weight']==pytest.approx(view['reference_log_weight'])
    assert t['full_log_q']==pytest.approx(view['reference_full_log_q'])
    assert road.log_weight==pytest.approx(2+t['log_weight'])
    assert a['log_p']-a['log_q']==pytest.approx(t['log_weight'])
    assert set(actions)=={1,2}


def test_natural_never_calls_target_policy(monkeypatch):
    road=SimpleNamespace(rng=np.random.default_rng(4))
    actions,t,a=native.act(road,[dict(ids=[1],p=np.array([.4,.6]))],{}, {},'natural',None,0.)
    assert t is None and a['log_p']==a['log_q'] and set(actions)=={1}


def test_protocol_disjoint_budget_and_all_checkpoints():
    c=cfg();sources=set(range(c['source']['seed_start'],c['source']['seed_start']+c['source']['episodes']))
    assert sources==set(range(1800010000,1800011024))
    assert not sources&set(range(1800000000,1800004096))
    assert c['checkpoint_iterations']==list(range(0,201,10))
    assert (len(c['training']['seeds'])*2*len(c['checkpoint_iterations'])+1)*c['draws_per_checkpoint']==864256
    assert c['state_model'] is None and c['state_likelihood_ratio']==1


def row(source=1800010000):
    return dict(seed=source,state_index=0,actor_ids=[1,2],time_s=30.,mechanism='test',
        views=dict(production_strict=True),metrics=dict(positive_interaction_mass=.1),ranker_features=[0,0,.8])


def chunk():
    start=1800010000;state=dict(seed=start,time_s=30.)
    rows=[row()];indices,funnel=runner.expansion.selected_rows(rows,list(range(start,start+64)))
    return dict(source=dict(seed_start=start),bank=[state],rows=rows,selected_indices=indices,funnel=funnel,
        report=dict(episodes=[dict(seed=i) for i in range(start,start+64)]))


def test_bank_retains_abstentions_and_selects_no_native_labels():
    c=cfg();c['source']['episodes']=64;x=chunk()
    b=runner.build_bank([('chunk.json',x)],c)
    assert len(b['source_records'])==64 and len(b['cases'])==1
    assert b['cases'][0]['alpha']==1. and not b['selection_uses_outcomes']
    x['selected_indices']=[]
    with pytest.raises(ValueError,match='selection'):runner.build_bank([('chunk.json',x)],c)


def record(case,seed):
    return dict(seed=seed,case_id=case['case_id'],source_seed=case['source_seed'],event_result=False,
        event_type='horizon_no_collision',state_target_mass=case['alpha'],state_proposal_mass=case['alpha'],
        state_likelihood_ratio=1.,state_log_weight=0.,generation_log_weight=0.,action_log_weight=0.,
        cav_id=99,collision_pairs=[],elapsed_s=4.,steps=[],cpu_s=.1,wall_time_s=.2,
        native_decisions=[dict(log_p=-1.,log_q=-1.,time_s=0.,actions={'1':0},decision_cpu_s=.01)])


@pytest.mark.parametrize('key,value',[('seed',9),('event_result',True),('state_target_mass',.5),
    ('state_likelihood_ratio',2),('generation_log_weight',1),('elapsed_s',5),('event_type','cav_first_collision')])
def test_tampered_native_record_rejected(key,value):
    case=dict(case_id=0,source_seed=10,alpha=1.);s=record(case,7)
    assert runner.check_record(s,case,7,None,'natural')==0.
    s[key]=value
    with pytest.raises(ValueError):runner.check_record(s,case,7,None,'natural')


def test_canonical_replay_excludes_execution_time_only():
    a=record(dict(case_id=0,source_seed=10,alpha=1.),7);b=copy.deepcopy(a)
    b['cpu_s']=100;b['wall_time_s']=100;b['native_decisions'][0]['decision_cpu_s']=100
    assert canonical_path(a)==canonical_path(b)
    b['native_decisions'][0]['actions']['1']=1
    assert canonical_path(a)!=canonical_path(b)


def test_complete_synthetic_validation_pipeline_and_sparse_guard(tmp_path,monkeypatch):
    c=cfg();c['draws_per_checkpoint']=4;c['batch_size']=2;c['training']['seeds']=[7]
    c['checkpoint_iterations']=[0,10];c['source']['episodes']=64;c['source_claim']=str(tmp_path/'claim.json')
    arms=[dict(label='natural',mode='natural',seed=None,iteration=None,policy_path=None)]
    arms.extend(dict(label=f'{m}_seed7_iter{i}',mode=m,seed=7,iteration=i,policy_path=None)
                for m in ('single','dual') for i in (0,10))
    lock=dict(config=c,arms=arms);b=runner.build_bank([('chunk_1800010000.json',chunk())],c)
    bank=tmp_path/'bank'
    monkeypatch.setattr(runner,'checked',lambda *args:(c,{},tmp_path,lock))
    monkeypatch.setattr(runner,'ProcessPoolExecutor',lambda *args,**kwargs:ThreadPoolExecutor(1))
    def discovery(task):
        start,output=task;name=f'chunk_{start}.json';save(output+'/'+name,chunk())
        return name,digest(output+'/'+name)
    monkeypatch.setattr(runner,'discovery_worker',discovery)
    assert runner.bank('unused','unused',bank)['cases']==1
    assert runner.load_bank(bank,lock,True)['cases']==b['cases']
    with pytest.raises(FileExistsError):runner.bank('unused','unused',tmp_path/'second_bank')
    monkeypatch.setattr(runner,'batch_worker',lambda task:[record(b['cases'][k],s) for k,s in task[1]])
    monkeypatch.setattr(runner,'PairModelQueries',lambda *args,**kwargs:None)
    monkeypatch.setattr(native,'rollout',lambda model,assets,phys,mode,seed,case,policy:record(case,seed))
    out=tmp_path/'native';result=runner.run('unused','unused',bank,out)
    assert result['complete'] and not result['readiness']['passed']
    selected=read(out/'selection.json')['selected']
    assert all(a['iteration']==0 and a['reason']=='insufficient_baseline_event_evidence' for a in selected)
    result=runner.audit('unused','unused',bank,out,tmp_path/'audit.json',True)
    assert result['verified'] and result['native_replays']==5
    assert read(out/'summary.json')['logical_continuations']==20
    with pytest.raises(FileExistsError):runner.run('unused','unused',bank,out)
    path=out/'natural/batch_0000.json';path.write_text('[]')
    with pytest.raises(ValueError,match='artifact'):runner.audit('unused','unused',bank,out,tmp_path/'bad.json')


def test_freeze_never_opens_sources_or_relabels_previous_claim(tmp_path,monkeypatch):
    c=cfg();c['source_claim']=str(tmp_path/'claim.json')
    monkeypatch.setattr(runner,'protocol',lambda *args:(c,{},tmp_path,dict(config_sha256='test')))
    monkeypatch.setattr(runner,'verify_training',lambda *args:({},[]))
    result=runner.freeze('unused',tmp_path/'lock.json')
    assert not result['validation_opened'] and not (tmp_path/'claim.json').exists()
    save(tmp_path/'claim.json',{'opened':True})
    with pytest.raises(ValueError,match='already opened'):runner.freeze('unused',tmp_path/'new_lock.json')
