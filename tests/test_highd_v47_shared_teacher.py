import copy
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_shared_teacher as runner
from scenario_reconstruction import highd_v47_shared_teacher_native as native
from scenario_reconstruction.highd_v46_assets import read,save,digest
from scenario_reconstruction.highd_v46_contract import enrich_observation
from scenario_reconstruction.highd_v46_cross_unit import proposal_factors
from scenario_reconstruction.highd_v46_d2rl import identity
from d2rl_training.highd_v46_sequence_env import log_weight,V46SequenceEnv
from d2rl_training.highd_v47_shared_teacher_env import SharedTeacherEnv


def test_export_audit_stage_wiring(tmp_path,monkeypatch):
    """Synthetic complete handoff; opens no native source or fresh outcome."""
    d=data();cases=copy.deepcopy(d['state_law'])
    bank=tmp_path/'bank';old=tmp_path/'old';out=tmp_path/'export'
    cfg=dict(bank=str(bank),data=str(old),workers=1,data_summary_sha256='same_behavior',
        target_rule='formal',minimum_relative_derivative=1e-6,minimum_usable_events=20,
        minimum_usable_sources=2,minimum_second_factor_events=10,minimum_second_factor_sources=2)
    parent=dict(natural_target_sha256='natural',physical_target_sha256='physical')
    lock=dict(config=cfg,components={'scenario_reconstruction/highd_v47_shared_teacher_native.py':'native'})
    for c in cases:
        c['design_file']=f'design_{c["case_id"]}.json';save(bank/c['design_file'],{'case_id':c['case_id']})
        save(old/f'case_{c["case_id"]:05d}.json',dict(sequences=[dict(
            seed=c['continuation_seed_start']+j,event_result=j%4==0,
            collision_pairs=[[99,1]] if j%4==0 else [],generation_log_weight=float(np.log(2.)),
            initial_likelihood_audit={}) for j in range(64)]))
    def replay(design,s,*args):
        active=(s['seed']-1000)%64!=0
        views={m:[native.target_step(description(m),{1:0,2:1,3:0},-3.,0.)] if active else []
               for m in ('single','dual')}
        return dict(seed=s['seed'],event_result=s['event_result'],collision_pairs=s['collision_pairs'],
            generation_log_weight=s['generation_log_weight'],cav_id=99,behavior_initial_audit={},views=views,
            native_decisions=[dict(time_s=0.,actions={'1':0,'2':1,'3':0},behavior_full_log_p=-3.,
                behavior_full_log_q=-3.-s['generation_log_weight'],target_active={m:active for m in views})])
    monkeypatch.setattr(runner,'checked',lambda *args:(cfg,lock,{},tmp_path,parent,cases))
    monkeypatch.setattr(runner,'worker_init',lambda *args:None)
    monkeypatch.setattr(runner,'_WORK',(None,tmp_path,{},cfg))
    monkeypatch.setattr(runner,'ProcessPoolExecutor',lambda n,**kw:ThreadPoolExecutor(n))
    monkeypatch.setattr(native,'replay',replay)
    monkeypatch.setattr(runner,'audit_sample',lambda *args:None)
    monkeypatch.setattr(runner,'PairModelQueries',lambda *args,**kwargs:None)
    result=runner.export('unused','unused',out)
    assert result['complete'] and result['physical_paths']==192 and result['new_samples']==0
    assert result['training_signal_ready']
    result=runner.audit('unused','unused',out,tmp_path/'audit.json',True)
    assert result['verified'] and result['native_replays']==3
    assert read(out/'summary.json')['readiness']['modes']['single']['target_inactive_event_constants']==3
    with pytest.raises(FileExistsError):runner.export('unused','unused',out)


def description(mode='single',fallback=False):
    pair=mode=='dual' and not fallback
    p=np.array([[.3,.2],[.1,.4]]) if pair else np.array([.4,.6])
    h=np.array([[.1,.6],[.2,.1]]) if pair else np.array([.8,.2])
    return dict(actor_ids=[1,2] if pair else [3],dimension=1 if mode=='single' else 2,
        factor_mask=[1,1] if pair else [1] if mode=='single' else [1,0],
        observation=enrich_observation([0.]*14,.1,.4,0.,0.,4,[1,int(pair)]),
        time_s=0.,source_time_s=10.,P=p,H=h)


def data(mode='single'):
    cases=[dict(case_id=i,source_seed=s,alpha=a,state_sha256=str(i),actor_ids=[1,2],continuation_seed_start=1000+i*64)
           for i,(s,a) in enumerate([(10,.25),(10,.25),(20,.5)])]
    seq=[]
    for c in cases:
        for j in range(64):
            event=j%4==0
            t=native.target_step(description(mode),{1:0,2:1,3:0},-3.,0.)
            seq.append(dict(seed=c['continuation_seed_start']+j,source_seed=c['source_seed'],case_id=c['case_id'],
                cav_id=99,event_result=event,collision_pairs=[[99,1]] if event else [],
                generation_log_weight=np.log(2.),state_likelihood_ratio=1.,state_log_weight=0.,
                target_mass=c['alpha']/64,steps=[native.compact_step(t)] if j!=0 else []))
    parent=dict(natural_target_sha256='natural',physical_target_sha256='physical')
    lock=dict(config=dict(data_summary_sha256='same_behavior',target_rule='formal'),
              components={'scenario_reconstruction/highd_v47_shared_teacher_native.py':'native'})
    return runner.dataset(mode,seq,cases,parent,lock,'same_shared_paths')


def write(path,d):
    save(path/'sequences.json',d)
    save(path/'manifest.json',dict(data_path='sequences.json',sha256=digest(path/'sequences.json'),**{k:d[k] for k in
        ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256','state_law_sha256',
         'proposal_structure_sha256','behavior_sha256','shared_paths_sha256','intervention_mode')}))
    return path/'manifest.json'


@pytest.mark.parametrize('mode',['single','dual'])
def test_exact_stratified_shared_reward_includes_inactive_constants(tmp_path,mode):
    d=data(mode);env=SharedTeacherEnv(dict(sequence_manifest=write(tmp_path,d),replay_positive_fraction=.5))
    values=env.objective_values(lambda _:[.1]*env.dimension)
    direct=[]
    for s in d['sequences']:
        lt=sum(log_weight(t,[.1]*env.dimension) for t in s['steps'])
        direct.append(s['target_mass']*s['event_result']*np.exp(s['generation_log_weight']+lt-env.log_scale))
    assert values.mean()==pytest.approx(sum(direct))
    assert all(values[i]>0 for i in (0,64,128))  # no-target-step events are retained constants
    for _ in range(15):
        env.reset();s=env.sequence;_,reward,done,_=env.step([.1]*env.dimension)
        group=env.positive if s['event_result'] else env.negative
        expected=-len(group)/(len(env.sequences)*.5)*env.loss(s,sum(log_weight(t,[.1]*env.dimension) for t in s['steps']))
        assert done and reward==pytest.approx(expected)
    with pytest.raises(ValueError):V46SequenceEnv(dict(sequence_manifest=tmp_path/'manifest.json'))


@pytest.mark.parametrize('mutation',['behavior_lr','mass','source','seed','mask','prefix','label','reference_q','tv','timeline','state_lr'])
def test_bad_shared_contract_rejected(tmp_path,mutation):
    d=data('dual');s=d['sequences'][1];t=s['steps'][0]
    if mutation=='behavior_lr':s['generation_log_weight']=float('inf')
    elif mutation=='mass':s['target_mass']=.1
    elif mutation=='source':s['source_seed']=999
    elif mutation=='seed':s['seed']+=1
    elif mutation=='mask':t['factor_mask']=[1,0]
    elif mutation=='prefix':t['observation'][16]=1
    elif mutation=='label':s['event_result']=True
    elif mutation=='reference_q':t['reference_full_log_q']+=1
    elif mutation=='tv':t['factor_tv'][0]=-1
    elif mutation=='timeline':t['time_s']=4
    else:s['state_likelihood_ratio']=2
    # JSON disallows Infinity too; both serialization and runtime reject it.
    with pytest.raises(ValueError):SharedTeacherEnv(dict(sequence_manifest=write(tmp_path,d)))


def test_actor_selection_is_independent_not_teacher_first_factor(monkeypatch):
    calls=[]
    def choose(self,units,actors,pmfs,mode):
        calls.append(mode);return {'ids':[9]} if mode=='single' else {'ids':[7,8]}
    monkeypatch.setattr(native.NativeExperiment,'choose',choose)
    result=native.choices(object(),[],{}, {})
    assert result=={'single':{'ids':[9]},'dual':{'ids':[7,8]}}
    assert calls==['single','dual']


def test_common_gate_and_dual_single_fallback(monkeypatch):
    monkeypatch.setattr(native.NativeExperiment,'choose',lambda *args:None)
    assert native.choices(object(),[],{}, {})==dict(single=None,dual=None)
    monkeypatch.setattr(native.NativeExperiment,'choose',lambda self,u,a,p,m:dict(ids=[4]))
    assert native.choices(object(),[],{}, {})['dual']['ids']==[4]
    t=native.target_step(description('dual',True),{3:0},-3.,0.)
    assert t['factor_mask']==[1,0] and t['p_factors'][1]==t['h_factors'][1]==1
    assert t['factor_tv'][1]==0
    assert log_weight(t,[.2,.05])==log_weight(t,[.2,1.])


def test_correlated_natural_joint_shared_behavior_two_different_targets():
    # Behavior biases actors 1 and 2; Single target independently biases actor 3.
    # Actors 1/2 are naturally correlated; actor 3 remains natural under behavior.
    p12=np.array([[.3,.2],[.1,.4]]);p3=np.array([.7,.3])
    hb=np.array([[.1,.6],[.2,.1]]);q1,q2=proposal_factors(p12,hb,[.1,.1])[-2:]
    qb=q1[:,None]*q2;h3=np.array([.1,.9])
    hd=np.array([[.45,.05],[.05,.45]])
    def objective(e,mode):
        actual=truth=0.
        qt3=e*p3+(1-e)*h3
        z1,z2=proposal_factors(p12,hd,[e,e])[-2:];qt12=z1[:,None]*z2
        for a,b,c in product(range(2),repeat=3):
            pp=p12[a,b]*p3[c];behavior=qb[a,b]*p3[c]
            target=p12[a,b]*qt3[c] if mode=='single' else qt12[a,b]*p3[c]
            event=(a!=b and c==1) or (a==b==c==0)
            actual+=behavior*event*(pp/behavior)*(pp/target)
            truth+=event*pp**2/target
        return actual,truth
    for mode in ('single','dual'):
        for e in (.05,.1,.5,1.):
            actual,truth=objective(e,mode);assert actual==pytest.approx(truth,rel=1e-13)
        alo,tlo=objective(.3-1e-6,mode);ahi,thi=objective(.3+1e-6,mode)
        assert (ahi-alo)/2e-6==pytest.approx((thi-tlo)/2e-6)


def test_target_prefix_not_behavior_terminal_weight():
    t=native.target_step(description(),{3:0},-3.,0.)
    assert t['observation'][16]==0
    t2=native.target_step(description(),{3:1},-2.,t['reference_log_weight'])
    assert t2['observation'][16]==pytest.approx(t['reference_log_weight']/30.)
    assert t2['full_log_p']-t2['reference_full_log_q']==pytest.approx(t2['reference_log_weight'])


def test_training_readiness_separate_from_raw_collision_counts(tmp_path):
    envs={m:SharedTeacherEnv(dict(sequence_manifest=write(tmp_path/m,data(m)))) for m in ('single','dual')}
    cfg=read(runner.DEFAULT)
    report=runner.readiness(envs,cfg)
    assert not report['passed']
    assert report['modes']['single']['CAV_events']==48
    assert report['modes']['single']['usable_events']==45
    assert report['modes']['single']['target_inactive_event_constants']==3
    assert report['modes']['dual']['second_factor_usable_sources']==2
    envs['single'].sequences[0]['generation_log_weight']+=.1
    with pytest.raises(ValueError,match='identical'):runner.readiness(envs,cfg)


def test_runtime_adapter_restores_original_environment_and_metrics():
    from d2rl_training import highd_v46_sequence_env as base
    from scenario_reconstruction import highd_v47_formal_training_metrics as metrics
    old=base.V46SequenceEnv;old_metrics=metrics.diagnostics
    with runner.adapter():
        assert base.V46SequenceEnv is SharedTeacherEnv and metrics.diagnostics is runner.diagnostics
    assert base.V46SequenceEnv is old and metrics.diagnostics is old_metrics


def test_failed_signal_never_starts_ppo(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'training_inputs',lambda *args:({}, {}, {'readiness':dict(passed=False,reasons=['no signal'])}))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Must not start PPO'))
    result=runner.train('x','x','x','x',tmp_path/'training')
    assert not result['PPO_started'] and not (tmp_path/'training').exists()


def test_compact_diagnostic_tv_matches_full_target_pmfs(tmp_path):
    d=data('dual');env=SharedTeacherEnv(dict(sequence_manifest=write(tmp_path,d)))
    r=runner.diagnostics(env,lambda _:[.3,.6])
    t=native.target_step(description('dual'),{1:0,2:1},-3.,0.)
    for i,e in enumerate((.3,.6)):
        p,h=np.asarray(t['factor_pmfs'][i]['p']),np.asarray(t['factor_pmfs'][i]['h'])
        assert r['conditional_QP_TV_at_logged_action_rows'][i]==pytest.approx(.5*np.abs(e*p+(1-e)*h-p).sum())
