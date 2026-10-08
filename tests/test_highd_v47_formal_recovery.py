import copy
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest

from scenario_reconstruction import highd_v47_formal_recovery as runner
from scenario_reconstruction import highd_v47_formal_recovery_native as native
from scenario_reconstruction.highd_v46_assets import read,save,digest
from scenario_reconstruction.highd_v46_contract import enrich_observation
from scenario_reconstruction.highd_v46_cross_unit import proposal_factors
from d2rl_training.highd_v46_sequence_env import V46SequenceEnv,log_weight
from d2rl_training.highd_v47_offpolicy_sequence_env import OffPolicySequenceEnv,CONTRACT,BEHAVIOR
from test_highd_v47_fixed_alpha_ppo import dataset,FixedRng


def data(mode='dual',n=12):
    d=dataset(mode,n);dim=d['action_dimension']
    d.update(contract=CONTRACT,behavior_H=BEHAVIOR,target_H='unchanged_original_formal_Single_or_Dual_H',
        behavior_sha256='behavior',proposal_sha256='fixed_target_epsilon')
    for seq in d['sequences']:
        total=0.;decisions=[]
        for t,step in enumerate(seq['steps']):
            p=np.full(dim,.4);g=np.full(dim,.5);q=.1*p+.9*g
            step.update(generation_h_factors=g.tolist(),q_factors=q.tolist(),action_indices=[0]*dim,
                factor_pmfs=[dict(p=[.4,.6],h=[.7,.3]) for _ in range(dim)],
                generation_factor_pmfs=[dict(p=[.4,.6],h=[.5,.5]) for _ in range(dim)],
                full_log_p=float(np.log(p).sum()-.2),full_log_q=float(np.log(q).sum()-.2),
                log_weight=float(np.log(p/q).sum()),
                observation=enrich_observation([0.]*14,.1,.5,total,t,4,[1,1] if dim==2 else [1,0]))
            total+=step['log_weight'];decisions.append(dict(log_p=step['full_log_p'],log_q=step['full_log_q']))
        seq.update(cav_id=0,collision_pairs=[[0,1]] if seq['event_result'] else [[1,2]],native_decisions=decisions,
            generation_log_weight=total,action_log_weight=total)
    return d


def write(path,d):
    save(path/'sequences.json',d)
    save(path/'manifest.json',dict(data_path='sequences.json',sha256=digest(path/'sequences.json'),**{k:d[k] for k in
        ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256','state_law_sha256',
         'proposal_structure_sha256','intervention_mode','behavior_sha256')}))
    return path/'manifest.json'


@pytest.mark.parametrize('mode',['single','dual'])
def test_behavior_target_LR_and_numerical_scale_are_distinct(tmp_path,mode):
    d=data(mode);env=OffPolicySequenceEnv(dict(sequence_manifest=write(tmp_path,d)))
    s=d['sequences'][0];target=sum(log_weight(t,[.1]*env.dimension) for t in s['steps'])
    assert env.log_scale==pytest.approx(s['generation_log_weight']+target)
    assert env.log_scale!=pytest.approx(2*s['generation_log_weight'])
    assert env.empirical_objective(lambda obs:[.1]*env.dimension)==pytest.approx(1/3)
    assert env.dataset['contract']==CONTRACT
    # The frozen old constructor refuses the new action-role contract.
    with pytest.raises(ValueError,match='contract'):V46SequenceEnv(dict(sequence_manifest=tmp_path/'manifest.json'))


@pytest.mark.parametrize('mode',['single','dual'])
def test_positive_negative_replay_corrects_to_full_offpolicy_collection(tmp_path,mode):
    env=OffPolicySequenceEnv(dict(sequence_manifest=write(tmp_path,data(mode)),replay_positive_fraction=.5))
    seen=set()
    for _ in range(40):
        env.reset();done=False
        while not done:_,reward,done,_=env.step([.1]*env.dimension)
        seen.add(env.sequence['event_result'])
        assert reward==pytest.approx(-2/3 if env.sequence['event_result'] else 0.)
    assert seen=={True,False}


@pytest.mark.parametrize('mutation',['behavior_q','target_h','generation_h','prefix','full','label','LR','source_mass','contract','pmf','epsilon'])
def test_no_silent_behavior_target_or_event_identity_conversion(tmp_path,mutation):
    d=data();s=d['sequences'][0];t=s['steps'][0]
    if mutation=='behavior_q':t['q_factors'][0]+=.1
    elif mutation=='target_h':t['h_factors'][0]=.5
    elif mutation=='generation_h':t['generation_h_factors'][0]=.7
    elif mutation=='prefix':t['observation'][16]=1.
    elif mutation=='full':s['native_decisions'][0]['log_q']+=.2
    elif mutation=='label':s['collision_pairs']=[[1,2]]
    elif mutation=='LR':s['state_likelihood_ratio']=2.
    elif mutation=='source_mass':s['state_proposal_mass']=.7
    elif mutation=='contract':d['contract']='highd_v47_fixed_alpha_d2rl_sequence_v1'
    elif mutation=='epsilon':t['generation_epsilon'][0]=.2
    else:t['factor_pmfs'][0]['h']=[.8,.3]
    with pytest.raises(ValueError):OffPolicySequenceEnv(dict(sequence_manifest=write(tmp_path,d)))


def test_exact_two_decision_offpolicy_M2_and_epsilon_derivative():
    p=np.array([.35,.65]);target_h=np.array([.8,.2]);g=np.array([.5,.5]);qb=.1*p+.9*g
    def objective(e):
        qt=e*p+(1-e)*target_h
        result=truth=0.
        for a,b in product(range(2),repeat=2):
            # Include both negative endpoints and a history-dependent event.
            y=a!=b;pp=p[a]*p[b];behavior=qb[a]*qb[b];target=qt[a]*qt[b]
            result+=behavior*y*(pp/behavior)*(pp/target)
            truth+=y*pp**2/target
        assert result==pytest.approx(truth,rel=1e-13)
        return result,truth
    for e in (.05,.1,.3,.9,1.):objective(e)
    lo,hi=objective(.3-1e-6),objective(.3+1e-6)
    assert (hi[0]-lo[0])/2e-6==pytest.approx((hi[1]-lo[1])/2e-6)


def test_uniform_teacher_has_no_unsupported_mass():
    assert np.allclose(native.supported_uniform([[.2,0.],[.3,.5]]),[[1/3,0.],[1/3,1/3]])
    with pytest.raises(ValueError):native.supported_uniform([0.,0.])


@pytest.mark.parametrize('mode,ids',[('single',[1]),('dual',[1]),('dual',[1,2])])
def test_native_logs_behavior_and_original_target_separately(monkeypatch,mode,ids):
    p=np.array([[.15,.25],[.1,.5]]);unit=dict(ids=[1,2],p=p)
    target=np.array([[0.,0.],[.6,.4]]) if len(ids)==2 else np.array([.8,.2])
    cars={i:SimpleNamespace(x=i*10.,length=5.,v=30.) for i in (1,2)}
    road=SimpleNamespace(config={'horizon_s':60},time=12.,log_weight=0.,steps=[],rng=FixedRng([.2,.7,.3,.4]))
    monkeypatch.setattr(native.native,'common_choice',lambda *args:dict(ids=ids,h=target,mass=.1,relative_contrast=.2,unit=unit))
    monkeypatch.setattr(native.native,'observation',lambda *args:[0.]*14)
    actions,step,audit=native.act(road,[unit],cars,{},mode)
    assert set(actions)=={1,2}  # unselected partner still follows correlated P
    assert audit['log_p']-audit['log_q']==pytest.approx(step['log_weight'])
    assert np.log(np.asarray(step['p_factors'])/step['q_factors']).sum()==pytest.approx(road.log_weight)
    if len(ids)==2:
        a,b=step['action_indices'];p1,p2,h1,h2,_,_=proposal_factors(p,target,[.1,.1])
        assert step['h_factors']==pytest.approx([h1[a],h2[a,b]])
        assert step['h_factors'][0]==0  # original zero row: conditional P fallback retained
        assert step['generation_h_factors'][0]>0
    else:
        assert step['h_factors'][0]==target[step['action_indices'][0]]
        if mode=='dual':
            assert step['factor_mask']==[1,0]
            assert all(step[k][1]==1 for k in ('p_factors','h_factors','generation_h_factors','q_factors'))


def test_replay_adapter_is_scoped_and_old_source_not_edited():
    from d2rl_training import highd_v46_sequence_env as base
    old=base.V46SequenceEnv
    with runner.replay_adapter():assert base.V46SequenceEnv is OffPolicySequenceEnv
    assert base.V46SequenceEnv is old


def test_training_block_has_no_Ray_subprocess_or_output_directory(tmp_path,monkeypatch):
    cfg=read(runner.DEFAULT);envs={}
    for mode in ('single','dual'):
        d=data(mode)
        for s in d['sequences']:s['event_result']=False;s['collision_pairs']=[[1,2]]
        envs[mode]=OffPolicySequenceEnv(dict(sequence_manifest=write(tmp_path/mode,d),evaluation_only=True))
    summary=dict(complete=True,training_readiness=runner.training_readiness(envs,cfg))
    plan=dict(case_indices=[0,1]);lock={'parent':1};r={'recovery':2}
    monkeypatch.setattr(runner,'checked',lambda *a:(cfg,{},None,None,lock,r,{},plan))
    monkeypatch.setattr(runner,'load_data',lambda *a:(summary,envs))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('PPO was started with zero reward'))
    from scenario_reconstruction.highd_v46_d2rl import identity
    save(tmp_path/'data/summary.json',summary)
    save(tmp_path/'data/audit.json',dict(verified=True,lock=lock,recovery_lock_sha256=identity(r),
        collection_sha256=digest(tmp_path/'data/summary.json'),native_replays=4))
    result=runner.train('cfg','frozen',tmp_path/'data',tmp_path/'ppo')
    assert result['phase']=='training_blocked' and not result['PPO_started'] and not result['validation_opened']
    assert not (tmp_path/'ppo').exists()


def test_sparse_positive_labels_do_not_pass_both_mode_signal_gate(tmp_path):
    envs={m:OffPolicySequenceEnv(dict(sequence_manifest=write(tmp_path/m,data(m)),evaluation_only=True)) for m in ('single','dual')}
    report=runner.training_readiness(envs,read(runner.DEFAULT))
    assert not report['passed'] and len(report['reasons'])==2
    assert report['modes']['single']['usable_events']==4
    assert report['modes']['dual']['second_factor_usable_events']==4


def test_dual_cannot_start_as_only_a_padded_single_agent(tmp_path):
    envs={m:OffPolicySequenceEnv(dict(sequence_manifest=write(tmp_path/m,data(m)),evaluation_only=True)) for m in ('single','dual')}
    # All event reward sensitivity belongs to factor 1; factor 2 is natural.
    for s in envs['dual'].sequences:
        for t in s['steps']:t['h_factors'][1]=t['p_factors'][1]
    report=runner.training_readiness(envs,read(runner.DEFAULT))
    assert report['modes']['dual']['second_factor_usable_events']==0
    assert 'dual: no event-weighted active second-factor target contrast' in report['reasons']


def test_recovery_does_not_change_parent_state_schedule_or_PPO_budget():
    cfg=read(runner.DEFAULT);old=read(cfg['parent_config'])
    assert cfg['state_model'] is None and cfg['state_likelihood_ratio']==1
    for key in ('draws_per_mode','state_seed','seeds','iterations','checkpoint_interval'):
        assert cfg['training'][key]==old['training'][key]
    assert cfg['training']['continuation_seed_start']!=old['training']['continuation_seed_start']
    assert not cfg['validation_or_final_test_during_recovery']


def test_freeze_locks_whole_plan_without_simulating_and_detects_input_changes(tmp_path,monkeypatch):
    cfg=read(runner.DEFAULT);pc=read(cfg['parent_config']);lock=dict(config=pc,parent='unchanged')
    law=dataset()['state_law'];bank=dict(cases=[dict(case_id=s['case_id']) for s in law],state_law=law)
    for c,s in zip(bank['cases'],law):c['source_seed']=s['source_seed']
    pf=tmp_path/'parent';bp=tmp_path/'bank';old=tmp_path/'old'
    paths=[pf/'lock.json',pf/'historical_inventory.json',bp/'bank.json',bp/'summary.json',old/'summary.json',old/'audit.json',old/'shared_schedule.json']
    paths += [pf/(s+'_source_manifest.json') for s in pc['source_splits']]
    paths += [old/m/name for m in ('single','dual') for name in ('sequences.json','sequence_manifest.json')]
    for path in paths:save(path,dict(unchanged=str(path.relative_to(tmp_path))))
    save(tmp_path/'config.json',cfg)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'implementation',lambda *a:(cfg,{'config.json':digest(tmp_path/'config.json')}))
    monkeypatch.setattr(runner.parent,'checked',lambda *a:(pc,{},None,lock))
    monkeypatch.setattr(runner.compat,'load_bank',lambda *a:bank)
    monkeypatch.setattr(runner,'original_evidence',lambda *a:dict(modes={m:{'CAV_events':0} for m in ('single','dual')}))
    monkeypatch.setattr(runner,'worker',lambda *a:pytest.fail('Freeze simulated a native trajectory'))
    result=runner.freeze('cfg',pf,bp,old,tmp_path/'frozen')
    assert not result['new_native_outcomes_opened']
    plan=runner.checked('cfg',tmp_path/'frozen')[-1]
    assert len(plan['case_indices'])==8192 and plan['seeds'][0]==1300000000
    assert plan['case_indices']==runner.state_schedule(law,8192,pc['training']['state_seed'])
    # Changes to a previously negative artifact also invalidate the recovery.
    (old/'single/sequences.json').write_text('{}')
    with pytest.raises(ValueError,match='Protected input'):runner.checked('cfg',tmp_path/'frozen')


@pytest.mark.parametrize('key',['proposal_structure_sha256','horizon_s','continuation_seed_start','state_schedule_seed','behavior_sha256'])
def test_loader_requires_original_target_identity_and_exact_new_behavior_plan(tmp_path,monkeypatch,key):
    cfg=read(runner.DEFAULT);cfg['training'].update(draws_per_mode=12,state_seed=37,continuation_seed_start=1000)
    lock=dict(config={'unused':1});r=dict(config=cfg);b=dict(state_law=dataset()['state_law']);envs={};hashes={}
    from scenario_reconstruction.highd_v46_d2rl import identity
    for mode in ('single','dual'):
        d=data(mode);d['recovery_lock_sha256']=identity(r);d['behavior_sha256']=identity(dict(recovery=identity(r),mode=mode))
        manifest=write(tmp_path/mode,d);hashes[mode]=digest(tmp_path/mode/'sequences.json')
        save(tmp_path/mode/'sequence_manifest.json',read(manifest))
        envs[mode]=OffPolicySequenceEnv(dict(sequence_manifest=manifest,evaluation_only=True))
    summary=dict(complete=True,lock=lock,recovery_lock_sha256=identity(r),sequence_hashes=hashes,
        training_readiness=runner.training_readiness(envs,cfg))
    save(tmp_path/'summary.json',summary)
    monkeypatch.setattr(runner.parent,'dataset',lambda mode,*a:data(mode))
    assert runner.load_data(tmp_path,lock,r,b)[0]==summary
    # Force an otherwise checksummed, internally consistent but wrong target.
    envs['single'].dataset[key]='wrong'
    monkeypatch.setattr(runner,'OffPolicySequenceEnv',lambda c:envs['single'] if 'single' in c['sequence_manifest'] else envs['dual'])
    with pytest.raises(ValueError,match='data/law/budget'):runner.load_data(tmp_path,lock,r,b)
