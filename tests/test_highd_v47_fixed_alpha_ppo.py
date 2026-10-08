import copy
from types import SimpleNamespace

import numpy as np
import pytest

from d2rl_training.highd_v46_sequence_env import V46SequenceEnv, log_weight
from scenario_reconstruction.highd_v46_assets import save, digest, read
from scenario_reconstruction.highd_v46_contract import FIXED_ALPHA_OBSERVATION, enrich_observation, independent_split
from scenario_reconstruction.highd_v46_d2rl import identity
from scenario_reconstruction.highd_v46_pair_synergy import response_risk
from scenario_reconstruction import highd_v47_fixed_alpha_native as native
from scenario_reconstruction import highd_v47_fixed_alpha_contract as contract
from scenario_reconstruction import highd_v47_fixed_alpha_ppo as runner


def dataset(mode='dual',n=12):
    law=[dict(case_id=str(i),source_seed=260000+i,state_sha256=str(i),alpha=.5) for i in range(2)]
    schedule=contract.state_schedule(law,n,37); dim=2 if mode=='dual' else 1; sequences=[]
    for i,k in enumerate(schedule):
        steps=[]; prefix=0.
        for t in range(4):
            p=np.full(dim,.4); h=np.full(dim,.7); e=np.full(dim,.1); q=e*p+(1-e)*h
            obs=enrich_observation([0.]*14,.1,.5,prefix,t,4,[1,1] if dim==2 else [1,0])
            step=dict(time_s=t,observation=obs,actor_ids=list(range(1,dim+1)),factor_mask=[1]*dim,
                p_factors=p.tolist(),h_factors=h.tolist(),q_factors=q.tolist(),generation_epsilon=e.tolist())
            prefix+=log_weight(step,e); steps.append(step)
        sequences.append(dict(seed=1000+i,draw_index=i,case_id=str(k),source_seed=law[k]['source_seed'],steps=steps,
            event_result=i%3==0,state_target_mass=.5,state_proposal_mass=.5,state_log_weight=0.,state_likelihood_ratio=1.,
            generation_log_weight=prefix,action_log_weight=prefix))
    return dict(contract=contract.CONTRACT,observation_contract=FIXED_ALPHA_OBSERVATION,action_dimension=dim,
        intervention_mode=mode,sampling_scope=contract.SCOPE,natural_target_sha256='P',experiment_target_sha256='family',
        conditional_target_sha256='target',state_law_sha256=identity(law),proposal_structure_sha256='proposal',
        state_law=law,state_sampling='Q_S_equals_alpha',state_likelihood_ratio=1.,state_model=None,
        horizon_s=4,state_schedule_seed=37,continuation_seed_start=1000,sequences=sequences)


def write(tmp_path,data):
    save(tmp_path/'sequences.json',data)
    manifest=dict(data_path='sequences.json',sha256=digest(tmp_path/'sequences.json'),
        **{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256',
            'state_law_sha256','proposal_structure_sha256','intervention_mode')})
    save(tmp_path/'manifest.json',manifest);return tmp_path/'manifest.json'


@pytest.mark.parametrize('mode',['single','dual'])
def test_fixed_alpha_multistep_replay_has_only_action_and_replay_correction(tmp_path,mode):
    data=dataset(mode); env=V46SequenceEnv(dict(sequence_manifest=write(tmp_path,data),replay_positive_fraction=.5))
    assert env.dimension==(1 if mode=='single' else 2)
    assert env.empirical_objective(lambda obs:[.1]*env.dimension)==pytest.approx(1/3)
    seen_positive=seen_negative=False
    for _ in range(50):
        env.reset(); done=False
        while not done: _,reward,done,_=env.step([.1]*env.dimension)
        if env.sequence['event_result']:
            seen_positive=True; assert reward==pytest.approx(-2/3)
        else:seen_negative=True; assert reward==0.
    assert seen_positive and seen_negative
    assert env.objective_standard_error(env.objective_values(lambda obs:[.1]*env.dimension))==pytest.approx(np.std([1,0,0]*4,ddof=1)/np.sqrt(12))


@pytest.mark.parametrize('mutation',['q','state_lr','state_log','state_model','mass','schedule','case','total','hash'])
def test_fixed_state_likelihood_is_enforced(tmp_path,mutation):
    d=dataset(); s=d['sequences'][0]
    if mutation=='q':s['state_proposal_mass']=.8
    elif mutation=='state_lr':s['state_likelihood_ratio']=1.01
    elif mutation=='state_log':s['state_log_weight']=.01
    elif mutation=='state_model':d['state_model']='old_lookup'
    elif mutation=='mass':s['target_mass']=1/12
    elif mutation=='schedule':d['state_schedule_seed']=38
    elif mutation=='case':s['case_id']='wrong'
    elif mutation=='total':s['generation_log_weight']+=1
    else:d['state_law_sha256']='changed'
    with pytest.raises(ValueError):V46SequenceEnv(dict(sequence_manifest=write(tmp_path,d)))


def test_source_law_retains_unobserved_states_and_split_guard():
    d=dataset(n=1); v=copy.deepcopy(d)
    for s in v['sequences']:s['seed']+=1000; s['source_seed']+=1000
    # Bank membership, not just sampled sequences, defines source leakage.
    with pytest.raises(ValueError,match='source episodes'):independent_split(d,v)
    for s in v['state_law']:s['source_seed']+=1000
    independent_split(d,v)
    with pytest.raises(ValueError,match='uniform source'):
        contract.state_schedule([dict(case_id='0',source_seed=1,alpha=.8),dict(case_id='1',source_seed=2,alpha=.2)],10,1)


class FixedRng:
    def __init__(self,values):self.values=iter(values)
    def random(self):return next(self.values)


@pytest.mark.parametrize('actor',[1,2])
def test_single_sampling_preserves_correlated_partner_and_exact_IS(actor):
    p=np.array([[.15,.25],[.1,.5]]); units=[dict(ids=[1,2],p=p)]
    marginal=native.single_marginal(units,actor); h=np.array([.8,.2]); eps=.1; q=eps*marginal+(1-eps)*h
    weighted=0.; mean_weight=0.
    for a in range(2):
        for b in range(2):
            chosen=a if actor==1 else b; other=b if actor==1 else a
            conditional=(p[chosen]/marginal[chosen]) if actor==1 else (p[:,chosen]/marginal[chosen])
            u=float(q[:chosen].sum()+q[chosen]/2); v=float(conditional[:other].sum()+conditional[other]/2)
            rng=FixedRng([u,.5 if actor==1 else v,v if actor==1 else .5])
            actions,log=native.draw_single(units,actor,h,[eps],rng)
            assert actions=={1:a,2:b}
            full_q=q[chosen]*conditional[other]
            assert np.exp(log['full_log_q'])==pytest.approx(full_q)
            assert np.exp(log['full_log_p'])==pytest.approx(p[a,b])
            w=np.exp(log['log_weight']); mean_weight+=full_q*w
            weighted+=full_q*w*(a==b)
    assert mean_weight==pytest.approx(1.) and weighted==pytest.approx(np.trace(p))


def road_fixture():
    cars=[SimpleNamespace(id=i,x=x,v=v,lane=1,y=4.,length=5.,width=2.,target=-1,elapsed=0.)
        for i,x,v in [(0,100.,29.),(1,87.,31.),(2,117.,27.),(3,128.,26.)]]
    model=SimpleNamespace(a=np.linspace(-4,2,31),config={'speed_grid':[20.,40.]},summary={'idm':{'parameters':[1.5,35.,4.,2.,2.,.5]}})
    physical={'cav':dict(accel=2.,decel=4.,emergency_decel=9.,tau=.2,min_gap=2.,desired_speed=40.)}
    return SimpleNamespace(vehicles=cars,cav=cars[0],model=model,config=physical),{c.id:np.full(33,1/33) for c in cars}


@pytest.mark.parametrize('horizon',[1.,2.,4.])
def test_fast_single_proxy_matches_existing_reactive_proxy(horizon):
    road,pmfs=road_fixture(); cfg=dict(step_s=.2,horizon_s=horizon,clearance_scale_m=2.)
    for actor in (1,2):
        assert np.allclose(native.single_risk(road,actor,pmfs,cfg),response_risk(road,[actor],pmfs,cfg,reactive=True)[:,0],rtol=1e-13,atol=1e-14)


def test_single_teacher_has_supported_mass_even_at_zero_risk():
    p=np.array([.2,.3,0.,.5]); r=np.array([0.,.4,1.,.9]); h=native.teacher(p,r,[0,.2,.4,.6,.8,1])
    assert h==pytest.approx([1/3,1/3,0.,1/3])


def test_independent_single_selector_can_choose_actor_outside_dual_pair(monkeypatch):
    road,pmfs=road_fixture(); state=dict(cav_id=0,vehicles=[vars(c) for c in road.vehicles],
        pmfs={str(k):v.tolist() for k,v in pmfs.items()},units=[dict(ids=[i],p=pmfs[i].tolist()) for i in (1,2,3)])
    monkeypatch.setattr(native,'single_risk',lambda road,actor,pmfs,cfg:np.full(33,actor/4))
    chosen,audit=native.choose_single(state,road.model,road.config,{},[0,.2,.4,.6,.8,1])
    assert chosen['ids']==[3] and len(audit)==3  # independent of a dual pair [1,2]


def test_formal_training_protocol_fixes_budget_and_no_state_fit():
    cfg=read(runner.DEFAULT); training=read(cfg['training_config'])
    assert cfg['state_model'] is None and cfg['state_likelihood_ratio']==1.
    assert cfg['modes']==['single','dual'] and cfg['training_seeds']==[7,17,29]
    assert cfg['native_draws_per_mode']==4096 and training['iterations']==100
    assert training['checkpoint_selection']=='fixed_iteration' and not training['require_validation']


def test_replay_objective_matches_finite_two_step_second_moment():
    # Full trajectory density factors multiply across decisions; state LR=1.
    p=np.array([.4,.6]); h=np.array([.8,.2]); qb=.1*p+.9*h; qt=.7*p+.3*h
    estimate=truth=0.
    for a in range(2):
        for b in range(2):
            y=float(a==b); pb=p[a]*p[b]; behavior=qb[a]*qb[b]; candidate=qt[a]*qt[b]
            estimate+=behavior*y*(pb/behavior)*(pb/candidate)
            truth+=y*pb*pb/candidate
    assert estimate==pytest.approx(truth)
