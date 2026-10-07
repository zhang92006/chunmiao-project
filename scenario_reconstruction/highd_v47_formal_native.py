"""Original Single critical states, matched action-only agents, frozen native P."""
import time
import numpy as np

from .highd_v46_d2rl import NativeExperiment, observation
from .highd_v46_contract import enrich_observation
from .highd_v46_cross_unit import draw_intervention, natural_log_probability, proposal_factors
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state
from .highd_v47_fixed_alpha_native import draw_single, single_marginal


def common_choice(road, units, actors, pmfs, mode):
    # Calling the ORIGINAL method is essential: dual eligibility cannot filter s.
    single=NativeExperiment.choose(road,units,actors,pmfs,'single')
    if single is None or mode=='single':return single
    dual=NativeExperiment.choose(road,units,actors,pmfs,'dual')
    return dual if dual is not None and len(dual['ids'])==2 else single


def act(road, units, actors, pmfs, mode, policy, epsilon=.1):
    begun=time.process_time()
    selected=None if mode=='natural' else common_choice(road,units,actors,pmfs,mode)
    if selected is None:
        actions=conditional_actions(units,{},road.rng)
        full=natural_log_probability(units,actions)
        return actions,None,dict(log_p=full,log_q=full,decision_cpu_s=time.process_time()-begun)
    ids=selected['ids']; dimension=1 if mode=='single' else 2; mask=[1,int(len(ids)==2)]
    pair=[actors[i] for i in ids]
    obs=enrich_observation(observation(road,ids),selected['mass'],selected['relative_contrast'],
        road.log_weight,road.time,road.config['horizon_s'],mask,
        pair[1].x-pair[1].length-pair[0].x if len(pair)==2 else 0.,
        pair[1].v-pair[0].v if len(pair)==2 else 0.)
    eps=np.asarray(policy(obs) if policy else [epsilon]*dimension,float)
    if eps.shape!=(dimension,) or not np.isfinite(eps).all() or np.any(eps<.05) or np.any(eps>1):
        raise ValueError('Invalid formal policy epsilon')
    h=selected['h']; distributions=[]
    if len(ids)==2:
        actions,logged=draw_intervention(units,ids,h,eps,road.rng)
        p=selected['unit']['p']; p1,p2,h1,h2,q1,q2=proposal_factors(p,h,eps)
        a=logged['action_indices'][0]
        distributions=[dict(p=p1.tolist(),h=h1.tolist()),dict(p=p2[a].tolist(),h=h2[a].tolist())]
        tv=[float(.5*np.abs(q1-p1).sum()),float(np.sum(q1*.5*np.abs(q2-p2).sum(axis=1)))]
    else:
        actions,logged=draw_single(units,ids[0],h,eps[:1],road.rng)
        p=single_marginal(units,ids[0]); distributions=[dict(p=p.tolist(),h=h.tolist())]
        tv=[float(.5*np.abs(eps[0]*p+(1-eps[0])*h-p).sum()),0.]
        if dimension==2:
            for key in ('p_factors','h_factors','q_factors'):logged[key].append(1.)
            logged['generation_epsilon']=eps.tolist()
            distributions.append(dict(p=[1.],h=[1.]))
    cpu=time.process_time()-begun
    step=dict(logged,time_s=float(road.time),observation=obs,actor_ids=ids,factor_mask=mask[:dimension],
        critical_mass=selected['mass'],relative_action_contrast=selected['relative_contrast'],
        factor_pmfs=distributions,actual_QP_TV=tv,decision_cpu_s=cpu)
    road.log_weight+=logged['log_weight'];road.steps.append(step)
    return actions,step,dict(log_p=logged['full_log_p'],log_q=logged['full_log_q'],decision_cpu_s=cpu)


def advance_prepared(road, pmfs, actions):
    road.action_pmfs=lambda:pmfs;road.sample=lambda *args:actions
    try:road.step_policy('natural',1.)
    finally:del road.action_pmfs;del road.sample


def rollout(model, assets, physical, mode, seed, case=None, policy=None):
    begun=time.process_time(); wall=time.perf_counter()
    if case is None:
        road=NativeExperiment(model,assets/'pair',assets/'lateral',seed,physical)
        initial=0.; end=physical['horizon_s']; prepared=None
    else:
        road=restore_prepared_state(model,assets,physical,case['state'],seed)
        road.last_acceleration={int(k):v for k,v in case['state']['last_acceleration'].items()}
        road.cav_acceleration=case['state']['cav_acceleration']
        initial=road.time; end=initial+case['horizon_s']; prepared=case['state']
    fleet=[vars(c).copy() for c in road.vehicles]; decisions=[]
    while not road.crashes and road.time<end-1e-9:
        if prepared is not None:
            pmfs={int(k):np.asarray(v) for k,v in prepared['pmfs'].items()}
            units=[dict(ids=u['ids'],p=np.asarray(u['p'])) for u in prepared['units']];prepared=None
            actors={c.id:c for c in road.vehicles}
        else:pmfs=road.action_pmfs();units,actors=road.prepare(pmfs)
        actions,step,audit=act(road,units,actors,pmfs,mode,policy)
        decisions.append(dict(audit,time_s=road.time,actions={str(k):int(v) for k,v in actions.items()}))
        advance_prepared(road,pmfs,actions)
    event=any(road.cav.id in pair for pair in road.crashes)
    result=dict(seed=seed,source_seed=case['source_seed'] if case else seed,event_result=event,
        event_type='cav_first_collision' if event else 'bv_only_competing_collision' if road.crashes else 'horizon_no_collision',
        generation_log_weight=float(road.log_weight),action_log_weight=float(road.log_weight),state_log_weight=0.,
        state_likelihood_ratio=1.,initial_fleet=fleet,cav_id=road.cav.id,collision_pairs=road.crashes,
        elapsed_s=road.time-initial,steps=road.steps,native_decisions=decisions,
        cpu_s=time.process_time()-begun,wall_time_s=time.perf_counter()-wall)
    if case:
        result.update(case_id=case['case_id'],state_target_mass=case['alpha'],state_proposal_mass=case['alpha'])
        for step in result['steps']:step['source_time_s']=step['time_s'];step['time_s']-=initial
    return result
