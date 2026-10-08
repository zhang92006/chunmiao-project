"""Real target-policy continuations matching shared-teacher training semantics."""
import time
import numpy as np
from . import highd_v47_shared_teacher_native as target
from .highd_v46_pair_witness import restore_prepared_state,conditional_actions
from .highd_v46_cross_unit import draw_intervention,natural_log_probability,proposal_factors
from .highd_v47_fixed_alpha_native import draw_single
from .highd_v47_formal_native import advance_prepared,common_choice


def act(road,units,actors,pmfs,mode,policy,start):
    begun=time.process_time()
    choice=None if mode=='natural' else common_choice(road,units,actors,pmfs,mode)
    description=target.describe(road,units,choice,mode,start,4) if choice else None
    if description is None:
        actions=conditional_actions(units,{},road.rng);full=natural_log_probability(units,actions)
        return actions,None,dict(log_p=full,log_q=full,decision_cpu_s=time.process_time()-begun)
    obs=list(description['observation']);obs[16]=float(np.clip(road.log_weight/30.,-5,5))
    eps=np.asarray(policy(obs),float);dim=description['dimension']
    if eps.shape!=(dim,) or not np.isfinite(eps).all() or np.any(eps<.05) or np.any(eps>1):
        raise ValueError('Invalid target epsilon')
    ids=description['actor_ids'];p=description['P'];h=description['H']
    if len(ids)==2:
        actions,logged=draw_intervention(units,ids,h,eps,road.rng)
        p1,p2,h1,h2,q1,q2=proposal_factors(p,h,eps);a=logged['action_indices'][0]
        pmf=[dict(p=p1.tolist(),h=h1.tolist()),dict(p=p2[a].tolist(),h=h2[a].tolist())]
        tv=[float(.5*np.abs(q1-p1).sum()),float(np.sum(q1*.5*np.abs(q2-p2).sum(axis=1)))]
    else:
        actions,logged=draw_single(units,ids[0],h,eps[:1],road.rng)
        pmf=[dict(p=p.tolist(),h=h.tolist())];tv=[float(.5*np.abs(eps[0]*p+(1-eps[0])*h-p).sum()),0.]
        if dim==2:
            for key in ('p_factors','h_factors','q_factors'):logged[key].append(1.)
            logged['generation_epsilon']=eps.tolist();pmf.append(dict(p=[1.],h=[1.]))
    step=dict(logged,observation=obs,time_s=road.time-start,source_time_s=road.time,
        actor_ids=ids,factor_mask=description['factor_mask'],factor_pmfs=pmf,actual_QP_TV=tv,
        decision_cpu_s=time.process_time()-begun)
    road.log_weight+=logged['log_weight']
    return actions,step,dict(log_p=logged['full_log_p'],log_q=logged['full_log_q'],decision_cpu_s=step['decision_cpu_s'])


def rollout(model,assets,physical,mode,seed,case,policy):
    begun=time.process_time();wall=time.perf_counter();state=case['state'];start=state['time_s']
    # Exactly the training snapshot restoration; do not inject formal-native's
    # extra acceleration-history restoration or global-time/60s observation.
    road=restore_prepared_state(model,assets,physical,state,seed)
    initial=[vars(c).copy() for c in road.vehicles];steps=[];decisions=[];first=True
    while not road.crashes and road.time<start+4-1e-9:
        if first:
            pmfs={int(k):np.asarray(v) for k,v in state['pmfs'].items()}
            units=[dict(ids=u['ids'],p=np.asarray(u['p'])) for u in state['units']]
            actors={c.id:c for c in road.vehicles};first=False
        else:pmfs=road.action_pmfs();units,actors=road.prepare(pmfs)
        actions,step,audit=act(road,units,actors,pmfs,mode,policy,start)
        if step:steps.append(step)
        decisions.append(dict(audit,time_s=road.time-start,actions={str(k):int(v) for k,v in actions.items()}))
        advance_prepared(road,pmfs,actions)
    event=any(road.cav.id in pair for pair in road.crashes)
    return dict(seed=seed,source_seed=case['source_seed'],case_id=case['case_id'],event_result=event,
        event_type='cav_first_collision' if event else 'bv_only_competing_collision' if road.crashes else 'horizon_no_collision',
        generation_log_weight=float(road.log_weight),action_log_weight=float(road.log_weight),state_log_weight=0.,
        state_likelihood_ratio=1.,state_target_mass=case['alpha'],state_proposal_mass=case['alpha'],
        initial_fleet=initial,cav_id=road.cav.id,collision_pairs=road.crashes,elapsed_s=road.time-start,
        steps=steps,native_decisions=decisions,cpu_s=time.process_time()-begun,wall_time_s=time.perf_counter()-wall)
