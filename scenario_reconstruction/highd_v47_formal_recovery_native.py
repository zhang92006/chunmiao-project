"""Training behavior exploration; the original evaluated target H is untouched."""
from contextlib import contextmanager
import time
import numpy as np

from . import highd_v47_formal_native as native
from .highd_v46_cross_unit import draw_intervention, proposal_factors, natural_log_probability
from .highd_v47_fixed_alpha_native import draw_single, single_marginal
from .highd_v46_contract import enrich_observation


def supported_uniform(p):
    p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1.):
        raise ValueError('Invalid natural action distribution')
    support=p>0
    return support.astype(float)/support.sum()


def act(road, units, actors, pmfs, mode, policy=None, epsilon=.1):
    if policy is not None or epsilon!=.1 or mode not in ('single','dual'):
        raise ValueError('Recovery behavior is fixed and training-only')
    begun=time.process_time();selected=native.common_choice(road,units,actors,pmfs,mode)
    if selected is None:
        actions=native.conditional_actions(units,{},road.rng)
        full=natural_log_probability(units,actions)
        return actions,None,dict(log_p=full,log_q=full,decision_cpu_s=time.process_time()-begun)
    ids=selected['ids'];dim=1 if mode=='single' else 2;mask=[1,int(len(ids)==2)]
    pair=[actors[i] for i in ids]
    obs=enrich_observation(native.observation(road,ids),selected['mass'],selected['relative_contrast'],
        road.log_weight,road.time,road.config['horizon_s'],mask,
        pair[1].x-pair[1].length-pair[0].x if len(pair)==2 else 0.,
        pair[1].v-pair[0].v if len(pair)==2 else 0.)
    eps=np.full(dim,epsilon);h=selected['h']
    if len(ids)==2:
        p=selected['unit']['p'];g=supported_uniform(p)
        actions,logged=draw_intervention(units,ids,g,eps,road.rng)
        p1,p2,h1,h2,_,_=proposal_factors(p,h,eps)
        _,_,g1,g2,q1,q2=proposal_factors(p,g,eps)
        a,b=logged['action_indices']
        target_h=[float(h1[a]),float(h2[a,b])]
        distributions=[dict(p=p1.tolist(),h=h1.tolist()),dict(p=p2[a].tolist(),h=h2[a].tolist())]
        generation=[dict(p=p1.tolist(),h=g1.tolist()),dict(p=p2[a].tolist(),h=g2[a].tolist())]
        tv=[float(.5*np.abs(q1-p1).sum()),float(np.sum(q1*.5*np.abs(q2-p2).sum(axis=1)))]
    else:
        p=single_marginal(units,ids[0]);g=supported_uniform(p)
        actions,logged=draw_single(units,ids[0],g,eps[:1],road.rng)
        a=logged['action_indices'][0];target_h=[float(h[a])]
        distributions=[dict(p=p.tolist(),h=h.tolist())];generation=[dict(p=p.tolist(),h=g.tolist())]
        tv=[float(.5*np.abs(epsilon*p+(1-epsilon)*g-p).sum()),0.]
        if dim==2:
            for key in ('p_factors','h_factors','q_factors'):logged[key].append(1.)
            logged['generation_epsilon']=eps.tolist();target_h.append(1.)
            distributions.append(dict(p=[1.],h=[1.]));generation.append(dict(p=[1.],h=[1.]))
    # q_factors/full_log_q describe the actual behavior, never the target H.
    logged['generation_h_factors']=logged['h_factors'];logged['h_factors']=target_h
    cpu=time.process_time()-begun
    step=dict(logged,time_s=float(road.time),observation=obs,actor_ids=ids,factor_mask=mask[:dim],
        critical_mass=selected['mass'],relative_action_contrast=selected['relative_contrast'],
        factor_pmfs=distributions,generation_factor_pmfs=generation,
        actual_QP_TV=tv,decision_cpu_s=cpu)
    road.log_weight+=logged['log_weight'];road.steps.append(step)
    return actions,step,dict(log_p=logged['full_log_p'],log_q=logged['full_log_q'],decision_cpu_s=cpu)


@contextmanager
def behavior_adapter():
    original=native.act;native.act=act
    try:yield
    finally:native.act=original


def rollout(model,assets,physical,mode,seed,case):
    with behavior_adapter():return native.rollout(model,assets,physical,mode,seed,case)
