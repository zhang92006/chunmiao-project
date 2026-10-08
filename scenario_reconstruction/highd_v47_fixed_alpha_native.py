"""Real Single-BV / Dual-BV native critical sequences from fixed alpha states."""
import time
from types import SimpleNamespace

import numpy as np

from .highd_paper_data import idm
from .highd_paper_discrete_pairs import inverse_cdf
from .highd_v46_contract import enrich_observation
from .highd_v46_cross_unit import draw_intervention, natural_log_probability
from .highd_v46_d2rl import idm_acceleration, observation
from .highd_v46_pair_synergy import bounded_motion, pair_marginal, response_risk
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state
from .highd_v47_action_teacher import exploration_h


def single_marginal(units, actor):
    matches=[u for u in units if actor in u['ids']]
    if len(matches)!=1:raise ValueError('Single actor must belong to exactly one natural unit')
    unit=matches[0]; p=np.asarray(unit['p'],float)
    return p if len(unit['ids'])==1 else p.sum(axis=1 if actor==unit['ids'][0] else 0)


def single_risk(road, actor, pmfs, cfg):
    """33-cell version of the unchanged reactive response proxy.

Equivalent to response_risk(road, [actor], ... )[:, 0], without replicating
every candidate action 33 times. Candidate actions last the first second.
"""
    dt=float(cfg['step_s']); horizon=float(cfg['horizon_s'])
    if dt<=0 or horizon<=0 or not np.isclose(1/dt,round(1/dt)) or not np.isclose(horizon/dt,round(horizon/dt)):
        raise ValueError('Proxy step must divide one second and horizon')
    cars=road.vehicles; ci=cars.index(road.cav); n=33; m=len(cars)
    x=np.broadcast_to([c.x for c in cars],(n,m)).copy(); v=np.broadcast_to([c.v for c in cars],(n,m)).copy()
    y0=np.broadcast_to([c.y for c in cars],(n,m)).copy(); target=y0.copy()
    lengths=np.array([c.length for c in cars]); widths=np.array([c.width for c in cars])
    av=np.r_[0.,road.model.a,0.]; held=np.broadcast_to([pmfs[c.id]@av for c in cars],(n,m)).copy()
    j=next(k for k,c in enumerate(cars) if c.id==actor); codes=np.arange(33)
    held[:,j]=av[codes]; target[:,j]+=np.where(codes==0,4.,np.where(codes==32,-4.,0.))
    lower=np.full(m,road.model.config['speed_grid'][0]); lower[ci]=0.
    upper=np.full(m,road.model.config['speed_grid'][1]); upper[ci]=40.
    alive=np.ones(n,bool); risk=np.zeros(n); diagonal=np.eye(m,dtype=bool)[None,:,:]
    for tick in range(round(horizon/dt)):
        y=y0+(target-y0)*min(tick*dt,1.); dx=x[:,None,:]-x[:,:,None]
        lateral=np.abs(y[:,None,:]-y[:,:,None])<(widths[:,None]+widths[None,:])/2
        gaps=np.where((dx>0)&lateral&~diagonal,dx-lengths[None,None,:],np.inf)
        lead=np.argmin(gaps,axis=2); gap=np.min(gaps,axis=2); lead_v=np.take_along_axis(v,lead,axis=1)
        acc=held.copy() if tick*dt<1.-1e-9 else np.clip(
            idm(road.model.summary['idm']['parameters'],v,gap,lead_v-v),road.model.a[0],road.model.a[-1])
        acc[:,ci]=idm_acceleration(v[:,ci],gap[:,ci],lead_v[:,ci],road.config['cav'])
        nx,nv=bounded_motion(x,v,acc,dt,lower,upper)
        x=np.where(alive[:,None],nx,x); v=np.where(alive[:,None],nv,v)
        y=y0+(target-y0)*min((tick+1)*dt,1.); center=x-lengths/2
        clearance=np.abs(center[:,None,:]-center[:,:,None])-(lengths[:,None]+lengths[None,:])/2
        lateral=np.abs(y[:,None,:]-y[:,:,None])<(widths[:,None]+widths[None,:])/2
        cav_gap=np.min(np.where((lateral&~diagonal)[:,ci,:],clearance[:,ci,:],np.inf),axis=1)
        score=np.exp(-np.maximum(cav_gap,0.)/cfg['clearance_scale_m'])
        risk=np.maximum(risk,np.where(alive,score,0.))
        collision_lateral=np.abs(y[:,None,:]-y[:,:,None])<(widths[:,None]+widths[None,:])/2-1e-9
        alive &= ~np.any(collision_lateral&~diagonal&(clearance < -1e-9),axis=(1,2))
        if not alive.any():break
    return risk


def teacher(p, risk, edges):
    p,risk=np.asarray(p,float),np.asarray(risk,float)
    if p.ndim==1:return exploration_h(p[:,None],risk[:,None],edges)[0][:,0]
    return exploration_h(p,risk,edges)[0]


def choose_single(state, model, physical, proxy, edges):
    cars=[SimpleNamespace(**v) for v in state['vehicles']]; actors={c.id:c for c in cars}
    road=SimpleNamespace(vehicles=cars,cav=actors[state['cav_id']],model=model,config=physical)
    pmfs={int(k):np.asarray(v) for k,v in state['pmfs'].items()}
    nearby=sorted((abs(c.x-road.cav.x),c.id) for c in cars if c.id!=road.cav.id and abs(c.x-road.cav.x)<=115.)[:8]
    candidates=[]
    for _,actor in nearby:
        p=single_marginal(state['units'],actor); risk=single_risk(road,actor,pmfs,proxy)
        candidates.append(dict(ids=[actor],p=p.tolist(),risk=risk.tolist(),h=teacher(p,risk,edges).tolist(),
            natural_risk_mass=float(p@risk)))
    if not candidates:raise ValueError('Locked selected state has no single candidate')
    return min(candidates,key=lambda c:(-c['natural_risk_mass'],c['ids'][0])),[
        dict(actor_ids=c['ids'],natural_risk_mass=c['natural_risk_mass']) for c in candidates]


def draw_single(units, actor, h, epsilon, rng):
    p=single_marginal(units,actor); h=np.asarray(h,float); e=np.asarray(epsilon,float)
    if (e.shape!=(1,) or not np.isfinite(e).all() or np.any(e<.05) or np.any(e>1)
            or h.shape!=p.shape or not np.isfinite(h).all() or np.any(h<0)
            or not np.isclose(h.sum(),1.) or np.any(h[p==0]>0)):
        raise ValueError('Invalid Single-BV supported proposal')
    q=e[0]*p+(1-e[0])*h; a=inverse_cdf(q,float(rng.random()))
    actions=conditional_actions(units,{actor:a},rng)
    log_p=natural_log_probability(units,actions); conditional=log_p-float(np.log(p[a]))
    log_q=float(np.log(q[a]))+conditional; ratio=float(np.log(p[a]/q[a]))
    if not np.isclose(log_p-log_q,ratio,atol=1e-10,rtol=1e-10):raise ArithmeticError('Single full/factor likelihood mismatch')
    return actions,dict(p_factors=[float(p[a])],h_factors=[float(h[a])],q_factors=[float(q[a])],
        generation_epsilon=e.tolist(),log_weight=ratio,full_log_p=log_p,full_log_q=log_q,
        cancelled_log_conditional=conditional,action_indices=[int(a)])


def rollout(case, mode, seed, model, assets, physical, proxy, cfg, policy=None):
    started=time.perf_counter(); state=case['state']; initial=state['time_s']; horizon=cfg['horizon_s']
    road=restore_prepared_state(model,assets,physical,state,seed)
    chosen=case['choices'][mode]; ids=chosen['ids']; steps=[]; log_weight=0.
    while not road.crashes and road.time<initial+horizon-1e-9:
        elapsed=road.time-initial; first=not steps
        if first:
            pmfs={int(k):np.asarray(v) for k,v in state['pmfs'].items()}; units=state['units']
            p=np.asarray(chosen['p']); risk=np.asarray(chosen['risk']); h=np.asarray(chosen['h'])
        else:
            pmfs=road.action_pmfs(); units,_=road.prepare(pmfs)
            local=dict(proxy,horizon_s=float(horizon-round(elapsed)))
            if mode=='single':p=single_marginal(units,ids[0]); risk=single_risk(road,ids[0],pmfs,local)
            else:p=pair_marginal(units,ids)[0]; risk=response_risk(road,ids,pmfs,local,reactive=True)
            h=teacher(p,risk,cfg['risk_edges'])
        actors={c.id:c for c in road.vehicles}; pair=[actors[i] for i in ids]
        mask=[1,1] if mode=='dual' else [1,0]
        mass=float(np.sum(p*risk)); contrast=float(np.ptp(risk[p>0])/max(float(risk[p>0].max()),1e-30))
        obs=enrich_observation(observation(road,ids),mass,contrast,log_weight,elapsed,horizon,mask,
            pair[1].x-pair[1].length-pair[0].x if len(pair)==2 else 0.,pair[1].v-pair[0].v if len(pair)==2 else 0.)
        epsilon=policy(obs) if policy else [cfg['generation_epsilon']]*len(ids)
        actions,logged=(draw_single(units,ids[0],h,epsilon,road.rng) if mode=='single'
            else draw_intervention(units,ids,h,epsilon,road.rng))
        steps.append(dict(logged,time_s=float(elapsed),source_time_s=float(road.time),observation=obs,
            actor_ids=ids,factor_mask=[1]*len(ids),critical_mass=mass,
            natural_action_indices={str(k):int(v) for k,v in actions.items()},
            natural_unit_ids=[list(u['ids']) for u in units]))
        log_weight+=logged['log_weight']
        # This snapshot/decision has already prepared natural history exactly once.
        road.action_pmfs=lambda:pmfs; road.sample=lambda *args:actions
        road.step_policy('natural',1.)
        del road.action_pmfs; del road.sample
    event=any(road.cav.id in pair for pair in road.crashes)
    return dict(seed=seed,source_seed=state['seed'],case_id=case['case_id'],
        event_result=event,bv_only_collision=bool(road.crashes) and not event,
        event_type='cav_first_collision' if event else 'bv_only_competing_collision' if road.crashes else 'horizon_no_collision',
        elapsed_s=float(road.time-initial),collision_pairs=road.crashes,actor_ids=ids,steps=steps,
        state_target_mass=case['alpha'],state_proposal_mass=case['alpha'],state_log_weight=0.,state_likelihood_ratio=1.,
        action_log_weight=float(log_weight),generation_log_weight=float(log_weight),
        minimum_gap=float(road.minimum_gap) if np.isfinite(road.minimum_gap) else None,
        wall_time_s=time.perf_counter()-started)
