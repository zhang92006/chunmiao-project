"""Training-only natural-state pool and supported pair-risk counterfactual audit.

This module does not change the frozen NDE, the online sampler or PPO. Its
alternative response model is a surrogate, not a natural transition kernel.
"""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

from .highd_v46_assets import ensure_assets, read, save, digest
from .highd_v46_d2rl import NativeExperiment, challenge, factor_influence, idm_acceleration
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_paper_data import idm


def pair_marginal(units, ids):
    """Exact pair marginal of the current product of disjoint natural units."""
    found=[]
    for key in ids:
        matches=[u for u in units if key in u['ids']]
        if len(matches)!=1:
            raise ValueError('Actor must occur in exactly one natural unit')
        found.append(matches[0])
    if found[0] is found[1]:
        p=np.asarray(found[0]['p'],float)
        return (p if found[0]['ids']==list(ids) else p.T),True
    marginals=[]
    for key,unit in zip(ids,found):
        p=np.asarray(unit['p'],float)
        marginals.append(p if p.ndim==1 else p.sum(axis=1 if unit['ids'][0]==key else 0))
    return np.outer(*marginals),False


def pair_metrics(p, risk, thresholds):
    """Conditional leverage plus a joint-mode-anchored interaction contrast.

    R(a,b)-E_P[R(a,B)|a] averages the natural second action, rather than using
    one arbitrarily drawn action. Four-corner contrasts additionally remove
    single-actor effects; all four corners must be in natural support.
    """
    p=np.asarray(p,float);r=np.asarray(risk,float)
    if (p.ndim!=2 or p.shape!=r.shape or not np.isfinite(p).all()
            or not np.isfinite(r).all() or np.any(p<0) or not np.isclose(p.sum(),1.)
            or np.any(r<0) or np.any(r>1)):
        raise ValueError('Expected normalized joint P and finite [0,1] risk matrix')
    support=p>0;p1=p.sum(1);p2=p.sum(0)
    conditional_j=np.divide((p*r).sum(1),p1,out=np.zeros_like(p1),where=p1>0)
    conditional_i=np.divide((p*r).sum(0),p2,out=np.zeros_like(p2),where=p2>0)
    delta_j=r-conditional_j[:,None];delta_i=r-conditional_i[None,:]
    a0,b0=np.unravel_index(np.argmax(p),p.shape);base=float(r[a0,b0])
    corners=support & support[:,b0,None] & support[a0,None,:]
    interaction=r-r[:,b0,None]-r[a0,None,:]+base
    positive=np.where(corners,np.maximum(interaction,0.),0.)
    ci=max(0.,float(np.max(r[:,b0][support[:,b0]]))-base)
    cj=max(0.,float(np.max(r[a0,:][support[a0,:]]))-base)
    cij=max(0.,float(np.max(r[support]))-base)
    mass=float((p*r).sum());h=p*r/mass if mass>0 else p.copy()
    tv=factor_influence(p,h)
    values={'natural_risk_mass':mass,'baseline_actions':[int(a0),int(b0)],
        'baseline_risk':base,'C_i_increment':ci,'C_j_increment':cj,'C_ij_increment':cij,
        'joint_excess_over_single_sum':cij-ci-cj,
        'max_delta_second':float(np.max(delta_j[support])),
        'max_delta_first':float(np.max(delta_i[support])),
        'max_positive_interaction':float(positive.max()),
        'positive_interaction_mass':float((p*positive).sum()),
        'max_interaction_actions':[int(i) for i in np.unravel_index(np.argmax(positive),p.shape)],
        'max_delta_second_actions':[int(i) for i in np.unravel_index(np.argmax(np.where(support,delta_j,-np.inf)),p.shape)],
        'four_corner_support_mass':float(p[corners].sum()),
        'factor_tv':tv,'supported_actions':int(support.sum())}
    values['synergy_eligible']=bool(
        min(values['max_delta_first'],values['max_delta_second'])>=thresholds['minimum_delta']
        and values['max_positive_interaction']>=thresholds['minimum_interaction']
        and values['joint_excess_over_single_sum']>=thresholds['minimum_joint_excess']
        and min(tv)>=thresholds['minimum_factor_tv']
        and values['positive_interaction_mass']>=thresholds['minimum_synergy_mass'])
    return values


def select_pair(rows):
    eligible=[r for r in rows if r['metrics']['synergy_eligible']]
    return max(eligible,key=lambda r:(r['metrics']['positive_interaction_mass'],
                                      r['metrics']['natural_risk_mass'],tuple(r['actor_ids']))) if eligible else None


def bounded_motion(x,v,acc,dt,lower,upper):
    next_v=np.clip(v+acc*dt,lower,upper)
    t=np.full_like(v,dt)
    bounded=np.abs(next_v-(v+acc*dt))>1e-10
    np.divide(next_v-v,acc,out=t,where=bounded & (acc!=0))
    return x+v*t+.5*acc*t*t+next_v*(dt-t),next_v


def response_risk(road, ids, pmfs, cfg, reactive=True):
    """Enumerate 33x33 first-second actions, then mean or fitted-IDM responses.

    All candidate actions last one second. No later BV lane changes are
    forecast. The reactive variant updates longitudinal responses to current
    neighbors; it is deliberately distinct from the frozen stochastic NDE.
    """
    dt=float(cfg['step_s']);horizon=float(cfg['horizon_s'])
    if dt<=0 or horizon<=0 or not np.isclose(1/dt,round(1/dt)) or not np.isclose(horizon/dt,round(horizon/dt)):
        raise ValueError('Proxy step must divide one second and the horizon')
    cars=road.vehicles;ci=cars.index(road.cav);n=1089;m=len(cars)
    x=np.broadcast_to([c.x for c in cars],(n,m)).copy()
    v=np.broadcast_to([c.v for c in cars],(n,m)).copy()
    y0=np.broadcast_to([c.y for c in cars],(n,m)).copy();target=y0.copy()
    lengths=np.array([c.length for c in cars]);widths=np.array([c.width for c in cars])
    av=np.r_[0.,road.model.a,0.]
    means=np.broadcast_to([pmfs[c.id]@av for c in cars],(n,m)).copy()
    held=means.copy()
    for key,codes in zip(ids,(np.repeat(np.arange(33),33),np.tile(np.arange(33),33))):
        j=next(k for k,c in enumerate(cars) if c.id==key)
        held[:,j]=av[codes];target[:,j]+=np.where(codes==0,4.,np.where(codes==32,-4.,0.))
    lower=np.full(m,road.model.config['speed_grid'][0]);lower[ci]=0.
    upper=np.full(m,road.model.config['speed_grid'][1]);upper[ci]=40.
    alive=np.ones(n,bool);risk=np.zeros(n);diagonal=np.eye(m,dtype=bool)[None,:,:]
    for tick in range(round(horizon/dt)):
        y=y0+(target-y0)*min(tick*dt,1.)
        dx=x[:,None,:]-x[:,:,None]
        lateral=np.abs(y[:,None,:]-y[:,:,None])<(widths[:,None]+widths[None,:])/2
        gaps=np.where((dx>0)&lateral&~diagonal,dx-lengths[None,None,:],np.inf)
        lead=np.argmin(gaps,axis=2);gap=np.min(gaps,axis=2)
        lead_v=np.take_along_axis(v,lead,axis=1)
        if tick*dt<1.-1e-9:
            acc=held.copy()
        elif reactive:
            acc=np.clip(idm(road.model.summary['idm']['parameters'],v,gap,lead_v-v),road.model.a[0],road.model.a[-1])
        else:
            acc=means.copy()
        acc[:,ci]=idm_acceleration(v[:,ci],gap[:,ci],lead_v[:,ci],road.config['cav'])
        nx,nv=bounded_motion(x,v,acc,dt,lower,upper)
        x=np.where(alive[:,None],nx,x);v=np.where(alive[:,None],nv,v)
        y=y0+(target-y0)*min((tick+1)*dt,1.)
        center=x-lengths/2
        clearance=np.abs(center[:,None,:]-center[:,:,None])-(lengths[:,None]+lengths[None,:])/2
        lateral=np.abs(y[:,None,:]-y[:,:,None])<(widths[:,None]+widths[None,:])/2
        close=lateral&~diagonal
        cav_gap=np.min(np.where(close[:,ci,:],clearance[:,ci,:],np.inf),axis=1)
        score=np.exp(-np.maximum(cav_gap,0.)/cfg['clearance_scale_m'])
        risk=np.maximum(risk,np.where(alive,score,0.))
        collision_lateral=np.abs(y[:,None,:]-y[:,:,None])<(widths[:,None]+widths[None,:])/2-1e-9
        alive &= ~np.any(collision_lateral&~diagonal&(clearance < -1e-9),axis=(1,2))
        if not alive.any():break
    return risk.reshape(33,33)


def state_features(road,pmfs,cfg):
    cav=road.cav;gaps=[];ttcs=[];closing_near=False;cut_in=False
    for c in road.vehicles:
        if c.id==cav.id:continue
        dx=(c.x-c.length/2)-(cav.x-cav.length/2)
        gap=abs(dx)-(c.length+cav.length)/2
        closing=np.sign(dx)*(cav.v-c.v)
        if c.lane==cav.lane:
            gaps.append(gap)
            if closing>0 and gap>0:ttcs.append(gap/closing)
            closing_near |= closing>=cfg['closing_rate_mps'] and gap<=cfg['closing_gap_m']
        elif abs(c.lane-cav.lane)==1 and abs(dx)<=cfg['cut_in_distance_m']:
            code=0 if cav.lane>c.lane else 32
            cut_in |= pmfs[c.id][code]>=cfg['cut_in_probability']
    reasons=[]
    if gaps and min(gaps)<=cfg['gap_m']:reasons.append('small_gap')
    if ttcs and min(ttcs)<=cfg['ttc_s']:reasons.append('low_ttc')
    if closing_near:reasons.append('closing')
    if cut_in:reasons.append('cut_in_opportunity')
    return {'minimum_gap_m':min(gaps) if gaps else None,'minimum_ttc_s':min(ttcs) if ttcs else None,
            'reasons':reasons,'stratum':'high_risk' if reasons else 'control'}


def snapshot(road,units,pmfs,seed,features):
    return {'contract':'v46_natural_predraw_diagnostic_state_v1','seed':seed,'time_s':road.time,
        'phase':'after_prepare_before_action_draw','cav_id':road.cav.id,
        'vehicles':[vars(c).copy() for c in road.vehicles],
        'units':[{'ids':list(u['ids']),'p':np.asarray(u['p']).tolist()} for u in units],
        'pmfs':{str(k):p.tolist() for k,p in pmfs.items()},
        'natural_histories':{str(k):list(v) for k,v in road.natural.histories.items()},
        'natural_previous':[list(p) for p in sorted(road.natural.previous)],
        'rng_state':road.rng.bit_generator.state,'features':features}


def candidate_pairs(road,units,cfg):
    near=sorted((abs(c.x-road.cav.x),c.id) for c in road.vehicles
                if c.id!=road.cav.id and abs(c.x-road.cav.x)<=cfg['observation_m'])
    anchors={key for _,key in near[:cfg['nearest_actors']]}
    pairs={tuple(sorted(p)) for p in combinations(sorted(anchors),2)}
    # Include old natural candidates even when their partner is outside top k.
    pairs.update(tuple(sorted(u['ids'])) for u in units if len(u['ids'])==2 and anchors.intersection(u['ids']))
    return sorted(pairs)


def run(config_path,output):
    cfg=read(config_path);exp=read(cfg['experiment_config']);assets,lock=ensure_assets()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    pools={'high_risk':[],'control':[]};counts=Counter();events=Counter()
    rng=np.random.default_rng(cfg['pool']['reservoir_seed']);capacity=cfg['pool']['capacity_per_stratum']
    if capacity<1 or cfg['rollouts']['episodes']<1 or cfg['rollouts']['snapshot_stride_s']<1:
        raise ValueError('Positive pool, rollout and integer stride sizes required')
    started=time.perf_counter()

    class PoolRoad(NativeExperiment):
        def choose(self,units,actors,pmfs,mode):
            if mode!='natural':raise ValueError('Diagnostic pool requires natural rollouts')
            if round(self.time)%cfg['rollouts']['snapshot_stride_s']:return None
            features=state_features(self,pmfs,cfg['pool']);key=features['stratum'];counts[key]+=1
            slot=len(pools[key]) if len(pools[key])<capacity else int(rng.integers(counts[key]))
            if slot<capacity:
                item=snapshot(self,units,pmfs,seed,features)
                if slot==len(pools[key]):pools[key].append(item)
                else:pools[key][slot]=item
            return None

    for seed in range(cfg['rollouts']['seed_start'],cfg['rollouts']['seed_start']+cfg['rollouts']['episodes']):
        road=PoolRoad(model,assets/'pair',assets/'lateral',seed,exp)
        while road.time<exp['horizon_s']-1e-9:
            if not road.step_policy('natural',1.):break
        event='cav_first_collision' if any(road.cav.id in p for p in road.crashes) else 'bv_only' if road.crashes else 'horizon'
        events[event]+=1
    states=sorted(pools['high_risk']+pools['control'],key=lambda s:(s['seed'],s['time_s']))
    save(output/'state_pool.json',{'scope':'Training diagnostic only; not a population-risk or PPO sequence dataset',
        'config':cfg,'natural_target_sha256':lock['natural_target_sha256'],
        'population_stratum_counts':dict(counts),'states':states})
    rows=[];decisions=[]
    for index,state in enumerate(states):
        cars=[SimpleNamespace(**c) for c in state['vehicles']]
        road=SimpleNamespace(vehicles=cars,cav=next(c for c in cars if c.id==state['cav_id']),model=model,config=exp)
        units=[{'ids':u['ids'],'p':np.asarray(u['p'])} for u in state['units']]
        pmfs={int(k):np.asarray(p) for k,p in state['pmfs'].items()}
        state_rows=[]
        for ids in candidate_pairs(road,units,cfg['candidate']):
            p,same=pair_marginal(units,ids)
            for variant in cfg['proxy']['variants']:
                if variant=='legacy_hold':risk=challenge(road,list(ids),pmfs,cfg['proxy'])
                elif variant in ('mean_1s','reactive_1s'):
                    risk=response_risk(road,ids,pmfs,cfg['proxy'],reactive=variant=='reactive_1s')
                else:raise ValueError('Unknown proxy variant: '+variant)
                metrics=pair_metrics(p,risk,cfg['selection'])
                row={'state_index':index,'seed':state['seed'],'time_s':state['time_s'],
                    'stratum':state['features']['stratum'],'actor_ids':list(ids),'same_natural_unit':same,
                    'proxy':variant,'metrics':metrics}
                rows.append(row);state_rows.append(row)
                np.savez_compressed(output/f'matrix_{index:03d}_{ids[0]}_{ids[1]}_{variant}.npz',p=p,risk=risk)
        for variant in cfg['proxy']['variants']:
            candidates=[r for r in state_rows if r['proxy']==variant];chosen=select_pair(candidates)
            decisions.append({'state_index':index,'proxy':variant,'selected_pair':chosen['actor_ids'] if chosen else None,
                'same_natural_unit':chosen['same_natural_unit'] if chosen else None,
                'score':chosen['metrics']['positive_interaction_mass'] if chosen else None})
        print({'states_analyzed':index+1,'total_states':len(states),'pairs':len(candidate_pairs(road,units,cfg['candidate']))},flush=True)
    save(output/'pair_diagnostics.json',{'rows':rows,'selections':decisions})
    summaries=[]
    for variant in cfg['proxy']['variants']:
        for stratum in ('high_risk','control'):
            group=[r for r in rows if r['proxy']==variant and r['stratum']==stratum]
            if not group:continue
            summaries.append({'proxy':variant,'stratum':stratum,'pairs':len(group),
                'same_natural_unit_pairs':sum(r['same_natural_unit'] for r in group),
                'synergy_eligible_pairs':sum(r['metrics']['synergy_eligible'] for r in group),
                'weak_delta_second_fraction':float(np.mean([r['metrics']['max_delta_second']<cfg['selection']['minimum_delta'] for r in group])),
                'weak_second_factor_tv_fraction':float(np.mean([r['metrics']['factor_tv'][1]<cfg['selection']['minimum_factor_tv'] for r in group])),
                'max_joint_excess':max(r['metrics']['joint_excess_over_single_sum'] for r in group),
                'max_interaction':max(r['metrics']['max_positive_interaction'] for r in group)})
    report={'schema_version':1,'config':cfg,'config_sha256':digest(config_path),
        'implementation_sha256':digest(__file__),'collector_sha256':digest(Path(__file__).with_name('highd_v46_d2rl.py')),
        'natural_target_sha256':lock['natural_target_sha256'],'state_pool_sha256':digest(output/'state_pool.json'),
        'pair_diagnostics_sha256':digest(output/'pair_diagnostics.json'),
        'rollout_event_counts':dict(events),'population_stratum_counts':dict(counts),
        'pool_stratum_counts':{k:len(v) for k,v in pools.items()},'summaries':summaries,
        'selected_state_pairs_by_proxy':dict(Counter(d['proxy'] for d in decisions if d['selected_pair'] is not None)),
        'wall_time_s':time.perf_counter()-started,
        'scope':'Surrogate sensitivity and pair selection only. Correlated states/pairs are not independent test episodes. No calibrated crash probability, PPO benefit, natural risk or acceleration claim. Cross-unit deployment needs a full joint conditional sampler and likelihood audit.'}
    save(output/'summary.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/highd_v46_pair_synergy_diagnostic_v1.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();print(run(args.config,args.output))


if __name__=='__main__':main()
