"""Native V46 D2RL collection: one biased BV or one biased following pair.

P is the frozen V46 BV law conditional on a fixed IDM CAV. Q alone changes.
The event is CAV involvement in the first collision before the fixed horizon.
"""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import time
import numpy as np

from .highd_v46_assets import ensure_assets, read, save, digest, ROOT
from .highd_paper_discrete_pairs import PairModelQueries, DiscretePairs, SINGLE_OPTIONS, joint_group, inverse_cdf
from .highd_paper_discrete import DiscreteRoad
from .highd_paper_dynamics import advance, collisions
from .highd_paper_pairs import update_histories, joint33
from .highd_dynamic_pairs import partition, pair_key
from .highd_v46_contract import LEGACY_OBSERVATION, CONTEXT_OBSERVATION, OBSERVATION_DIMENSIONS, enrich_observation


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def experiment_target(lock, config):
    # The natural risk target excludes the proposal's epsilon, risk proxy and
    # observation choices. All comparison arms retain the same physical P.
    return {'schema_version': 2, 'natural_target_sha256': lock['natural_target_sha256'],
            'experiment': {k: config[k] for k in ('horizon_s', 'cav')},
            'cav_role': 'uniform_initialized_actor_with_observed_current_leader',
            'event': 'CAV_in_first_any_actor_collision_by_horizon; BV_only_is_competing_negative',
            'runtime_sha256': digest(__file__)}


def proposal_structure(config, mode):
    return {'mode':mode,'criticality':config['criticality'],
            'observation_contract':config.get('observation_contract',LEGACY_OBSERVATION),
            'dual_single_fallback':config.get('dual_single_fallback',False),
            'epsilon_min':config['epsilon_min']}


def factor_influence(p, h):
    """Total variation of the first and conditional second proposal factors."""
    if p.ndim == 1:
        return [float(.5*np.abs(p-h).sum()), 0.]
    p1, h1 = p.sum(axis=1), h.sum(axis=1)
    first = float(.5*np.abs(p1-h1).sum())
    rows = (p1 > 0) & (h1 > 0)
    second = float(np.sum(h1[rows]*.5*np.abs(h[rows]/h1[rows, None]-p[rows]/p1[rows, None]).sum(axis=1)))
    return [first, second]


def idm_acceleration(speed, gap, lead_speed, cfg):
    desired = cfg['min_gap']+np.maximum(0., speed*cfg['tau']+
              speed*(speed-lead_speed)/(2*np.sqrt(cfg['accel']*cfg['decel'])))
    return np.clip(cfg['accel']*(1-(speed/cfg['desired_speed'])**4-(desired/np.maximum(gap,.01))**2),
                   -cfg['emergency_decel'], cfg['accel'])


def observation(road, ids):
    cav = road.cav
    front = min((c for c in road.vehicles if c.id != cav.id and c.lane == cav.lane and c.x > cav.x),
                key=lambda c:c.x, default=None)
    has = front is not None and front.x-front.length-cav.x <= 115
    result = [cav.v/40, road.cav_acceleration/9, float(cav.lane), float(has),
              (front.x-front.length-cav.x)/115 if has else 1., (front.v-cav.v)/20 if has else 0.]
    actors = {c.id:c for c in road.vehicles}
    for key in ids:
        c = actors[key]
        result.extend([(c.x-cav.x)/115, c.v/40, float(c.lane-cav.lane), road.last_acceleration.get(key,0.)/4])
    if len(ids) == 1:
        result.extend([0.,0.,0.,0.])
    return np.clip(np.asarray(result),-5,5).tolist()


def challenge(road, ids, pmfs, cfg):
    """Pre-draw deterministic continuation proxy, not a learned crash probability."""
    codes = np.arange(33)
    choices = [codes] if len(ids)==1 else [np.repeat(codes,33),np.tile(codes,33)]
    n = len(choices[0]); cars=road.vehicles; ci=cars.index(road.cav)
    x=np.broadcast_to([c.x for c in cars],(n,len(cars))).copy()
    v=np.broadcast_to([c.v for c in cars],x.shape).copy()
    y0=np.broadcast_to([c.y for c in cars],x.shape).copy(); target=y0.copy()
    av=np.r_[0.,road.model.a,0.]
    acceleration=np.broadcast_to([float(pmfs[c.id]@av) for c in cars],x.shape).copy()
    for key,code in zip(ids,choices):
        j=next(i for i,c in enumerate(cars) if c.id==key)
        acceleration[:,j]=av[code]
        target[:,j]+=np.where(code==0,4.,np.where(code==32,-4.,0.))
    lengths=np.array([c.length for c in cars]); widths=np.array([c.width for c in cars])
    minimum=np.full(n,np.inf); dt=cfg['step_s']
    for tick in range(round(cfg['horizon_s']/dt)):
        y=y0+(target-y0)*min((tick*dt),1.)
        dx=x-x[:,ci,None]; lateral=np.abs(y-y[:,ci,None]) < (widths+widths[ci])/2
        valid=lateral & (dx>0);valid[:,ci]=False
        gaps=np.where(valid,dx-lengths,np.inf);lead=np.argmin(gaps,axis=1)
        gap=gaps[np.arange(n),lead];lead_v=v[np.arange(n),lead]
        acceleration[:,ci]=idm_acceleration(v[:,ci],gap,lead_v,road.config['cav'])
        next_v=np.clip(v+acceleration*dt,20.,40.)
        next_v[:,ci]=np.clip(v[:,ci]+acceleration[:,ci]*dt,0.,40.)
        x+=(v+next_v)*(.5*dt);v=next_v
        y=y0+(target-y0)*min((tick+1)*dt,1.)
        center=x-lengths/2;delta=np.abs(center-center[:,ci,None])-(lengths+lengths[ci])/2
        overlap=np.abs(y-y[:,ci,None]) < (widths+widths[ci])/2
        overlap[:,ci]=False
        minimum=np.minimum(minimum,np.min(np.where(overlap,delta,np.inf),axis=1))
    score=np.exp(-np.maximum(minimum,0.)/cfg['clearance_scale_m'])
    return score if len(ids)==1 else score.reshape(33,33)


class NativeExperiment(DiscreteRoad):
    def __init__(self,model,pair_root,lateral_root,seed,config):
        super().__init__(model,seed)
        self.config=config;self.decisions=[];self.last_acceleration={};self.cav_acceleration=0.
        # Uniform CAV role among initialized vehicles with an observed leader.
        candidates=[c for c in self.vehicles if any(o.id!=c.id and o.lane==c.lane and 0<o.x-o.length-c.x<=115 for o in self.vehicles)]
        if not candidates:raise ValueError('No eligible CAV in initialized fleet')
        self.cav=candidates[int(self.rng.integers(len(candidates)))]
        self.natural=DiscretePairs(pair_root,model,lateral_root,supported_pairs_only=True)
        self.steps=[];self.log_weight=0.;self.max_intervened=0
        self.minimum_ttc=np.inf;self.minimum_gap=np.inf;self.collision_context=[]
        self.physics_trace=[]

    def prepare(self,pmfs):
        sampler=self.natural
        sampler.histories,fronts=update_histories(self.vehicles,sampler.histories,1.)
        candidates=[];actors={c.id:c for c in self.vehicles}
        for car in self.vehicles:
            front=fronts[car.id]
            if car.id==self.cav.id or front is None or front.id==self.cav.id:continue
            gap=front.x-front.length-car.x
            if not 0<gap<=115 or min(sampler.histories[i][1] for i in (car.id,front.id))<3:continue
            ff=fronts[front.id]
            if not sampler.direct(car,front) or not sampler.direct(front,ff):continue
            group=int(joint_group(gap,front.v-car.v,ff is None or ff.x-ff.length-front.x>115,car.v,sampler.model.get('speed_edges',())))
            cell=sampler.model['cells'][str(group)]
            if cell['independent_mass']>=1:continue
            candidates.append({'actor_ids':(car.id,front.id),'relation':'following','priority':(1,0.,gap,car.id,front.id), 'cell':cell})
        selected,_=partition(candidates,sampler.previous)
        sampler.previous={pair_key(c) for c in selected}
        units=[];used=set()
        for item in selected:
            a,b=item['actor_ids'];cell=item['cell']
            p=joint33(pmfs[a],pmfs[b],cell['weights'],cell['independent_mass'])
            units.append({'ids':[a,b],'p':p});used.update((a,b))
        units.extend({'ids':[c.id],'p':pmfs[c.id]} for c in self.vehicles if c.id!=self.cav.id and c.id not in used)
        return units,actors

    def choose(self,units,actors,pmfs,mode):
        if mode=='natural':return None
        cfg=self.config['criticality'];candidates=[]
        fallback=mode=='dual' and self.config.get('dual_single_fallback',False)
        nearby=sorted((abs(c.x-self.cav.x),c.id) for c in actors.values()
                      if c.id!=self.cav.id and abs(c.x-self.cav.x)<=cfg['observation_m'])
        anchors={i for _,i in nearby[:cfg['max_candidates']]}
        for unit in units:
            choices=[[i] for i in unit['ids']] if mode=='single' or fallback else []
            if mode=='dual' and len(unit['ids'])==2:choices.append(unit['ids'])
            for ids in choices:
                distance=min(abs(actors[i].x-self.cav.x) for i in ids)
                if distance<=cfg['observation_m'] and (not fallback or any(i in anchors for i in ids)):
                    candidates.append((distance,tuple(ids),unit))
        candidates=sorted(candidates,key=lambda c:(c[0],c[1]))
        if not fallback:candidates=candidates[:cfg['max_candidates']]
        self.selection_diagnostics={'nearby_actors':len(nearby),'evaluated_candidates':len(candidates),
            'pair_candidates':sum(len(ids)==2 for _,ids,_ in candidates),'eligible_candidates':0,'eligible_pair_candidates':0}
        best=None;best_pair=None
        for _,ids,unit in candidates:
            p=unit['p']
            if len(ids)==1 and p.ndim==2:p=p.sum(axis=1 if ids[0]==unit['ids'][0] else 0)
            score=challenge(self,list(ids),pmfs,cfg)
            mass=float(np.sum(p*score));support=p>0
            contrast=float(np.ptp(score[support]))
            relative=contrast/max(float(np.max(score[support])),1e-30)
            if (mass<cfg['minimum_critical_mass'] or contrast<cfg['minimum_action_contrast']
                    or relative<cfg.get('minimum_relative_action_contrast',0.)):continue
            self.selection_diagnostics['eligible_candidates']+=1
            h=p*score/mass
            item={'unit':unit,'ids':list(ids),'h':h,'mass':mass,
                  'relative_contrast':relative,'factor_influence':factor_influence(p,h)}
            if best is None or mass>best['mass']:best=item
            if len(ids)==2:
                self.selection_diagnostics['eligible_pair_candidates']+=1
                if best_pair is None or mass>best_pair['mass']:best_pair=item
        # Preserve a genuine two-factor experiment whenever an actionable pair
        # exists. A single candidate supplies coverage only when no pair passes
        # the same pre-draw risk/action-contrast tests.
        return best_pair if fallback and best_pair is not None else best

    def sample(self,pmfs,mode,epsilon,policy=None):
        units,actors=self.prepare(pmfs);chosen=self.choose(units,actors,pmfs,mode)
        uniforms={c.id:float(self.rng.random()) for c in self.vehicles if c.id!=self.cav.id}
        actions={};training=None
        for unit in units:
            ids=unit['ids'];p=unit['p'];active=chosen is not None and chosen['unit'] is unit
            one_factor=active and len(chosen['ids'])==1
            if one_factor and ids[0]!=chosen['ids'][0]:ids=ids[::-1];p=p.T
            p1=p if p.ndim==1 else p.sum(axis=1)
            eps=np.ones(1 if mode!='dual' else 2)
            if active:
                obs=observation(self,chosen['ids'])
                mask=[1,int(len(chosen['ids'])==2)]
                if self.config.get('observation_contract',LEGACY_OBSERVATION)==CONTEXT_OBSERVATION:
                    pair=[actors[i] for i in chosen['ids']]
                    gap=pair[1].x-pair[1].length-pair[0].x if len(pair)==2 else 0.
                    rate=pair[1].v-pair[0].v if len(pair)==2 else 0.
                    obs=enrich_observation(obs,chosen['mass'],chosen.get('relative_contrast',0.),
                        self.log_weight,self.time,self.config['horizon_s'],mask,gap,rate)
                dimension=2 if mode=='dual' else 1
                eps=np.asarray(policy(obs) if policy is not None else np.broadcast_to(epsilon,(dimension,)),float)
                if eps.shape!=(dimension,) or not np.isfinite(eps).all() or np.any(eps<self.config['epsilon_min']) or np.any(eps>1):raise ValueError('Invalid policy epsilon')
                h1=chosen['h'] if one_factor else chosen['h'].sum(axis=1)
                q1=eps[0]*p1+(1-eps[0])*h1
            else:h1=p1;q1=p1
            a=inverse_cdf(q1,uniforms[ids[0]]);actions[ids[0]]=a
            pf=[float(p1[a])];hf=[float(h1[a])];qf=[float(q1[a])]
            if len(ids)==2:
                p2=p[a]/p1[a];h2=p2
                if active and not one_factor:h2=chosen['h'][a]/h1[a] if h1[a]>0 else p2
                q2=eps[1]*p2+(1-eps[1])*h2 if active and not one_factor else p2
                b=inverse_cdf(q2,uniforms[ids[1]]);actions[ids[1]]=b
                if active and not one_factor:pf.append(float(p2[b]));hf.append(float(h2[b]));qf.append(float(q2[b]))
            if active:
                if mode=='dual' and one_factor:pf.append(1.);hf.append(1.);qf.append(1.)
                weight=float(np.log(np.asarray(pf)/qf).sum());self.log_weight+=weight
                training={'time_s':self.time,'observation':obs,'actor_ids':chosen['ids'],'natural_unit_ids':ids,
                    'action_indices':[actions[i] for i in chosen['ids']],'p_factors':pf,'h_factors':hf,'q_factors':qf,
                    'generation_epsilon':eps.tolist(),'log_weight':weight,'critical_mass':chosen['mass'],
                    'factor_mask':mask[:dimension],'relative_action_contrast':chosen.get('relative_contrast'),
                    'factor_influence_tv':chosen.get('factor_influence'),
                    'selection_diagnostics':getattr(self,'selection_diagnostics',{})}
                self.max_intervened=max(self.max_intervened,len(chosen['ids']))
        if hasattr(self,'diagnostics'):
            self.diagnostics['decision_steps']+=1
            self.diagnostics['pair_available_steps']+=int(any(len(u['ids'])==2 for u in units))
            self.diagnostics['critical_steps']+=int(chosen is not None)
            self.diagnostics['dual_steps']+=int(chosen is not None and len(chosen['ids'])==2)
            self.diagnostics['single_steps']+=int(chosen is not None and len(chosen['ids'])==1)
            for key,value in getattr(self,'selection_diagnostics',{}).items():self.diagnostics[key]+=value
        if len(actions)!=len(self.vehicles)-1:raise ValueError('Duplicate or missing BV decision')
        if training:self.steps.append(training)
        return actions

    def step_policy(self,mode,epsilon,policy=None):
        pmfs=self.action_pmfs();actions=self.sample(pmfs,mode,epsilon,policy)
        bvs=[c for c in self.vehicles if c.id!=self.cav.id];acc=[];changes={}
        for car in bvs:
            code=actions[car.id]
            if code in (0,32):
                target=car.lane+(1 if code==0 else -1);changes[car.id]=(car.lane,target)
                car.target=target;car.elapsed=0.;acc.append(0.);self.diagnostics['lane_change_onsets']+=1
            else:acc.append(float(self.model.a[code-1]))
        start=self.time;dt=1/15
        for k in range(15):
            cav=self.cav
            ahead=[c for c in bvs if c.x>cav.x and abs(c.y-cav.y)<(c.width+cav.width)/2]
            front=min(ahead,key=lambda c:c.x,default=None)
            gap=np.inf if front is None else front.x-front.length-cav.x
            self.cav_acceleration=float(idm_acceleration(cav.v,gap,cav.v if front is None else front.v,self.config['cav']))
            old_v=cav.v;new_v=float(np.clip(old_v+self.cav_acceleration*dt,0,40))
            t=dt if new_v==old_v+self.cav_acceleration*dt else (new_v-old_v)/self.cav_acceleration
            cav.x+=old_v*t+.5*self.cav_acceleration*t*t+new_v*(dt-t);cav.v=new_v
            bounded,error,distance=advance(bvs,acc,dt,self.model.config)
            for car in bvs:
                if car.id in changes:
                    old,target=changes[car.id];car.y=(old+(target-old)*(k+1)/15)*4
            self.time=start+(k+1)/15;self.crashes=collisions(self.vehicles)
            for car in bvs:
                if abs(car.y-cav.y)>=(car.width+cav.width)/2:continue
                dx=(car.x-car.length/2)-(cav.x-cav.length/2)
                gap=abs(dx)-(car.length+cav.length)/2
                self.minimum_gap=min(self.minimum_gap,gap)
                closing=np.sign(dx)*(cav.v-car.v)
                if closing>0 and gap>0:self.minimum_ttc=min(self.minimum_ttc,gap/closing)
            if self.config.get('record_physics_trace',False):
                self.physics_trace.append({'time_s':self.time,'actors':[vars(c).copy() for c in self.vehicles]})
            if self.crashes:
                self.collision_context=[{'actor_ids':pair,'actors':[vars(c).copy() for c in self.vehicles if c.id in pair],
                    'longitudinal_speed_difference_mps':abs(next(c.v for c in self.vehicles if c.id==pair[0])-next(c.v for c in self.vehicles if c.id==pair[1]))}
                    for pair in self.crashes]
            if self.crashes:break
        self.last_acceleration={c.id:a for c,a in zip(bvs,acc)}
        return not self.crashes


def collect(config_path,output,mode,episodes,seed,assets=None,policy=None,epsilon=None):
    destination,lock=ensure_assets(assets);config=read(config_path)
    if mode not in ('single','dual','natural'):raise ValueError('Invalid intervention mode')
    if episodes<1:raise ValueError('Episodes must be positive')
    contract=config.get('observation_contract',LEGACY_OBSERVATION)
    if contract not in OBSERVATION_DIMENSIONS:raise ValueError('Unknown observation contract')
    if config['epsilon_min']!=.05:raise ValueError('Current PPO epsilon head requires support floor 0.05')
    if config.get('dual_single_fallback') and contract!=CONTEXT_OBSERVATION:raise ValueError('Fallback requires the context observation and active-factor mask')
    generation_epsilon=config['generation_epsilon'] if epsilon is None else epsilon
    if not np.isfinite(generation_epsilon).all() or np.any(np.asarray(generation_epsilon)<config['epsilon_min']) or np.any(np.asarray(generation_epsilon)>1):
        raise ValueError('Invalid fixed proposal epsilon')
    if policy is not None:
        metadata=getattr(policy,'metadata',{})
        if metadata.get('observation_contract',contract)!=contract:raise ValueError('Policy observation differs from collection')
        if metadata.get('proposal_structure_sha256',identity(proposal_structure(config,mode)))!=identity(proposal_structure(config,mode)):
            raise ValueError('Policy candidate/risk-proxy structure differs from collection')
        excluded=set(metadata.get('training_seeds',[]))|set(metadata.get('validation_seeds',[]))
        if excluded & set(range(seed,seed+episodes)):raise ValueError('Evaluation seeds overlap training/validation')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model=PairModelQueries(destination/'model',destination/'lateral',**SINGLE_OPTIONS)
    target=experiment_target(lock,config)
    proposal={'mode':mode,'config':config,'generation_epsilon':generation_epsilon,
              'policy_weights_sha256':getattr(policy,'metadata',{}).get('weights_sha256')}
    target_hash=identity(target);sequences=[]
    started=time.perf_counter()
    for k in range(episodes):
        episode_started=time.perf_counter()
        road=NativeExperiment(model,destination/'pair',destination/'lateral',seed+k,config)
        initial=[vars(c).copy() for c in road.vehicles]
        while road.time<config['horizon_s']-1e-9:
            if not road.step_policy(mode,generation_epsilon,policy):break
        positive=any(road.cav.id in pair for pair in road.crashes)
        event='cav_first_collision' if positive else 'bv_only_competing_collision' if road.crashes else 'horizon_no_collision'
        seq={'seed':seed+k,'cav_id':road.cav.id,'event_result':positive,'event_type':event,'elapsed_s':road.time,
             'collision_pairs':road.crashes,'generation_log_weight':road.log_weight,'steps':road.steps,
             'maximum_intervened_bvs':road.max_intervened,'lane_changes':road.diagnostics.get('lane_change_onsets',0),
             'diagnostics':dict(road.diagnostics),'wall_time_s':time.perf_counter()-episode_started,
             'minimum_ttc_s':float(road.minimum_ttc) if np.isfinite(road.minimum_ttc) else None,
             'minimum_bumper_gap_m':float(road.minimum_gap) if np.isfinite(road.minimum_gap) else None,
             'collision_context':road.collision_context}
        details=dict(seq,initial_fleet=initial)
        if config.get('record_physics_trace',False):details['physics_trace']=road.physics_trace
        save(output/f'episode_{k:05d}.json',details);sequences.append(seq)
        print(json.dumps({'episode':k+1,'seed':seed+k,'event':event,'critical_steps':len(road.steps),'log_weight':road.log_weight}),flush=True)
    dataset={'contract':'highd_v46_native_d2rl_sequence_v2' if contract==CONTEXT_OBSERVATION else 'highd_v46_native_d2rl_sequence_v1','observation_contract':contract,
        'action_dimension':1 if mode!='dual' else 2,'intervention_mode':mode,'natural_target_sha256':lock['natural_target_sha256'],
        'experiment_target':target,'experiment_target_sha256':target_hash,'proposal':proposal,
        'proposal_sha256':identity(proposal),'proposal_structure_sha256':identity(proposal_structure(config,mode)),
        'sequences':sequences}
    save(output/'sequences.json',dataset)
    save(output/'sequence_manifest.json',{'data_path':'sequences.json','sha256':digest(output/'sequences.json'),
          'natural_target_sha256':lock['natural_target_sha256'],'experiment_target_sha256':target_hash,'intervention_mode':mode,
          'proposal_sha256':dataset['proposal_sha256'],'proposal_structure_sha256':dataset['proposal_structure_sha256']})
    from .highd_v46_statistics import summarize_sequences
    statistics=summarize_sequences(sequences)
    totals=Counter()
    for seq in sequences:totals.update(seq['diagnostics'])
    summary={'episodes':episodes,'event_counts':dict(Counter(s['event_type'] for s in sequences)),
        'critical_steps':sum(len(s['steps']) for s in sequences),
        'mode':mode,'experiment_target_sha256':target_hash,'natural_target_sha256':lock['natural_target_sha256'],
        'proposal_sha256':dataset['proposal_sha256'],'wall_time_s':time.perf_counter()-started,
        'diagnostics':dict(totals),'critical_step_fraction':totals['critical_steps']/max(totals['decision_steps'],1),
        'pair_available_fraction':totals['pair_available_steps']/max(totals['decision_steps'],1),**statistics}
    save(output/'collection_summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/highd_v46_d2rl_experiment.json');p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['natural','single','dual'],default='single')
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--seed',type=int,default=1000);p.add_argument('--assets')
    p.add_argument('--epsilon',type=float,help='Override fixed proposal epsilon; physical P is unchanged')
    a=p.parse_args();print(json.dumps(collect(a.config,a.output,a.mode,a.episodes,a.seed,a.assets,epsilon=a.epsilon),indent=2))
