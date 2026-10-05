"""Native V46 D2RL collection: one biased BV or one biased following pair.

P is the frozen V46 BV law conditional on a fixed IDM CAV. Q alone changes.
The event is CAV involvement in the first collision before the fixed horizon.
"""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np

from .highd_v46_assets import ensure_assets, read, save, digest, ROOT
from .highd_paper_discrete_pairs import PairModelQueries, DiscretePairs, SINGLE_OPTIONS, joint_group, inverse_cdf
from .highd_paper_discrete import DiscreteRoad
from .highd_paper_dynamics import advance, collisions
from .highd_paper_pairs import update_histories, joint33
from .highd_dynamic_pairs import partition, pair_key


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


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
        for unit in units:
            choices=[[i] for i in unit['ids']] if mode=='single' else [unit['ids']] if len(unit['ids'])==2 else []
            for ids in choices:
                distance=min(abs(actors[i].x-self.cav.x) for i in ids)
                if distance<=cfg['observation_m']:candidates.append((distance,tuple(ids),unit))
        best=None
        for _,ids,unit in sorted(candidates,key=lambda c:(c[0],c[1]))[:cfg['max_candidates']]:
            p=unit['p']
            if len(ids)==1 and p.ndim==2:p=p.sum(axis=1 if ids[0]==unit['ids'][0] else 0)
            score=challenge(self,list(ids),pmfs,cfg)
            mass=float(np.sum(p*score));support=p>0
            if mass<cfg['minimum_critical_mass'] or np.ptp(score[support])<cfg['minimum_action_contrast']:continue
            if best is None or mass>best['mass']:
                best={'unit':unit,'ids':list(ids),'h':p*score/mass,'mass':mass}
        return best

    def sample(self,pmfs,mode,epsilon,policy=None):
        units,actors=self.prepare(pmfs);chosen=self.choose(units,actors,pmfs,mode)
        uniforms={c.id:float(self.rng.random()) for c in self.vehicles if c.id!=self.cav.id}
        actions={};training=None
        for unit in units:
            ids=unit['ids'];p=unit['p'];active=chosen is not None and chosen['unit'] is unit
            if active and mode=='single' and ids[0]!=chosen['ids'][0]:ids=ids[::-1];p=p.T
            p1=p if p.ndim==1 else p.sum(axis=1)
            eps=np.ones(1 if mode!='dual' else 2)
            if active:
                obs=observation(self,chosen['ids'])
                eps=np.asarray(policy(obs) if policy is not None else [epsilon]*len(chosen['ids']),float)
                if eps.shape!=(len(chosen['ids']),) or not np.isfinite(eps).all() or np.any(eps<self.config['epsilon_min']) or np.any(eps>1):raise ValueError('Invalid policy epsilon')
                h1=chosen['h'] if mode=='single' else chosen['h'].sum(axis=1)
                q1=eps[0]*p1+(1-eps[0])*h1
            else:h1=p1;q1=p1
            a=inverse_cdf(q1,uniforms[ids[0]]);actions[ids[0]]=a
            pf=[float(p1[a])];hf=[float(h1[a])];qf=[float(q1[a])]
            if len(ids)==2:
                p2=p[a]/p1[a];h2=p2
                if active and mode=='dual':h2=chosen['h'][a]/h1[a] if h1[a]>0 else p2
                q2=eps[1]*p2+(1-eps[1])*h2 if active and mode=='dual' else p2
                b=inverse_cdf(q2,uniforms[ids[1]]);actions[ids[1]]=b
                if active and mode=='dual':pf.append(float(p2[b]));hf.append(float(h2[b]));qf.append(float(q2[b]))
            if active:
                weight=float(np.log(np.asarray(pf)/qf).sum());self.log_weight+=weight
                training={'time_s':self.time,'observation':obs,'actor_ids':chosen['ids'],'natural_unit_ids':ids,
                    'action_indices':[actions[i] for i in chosen['ids']],'p_factors':pf,'h_factors':hf,'q_factors':qf,
                    'generation_epsilon':eps.tolist(),'log_weight':weight,'critical_mass':chosen['mass']}
                self.max_intervened=max(self.max_intervened,len(chosen['ids']))
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
            if self.crashes:break
        self.last_acceleration={c.id:a for c,a in zip(bvs,acc)}
        return not self.crashes


def collect(config_path,output,mode,episodes,seed,assets=None,policy=None):
    destination,lock=ensure_assets(assets);config=read(config_path)
    if mode not in ('single','dual','natural'):raise ValueError('Invalid intervention mode')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model=PairModelQueries(destination/'model',destination/'lateral',**SINGLE_OPTIONS)
    target={'natural_target_sha256':lock['natural_target_sha256'],'experiment':config,
            'cav_role':'uniform_initialized_actor_with_observed_current_leader',
            'event':'CAV_in_first_any_actor_collision_by_horizon; BV_only_is_competing_negative',
            'runtime_sha256':digest(__file__)}
    target_hash=identity(target);sequences=[]
    for k in range(episodes):
        road=NativeExperiment(model,destination/'pair',destination/'lateral',seed+k,config)
        initial=[vars(c).copy() for c in road.vehicles]
        while road.time<config['horizon_s']-1e-9:
            if not road.step_policy(mode,config['generation_epsilon'],policy):break
        positive=any(road.cav.id in pair for pair in road.crashes)
        event='cav_first_collision' if positive else 'bv_only_competing_collision' if road.crashes else 'horizon_no_collision'
        seq={'seed':seed+k,'cav_id':road.cav.id,'event_result':positive,'event_type':event,'elapsed_s':road.time,
             'collision_pairs':road.crashes,'generation_log_weight':road.log_weight,'steps':road.steps,
             'maximum_intervened_bvs':road.max_intervened,'lane_changes':road.diagnostics.get('lane_change_onsets',0)}
        save(output/f'episode_{k:05d}.json',dict(seq,initial_fleet=initial));sequences.append(seq)
        print(json.dumps({'episode':k+1,'seed':seed+k,'event':event,'critical_steps':len(road.steps),'log_weight':road.log_weight}),flush=True)
    dataset={'contract':'highd_v46_native_d2rl_sequence_v1','observation_contract':'native_cav6_bv4x2_v1',
        'action_dimension':1 if mode!='dual' else 2,'intervention_mode':mode,'natural_target_sha256':lock['natural_target_sha256'],
        'experiment_target':target,'experiment_target_sha256':target_hash,'sequences':sequences}
    save(output/'sequences.json',dataset)
    save(output/'sequence_manifest.json',{'data_path':'sequences.json','sha256':digest(output/'sequences.json'),
          'natural_target_sha256':lock['natural_target_sha256'],'experiment_target_sha256':target_hash,'intervention_mode':mode})
    weights=np.array([np.exp(s['generation_log_weight']) if s['event_result'] else 0. for s in sequences])
    summary={'episodes':episodes,'event_counts':dict(Counter(s['event_type'] for s in sequences)),
        'critical_steps':sum(len(s['steps']) for s in sequences),'weighted_first_collision_mean':float(weights.mean()),
        'weighted_second_moment':float(np.mean(weights**2)),'weighted_standard_error':float(weights.std(ddof=1)/np.sqrt(episodes)) if episodes>1 else None,
        'mode':mode,'experiment_target_sha256':target_hash,'natural_target_sha256':lock['natural_target_sha256']}
    save(output/'collection_summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/highd_v46_d2rl_experiment.json');p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['natural','single','dual'],default='single')
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--seed',type=int,default=1000);p.add_argument('--assets')
    a=p.parse_args();print(json.dumps(collect(a.config,a.output,a.mode,a.episodes,a.seed,a.assets),indent=2))
