"""One-second 33-action NDE baseline; collision-first development runner.

All actions are sampled simultaneously and held for one second. LC acceleration
is zero, LC finishes in that second, and adjacency is rebuilt before each draw.
The 15 Hz executor uses direct longitudinal kinematics and a linear lateral
path, not the paper's bicycle dynamics. No overlap blend or joint model.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from .highd_paper_data import read,write,sha,clock,onset_labels,context_key,idm,index
from .highd_paper_dynamics import EmpiricalModel,Road,advance,collisions,neighbors


def fit_lateral(model_root,source,output):
    model=EmpiricalModel(model_root);c={**model.config,'decision_hz':1};root=Path(output);root.mkdir(parents=True,exist_ok=False)
    table={};counts=Counter();sources={};samples=[];rng=np.random.default_rng(1901)
    columns=['frame','id','x','y','width','height','xVelocity','yVelocity','xAcceleration','dhw','precedingXVelocity','precedingId','frontSightDistance','laneId',
        'leftPrecedingId','leftAlongsideId','leftFollowingId','rightPrecedingId','rightAlongsideId','rightFollowingId']
    for rec in model.summary['train_recordings']:
        base=Path(source)/'data';sources[rec]={k:sha(base/f'{rec}_{k}.csv') for k in ('tracks','tracksMeta','recordingMeta')}
        if sources[rec]!=model.summary['source_sha256'][rec]:raise ValueError('Training source changed')
        meta=pd.read_csv(base/f'{rec}_recordingMeta.csv').iloc[0]
        if int(meta.frameRate)!=25:raise ValueError('Expected 25 Hz source')
        tm=pd.read_csv(base/f'{rec}_tracksMeta.csv');t=pd.read_csv(base/f'{rec}_tracks.csv',usecols=columns).sort_values(['id','frame']).reset_index(drop=True)
        t['direction']=t.id.map(dict(zip(tm.id,tm.drivingDirection)));labels,available=onset_labels(t,c);t['lane_action']=labels
        supported=tm.loc[tm['class'].isin(c['vehicle_classes'])&(tm.numFrames>=25*c['minimum_track_s']),'id']
        legal={int(d):set(p.laneId.unique()) for d,p in t.groupby('direction')}
        mask=available&t.id.isin(supported)&t.xVelocity.abs().between(20,40)&(t.precedingId>0)&t.dhw.between(.00001,115)
        raw=t.loc[mask];lookup=t.set_index(['frame','id']);record_counts=Counter()
        for side in ('left','right'):
            vectors={role:lookup.reindex(pd.MultiIndex.from_arrays([raw.frame,raw[side+role+'Id']]))[['x','width','xVelocity','direction']].to_numpy() for role in ('Preceding','Following','Alongside')}
            chosen=set(rng.choice(len(raw),size=min(len(raw),max(1,c['idm_max_fit_samples']//(2*len(model.summary['train_recordings'])))),replace=False))
            for i,row in enumerate(raw.itertuples()):
                delta=(1 if row.direction==1 else -1)*(1 if side=='left' else -1)
                if row.laneId+delta not in legal[int(row.direction)]:continue
                if np.isfinite(vectors['Alongside'][i,0]):record_counts['alongside_excluded']+=1;continue
                v=abs(row.xVelocity);g=row.dhw;r=abs(row.precedingXVelocity)-v;sign=-1 if row.direction==1 else 1
                own=sign*(row.x+row.width/2)+row.width/2;nbs=[]
                for role in ('Preceding','Following'):
                    nx,nl,nv,nd=vectors[role][i]
                    if not np.isfinite(nx) or nd!=row.direction:nbs.append(None);continue
                    front=sign*(nx+nl/2)+nl/2;gap=front-nl-own if role=='Preceding' else own-row.width-front
                    rr=abs(nv)-v if role=='Preceding' else v-abs(nv)
                    nbs.append((float(gap),float(rr)) if 0<=gap<=115 else None)
                key=context_key(v,g,r,*nbs,c);positive=int(row.lane_action==(1 if side=='left' else -1))
                cell=table.setdefault(key,[0,0]);cell[0]+=1;cell[1]+=positive
                record_counts['exposures']+=1;record_counts['onsets']+=positive;record_counts['context_'+str(key[0])]+=1
                if i in chosen:
                    f,b=nbs;samples.append([v,g,r,np.inf if f is None else f[0],0. if f is None else f[1],np.inf if b is None else b[0],0. if b is None else b[1],positive])
        counts.update(record_counts);print({'recording':rec,**record_counts},flush=True)
    ms=np.asarray(samples);par=model.summary['idm']['parameters'];a=model.a
    old=np.clip(idm(par,ms[:,0],ms[:,1],ms[:,2]),a[0],a[-1]);new=np.clip(idm(par,ms[:,0],ms[:,3],ms[:,4]),a[0],a[-1])
    follower=np.where(np.isfinite(ms[:,5]),np.clip(idm(par,ms[:,0]-ms[:,6],ms[:,5],ms[:,6]),a[0],a[-1]),0.);y=ms[:,7]
    def loss(beta):
        z=np.clip((new-old+beta[0]*follower-beta[1])/beta[2]+beta[3],-30,30)
        return float(np.mean(np.logaddexp(0,z)-y*z))
    fit=minimize(loss,[.1,.2,1.,-5.],bounds=[(0.,1.),(-2.,2.),(.1,5.),(-15.,0.)],method='L-BFGS-B')
    if not fit.success:raise RuntimeError('One-second fallback fit failed: '+str(fit.message))
    write(root/'lateral.json',{'states':[[list(k),v] for k,v in table.items()]})
    write(root/'fit_summary.json',{'decision_interval_s':1.,'base_model_sha256':model.summary['model_sha256'],'train_recordings':model.summary['train_recordings'],
        'source_sha256':sources,'counts':dict(counts),'lateral_sha256':sha(root/'lateral.json'),'mobil_parameters':fit.x.tolist(),
        'scope':'Only empirical lateral exposure/onset counts and the existing stochastic missing-cell fallback are recomputed at the 1 Hz clock. Frozen longitudinal PMFs, initializer and data split unchanged. No joint PMF fit. highD onset is mapped to its preceding 1 s decision; this is the declared highD adaptation.'})


class DiscreteModel(EmpiricalModel):
    def __init__(self,model_root,lateral_root,empirical_backoff=False,pooled_lateral=False,weighted_lateral=False,reference_lateral=False,train_headway=False,one_second_feasibility=False):
        super().__init__(model_root)
        if not reference_lateral:
            r=read(Path(lateral_root)/'fit_summary.json')
            if r['base_model_sha256']!=self.summary['model_sha256'] or sha(Path(lateral_root)/'lateral.json')!=r['lateral_sha256']:raise ValueError('Lateral artifact differs')
            self.lateral={tuple(k):v for k,v in read(Path(lateral_root)/'lateral.json')['states']}
            self.summary['mobil']['parameters']=r['mobil_parameters']
        self.config={**self.config,'decision_hz':1,'lane_change_duration_s':1.}
        self.one_second_feasibility=one_second_feasibility;self.feasibility_exclusions=Counter()
        self.headway_estimate=None
        if train_headway:
            weights=self.arrays['cf_counts'][:,:,abs(self.rr)<=1,:][:,:,:,abs(self.a)<=.200001].sum(axis=(2,3),dtype=float)
            values=np.maximum(self.g[None,:]-self.summary['idm']['parameters'][4],0)/self.v[:,None]
            order=np.argsort(values.ravel());cumulative=np.cumsum(weights.ravel()[order])
            value=float(values.ravel()[order[np.searchsorted(cumulative,cumulative[-1]/2)]])
            self.headway_estimate={'headway_s':value,'train_rows':int(cumulative[-1]),'definition':'Exposure-weighted median (gap-s0)/speed among direct train CF counts with abs(rr)<=1 m/s and abs(a)<=0.2 m/s2; steady-gap surrogate, not an identified desired headway.'}
            self.summary['idm']['parameters'][-1]=value
        self.empirical_backoff=empirical_backoff;self.pooled_lateral=pooled_lateral;self.weighted_lateral=weighted_lateral;self.context_trees={}
        if empirical_backoff:
            for code in range(4):
                rows=[(key,value) for key,value in self.lateral.items() if key[0]==code]
                x=np.asarray([self.distance_features(k,pooled_lateral) for k,_ in rows]);counts=np.asarray([v for _,v in rows])
                self.context_trees[code]=(cKDTree(x),counts)

    @staticmethod
    def distance_features(key,closing=False):
        x=np.asarray(key[1:],dtype=float)/5.
        # Distances use speed differences and log net gaps, within the same
        # 0/front-only/rear-only/2-neighbor category. No collision label input.
        x[1::2]=np.log1p(np.maximum(np.asarray(key[1:],dtype=float)[1::2],0)/5.)
        if closing:
            values=np.asarray(key[1:],dtype=float)
            x=np.r_[x,np.maximum(-values[2::2],0)/np.maximum(values[1::2],1.)]
        return x

    def longitudinal(self,v,gap=None,rr=0.):
        p,status=super().longitudinal(v,gap,rr)
        if self.empirical_backoff and gap is not None and 0<=gap<=115 and abs(rr)<=20 and 20<=v<=40:
            key=(int(index(v,self.v)),int(index(gap,self.g)),int(index(rr,self.rr)))
            if self.arrays['cf_counts'][key].sum()==0:
                return self.fallback(v,gap,rr),'direct_state_idm_fallback'
        return p,status

    def lateral_probability(self,v,gap,rr,front,rear):
        if self.one_second_feasibility:
            braking=self.config['inevitable_state_deceleration_mps2']
            for name,neighbor in [('front',front),('rear',rear)]:
                if neighbor is not None:
                    g,r=neighbor
                    if g+r<=min(r,0.)**2/(2*braking):
                        self.feasibility_exclusions[name]+=1
                        return 0.,'one_second_target_'+name+'_infeasible'
        if not self.empirical_backoff:return super().lateral_probability(v,gap,rr,front,rear)
        key=context_key(v,gap,rr,front,rear,self.config);p,status=self.lateral_frequency(key)
        if p is not None and not self.pooled_lateral:return p,status
        tree,counts=self.context_trees[key[0]];distance,ids=tree.query(self.distance_features(key,self.pooled_lateral),k=min(64,len(counts)))
        weights=1/np.maximum(np.atleast_1d(distance),.05)**2 if self.weighted_lateral else np.ones_like(np.atleast_1d(distance))
        total=(counts[np.atleast_1d(ids)]*weights[:,None]).sum(axis=0)
        return float(total[1]/total[0]),'same_context_empirical_backoff'


class DiscreteRoad(Road):
    def step(self):
        if any(c.target>=0 for c in self.vehicles):raise ValueError('Maneuver remained active at the next one-second decision')
        before={c.id:vars(c).copy() for c in self.vehicles};pmfs=self.action_pmfs()
        decisions=(self.pair_sampler.draw(self.vehicles,pmfs,self.rng) if self.pair_sampler is not None else
            {c.id:int(self.rng.choice(33,p=pmfs[c.id])) for c in self.vehicles});actions=[];changes=[];contexts=[]
        for car in self.vehicles:
            code=decisions[car.id]
            if code in (0,32):
                target=car.lane+(1 if code==0 else -1);changes.append((car,target));actions.append(0.)
                ft,rt,overlap=neighbors(self.vehicles,car,target,115.)
                contexts.append({'actor_id':car.id,'target_lane':target,'front':ft,'rear':rt,'alongside':overlap})
                self.diagnostics['lane_change_onsets']+=1
            else:actions.append(float(self.model.a[code-1]))
        start=self.time
        for car,target in changes:car.target=target;car.elapsed=0.
        for substep in range(15):
            # The public reference executor supplies exact longitudinal motion.
            # Linear lateral interpolation is explicit and collision-visible.
            bounded,error,distance=advance(self.vehicles,actions,1/15,self.model.config)
            for car,target in changes:
                fraction=(substep+1)/15
                car.y=(before[car.id]['lane']+(target-before[car.id]['lane'])*fraction)*self.model.config['lane_width_m']
            self.time=start+(substep+1)/15;self.max_action_error=max(self.max_action_error,error)
            self.diagnostics['speed_bound_actions']+=bounded;self.diagnostics['distance_m']+=distance
            self.crashes=collisions(self.vehicles)
            if self.crashes:
                ids={i for p in self.crashes for i in p}
                self.crash_context=[{'decision_before':before[c.id],'at_collision':vars(c).copy(),'held_acceleration':a,'action_index':decisions[c.id]} for c,a in zip(self.vehicles,actions) if c.id in ids]
                break
        self.decisions.append({'time_s':start,'actions':decisions,'accelerations':{c.id:a for c,a in zip(self.vehicles,actions)},'lane_contexts':contexts})
        self.diagnostics['maneuver_decisions']+=len(self.vehicles)
        return not self.crashes


def run(model_root,lateral_root,bank,output,seeds=(7,19,29),duration=60.,empirical_backoff=False,pooled_lateral=False,weighted_lateral=False,reference_lateral=False,train_headway=False,one_second_feasibility=False):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);m=DiscreteModel(model_root,lateral_root,empirical_backoff,pooled_lateral,weighted_lateral,reference_lateral,train_headway,one_second_feasibility);results=[]
    for seed in seeds:
        initial=read(Path(bank)/f'initial_seed{seed}.json');road=DiscreteRoad(m,seed,initial);road.decisions=[]
        while road.time<duration-1e-9:
            if not road.step():break
        result={'seed':seed,'elapsed_s':road.time,'complete':not bool(road.crashes) and road.time>=duration-1e-9,'collision_pairs':road.crashes,'collision_context':road.crash_context,
            'diagnostics':dict(road.diagnostics),'initial_fleet_sha256':sha(Path(bank)/f'initial_seed{seed}.json'),'decisions':road.decisions}
        results.append(result);write(root/f'run_seed{seed}.json',result);print({k:result[k] for k in ('seed','elapsed_s','complete','collision_pairs')},flush=True)
    report={'runs':[{k:v for k,v in r.items() if k!='decisions'} for r in results],'complete_run_count':sum(r['complete'] for r in results),
        'model_sha256':m.summary['model_sha256'],'lateral_sha256':sha(Path(model_root if reference_lateral else lateral_root)/'lateral.json'),'implementation_sha256':sha(__file__),
        'decision_interval_s':1.,'physics_hz':15,'lane_change_duration_s':1.,'lane_change_acceleration':0.,'joint_model':None,
        'empirical_backoff':empirical_backoff,
        'pooled_lateral':pooled_lateral,
        'weighted_lateral':weighted_lateral,
        'reference_lateral':reference_lateral,
        'train_headway_estimate':m.headway_estimate,
        'one_second_feasibility':one_second_feasibility,'feasibility_excluded_queries':dict(m.feasibility_exclusions),
        'scope':'Collision screen only. All 33 actions are drawn independently/simultaneously every second and held for the second; neighbors rebuilt before each decision. No density scoring or acceptance inferred. Direct kinematics plus linear lateral path differs from the paper bicycle model.'}
    if empirical_backoff:report['scope']+=' Empty raw CF states use the existing highD-fitted IDM probability distribution. Missing lateral cells use pooled exposure/onset counts from 64 nearest populated cells in the identical adjacent category, using log-gap and speed features, instead of stochastic MOBIL extrapolation. Both change declared pre-draw distributions; no sampled-action override or collision filter.'
    if pooled_lateral:report['scope']+=' Empirical pooling also replaces low-count exact/axis-smoothed lateral cells; neighbor distance includes closing-rate/net-gap for current, target-front and target-rear relationships. No collision-conditioned action suppression.'
    if weighted_lateral:report['scope']+=' Nearby exposure and event counts receive inverse-squared state-distance weights with a 0.05 distance floor; distant positive labels are not equally weighted with closer negative exposures.'
    if reference_lateral:report['scope']+=' Original empirical lateral table queried once per second with no hazard conversion or refitting. This isolates decision/execution-clock changes; the original onset labels/counts were produced on a 10 Hz source decision clock, so per-second natural onset calibration is not established.'
    if one_second_feasibility:report['scope']+=' EXPLICIT EXTRA ACTION-SUPPORT RESTRICTION: before sampling, target front/rear projected gap at 1 s under unchanged neighbor speeds must exceed relative stopping distance rr_minus^2/(2*4). Applies to the declared compressed 1 s zero-acceleration LC, not a rule established by the paper. Actual neighbors may accelerate, so it is not a safety proof. All exclusions counted. Source-positive exclusions require later assessment; no full natural-law acceptance from crash-free runs.'
    write(root/'collision_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-root');p.add_argument('--lateral-model');p.add_argument('--initial-bank');p.add_argument('--seeds',nargs='+',type=int,default=[7,19,29]);p.add_argument('--duration',type=float,default=60.)
    p.add_argument('--empirical-backoff',action='store_true')
    p.add_argument('--pooled-lateral',action='store_true')
    p.add_argument('--weighted-lateral',action='store_true')
    p.add_argument('--reference-lateral',action='store_true')
    p.add_argument('--train-headway',action='store_true')
    p.add_argument('--one-second-feasibility',action='store_true')
    a=p.parse_args()
    if a.source_root:fit_lateral(a.model,a.source_root,a.output)
    else:run(a.model,a.lateral_model,a.initial_bank,a.output,a.seeds,a.duration,a.empirical_backoff,a.pooled_lateral,a.weighted_lateral,a.reference_lateral,a.train_headway,a.one_second_feasibility)
