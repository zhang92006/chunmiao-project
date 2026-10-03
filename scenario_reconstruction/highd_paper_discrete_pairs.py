"""Dynamic disjoint following pairs on the user-accepted V34 single baseline.

Only dependence in the both-stay block changes. Both 33-action marginals,
including lateral probabilities, are exactly retained. Each actor receives
one RNG uniform per second; the independent arm reproduces the single baseline.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from .highd_paper_data import read,write,sha,index
from .highd_paper_discrete import DiscreteModel,DiscreteRoad
from .highd_paper_pairs import joint33,following_group,update_histories
from .highd_paper_pair_calibration import samples,select_strength
from .highd_dual_ndd import observed_components,rank_fractions,rank_kernel
from .highd_dynamic_pairs import partition,pair_key,PARTITION_VERSION


SINGLE_OPTIONS={'empirical_backoff':True,'pooled_lateral':True,'weighted_lateral':True,
    'reference_lateral':False,'train_headway':False,'one_second_feasibility':True}


class PairModelQueries(DiscreteModel):
    def query_rows_custom(self,rows):
        speed=rows.xVelocity.abs().to_numpy();gap=rows.dhw.to_numpy();rate=rows.precedingXVelocity.abs().to_numpy()-speed
        vi=index(speed,self.v);free=(rows.precedingId.to_numpy()==0)|(gap>115)
        pdf=self.arrays['ff_probability'][vi].astype(float).copy();cf=~free
        keys=(vi[cf],index(gap[cf],self.g),index(rate[cf],self.rr))
        pdf[cf]=self.arrays['cf_probability'][keys]
        unsupported=np.zeros(len(rows),bool);unsupported[cf]=self.arrays['cf_counts'][keys].sum(axis=-1)==0
        for k in np.flatnonzero(unsupported|(pdf.sum(axis=1)<=0)):
            pdf[k]=self.fallback(speed[k],np.inf if free[k] else gap[k],rate[k])
        return pdf/pdf.sum(axis=1,keepdims=True)


def contract(model,lateral_root):
    return {'single_version':'user_accepted_v34','model_sha256':model.summary['model_sha256'],
        'lateral_sha256':sha(Path(lateral_root)/'lateral.json'),'options':SINGLE_OPTIONS,
        'decision_interval_s':1.,'lane_change_duration_s':1.,'physics_hz':15,'action_count':33}


def fit_balanced(chunks):
    chunks=[c for c in chunks if len(c)];components=np.concatenate(chunks)
    exposure=np.concatenate([np.full(len(c),1/(len(c)*len(chunks))) for c in chunks])
    def loss(weights):
        ratio=.1+.9*(components@weights)
        return float(-exposure@np.log(ratio)),-.9*(components.T@(exposure/ratio))
    with threadpool_limits(limits=1):
        result=minimize(loss,np.full(6,1/6),jac=True,method='SLSQP',bounds=[(0,1)]*6,
            constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(6)},options={'maxiter':100,'ftol':1e-10})
    if not result.success:raise RuntimeError('Joint fit failed: '+str(result.message))
    w=np.maximum(result.x,0);return w/w.sum()


def fit(model_root,lateral_root,source,output):
    m=PairModelQueries(model_root,lateral_root,**SINGLE_OPTIONS);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    splits=read(Path(__file__).parents[1]/'configs/highd_paper_nde_protocol_v1.json')['splits']
    train=m.summary['train_recordings'];cal=splits['calibration']
    if set(train)&set(cal) or set(train+cal)&set(splits['validation']+splits['test']):raise ValueError('Overlapping splits')
    data={};counts={}
    for rec in train+cal:
        d,info=samples(m,source,rec,return_data=True,decision_hz=1);mask=d['direct']&d['support']
        components=observed_components(d['first'][mask],d['second'][mask],d['a'][mask],d['b'][mask]);groups=d['groups'][mask]
        counts[rec]={**info,'direct_supported_pairs':int(mask.sum())};data[rec]=(components,groups)
        np.savez_compressed(root/f'components_{rec}.npz',components=components,groups=groups)
        print({'recording':rec,'pairs':len(components)},flush=True)
    cells={}
    for group in range(12):
        chunks=[data[r][0][data[r][1]==group] for r in train]
        selected=[data[r][0][data[r][1]==group] for r in cal];selected=[c for c in selected if len(c)>=20]
        info={'train_pairs':sum(len(c) for c in chunks),'train_records':sum(len(c)>=20 for c in chunks),'calibration_records':len(selected)}
        if info['train_pairs']<200 or info['train_records']<3 or len(selected)<3:
            cells[str(group)]={**info,'weights':[1/6]*6,'independent_mass':1.,'reason':'insufficient_recording_support'};continue
        w=fit_balanced(chunks);strength,scores=select_strength([.1+.9*(c@w) for c in selected])
        cells[str(group)]={**info,'weights':w.tolist(),'independent_mass':1-.9*strength,'calibration_scores':scores,
            'reason':'calibrated' if strength else 'calibration_prefers_independence'}
    result={'schema_version':2,'single_contract':contract(m,lateral_root),'train_recordings':train,'calibration_recordings':cal,
        'base_model_sha256':m.summary['model_sha256'],'decision_hz':1,'history_s':3.,'direct_support_required':True,
        'cells':cells,'by_recording':counts,'partition_version':PARTITION_VERSION,'implementation_sha256':sha(__file__),
        'scope':'Fresh 1 Hz synchronized highD fit against accepted V34 query laws; equal-record training and calibration shrinkage. Both actors need direct empirical support and 3 s past-only stable leader/lane history. Dynamic disjoint following pairs, many pairs per road. Only the 31x31 both-stay block has fitted dependence; all 33-action marginals and lane-change cross-block independence retained. No imported 10 Hz weights, new safety mask, marginal fit or execution change.'}
    write(root/'pair_model.json',result);print({'independent_mass':{k:v['independent_mass'] for k,v in cells.items()}},flush=True)


def inverse_cdf(p,u):
    cdf=np.cumsum(p);cdf/=cdf[-1]
    return int(np.searchsorted(cdf,u,side='right'))


class DiscretePairs:
    def __init__(self,pair_root,model,lateral_root,mode='fitted',dependence_scale=1.,gap_taper=False,supported_pairs_only=False):
        self.model=read(Path(pair_root)/'pair_model.json');self.empirical=model
        if self.model['single_contract']!=contract(model,lateral_root):raise ValueError('Pair model has a different single baseline')
        if mode not in ('fitted','independent_control'):raise ValueError('Unknown dependence mode')
        if not np.isfinite(dependence_scale) or not 0<=dependence_scale<=1:raise ValueError('Dependence scale must be in [0,1]')
        self.dependence_scale=float(dependence_scale)
        self.gap_taper=gap_taper
        self.supported_pairs_only=supported_pairs_only
        self.mode=mode;self.previous=set();self.histories={};self.records=[];self.maximum_marginal_error=0.

    def direct(self,car,front):
        m=self.empirical
        if not 20<=car.v<=40:return False
        vi=int(index(car.v,m.v));g=None if front is None else front.x-front.length-car.x
        if g is None or g>115:return bool(m.arrays['ff_counts'][vi].sum()>0 and m.arrays['ff_probability'][vi].sum()>0)
        rr=front.v-car.v
        if not 0<g<=115 or abs(rr)>20:return False
        key=(vi,int(index(g,m.g)),int(index(rr,m.rr)))
        return bool(m.arrays['cf_counts'][key].sum()>0 and m.arrays['cf_probability'][key].sum()>0)

    def draw(self,vehicles,pmfs,rng):
        self.histories,fronts=update_histories(vehicles,self.histories,1.)
        actors={v.id:v for v in vehicles};candidates=[]
        for car in vehicles:
            front=fronts[car.id]
            if front is not None and 0<front.x-front.length-car.x<=115:
                if self.supported_pairs_only and (min(self.histories[i][1] for i in (car.id,front.id))<3. or
                        not self.direct(car,front) or not self.direct(front,fronts[front.id])):continue
                candidates.append({'actor_ids':(car.id,front.id),'relation':'following','priority':(1,0.,front.x-front.length-car.x,car.id,front.id)})
        selected,_=partition(candidates,self.previous)
        uniforms={v.id:float(rng.random()) for v in vehicles};decisions={};units=[]
        for item in selected:
            first,second=item['actor_ids'];a,b=actors[first],actors[second];front=fronts[second]
            group=int(following_group(b.x-b.length-a.x,b.v-a.v,front is None or front.x-front.length-b.x>115))
            cell=self.model['cells'][str(group)];w=cell['weights'];mass=cell['independent_mass'];reason=cell['reason']
            mass=1-self.dependence_scale*(1-mass)
            if self.gap_taper:mass=1-float(np.clip((115-(b.x-b.length-a.x))/85,0,1))*(1-mass)
            if min(self.histories[i][1] for i in (first,second))<3.:mass=1.;reason='insufficient_past_history'
            elif not self.direct(a,b) or not self.direct(b,front):mass=1.;reason='no_direct_pair_support'
            if self.mode=='independent_control':mass=1.;reason='independent_control'
            p=joint33(pmfs[first],pmfs[second],w,mass)
            error=max(float(abs(p.sum(axis=1)-pmfs[first]).max()),float(abs(p.sum(axis=0)-pmfs[second]).max()))
            if error>1e-10:raise ValueError('Joint action changed a single marginal')
            self.maximum_marginal_error=max(self.maximum_marginal_error,error)
            ca=inverse_cdf(pmfs[first],uniforms[first])
            cb=inverse_cdf(pmfs[second] if mass==1 else p[ca]/p[ca].sum(),uniforms[second])
            decisions[first]=ca;decisions[second]=cb
            units.append({'actor_ids':[first,second],'action_indices':[ca,cb],'group':group,'independent_mass':mass,'reason':reason,
                'selected_single_probabilities':[float(pmfs[first][ca]),float(pmfs[second][cb])],
                'selected_joint_probability':float(p[ca,cb]),'log_p':float(np.log(p[ca,cb]))})
        for car in vehicles:
            if car.id in decisions:continue
            code=inverse_cdf(pmfs[car.id],uniforms[car.id]);decisions[car.id]=code
            units.append({'actor_ids':[car.id],'action_indices':[code],'log_p':float(np.log(pmfs[car.id][code]))})
        if len(decisions)!=len(vehicles) or sum(len(u['actor_ids']) for u in units)!=len(vehicles):raise ValueError('Duplicate or missing action')
        current={pair_key(item) for item in selected}
        self.records.append({'pairs':[list(item['actor_ids']) for item in selected],'added':sorted(current-self.previous),'removed':sorted(self.previous-current),
            'units':units,'log_natural_probability':sum(u['log_p'] for u in units),'log_importance_ratio':0.})
        self.previous=current;return decisions


def run(model_root,lateral_root,pair_root,bank,output,mode='fitted',seeds=(7,19,29),duration=60.,dependence_scale=1.,gap_taper=False,supported_pairs_only=False):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);m=PairModelQueries(model_root,lateral_root,**SINGLE_OPTIONS);runs=[]
    for seed in seeds:
        sampler=DiscretePairs(pair_root,m,lateral_root,mode,dependence_scale,gap_taper,supported_pairs_only);initial=Path(bank)/f'initial_seed{seed}.json'
        road=DiscreteRoad(m,seed,read(initial),sampler);road.decisions=[]
        while road.time<duration-1e-9:
            if not road.step():break
        paired=[u for step in sampler.records for u in step['units'] if len(u['actor_ids'])==2]
        row={'seed':seed,'elapsed_s':road.time,'complete':not bool(road.crashes) and road.time>=duration-1e-9,
            'collision_pairs':road.crashes,'collision_context':road.crash_context,'diagnostics':dict(road.diagnostics),
            'initial_fleet_sha256':sha(initial),'decisions':road.decisions,
            'pair_statistics':{'paired_decisions':len(paired),'correlated_pair_decisions':sum(u['independent_mass']<1 for u in paired),
                'maximum_simultaneous_pairs':max(len(d['pairs']) for d in sampler.records),
                'pair_additions':sum(len(d['added']) for d in sampler.records),'pair_removals':sum(len(d['removed']) for d in sampler.records),
                'maximum_marginal_error':sampler.maximum_marginal_error}}
        write(root/f'run_seed{seed}.json',row);write(root/f'pair_decisions_seed{seed}.json',sampler.records);runs.append(row)
        print({k:row[k] for k in ('seed','elapsed_s','complete','collision_pairs','pair_statistics')},flush=True)
    write(root/'collision_summary.json',{'runs':[{k:v for k,v in r.items() if k!='decisions'} for r in runs],
        'complete_run_count':sum(r['complete'] for r in runs),'model_sha256':m.summary['model_sha256'],'joint_model':sha(Path(pair_root)/'pair_model.json'),
        'pair_mode':mode,'dependence_scale':dependence_scale,'gap_taper':gap_taper,'supported_pairs_only':supported_pairs_only,
        'single_contract':contract(m,lateral_root),'implementation_sha256':sha(__file__),
        'scope':'V34 single laws/execution unchanged. Many dynamic disjoint following pairs; rank dependence only in both-stay actions. Independent arm uses exactly the original single per-actor RNG draws. Collision-first screen, no acceptance by likelihood.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','lateral-model','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--source-root');p.add_argument('--pair-model');p.add_argument('--initial-bank');p.add_argument('--mode',default='fitted',choices=['fitted','independent_control'])
    p.add_argument('--dependence-scale',type=float,default=1.)
    p.add_argument('--gap-taper',action='store_true')
    p.add_argument('--supported-pairs-only',action='store_true')
    a=p.parse_args()
    if a.source_root:fit(a.model,a.lateral_model,a.source_root,a.output)
    else:run(a.model,a.lateral_model,a.pair_model,a.initial_bank,a.output,a.mode,dependence_scale=a.dependence_scale,gap_taper=a.gap_taper,supported_pairs_only=a.supported_pairs_only)
