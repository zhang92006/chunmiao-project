"""Dynamic disjoint following pairs on the frozen paper-native single NDD.

Reuse the SUMO-line partition and marginal-preserving rank-coupling utilities,
not its old marginals, history models, 93-action executor or intervention policy.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .highd_dynamic_pairs import partition,pair_key,PARTITION_VERSION
from .highd_dual_ndd import rank_joint,observed_components
from .highd_paper_data import read,write,sha,index,clock
from .highd_paper_dynamics import EmpiricalModel


def query_rows(model,rows):
    if hasattr(model,'query_rows_custom'):
        return model.query_rows_custom(rows)
    speed=np.abs(rows.xVelocity.to_numpy());gap=rows.dhw.to_numpy()
    rate=np.abs(rows.precedingXVelocity.to_numpy())-speed
    vi=index(speed,model.v);free=(rows.precedingId.to_numpy()==0)|(gap>115)
    pdf=model.arrays['ff_probability'][vi].astype(float).copy()
    cf=~free;pdf[cf]=model.arrays['cf_probability'][vi[cf],index(gap[cf],model.g),index(rate[cf],model.rr)]
    for k in np.flatnonzero(pdf.sum(axis=1)<=0):
        pdf[k]=model.fallback(speed[k],np.inf if free[k] else gap[k],rate[k])
    return pdf/pdf.sum(axis=1,keepdims=True)


def fit(model_root,source_root,output,independent_mass=.1):
    if not 0<independent_mass<=1:
        raise ValueError('Positive independence floor required')
    model=EmpiricalModel(model_root);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    all_components=[];records=[]
    columns=['frame','id','xVelocity','yVelocity','xAcceleration','dhw','precedingXVelocity','precedingId','frontSightDistance','laneId']
    for rec in model.summary['train_recordings']:
        base=Path(source_root)/'data'
        for kind in ('tracks','tracksMeta'):
            if sha(base/f'{rec}_{kind}.csv')!=model.summary['source_sha256'][rec][kind]:
                raise ValueError('Training source hash changed')
        meta=pd.read_csv(base/f'{rec}_tracksMeta.csv')
        supported=meta.loc[meta['class'].isin(model.config['vehicle_classes'])&(meta.numFrames>=75),'id']
        rows=pd.read_csv(base/f'{rec}_tracks.csv',usecols=columns).sort_values(['id','frame']).reset_index(drop=True)
        direction=rows.id.map(dict(zip(meta.id,meta.drivingDirection))).to_numpy()
        acc=rows.xAcceleration.to_numpy()*np.where(direction==1,-1.,1.)
        speed=np.abs(rows.xVelocity.to_numpy());gap=rows.dhw.to_numpy();rr=np.abs(rows.precedingXVelocity.to_numpy())-speed
        has=rows.precedingId.to_numpy()>0;cf=has&(gap>0)&(gap<=115)&(rr>=-20)&(rr<=20)
        ff=(has&(gap>115))|(~has&(rows.frontSightDistance.to_numpy()>=115))
        segment=(rows.id.ne(rows.id.shift())|rows.precedingId.ne(rows.precedingId.shift())|rows.frame.ne(rows.frame.shift()+1)).cumsum()
        stable=segment.groupby(segment).transform('size').to_numpy()>=75
        valid=rows.id.isin(supported).to_numpy()&(speed>=20)&(speed<=40)&(acc>=-4)&(acc<=2)&(np.abs(rows.yVelocity.to_numpy())<.2)&stable&(cf|ff)
        positions=np.flatnonzero(valid&cf&clock(rows.frame.to_numpy(),25,10))
        lookup=pd.MultiIndex.from_frame(rows[['frame','id']])
        leaders=lookup.get_indexer(pd.MultiIndex.from_arrays([rows.frame.to_numpy()[positions],rows.precedingId.to_numpy()[positions]]))
        keep=leaders>=0;positions=positions[keep];leaders=leaders[keep]
        keep=valid[leaders]&(rows.laneId.to_numpy()[positions]==rows.laneId.to_numpy()[leaders])&(direction[positions]==direction[leaders])
        positions=positions[keep];leaders=leaders[keep]
        first=query_rows(model,rows.iloc[positions]);second=query_rows(model,rows.iloc[leaders])
        a=index(acc[positions],model.a);b=index(acc[leaders],model.a)
        support=(first[np.arange(len(a)),a]>0)&(second[np.arange(len(b)),b]>0)
        components=observed_components(first[support],second[support],a[support],b[support])
        all_components.append(components)
        records.append({'recording':rec,'eligible_pairs':len(a),'support_excluded':int((~support).sum()),'fit_pairs':len(components)})
        print(records[-1],flush=True)
    components=np.concatenate(all_components)
    if not len(components):
        raise ValueError('No supported synchronized following samples')
    def objective(weights):
        ratio=independent_mass+(1-independent_mass)*(components@weights)
        return float(-np.log(ratio).mean()),-(1-independent_mass)*np.mean(components/ratio[:,None],axis=0)
    solved=minimize(objective,np.full(6,1/6),jac=True,method='SLSQP',bounds=[(0,1)]*6,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(6)},options={'maxiter':100,'ftol':1e-10})
    if not solved.success:
        raise RuntimeError('Following coupling fit failed: '+str(solved.message))
    weights=np.maximum(solved.x,0);weights/=weights.sum()
    report={'schema_version':1,'status':'following_only_research_candidate','base_model_sha256':model.summary['model_sha256'],
        'train_recordings':model.summary['train_recordings'],'weights':weights.tolist(),'independent_mass':independent_mass,
        'fit_pairs':len(components),'by_recording':records,'training_nll_gain_vs_independent':-objective(weights)[0],
        'partition_version':PARTITION_VERSION,'action_contract':'33 paper actions; rank dependence only in both-stay longitudinal 31x31 block',
        'scope':'Global six-component rank coupling refit against this frozen empirical marginal. In-sample gain is not heldout validity. Raw observations are correlated. Lane-change residual dependence and adversarial Q are not implemented.',
        'implementation_sha256':sha(__file__)}
    write(root/'pair_model.json',report);return report


def joint33(first,second,weights,independent_mass):
    first=np.asarray(first,float);second=np.asarray(second,float)
    if first.shape!=(33,) or second.shape!=(33,):
        raise ValueError('Expected the paper 33-action contract')
    if any(not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-10,rtol=0) for p in (first,second)):
        raise ValueError('Invalid single action distribution')
    joint=np.outer(first,second);s1=first[1:-1].sum();s2=second[1:-1].sum()
    if s1>0 and s2>0:
        joint[1:-1,1:-1]=s1*s2*rank_joint(first[1:-1]/s1,second[1:-1]/s2,np.asarray(weights),independent_mass)
    return joint


def following_group(gap,rate,leader_free):
    """12 coarse pre-action contexts; identical offline and runtime coding."""
    return np.searchsorted([30.,60.],gap,side='right')*4+(np.asarray(rate)>=0).astype(int)*2+np.asarray(leader_free,dtype=int)


def update_histories(vehicles,previous,dt,lane_by_id=None):
    """Only past states; no future trajectory/stability labels available online."""
    out={};fronts={}
    for actor in vehicles:
        ahead=[v for v in vehicles if v.id!=actor.id and v.x>actor.x and
               ((v.lane==actor.lane or v.target==actor.lane) if lane_by_id is None else lane_by_id[v.id]==lane_by_id[actor.id])]
        leader=min(ahead,key=lambda v:v.x) if ahead else None
        fronts[actor.id]=leader
        key=None if actor.target>=0 or (leader is not None and leader.target>=0) else (actor.lane,None if leader is None else leader.id)
        old=previous.get(actor.id)
        age=old[1]+dt if key is not None and old is not None and key==old[0] else 0.
        out[actor.id]=(key,age)
    return out,fronts


class NativePairs:
    def __init__(self,model_root,base_hash,dependence_mode='fitted'):
        self.model=read(Path(model_root)/'pair_model.json')
        if self.model['base_model_sha256']!=base_hash:
            raise ValueError('Pair model was fitted against different marginals')
        if dependence_mode not in ('fitted','independent_control'):
            raise ValueError('Unknown dependence mode')
        self.dependence_mode=dependence_mode
        self.previous=set();self.records=[];self.maximum_marginal_error=0.
        self.histories={}

    def draw(self,vehicles,pmfs,rng):
        conditional=self.model.get('schema_version',1)>=2
        if conditional:
            self.histories,fronts=update_histories(vehicles,self.histories,1/self.model['decision_hz'])
        actors={v.id:v for v in vehicles}
        candidates=[]
        for actor in vehicles:
            if actor.id not in pmfs:
                continue
            ahead=[other for other in vehicles if other.id!=actor.id and other.x>actor.x and
                   (other.lane==actor.lane or other.target==actor.lane)]
            if not ahead:
                continue
            leader=min(ahead,key=lambda other:other.x)
            gap=leader.x-leader.length-actor.x
            if leader.id in pmfs and leader.lane==actor.lane and leader.target<0 and 0<gap<=115:
                candidates.append({'actor_ids':(actor.id,leader.id),'relation':'following','priority':(1,0.,gap,actor.id,leader.id)})
        selected,_=partition(candidates,self.previous)
        decisions={};units=[]
        for item in selected:
            first,second=item['actor_ids']
            audit={}
            if conditional:
                follower,leader=actors[first],actors[second];other=fronts[second]
                gap=leader.x-leader.length-follower.x;rate=leader.v-follower.v
                leader_gap=None if other is None else other.x-other.length-leader.x
                group=int(following_group(gap,rate,leader_gap is None or leader_gap>115))
                cell=self.model['cells'][str(group)];weights=cell['weights'];mass=cell['independent_mass']
                ages=[self.histories[v][1] for v in (first,second)]
                if min(ages)+1e-9<self.model['history_s']:
                    mass=1.;reason='insufficient_past_history'
                elif abs(rate)>20 or any(not 20<=v.v<=40 for v in (follower,leader)) or (leader_gap is not None and leader_gap<=115 and (leader_gap<=0 or abs(other.v-leader.v)>20)):
                    mass=1.;reason='outside_fit_state_domain'
                else:
                    reason=cell['reason']
                audit={'group':group,'history_s':ages,'dependence_reason':reason}
            else:
                weights=self.model['weights'];mass=self.model['independent_mass']
            if getattr(self,'dependence_mode','fitted')=='independent_control':
                mass=1.
            p=joint33(pmfs[first],pmfs[second],weights,mass)
            error=max(np.max(abs(p.sum(axis=1)-pmfs[first])),np.max(abs(p.sum(axis=0)-pmfs[second])))
            if error>1e-10 or np.any(p<0):
                raise ValueError('Coupling changed a frozen single marginal')
            self.maximum_marginal_error=max(self.maximum_marginal_error,float(error))
            a,b=np.unravel_index(rng.choice(p.size,p=p.ravel()),p.shape)
            decisions[first]=int(a);decisions[second]=int(b)
            units.append({'actor_ids':[first,second],'action_indices':[int(a),int(b)],'log_p':float(np.log(p[a,b])),
                          'independent_mass':mass,**audit})
        for actor in vehicles:
            if actor.id in pmfs and actor.id not in decisions:
                code=int(rng.choice(33,p=pmfs[actor.id]));decisions[actor.id]=code
                units.append({'actor_ids':[actor.id],'action_indices':[code],'log_p':float(np.log(pmfs[actor.id][code]))})
            elif actor.id not in pmfs:
                units.append({'actor_ids':[actor.id],'action_indices':[16],'log_p':0.,'locked_continuation':True})
        if sorted(decisions)!=sorted(pmfs):
            raise ValueError('Every unlocked actor must be sampled exactly once')
        current={pair_key(item) for item in selected};logp=sum(unit['log_p'] for unit in units)
        self.records.append({'pairs':[list(item['actor_ids']) for item in selected],
            'added':sorted(current-self.previous),'removed':sorted(self.previous-current),'units':units,
            'log_natural_probability':logp,'log_proposal_probability':logp,'log_importance_ratio':0.})
        self.previous=current;return decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','source-root','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();r=fit(a.model,a.source_root,a.output)
    print({k:r[k] for k in ('fit_pairs','training_nll_gain_vs_independent','weights')})
