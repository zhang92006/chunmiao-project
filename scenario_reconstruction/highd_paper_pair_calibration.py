"""Train-only conditional rank coupling; recording-balanced calibration shrinkage.

No single PMF, initialization, action semantics or collision rule is changed.
Validation/test recordings are never used for selecting dependence strength.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .highd_paper_data import read,write,sha,index,clock
from .highd_paper_dynamics import EmpiricalModel
from .highd_paper_pairs import query_rows,following_group
from .highd_dual_ndd import observed_components
from .highd_dynamic_pairs import PARTITION_VERSION


def causal_age(rows,hz):
    quiet=rows.yVelocity.abs()<.2
    changed=(rows.id.ne(rows.id.shift())|rows.precedingId.ne(rows.precedingId.shift())|
             rows.laneId.ne(rows.laneId.shift())|rows.frame.ne(rows.frame.shift()+1)|
             ~quiet|~quiet.shift(fill_value=False))
    segment=changed.cumsum()
    return rows.groupby(segment).cumcount().to_numpy()/hz


def samples(model,source_root,rec,return_data=False,decision_hz=10,one_second_actions=False):
    base=Path(source_root)/'data'
    hashes={kind:sha(base/f'{rec}_{kind}.csv') for kind in ('tracks','tracksMeta')}
    if rec in model.summary['train_recordings'] and any(value!=model.summary['source_sha256'][rec][kind] for kind,value in hashes.items()):
        raise ValueError('Training source changed')
    meta=pd.read_csv(base/f'{rec}_tracksMeta.csv')
    columns=['frame','id','xVelocity','yVelocity','xAcceleration','dhw','precedingXVelocity',
             'precedingId','frontSightDistance','laneId']
    rows=pd.read_csv(base/f'{rec}_tracks.csv',usecols=columns).sort_values(['id','frame']).reset_index(drop=True)
    direction=rows.id.map(dict(zip(meta.id,meta.drivingDirection))).to_numpy()
    acc=rows.xAcceleration.to_numpy()*np.where(direction==1,-1.,1.)
    speed=rows.xVelocity.abs().to_numpy();gap=rows.dhw.to_numpy()
    rr=rows.precedingXVelocity.abs().to_numpy()-speed
    has=rows.precedingId.to_numpy()>0
    cf=has&(gap>0)&(gap<=115)&(np.abs(rr)<=20)
    ff=(has&(gap>115))|(~has&(rows.frontSightDistance.to_numpy()>=115))
    classes=meta.loc[meta['class'].isin(model.config['vehicle_classes']),'id']
    age=causal_age(rows,25)
    eligible=(rows.id.isin(classes).to_numpy()&(age>=3.)&(speed>=20)&(speed<=40)&
              (acc>=-4)&(acc<=2)&(rows.yVelocity.abs().to_numpy()<.2)&(cf|ff))
    positions=np.flatnonzero(eligible&cf&clock(rows.frame.to_numpy(),25,decision_hz))
    lookup=pd.MultiIndex.from_frame(rows[['frame','id']])
    leaders=lookup.get_indexer(pd.MultiIndex.from_arrays([rows.frame.to_numpy()[positions],rows.precedingId.to_numpy()[positions]]))
    keep=leaders>=0;positions=positions[keep];leaders=leaders[keep]
    keep=eligible[leaders]&(rows.laneId.to_numpy()[positions]==rows.laneId.to_numpy()[leaders])&(direction[positions]==direction[leaders])
    positions=positions[keep];leaders=leaders[keep]
    future_excluded=0
    if one_second_actions:
        before=len(positions)
        def endpoints(ids):
            return lookup.get_indexer(pd.MultiIndex.from_arrays([rows.frame.to_numpy()[ids]+25,rows.id.to_numpy()[ids]]))
        end_a=endpoints(positions);end_b=endpoints(leaders)
        keep=(end_a>=0)&(end_b>=0)
        positions=positions[keep];leaders=leaders[keep];end_a=end_a[keep];end_b=end_b[keep]
        # Future velocities are training targets only; stable interval selection
        # defines the both-stay action block and excludes truncated tracks.
        aa=speed[end_a]-speed[positions];bb=speed[end_b]-speed[leaders]
        keep=(age[end_a]>=1.)&(age[end_b]>=1.)&(aa>=-4)&(aa<=2)&(bb>=-4)&(bb<=2)
        positions=positions[keep];leaders=leaders[keep]
        acc[positions]=aa[keep];acc[leaders]=bb[keep]
        future_excluded=before-len(positions)
    first=query_rows(model,rows.iloc[positions]);second=query_rows(model,rows.iloc[leaders])
    a=index(acc[positions],model.a);b=index(acc[leaders],model.a)
    support=(first[np.arange(len(a)),a]>0)&(second[np.arange(len(b)),b]>0)
    if return_data:
        direct=np.zeros(len(rows),dtype=bool)
        candidates=np.unique(np.r_[positions,leaders]);vi=index(speed[candidates],model.v)
        direct[candidates]=(model.arrays['ff_counts'][vi].sum(axis=-1)>0)&(model.arrays['ff_probability'][vi].sum(axis=-1)>0)
        local=cf[candidates];ids=candidates[local]
        keys=(vi[local],index(gap[ids],model.g),index(rr[ids],model.rr))
        direct[ids]=(model.arrays['cf_counts'][keys].sum(axis=-1)>0)&(model.arrays['cf_probability'][keys].sum(axis=-1)>0)
        states=np.column_stack([speed[positions],gap[positions],rr[positions],speed[leaders],
            np.where(ff[leaders],116.,gap[leaders]),np.where(ff[leaders],0.,rr[leaders])])
        return {'first':first,'second':second,'a':a,'b':b,'states':states,
            'groups':following_group(gap[positions],rr[positions],ff[leaders]),
            'direct':direct[positions]&direct[leaders],'support':support}, {'source_sha256':hashes,
            'eligible_pairs':len(a),'support_excluded':int((~support).sum()),'fit_pairs':int(support.sum()),
            'one_second_actions':one_second_actions,'future_interval_excluded':future_excluded}
    components=observed_components(first[support],second[support],a[support],b[support])
    groups=following_group(gap[positions],rr[positions],ff[leaders])[support]
    return components,groups,{'source_sha256':hashes,'eligible_pairs':len(a),
                             'support_excluded':int((~support).sum()),'fit_pairs':len(components)}


def fit_weights(components):
    def loss(w):
        ratio=.1+.9*(components@w)
        return -np.log(ratio).mean(),-.9*np.mean(components/ratio[:,None],axis=0)
    result=minimize(loss,np.full(6,1/6),jac=True,method='SLSQP',bounds=[(0,1)]*6,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(6)},
        options={'maxiter':100,'ftol':1e-10})
    if not result.success:
        raise RuntimeError(str(result.message))
    weights=np.maximum(result.x,0);return weights/weights.sum()


def select_strength(per_record_ratios):
    grid=[0.,.25,.5,.75,1.]
    # Each recording receives equal weight, not each correlated video frame.
    scores=[float(np.mean([np.log(1+a*(ratio-1)).mean() for ratio in per_record_ratios])) for a in grid]
    best=int(np.argmax(scores))
    return grid[best],scores


def calibrate(model_root,source_root,output,continuous_quiet=False):
    model=EmpiricalModel(model_root);protocol=read(Path(__file__).parents[1]/'configs/highd_paper_nde_protocol_v1.json')
    if continuous_quiet:
        from .highd_paper_probability_surface import ContinuousQueries
        model=ContinuousQueries(model)
    train=model.summary['train_recordings'];cal=protocol['splits']['calibration']
    if set(train)&set(cal) or (set(train)|set(cal))&(set(protocol['splits']['validation'])|set(protocol['splits']['test'])):
        raise ValueError('Overlapping splits')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    data={};counts={}
    for rec in train+cal:
        components,groups,info=samples(model,source_root,rec)
        data[rec]=(components,groups);counts[rec]=info
        np.savez_compressed(root/f'components_{rec}.npz',components=components,groups=groups)
        print({'recording':rec,'role':'train' if rec in train else 'calibration',**{k:v for k,v in info.items() if k!='source_sha256'}},flush=True)
    cells={}
    for group in range(12):
        chunks=[data[r][0][data[r][1]==group] for r in train]
        components=np.concatenate(chunks)
        selected=[data[r][0][data[r][1]==group] for r in cal]
        selected=[c for c in selected if len(c)>=100]
        info={'train_pairs':len(components),'train_recordings_with_support':sum(len(c)>=100 for c in chunks),
              'calibration_recordings_with_support':len(selected)}
        if len(components)<2000 or info['train_recordings_with_support']<3 or len(selected)<3:
            cells[str(group)]={**info,'weights':[1/6]*6,'independent_mass':1.,'reason':'insufficient_recording_support'}
            continue
        weights=fit_weights(components);ratios=[.1+.9*(c@weights) for c in selected]
        strength,scores=select_strength(ratios)
        cells[str(group)]={**info,'weights':weights.tolist(),'independent_mass':1-.9*strength,
            'selected_strength':strength,'calibration_gain_by_strength':scores,
            'reason':'calibrated' if strength else 'calibration_prefers_independence'}
    report={'schema_version':2,'status':'causal_conditional_following_candidate_not_frozen',
        'base_model_sha256':model.summary['model_sha256'],'train_recordings':train,'calibration_recordings':cal,
        'history_s':3.,'decision_hz':model.config['decision_hz'],'cells':cells,'by_recording':counts,
        'group_definition':'gap [0,30,60,115]; rr negative vs nonnegative; leader following vs free',
        'strength_grid':[0.,.25,.5,.75,1.],'partition_version':PARTITION_VERSION,
        'implementation_sha256':sha(__file__),
        'scope':'Train weights; calibration recording-balanced likelihood selects dependence strength. No validation/test used. At least 3 s past-only stable own-leader/lane history for both actors, reset on lane change. Marginals and lateral decisions unchanged. Not heldout acceptance or causal risk evidence.'}
    if continuous_quiet:
        report['marginal_surface']='trilinear_probability_with_current_state_fallback'
        report['surface_implementation_sha256']=sha(Path(__file__).with_name('highd_paper_probability_surface.py'))
        report['scope']=report['scope'].replace('Marginals and lateral decisions unchanged.', 'Quiet marginals changed explicitly; conditional lateral decisions unchanged.')
        report['scope']+=' Opt-in quiet marginal change: continuous interpolation of frozen probability vectors, current-state fallback for missing corners. No action/execution smoothing. Original single reference and lane-phase marginals untouched. New coupling fitted on these interpolated laws.'
    write(root/'pair_model.json',report)
    return report


def evaluate(model_root,source_root,pair_root,output):
    model=EmpiricalModel(model_root);pair=read(Path(pair_root)/'pair_model.json')
    if pair.get('marginal_surface'):
        from .highd_paper_probability_surface import ContinuousQueries
        model=ContinuousQueries(model)
    if pair['base_model_sha256']!=model.summary['model_sha256']:
        raise ValueError('Different frozen marginal')
    protocol=read(Path(__file__).parents[1]/'configs/highd_paper_nde_protocol_v1.json')
    recordings=protocol['splits']['validation']
    if set(recordings)&(set(pair['train_recordings'])|set(pair['calibration_recordings'])):
        raise ValueError('Validation leakage')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);records=[]
    for rec in recordings:
        components,groups,info=samples(model,source_root,rec)
        gains=np.empty(len(groups));by_group={}
        for group in range(12):
            mask=groups==group;cell=pair['cells'][str(group)];mass=cell['independent_mass']
            values=np.log(mass+(1-mass)*(components[mask]@np.asarray(cell['weights'])))
            gains[mask]=values
            by_group[str(group)]={'count':len(values),'gain_nats_per_pair':float(values.mean()) if len(values) else None}
        record={'recording':rec,**info,'gain_nats_per_supported_pair':float(gains.mean()) if len(gains) else None,'by_group':by_group}
        records.append(record);print({k:v for k,v in record.items() if k not in ('by_group','source_sha256')},flush=True)
    report={'schema_version':1,'pair_model_sha256':sha(Path(pair_root)/'pair_model.json'),'records':records,
        'recording_balanced_gain_nats_per_supported_pair':float(np.mean([r['gain_nats_per_supported_pair'] for r in records if r['fit_pairs']])),
        'support_excluded':sum(r['support_excluded'] for r in records),'eligible_pairs':sum(r['eligible_pairs'] for r in records),
        'scope':'No fit or strength selection on these validation recordings. Relative joint-vs-independent log likelihood only where both frozen single marginals assign nonzero mass. Full-data NLL is infinite if any observed action has zero marginal mass; dependence alone cannot fix that. Correlated observations; no IID confidence claim. Not rollout fidelity acceptance.'}
    if pair.get('marginal_surface'):
        report['marginal_surface']=pair['marginal_surface']
        report['scope']=report['scope'].replace('both frozen single marginals','both declared interpolated quiet marginals')
    write(root/'validation_summary.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','source-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--evaluate-pair-model')
    parser.add_argument('--continuous-quiet',action='store_true')
    args=parser.parse_args()
    if args.evaluate_pair_model:
        report=evaluate(args.model,args.source_root,args.evaluate_pair_model,args.output)
        print({k:report[k] for k in ('recording_balanced_gain_nats_per_supported_pair','support_excluded','eligible_pairs')})
    else:
        report=calibrate(args.model,args.source_root,args.output,args.continuous_quiet)
        print({k:{f:v[f] for f in ('train_pairs','independent_mass','reason')} for k,v in report['cells'].items()})
