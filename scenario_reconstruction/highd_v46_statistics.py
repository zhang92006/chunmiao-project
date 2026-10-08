"""Ordinary IS risk estimates and paper-style 90% relative precision reports.

Every attempted episode stays in the denominator. Normal intervals and plug-in
sample-size projections are diagnostics, not guarantees for heavy-tailed IS.
"""
import argparse
from pathlib import Path
from statistics import NormalDist
import numpy as np
from .highd_v46_assets import read, save, digest


def episode_values(sequences):
    values=[]
    for seq in sequences:
        if not seq['event_result']:
            values.append(0.)
            continue
        log=float(seq['generation_log_weight'])
        if not np.isfinite(log) or not -350 <= log <= 350:
            raise FloatingPointError('Event weight outside supported float64 moment range; use log-space analysis, never clip weights')
        values.append(float(np.exp(log)))
    return np.asarray(values)


def summarize_values(values, confidence=.90, rhw_target=.30, minimum_events=20):
    values=np.asarray(values,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Need finite nonnegative episode contributions')
    if not 0<confidence<1 or not 0<rhw_target<1 or minimum_events<2:
        raise ValueError('Invalid precision protocol')
    n=len(values);events=int(np.count_nonzero(values));mean=float(values.mean())
    second=float(np.mean(values**2))
    if not np.isfinite(second):raise FloatingPointError('Second moment overflow; no clipping permitted')
    variance=float(values.var(ddof=1)) if n>1 else None
    se=float(np.sqrt(variance/n)) if variance is not None else None
    z=NormalDist().inv_cdf((1+confidence)/2)
    rhw=z*se/mean if mean>0 and se is not None else None
    interval=[mean-z*se,mean+z*se] if mean>0 and se is not None else None
    projected=int(np.ceil((z/rhw_target)**2*variance/mean**2)) if rhw is not None and events>=minimum_events else None
    status=('insufficient_events' if events<minimum_events else
            'precision_reached' if rhw is not None and rhw<=rhw_target else 'precision_not_reached')
    return {'episodes':n,'positive_episodes':events,'weighted_first_collision_mean':mean,
            'weighted_second_moment':second,'weighted_sample_variance':variance,
            'weighted_standard_error':se,'confidence_level':confidence,
            'confidence_interval_normal':interval,'relative_half_width':rhw,
            'relative_half_width_target':rhw_target,'minimum_events':minimum_events,
            'precision_status':status,'projected_required_tests':projected,
            'projection_scope':'Asymptotic plug-in projection; not an observed convergence result',
            'event_contribution_ess':float(values.sum()**2/(n*second)) if second>0 else 0.,
            'raw_event_rate':events/n}


def summarize_sequences(sequences, **kwargs):
    return summarize_values(episode_values(sequences),**kwargs)


def convergence(values, confidence=.90, rhw_target=.30, minimum_events=20, include_curve=True):
    values=np.asarray(values,float)
    if not len(values):raise ValueError('Empty convergence series')
    n=np.arange(1,len(values)+1);total=np.cumsum(values);squares=np.cumsum(values**2)
    means=total/n;variance=np.zeros(len(values))
    variance[1:]=np.maximum(squares[1:]-total[1:]**2/n[1:],0)/(n[1:]-1)
    se=np.sqrt(variance/n);rhw=np.full(len(values),np.inf)
    nonzero=(means>0)&(n>1)
    rhw[nonzero]=NormalDist().inv_cdf((1+confidence)/2)*se[nonzero]/means[nonzero]
    events=np.cumsum(values>0)
    passed=nonzero&(events>=minimum_events)&(rhw<=rhw_target)
    # Require the observed suffix to remain below the threshold. This avoids
    # selecting an early accidental crossing or zero-event NaNs as convergence.
    required=None
    if passed[-1]:
        failures=np.flatnonzero(~passed)
        required=int(failures[-1]+2) if len(failures) else 1
    curve=[{'tests':int(k),'mean':float(m),'standard_error':float(s),
            'relative_half_width':float(r) if np.isfinite(r) else None,
            'positive_episodes':int(e),'precision_reached':bool(p)}
           for k,m,s,r,e,p in zip(n,means,se,rhw,events,passed)] if include_curve else []
    return required,curve


def load_collection(path):
    path=Path(path);manifest_path=path/'sequence_manifest.json' if path.is_dir() else path
    manifest=read(manifest_path);source=manifest_path.parent/manifest['data_path']
    if digest(source)!=manifest['sha256']:raise ValueError('Collection checksum mismatch')
    dataset=read(source);seeds=[s['seed'] for s in dataset['sequences']]
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate evaluation seed')
    return dataset


def compare_collections(collections, output, reference='natural', bootstrap=1000,
                        seed=41, confidence=.90, rhw_target=.30, minimum_events=20):
    if reference not in collections:raise ValueError('Missing natural reference arm')
    if bootstrap<0:raise ValueError('Bootstrap repeats cannot be negative')
    datasets={name:load_collection(path) for name,path in collections.items()}
    target=datasets[reference]['experiment_target_sha256']
    for dataset in datasets.values():
        if dataset['experiment_target_sha256']!=target:raise ValueError('Cannot compare different physical P/event targets')
    if datasets[reference]['intervention_mode']!='natural':raise ValueError('Reference arm must use natural sampling')
    protocol={'confidence':confidence,'rhw_target':rhw_target,'minimum_events':minimum_events}
    values={name:episode_values(d['sequences']) for name,d in datasets.items()}
    arms={};curves={};rng=np.random.default_rng(seed)
    for name,v in values.items():
        required,curve=convergence(v,**protocol);curves[name]=curve
        resampled=[]
        for _ in range(bootstrap):
            tests,_=convergence(v[rng.integers(len(v),size=len(v))],include_curve=False,**protocol)
            resampled.append(tests)
        reached=[n for n in resampled if n is not None]
        arms[name]={**summarize_values(v,**protocol),'required_tests_observed':required,
                    'bootstrap':{'repeats':bootstrap,'seed':seed,
                                 'budget_censored_repeats':bootstrap-len(reached),
                                 'required_tests_quantiles_conditional_on_reaching':
                                 np.quantile(reached,[.05,.5,.95]).tolist() if reached else None},
                    'intervention_mode':datasets[name]['intervention_mode'],
                    'proposal_sha256':datasets[name].get('proposal_sha256'),
                    'wall_time_s':sum(s.get('wall_time_s',0.) for s in datasets[name]['sequences'])}
    ref=arms[reference]
    for arm in arms.values():
        n0,n1=ref['required_tests_observed'],arm['required_tests_observed']
        arm['test_acceleration_observed']=n0/n1 if n0 is not None and n1 else None
        p0,p1=ref['projected_required_tests'],arm['projected_required_tests']
        arm['test_acceleration_projected']=p0/p1 if p0 is not None and p1 else None
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'experiment_target_sha256':target,'reference':reference,
            'protocol':protocol,'arms':arms,
            'convergence_definition':'First point after the last failed prefix, with final RHW <= target and minimum event count; finite-budget evidence only',
            'bootstrap_scope':'Independent episode resampling within each arm; unreached replicates remain censored. No acceleration confidence interval is claimed.',
            'source_manifests':{name:str(Path(path)) for name,path in collections.items()}}
    save(output/'comparison.json',report);save(output/'convergence.json',curves)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collections',nargs='+',required=True,help='name=collection_directory')
    p.add_argument('--output',required=True);p.add_argument('--reference',default='natural')
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--minimum-events',type=int,default=20)
    a=p.parse_args();collections={}
    for item in a.collections:
        name,path=item.split('=',1)
        if name in collections:raise ValueError('Duplicate comparison arm')
        collections[name]=path
    print(compare_collections(collections,a.output,a.reference,a.bootstrap,a.seed,minimum_events=a.minimum_events))


if __name__=='__main__':main()
