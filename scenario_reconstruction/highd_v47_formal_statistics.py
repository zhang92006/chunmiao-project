"""Full-target IS diagnostics, source guards, native checkpoint selection."""
import numpy as np
from .highd_v46_statistics import summarize_values, convergence


def describe(sequences, cfg):
    logs=np.asarray([s['generation_log_weight'] for s in sequences],float)
    if not np.isfinite(logs).all() or np.any(logs>350):
        raise FloatingPointError('Use log-space analysis for unsupported moment range; never clip')
    weights=np.exp(logs); y=np.asarray([s['event_result'] for s in sequences],float)
    values=y*weights; squares=values**2; total=float(squares.sum()); sources=sorted({s['source_seed'] for s in sequences})
    contributions={str(k):float(squares[[s['source_seed']==k for s in sequences]].sum()) for k in sources}
    shares={k:v/total if total else None for k,v in contributions.items()}
    steps=[t for s in sequences for t in s['steps']]; dual=[t for t in steps if len(t['actor_ids'])==2]
    effective=[t for t in dual if min(t['actual_QP_TV'])>=cfg['screen']['minimum_second_factor_TV']]
    effective_event_sources={s['source_seed'] for s in sequences if s['event_result'] and any(
        len(t['actor_ids'])==2 and min(t['actual_QP_TV'])>=cfg['screen']['minimum_second_factor_TV'] for t in s['steps'])}
    summary=summarize_values(values,confidence=cfg['statistics']['confidence'],minimum_events=cfg['screen']['minimum_events'])
    summary['native_CAV_events']=int(y.sum())
    summary['raw_event_rate']=float(y.mean())
    summary['event_contribution_underflow_count']=int(np.count_nonzero((y>0)&(values==0)))
    weighted_sources={s['source_seed'] for s,v in zip(sequences,values) if v>0}
    supported=summary['positive_episodes']>=cfg['screen']['minimum_events'] and len(weighted_sources)>=cfg['screen']['minimum_event_sources']
    if not supported:
        summary.update(relative_half_width=None,confidence_interval_normal=None,projected_required_tests=None,precision_status='insufficient_independent_event_evidence')
    if np.ptp(values)==0:
        summary.update(relative_half_width=None,confidence_interval_normal=None,projected_required_tests=None,precision_status='degenerate_observed_boundary')
        supported=False
    required={};curves={}
    for target in cfg['statistics']['rhw_targets']:
        n,curve=convergence(values,confidence=cfg['statistics']['confidence'],rhw_target=target,minimum_events=cfg['screen']['minimum_events'])
        # Source evidence must also hold throughout the reported stable suffix.
        seen=set(); guard=[]
        for s,v in zip(sequences,values):
            if v>0:seen.add(s['source_seed'])
            guard.append(len(seen)>=cfg['screen']['minimum_event_sources'])
        bad=[i for i,(c,g) in enumerate(zip(curve,guard)) if not c['precision_reached'] or not g]
        n=(bad[-1]+2 if bad else 1) if curve[-1]['precision_reached'] and guard[-1] else None
        if not supported:n=None
        required[str(target)]=n;curves[str(target)]=curve
        running=np.cumsum(squares); running_mean=np.cumsum(values)
        for i,point in enumerate(curve,1):
            point['M2']=float(running[i-1]/i)
            point['estimator_variance']=float(max(0.,running[i-1]-running_mean[i-1]**2/i)/(i-1)/i) if i>1 else None
            point['independent_event_source_guard']=guard[i-1]
            point['precision_reached']=bool(point['precision_reached'] and guard[i-1] and supported)
    return dict(summary,event_sources=len({s['source_seed'] for s in sequences if s['event_result']}),weighted_event_sources=len(weighted_sources),
        effective_dual_event_sources=len(effective_event_sources),source_M2_shares=shares,
        maximum_source_M2_share=max(shares.values()) if total else None,
        effective_M2_sources=1/sum(v*v for v in shares.values()) if total else None,
        maximum_event_M2_share=float(squares.max()/total) if total else None,
        likelihood_weight_quantiles=dict(zip(('p95','p99','max'),np.quantile(weights,[.95,.99,1.]).tolist())),
        likelihood_log_quantiles=dict(zip(('min','p95','p99','max'),np.quantile(logs,[0,.95,.99,1.]).tolist())),
        critical_decisions=len(steps),dual_decisions=len(dual),effective_dual_fraction=len(effective)/len(steps) if steps else 0.,
        second_factor_TV_mean=float(np.mean([t['actual_QP_TV'][1] for t in dual])) if dual else None,
        cpu_per_episode=float(sum(s['cpu_s'] for s in sequences)/len(sequences)),
        cpu_per_critical_decision=float(sum(t['decision_cpu_s'] for t in steps)/len(steps)) if steps else None,
        likelihood_max_log_error=max((abs(sum(d['log_p']-d['log_q'] for d in s['native_decisions'])-s['generation_log_weight']) for s in sequences),default=0.),
        required_tests_observed=required,convergence_curves=curves,uncertainty_supported=supported)


def select_checkpoint(reports):
    if 0 not in reports:raise ValueError('Iteration 0 must be measured')
    n=reports[0]['episodes']
    if any(r['episodes']!=n for r in reports.values()):raise ValueError('Checkpoint budgets differ')
    # Never use sparse all-zero empirical moments to promote a trained model.
    supported={i:r for i,r in reports.items() if r['uncertainty_supported']}
    if not reports[0]['uncertainty_supported']:return 0,'insufficient_baseline_event_evidence'
    winner=min(supported,key=lambda i:(supported[i]['weighted_second_moment'],i))
    if winner==0 or reports[winner]['weighted_second_moment']>=reports[0]['weighted_second_moment']:
        return 0,'no_supported_improvement'
    return winner,'validation_native_M2'


def metric_failures(label,r,mode,rules):
    reasons=[]
    if not r['uncertainty_supported']:reasons.append(label+': insufficient event/source evidence')
    if r['maximum_source_M2_share'] is None or r['maximum_source_M2_share']>rules['maximum_source_M2_share']:reasons.append(label+': source concentration')
    if r['effective_M2_sources'] is None or r['effective_M2_sources']<rules['minimum_effective_M2_sources']:reasons.append(label+': too few effective M2 sources')
    if r['maximum_event_M2_share'] is None or r['maximum_event_M2_share']>rules['maximum_event_M2_share']:reasons.append(label+': event weight tail')
    if mode=='dual' and (r['second_factor_TV_mean'] is None or r['second_factor_TV_mean']<rules['minimum_second_factor_TV'] or r['effective_dual_fraction']<rules['minimum_effective_dual_fraction'] or r['effective_dual_event_sources']<rules['minimum_event_sources']):
        reasons.append(label+': second factor lacks effective event coverage')
    return reasons


def readiness(selected, cfg):
    rules=cfg['screen']; reasons=[]
    expected={(m,s) for m in ('single','dual') for s in cfg['training']['seeds']}
    if len(selected)!=len(expected) or {(a['mode'],a['seed']) for a in selected}!=expected:
        reasons.append('Every predeclared paired seed must be retained exactly once')
    for mode in ('single','dual'):
        arms=[r for r in selected if r['mode']==mode]
        if sum(r['iteration']>0 for r in arms)<rules['minimum_improved_seeds_per_mode']:
            reasons.append(mode+': fewer than three improved paired seeds')
        for arm in arms:
            r=arm['metrics']; label=mode+'_seed'+str(arm['seed'])
            reasons.extend(metric_failures(label,r,mode,rules))
    return dict(passed=not reasons,reasons=reasons)


def final_readiness(pairs,cfg):
    reasons=[]
    if set(pairs)!=set(map(str,cfg['training']['seeds'])):reasons.append('Missing paired FinalTest seeds')
    for seed,pair in pairs.items():
        for mode in ('single','dual'):reasons.extend(metric_failures(mode+'_seed'+seed,pair[mode],mode,cfg['screen']))
    return dict(passed=not reasons,reasons=reasons)


def paired_comparison(single, dual, cfg):
    if [s['source_seed'] for s in single]!=[s['source_seed'] for s in dual]:raise ValueError('Comparison source schedules differ')
    a=describe(single,cfg);b=describe(dual,cfg);rng=np.random.default_rng(cfg['final_test']['bootstrap_seed'])
    x=np.asarray([s['event_result']*np.exp(s['generation_log_weight']) for s in single]);y=np.asarray([s['event_result']*np.exp(s['generation_log_weight']) for s in dual])
    # Original-initialization test: one episode per unique source, paired resampling.
    if len({s['source_seed'] for s in single})!=len(single):raise ValueError('Bootstrap requires one full reset per source')
    ratios=[];censored=0;minimum=cfg['screen']['minimum_event_sources']; events=cfg['screen']['minimum_events']
    for _ in range(cfg['final_test']['bootstrap_replicates']):
        indices=rng.integers(len(x),size=len(x)); unique=set(indices.tolist())
        if min(sum(x[i]>0 for i in unique),sum(y[i]>0 for i in unique))<minimum or min(np.count_nonzero(x[indices]),np.count_nonzero(y[indices]))<events:
            censored+=1;continue
        ratios.append(float(np.mean(y[indices]**2)/np.mean(x[indices]**2)))
    alpha=(1-cfg['statistics']['confidence'])/(2*len(cfg['training']['seeds']))
    interval=np.quantile(ratios,[alpha,1-alpha]).tolist() if ratios and censored/ max(1,cfg['final_test']['bootstrap_replicates'])<=.1 else None
    acceleration={t:a['required_tests_observed'][t]/b['required_tests_observed'][t] if a['required_tests_observed'][t] and b['required_tests_observed'][t] else None for t in a['required_tests_observed']}
    return dict(single=a,dual=b,M2_ratio_point=b['weighted_second_moment']/a['weighted_second_moment'] if a['weighted_second_moment'] else None,
        estimator_variance_ratio_point=b['weighted_sample_variance']/a['weighted_sample_variance'] if a['weighted_sample_variance'] else None,
        paired_source_M2_ratio_interval=interval,bootstrap_censored=censored,acceleration_observed=acceleration,
        episode_CPU_ratio=b['cpu_per_episode']/a['cpu_per_episode'] if a['cpu_per_episode'] else None,
        supports_dual_lower_M2=bool(interval and interval[1]<1))
