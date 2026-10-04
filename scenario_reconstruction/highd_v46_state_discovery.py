"""Cheap per-second geometry discovery, episode-disjoint proxy gate audit.

Offline bank selection uses no future collision outcome. Episode-wide ranking
is not an online policy and the enriched bank is not a natural reset law.
"""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from .highd_v46_assets import read,save,digest,ensure_assets
from .highd_v46_d2rl import NativeExperiment,factor_influence,identity
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
from .highd_v46_pair_synergy import snapshot,pair_marginal,pair_metrics,response_risk
from .highd_v46_cross_unit import positive_interaction


def screen_pairs(road,pmfs,cfg):
    """Cheap, pre-draw pressure ranking; score is not a collision probability."""
    cav=road.cav;features={}
    nearby=sorted((abs(c.x-cav.x),c.id,c) for c in road.vehicles
                  if c.id!=cav.id and abs(c.x-cav.x)<=cfg['observation_m'])[:cfg['nearest_actors']]
    for _,key,c in nearby:
        dx=(c.x-c.length/2)-(cav.x-cav.length/2)
        gap=max(0.,abs(dx)-(c.length+cav.length)/2)
        closing=max(0.,np.sign(dx)*(cav.v-c.v))
        same=c.lane==cav.lane
        toward=float(pmfs[key][0 if cav.lane>c.lane else 32]) if abs(cav.lane-c.lane)==1 else 0.
        lane_factor=1. if same else (.5+.5*np.sqrt(toward) if toward>0 else 0.)
        projected=max(0.,gap-cfg['horizon_s']*closing)
        pressure=float(lane_factor*np.exp(-projected/cfg['gap_scale_m']))
        features[key]={'x':c.x,'dx':dx,'gap':gap,'closing':closing,'same_lane':same,
                       'toward_lane_probability':toward,'pressure':pressure}
    rows=[]
    for i,j in combinations(features,2):
        a,b=features[i],features[j]
        if a['same_lane'] and b['same_lane']:
            kind='bracket' if a['dx']*b['dx']<0 else 'same_side'
            factor=1. if kind=='bracket' else cfg['same_side_factor']
        else:kind='lateral';factor=cfg['lateral_factor']
        score=float(a['pressure']*b['pressure']*factor)
        ids=sorted([i,j],key=lambda k:(features[k]['x'],k))
        rows.append({'actor_ids':ids,'mechanism':kind,'score':score})
    return sorted(rows,key=lambda r:(-r['score'],tuple(r['actor_ids'])))


def choose_bank(states,method,cfg,rng):
    eligible=[i for i,s in enumerate(states) if method!='geometry_5s' or round(s['time_s'])%5==0]
    if method=='random_1s':order=list(rng.permutation(eligible))
    elif method in ('geometry_1s','geometry_5s'):
        order=sorted(eligible,key=lambda i:(-states[i]['screen_score'],states[i]['time_s']))
    else:raise ValueError('Unknown discovery method')
    chosen=[]
    for i in order:
        if all(abs(states[i]['time_s']-states[j]['time_s'])>=cfg['minimum_separation_s'] for j in chosen):
            chosen.append(int(i))
        if len(chosen)>=cfg['states_per_episode_per_method']:break
    return sorted(chosen)


def audit_candidates(rows,cfg,rng):
    """Rank top candidate, then prefer another mechanism, plus random audit."""
    chosen=[];remaining=list(rows)
    for _ in range(cfg['ranked_pairs']):
        if not remaining:break
        kinds={r['mechanism'] for r in chosen}
        index=next((i for i,r in enumerate(remaining) if r['mechanism'] not in kinds),0)
        chosen.append(dict(remaining.pop(index),selection='ranked'))
    if remaining:
        for i in rng.choice(len(remaining),size=min(cfg['random_remaining_pairs'],len(remaining)),replace=False):
            chosen.append(dict(remaining[int(i)],selection='random_audit'))
    return chosen


def audit_gates(p,risk,cfg):
    metrics=pair_metrics(p,risk,cfg);score=positive_interaction(p,risk)
    mass=float(np.sum(p*score));h=p*score/mass if mass>0 else p.copy()
    tv=factor_influence(p,h)
    gates={'leverage':min(metrics['max_delta_first'],metrics['max_delta_second'])>=cfg['minimum_delta'],
        'interaction':metrics['max_positive_interaction']>=cfg['minimum_interaction'],
        'global_excess':metrics['joint_excess_over_single_sum']>=cfg['minimum_joint_excess'],
        'risk_H_tv':min(metrics['factor_tv'])>=cfg['minimum_factor_tv'],
        'interaction_mass':mass>=cfg['minimum_synergy_mass'],
        'actual_H_tv':min(tv)>=cfg['minimum_factor_tv']}
    views={'production_strict':all(gates.values()),
        'drop_global_excess':all(v for k,v in gates.items() if k!='global_excess'),
        'local_interaction_actual_H':all(v for k,v in gates.items() if k not in ('global_excess','risk_H_tv'))}
    assert views['production_strict']==bool(metrics['synergy_eligible'] and gates['actual_H_tv'])
    return {'metrics':metrics,'gates':gates,'views':views,'actual_H_tv':tv,
            'first_failed_gate':next((k for k,v in gates.items() if not v),'passed')}


def validate_split(cfg,split,training_lock=None,implementation=None,dependencies=None):
    ranges={k:set(range(cfg[k]['seed_start'],cfg[k]['seed_start']+cfg[k]['episodes'])) for k in ('train','validation')}
    if not all(ranges.values()) or ranges['train']&ranges['validation']:
        raise ValueError('Training/validation source episodes must be nonempty and disjoint')
    if (ranges['train']|ranges['validation']) & set(range(100000,101024)):
        raise ValueError('Reserved final-test seeds cannot be used for discovery')
    if split=='validation':
        if training_lock is None or training_lock.get('split')!='train':raise ValueError('Validation requires completed training lock')
        if training_lock['config_identity']!=identity(cfg) or training_lock['implementation_sha256']!=implementation:
            raise ValueError('Discovery rule changed since training; do not open this validation split')
        if training_lock.get('dependencies')!=dependencies:
            raise ValueError('Discovery dependency changed since training')


def run(config_path,split,output,training_lock=None):
    cfg=read(config_path);implementation=digest(__file__)
    dependencies={name:digest(Path(__file__).with_name(name)) for name in
                  ('highd_v46_pair_synergy.py','highd_v46_cross_unit.py','highd_v46_d2rl.py')}
    dependencies['physical_config']=digest(cfg['physical_config'])
    validate_split(cfg,split,read(training_lock) if training_lock else None,implementation,dependencies)
    if cfg['bank']['states_per_episode_per_method']<1 or cfg['audit']['ranked_pairs']<1:
        raise ValueError('Positive state/pair budgets required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    assets,lock=ensure_assets();physical=read(cfg['physical_config'])
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    bank=[];rows=[];counts=Counter();events=Counter();source_rows=[];started=time.perf_counter()
    rng=np.random.default_rng(cfg['bank']['random_seed']);cohort=cfg[split]
    scan_start=time.perf_counter()
    class Observed(NativeExperiment):
        def choose(self,units,actors,pmfs,mode):
            if mode!='natural':raise ValueError('Discovery uses natural rollouts only')
            fast=screen_pairs(self,pmfs,cfg['screen'])
            item=snapshot(self,units,pmfs,seed,{'discovery_only':True})
            item['screen_pairs']=fast;item['screen_score']=fast[0]['score'] if fast else 0.
            episode_states.append(item);counts['scanned_states']+=1;counts['cheap_pairs']+=len(fast)
            return None
    for seed in range(cohort['seed_start'],cohort['seed_start']+cohort['episodes']):
        episode_states=[];road=Observed(model,assets/'pair',assets/'lateral',seed,physical)
        while road.time<physical['horizon_s']-1e-9:
            if not road.step_policy('natural',1.):break
        # Bank ranking has no access to the terminal collision label below.
        chosen={}
        for method in cfg['bank']['methods']:
            for index in choose_bank(episode_states,method,cfg['bank'],rng):chosen.setdefault(index,[]).append(method)
        for index,methods in sorted(chosen.items()):bank.append(dict(episode_states[index],methods=methods))
        events['cav_first_collision' if any(road.cav.id in p for p in road.crashes) else 'bv_only' if road.crashes else 'horizon']+=1
        source_rows.append({'seed':seed,'states':len(episode_states),'bank_states':len(chosen)})
    scan_time=time.perf_counter()-scan_start
    save(output/'state_bank.json',{'scope':'Episode-stratified offline discovery bank; not a natural reset law or PPO dataset',
        'split':split,'config':cfg,'natural_target_sha256':lock['natural_target_sha256'],'states':bank})
    for index,state in enumerate(bank):
        cars=[SimpleNamespace(**c) for c in state['vehicles']]
        road=SimpleNamespace(vehicles=cars,cav=next(c for c in cars if c.id==state['cav_id']),model=model,config=physical)
        units=[{'ids':u['ids'],'p':np.asarray(u['p'])} for u in state['units']]
        pmfs={int(k):np.asarray(p) for k,p in state['pmfs'].items()}
        for candidate in audit_candidates(state['screen_pairs'],cfg['audit'],rng):
            ids=candidate['actor_ids'];p,same=pair_marginal(units,ids)
            risk=response_risk(road,ids,pmfs,cfg['proxy'],reactive=True)
            row={'state_index':index,'seed':state['seed'],'time_s':state['time_s'],'methods':state['methods'],
                **candidate,'same_natural_unit':same,**audit_gates(p,risk,cfg['selection'])}
            rows.append(row)
            np.savez_compressed(output/f'matrix_{index:03d}_{ids[0]}_{ids[1]}.npz',p=p,risk=risk)
        print({'split':split,'audited_states':index+1,'total':len(bank)},flush=True)
    methods={}
    for method in cfg['bank']['methods']:
        selected=[r for r in rows if method in r['methods']]
        views={}
        for view in cfg['funnel_views']:
            passing=[r for r in selected if r['views'][view]]
            views[view]={'pairs':len(passing),'states':len({r['state_index'] for r in passing}),
                'source_episodes':len({r['seed'] for r in passing}),
                'off_five_second_grid_states':len({r['state_index'] for r in passing if round(r['time_s'])%5}),
                'ranked_pairs':sum(r['selection']=='ranked' for r in passing),
                'random_audit_pairs':sum(r['selection']=='random_audit' for r in passing)}
        methods[method]={'bank_states':sum(method in s['methods'] for s in bank),'evaluated_pairs':len(selected),
            'first_failure_counts':dict(Counter(r['first_failed_gate'] for r in selected)),
            'individual_gate_pass_counts':{k:sum(r['gates'][k] for r in selected) for k in ('leverage','interaction','global_excess','risk_H_tv','interaction_mass','actual_H_tv')},
            'views':views}
    save(output/'pair_audit.json',{'rows':rows})
    summary={'split':split,'config_identity':identity(cfg),'implementation_sha256':implementation,'dependencies':dependencies,
        'config':cfg,'natural_target_sha256':lock['natural_target_sha256'],
        'proxy_sha256':digest(Path(__file__).with_name('highd_v46_pair_synergy.py')),
        'state_bank_sha256':digest(output/'state_bank.json'),'pair_audit_sha256':digest(output/'pair_audit.json'),
        'source_episodes':source_rows,'source_events':dict(events),'counts':dict(counts),'unique_bank_states':len(bank),
        'unique_evaluated_pairs':len(rows),'scan_wall_time_s':scan_time,'total_wall_time_s':time.perf_counter()-started,
        'methods':methods,'scope':'Proxy-qualified discovery yield only. Random competitor audit is incomplete, not global recall. Correlated states are not independent events. Ablated gates are reports, not production changes. No PPO or risk/acceleration estimate.'}
    save(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/highd_v46_state_discovery_v1.json')
    p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--training-lock');p.add_argument('--output',required=True)
    a=p.parse_args();print(run(a.config,a.split,a.output,a.training_lock))


if __name__=='__main__':main()
