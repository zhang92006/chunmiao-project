"""Check diagnostic four-corner witnesses with the actual frozen native dynamics."""
import argparse
from pathlib import Path
import numpy as np
from .highd_v46_assets import read,save,digest,ensure_assets
from .highd_v46_d2rl import NativeExperiment
from .highd_paper_dynamics import Vehicle
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS,inverse_cdf


def conditional_actions(units,forced,rng):
    """Draw all unforced actors under the full natural law conditional on forced actions."""
    ids=[i for u in units for i in u['ids']]
    if len(set(ids))!=len(ids) or not set(forced)<=set(ids):raise ValueError('Invalid natural partition or forced actors')
    uniforms={i:float(rng.random()) for i in sorted(ids)};actions={}
    for unit in units:
        keys=unit['ids'];p=np.asarray(unit['p'],float)
        if len(keys)==1:
            i=keys[0];a=forced[i] if i in forced else inverse_cdf(p,uniforms[i])
            if p[a]<=0:raise ValueError('Forced action outside natural support')
            actions[i]=int(a);continue
        i,j=keys
        if i in forced and j in forced:
            a,b=forced[i],forced[j]
            if p[a,b]<=0:raise ValueError('Forced pair outside natural support')
        elif i in forced:
            a=forced[i]
            if p[a].sum()<=0:raise ValueError('Forced action outside natural support')
            b=inverse_cdf(p[a]/p[a].sum(),uniforms[j])
        elif j in forced:
            b=forced[j]
            if p[:,b].sum()<=0:raise ValueError('Forced action outside natural support')
            a=inverse_cdf(p[:,b]/p[:,b].sum(),uniforms[i])
        else:
            marginal=p.sum(1);a=inverse_cdf(marginal,uniforms[i])
            b=inverse_cdf(p[a]/marginal[a],uniforms[j])
        actions[i]=int(a);actions[j]=int(b)
    return actions


def restore_prepared_state(model,assets,experiment,state,seed=None):
    if state['phase']!='after_prepare_before_action_draw':raise ValueError('Unsupported snapshot phase')
    road=NativeExperiment(model,assets/'pair',assets/'lateral',state['seed'],experiment)
    road.vehicles=[Vehicle(**c) for c in state['vehicles']]
    road.cav=next(c for c in road.vehicles if c.id==state['cav_id']);road.time=state['time_s']
    road.natural.histories={int(k):(tuple(v[0]) if v[0] is not None else None,v[1]) for k,v in state['natural_histories'].items()}
    road.natural.previous={tuple(p) for p in state['natural_previous']}
    if seed is None:road.rng.bit_generator.state=state['rng_state']
    else:road.rng=np.random.default_rng(seed)
    return road


def continue_state(model,assets,experiment,state,forced,seed,horizon,scale):
    road=restore_prepared_state(model,assets,experiment,state,seed)
    pmfs={int(k):np.asarray(p) for k,p in state['pmfs'].items()}
    # prepare already ran in the snapshot. Never age the histories a second time.
    road.action_pmfs=lambda:pmfs
    road.sample=lambda *args:conditional_actions(state['units'],forced,road.rng)
    road.step_policy('natural',1.)
    del road.action_pmfs;del road.sample
    while not road.crashes and road.time<state['time_s']+horizon-1e-9:
        road.step_policy('natural',1.)
    return {'clearance_risk':float(np.exp(-max(road.minimum_gap,0.)/scale)),
            'cav_first_collision':any(road.cav.id in pair for pair in road.crashes),
            'bv_only_collision':bool(road.crashes) and not any(road.cav.id in pair for pair in road.crashes)}


def run(diagnostic,config_path,output):
    diagnostic=Path(diagnostic);cfg=read(config_path)
    if cfg['replicates']<2 or cfg['horizon_s']<1 or cfg['horizon_s']!=int(cfg['horizon_s']):
        raise ValueError('Need >=2 replicates and a positive integer-second horizon')
    pool=read(diagnostic/'state_pool.json');source=read(diagnostic/'pair_diagnostics.json')
    source_summary=read(diagnostic/'summary.json')
    for key,name in [('state_pool_sha256','state_pool.json'),('pair_diagnostics_sha256','pair_diagnostics.json')]:
        if digest(diagnostic/name)!=source_summary[key]:raise ValueError('Diagnostic source checksum differs')
    assets,lock=ensure_assets()
    if pool['natural_target_sha256']!=lock['natural_target_sha256']:raise ValueError('Natural model differs')
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    experiment=read(pool['config']['experiment_config'])
    candidates=sorted((r for r in source['rows'] if r['proxy']==cfg['proxy'] and r['metrics']['synergy_eligible']),
                      key=lambda r:r['metrics']['positive_interaction_mass'],reverse=True)
    selected=[];seen=set()
    for row in candidates:
        if row['state_index'] in seen:continue
        selected.append(row);seen.add(row['state_index'])
        if len(selected)>=cfg['max_pairs']:break
    output=Path(output);output.mkdir(parents=True,exist_ok=False);records=[]
    for row in selected:
        state=pool['states'][row['state_index']];a0,b0=row['metrics']['baseline_actions']
        a,b=row['metrics']['max_interaction_actions'];corners=[(a0,b0),(a,b0),(a0,b),(a,b)]
        values=[];events=[];bv_events=[]
        for replicate in range(cfg['replicates']):
            results=[continue_state(model,assets,experiment,state,dict(zip(row['actor_ids'],corner)),
                cfg['continuation_seed_start']+replicate,cfg['horizon_s'],cfg['clearance_scale_m']) for corner in corners]
            values.append([r['clearance_risk'] for r in results])
            events.append([int(r['cav_first_collision']) for r in results]);bv_events.append([int(r['bv_only_collision']) for r in results])
        values=np.asarray(values);events=np.asarray(events)
        interaction=values[:,3]-values[:,1]-values[:,2]+values[:,0]
        record={'state_index':row['state_index'],'source_seed':state['seed'],'time_s':state['time_s'],
            'actor_ids':row['actor_ids'],'same_natural_unit':row['same_natural_unit'],
            'corners_base_first_second_joint':corners,'proxy_max_interaction':row['metrics']['max_positive_interaction'],
            'native_mean_clearance_risk':values.mean(0).tolist(),'native_cav_event_counts':events.sum(0).tolist(),
            'native_bv_only_event_counts':np.asarray(bv_events).sum(0).tolist(),
            'native_mean_interaction':float(interaction.mean()),
            'native_interaction_standard_error':float(interaction.std(ddof=1)/np.sqrt(len(interaction)))}
        records.append(record);save(output/f'witness_{row["state_index"]:03d}.json',dict(record,replicate_risk=values.tolist(),replicate_events=events.tolist()))
        print(record,flush=True)
    summary={'config':cfg,'config_sha256':digest(config_path),'implementation_sha256':digest(__file__),
        'source_diagnostic_sha256':digest(diagnostic/'summary.json'),'records':records,
        'scope':'Conditional training-state witness checks, common random numbers across four corners. Small selected sample; no population crash-rate, significance, PPO or acceleration claim. Native continuations preserve unforced natural partners.'}
    save(output/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',required=True)
    p.add_argument('--config',default='configs/highd_v46_pair_witness_v1.json');p.add_argument('--output',required=True)
    a=p.parse_args();print(run(a.diagnostic,a.config,a.output))


if __name__=='__main__':main()
