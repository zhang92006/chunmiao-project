"""Native conditional confirmation of episode-diverse discovery candidates."""
import argparse
from pathlib import Path
import numpy as np
from .highd_v46_assets import read,save,digest,ensure_assets
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
from .highd_v46_pair_witness import continue_state


def select_witnesses(rows,cfg):
    candidates=[r for r in rows if cfg['method'] in r['methods'] and r['views'][cfg['view']]
                and r['time_s']+cfg['horizon_s']<=cfg['source_horizon_s']
                and r['selection']==cfg['pair_selection']]
    candidates.sort(key=lambda r:(-r['metrics']['positive_interaction_mass'],r['seed'],r['time_s'],r['actor_ids']))
    chosen=[];seen=set()
    for row in candidates:
        if row['seed'] in seen:continue
        chosen.append(row);seen.add(row['seed'])
        if len(chosen)>=cfg['maximum_source_episodes']:break
    return chosen


def run(discovery,config_path,output):
    source=Path(discovery);cfg=read(config_path);summary=read(source/'summary.json')
    if cfg['maximum_source_episodes']<1 or cfg['replicates_per_corner']<2:raise ValueError('Invalid witness budget')
    for field,name in [('state_bank_sha256','state_bank.json'),('pair_audit_sha256','pair_audit.json')]:
        if digest(source/name)!=summary[field]:raise ValueError('Discovery source checksum differs')
    bank=read(source/'state_bank.json');rows=read(source/'pair_audit.json')['rows']
    assets,lock=ensure_assets()
    if bank['natural_target_sha256']!=lock['natural_target_sha256']:raise ValueError('Natural model changed')
    physical=read(bank['config']['physical_config'])
    if physical['horizon_s']!=cfg['source_horizon_s']:raise ValueError('Witness source horizon differs')
    if digest(bank['config']['physical_config'])!=summary['dependencies']['physical_config']:
        raise ValueError('Physical task changed since discovery')
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    selected=select_witnesses(rows,cfg);output=Path(output);output.mkdir(parents=True,exist_ok=False);records=[]
    for row in selected:
        state=bank['states'][row['state_index']];a0,b0=row['metrics']['baseline_actions']
        a,b=row['metrics']['max_interaction_actions'];corners=[(a0,b0),(a,b0),(a0,b),(a,b)]
        values=[];events=[]
        for seed in range(cfg[summary['split']+'_continuation_seed_start'],cfg[summary['split']+'_continuation_seed_start']+cfg['replicates_per_corner']):
            samples=[continue_state(model,assets,physical,state,dict(zip(row['actor_ids'],corner)),seed,cfg['horizon_s'],cfg['clearance_scale_m']) for corner in corners]
            values.append([s['clearance_risk'] for s in samples]);events.append([int(s['cav_first_collision']) for s in samples])
        values=np.asarray(values);interaction=values[:,3]-values[:,1]-values[:,2]+values[:,0]
        record={'seed':row['seed'],'time_s':row['time_s'],'actor_ids':row['actor_ids'],'mechanism':row['mechanism'],
            'production_strict':row['views']['production_strict'],'proxy_interaction':row['metrics']['max_positive_interaction'],
            'corners_base_first_second_joint':corners,'native_mean_risk':values.mean(0).tolist(),
            'native_cav_events':np.asarray(events).sum(0).tolist(),'native_interaction_mean':float(interaction.mean()),
            'native_interaction_se':float(interaction.std(ddof=1)/np.sqrt(len(interaction)))}
        records.append(record);print(record,flush=True)
        save(output/f'case_{row["seed"]}.json',dict(record,replicate_risk=values.tolist(),replicate_events=events))
    result={'split':summary['split'],'config':cfg,'config_sha256':digest(config_path),'implementation_sha256':digest(__file__),
        'source_summary_sha256':digest(source/'summary.json'),'records':records,
        'scope':'Proxy-selected conditional cases, unmodified stochastic native continuations. Four corners share random numbers. No population rate, global discovery recall or statistical-significance claim.'}
    save(output/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--discovery',required=True)
    p.add_argument('--config',default='configs/highd_v46_discovery_witness_v1.json');p.add_argument('--output',required=True)
    a=p.parse_args();print(run(a.discovery,a.config,a.output))


if __name__=='__main__':main()
