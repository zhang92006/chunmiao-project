"""Conditional one-decision proposal pilot on a known training state.

This deliberately exports no population-P sequence manifest or acceleration.
"""
import argparse
from pathlib import Path
import numpy as np
from .highd_v46_assets import ensure_assets,read,save,digest
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
from .highd_v46_pair_witness import restore_prepared_state
from .highd_v46_cross_unit import CrossUnitExperiment,draw_intervention,proposal_structure,validate_config
from .highd_v46_d2rl import identity
from .highd_v46_statistics import summarize_sequences


def run(pool_path,config_path,output):
    cfg=read(config_path)
    if digest(pool_path)!=cfg['state_pool_sha256']:raise ValueError('Probe source state bank differs')
    pool=read(pool_path);state=pool['states'][cfg['state_index']]
    proposal=read(cfg['proposal_config']);physical=read(proposal['physical_config']);validate_config(proposal,physical)
    if (cfg['replicates_per_arm']<1 or cfg['horizon_s']<1 or cfg['horizon_s']!=int(cfg['horizon_s'])
        or cfg['later_actions']!='Unchanged natural NDE; only initial decision is tilted'):
        raise ValueError('Unsupported conditional probe protocol')
    assets,lock=ensure_assets()
    if pool['natural_target_sha256']!=lock['natural_target_sha256']:raise ValueError('Natural target differs')
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    road=restore_prepared_state(model,assets,physical,state)
    road.proposal_config=proposal
    units=[{'ids':u['ids'],'p':np.asarray(u['p'])} for u in state['units']]
    actors={c.id:c for c in road.vehicles};pmfs={int(k):np.asarray(p) for k,p in state['pmfs'].items()}
    chosen=CrossUnitExperiment.select_pair(road,units,actors,pmfs)
    if chosen is None or chosen['ids']!=cfg['expected_selected_pair']:raise ValueError('Expected training-state pair not selected')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);summaries={}
    for name,epsilon in cfg['arms'].items():
        sequences=[]
        for seed in range(cfg['continuation_seed_start'],cfg['continuation_seed_start']+cfg['replicates_per_arm']):
            road=restore_prepared_state(model,assets,physical,state,seed)
            actions,logged=draw_intervention(units,chosen['ids'],chosen['h'],epsilon,road.rng)
            road.action_pmfs=lambda:pmfs;road.sample=lambda *args:actions
            road.step_policy('natural',1.)
            del road.action_pmfs;del road.sample
            while not road.crashes and road.time<state['time_s']+cfg['horizon_s']-1e-9:
                road.step_policy('natural',1.)
            seq={'seed':seed,'event_result':any(road.cav.id in p for p in road.crashes),
                'generation_log_weight':logged['log_weight'],'initial_likelihood_audit':logged,
                'collision_pairs':road.crashes,'clearance_risk':float(np.exp(-max(road.minimum_gap,0.)/proposal['proxy']['clearance_scale_m']))}
            sequences.append(seq)
        save(output/f'{name}_conditional_episodes.json',{'scope':cfg['scope'],'sequences':sequences})
        summaries[name]={**summarize_sequences(sequences),'epsilon':epsilon,
            'mean_clearance_risk':float(np.mean([s['clearance_risk'] for s in sequences])),
            'mean_likelihood_ratio':float(np.mean([np.exp(s['generation_log_weight']) for s in sequences]))}
        print({'arm':name,**summaries[name]},flush=True)
    result={'config':cfg,'config_sha256':digest(config_path),'implementation_sha256':digest(__file__),
        'proposal_structure_sha256':identity(proposal_structure(proposal,'dual')),
        'source_state':{'seed':state['seed'],'time_s':state['time_s'],'cav_id':state['cav_id']},
        'selected_pair':chosen['ids'],'synergy_mass':chosen['mass'],'factor_tv':chosen['factor_influence'],
        'arms':summaries,'scope':'Conditional risk given one previously selected training state. Later dynamics are natural. No reset-density correction or population/acceleration claim; not a final-test dataset.'}
    save(output/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-pool',required=True)
    p.add_argument('--config',default='configs/highd_v46_cross_unit_conditional_probe_v1.json');p.add_argument('--output',required=True)
    a=p.parse_args();print(run(a.state_pool,a.config,a.output))


if __name__=='__main__':main()
