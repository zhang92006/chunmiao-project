"""Describe candidate coverage and informative action factors before tuning Q."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .highd_v46_assets import save
from .highd_v46_statistics import load_collection,summarize_sequences


def distribution(values):
    values=np.asarray(values,float)
    if not len(values):return None
    return {'count':len(values),'mean':float(values.mean()),
            'quantiles_0_50_90_100':np.quantile(values,[0,.5,.9,1]).tolist()}


def diagnose(collection,output):
    data=load_collection(collection);sequences=data['sequences']
    steps=[step for seq in sequences for step in seq['steps']]
    totals=Counter()
    for seq in sequences:totals.update(seq.get('diagnostics',{}))
    first=[];second=[];contrast=[];collision_actor_covered=0;positive_count=0
    for step in steps:
        tv=step.get('factor_influence_tv')
        if tv is not None:
            first.append(tv[0])
            if len(step['actor_ids'])==2:second.append(tv[1])
        if step.get('relative_action_contrast') is not None:contrast.append(step['relative_action_contrast'])
    for seq in sequences:
        if not seq['event_result']:continue
        positive_count+=1
        crash_ids={i for pair in seq.get('collision_pairs',[]) for i in pair if i!=seq.get('cav_id')}
        selected={i for step in seq['steps'] for i in step['actor_ids']}
        collision_actor_covered+=bool(crash_ids&selected)
    decisions=totals['decision_steps']
    report={'schema_version':1,'experiment_target_sha256':data['experiment_target_sha256'],
            'proposal_structure_sha256':data.get('proposal_structure_sha256'),
            'mode':data['intervention_mode'],'episodes':len(sequences),'totals':dict(totals),
            'critical_step_fraction':len(steps)/decisions if decisions else None,
            'natural_pair_available_fraction':totals['pair_available_steps']/decisions if decisions else None,
            'two_factor_fraction_of_critical_steps':sum(len(s['actor_ids'])==2 for s in steps)/len(steps) if steps else None,
            'adversarial_first_factor_tv':distribution(first),'adversarial_second_factor_tv':distribution(second),
            'weak_second_factor_fraction_tv_below_0_001':float(np.mean(np.asarray(second)<.001)) if second else None,
            'relative_action_contrast':distribution(contrast),
            'terminal_log_weight':distribution([s['generation_log_weight'] for s in sequences]),
            'positive_episodes':positive_count,'positive_episodes_with_collision_BV_ever_selected':collision_actor_covered,
            'statistics':summarize_sequences(sequences),
            'interpretation':'Factor TV describes H versus P, not calibrated crash probability or causal proof. Actor coverage is descriptive. Threshold changes require fresh collection, not post-hoc deletion of episodes.'}
    out=Path(output);out.mkdir(parents=True,exist_ok=False);save(out/'diagnostics.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(diagnose(a.collection,a.output))


if __name__=='__main__':main()
