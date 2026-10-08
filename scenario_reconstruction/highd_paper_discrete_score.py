"""Fixed-bin, 10 Hz scoring of completed discrete-maneuver collision screens."""
import argparse
from pathlib import Path
import numpy as np
from .highd_paper_data import read,write,sha,histogram
from .highd_paper_dynamics import EmpiricalModel,Road,advance,collisions
from .highd_paper_experiment import metrics


def score(model_root,reference,bank,rollout,output):
    root=Path(rollout);screen=read(root/'collision_summary.json')
    if not all(r['complete'] for r in screen['runs']):raise ValueError('Collision screen must pass before distribution scoring')
    m=EmpiricalModel(model_root);ref=read(Path(reference)/'reference_summary.json')
    if m.summary['model_sha256']!=screen['model_sha256'] or sha(Path(reference)/'reference.npz')!=ref['reference_sha256']:raise ValueError('Model/reference mismatch')
    with np.load(Path(reference)/'reference.npz') as f:targets={k:f['reference_'+k+'_counts'] for k in ('speed','gap','rr')}
    specs={'speed':(20,40,1),'gap':(0,115,1),'rr':(-20,20,1)}
    collection={k:np.zeros_like(v,dtype=np.uint64) for k,v in targets.items()};bytime={};runs=[]
    for saved in screen['runs']:
        seed=saved['seed'];snapshot=Path(bank)/f'initial_seed{seed}.json'
        if sha(snapshot)!=saved['initial_fleet_sha256']:raise ValueError('Initial fleet changed')
        road=Road(m,seed,read(snapshot));recorded=read(root/f'run_seed{seed}.json');steps=recorded['decisions']
        last=round(saved['elapsed_s']*10);own={k:np.zeros_like(v,dtype=np.uint64) for k,v in targets.items()};timeline={};extra_collisions=[]
        for tick in range(last+1):
            sample=road.sample()
            for key,target in targets.items():
                counts=histogram(sample[key],*specs[key]).astype(np.uint64);own[key]+=counts;collection[key]+=counts
                if tick in (0,10,30,50,100,300,600):
                    time=str(tick/10);bytime.setdefault(time,{k:np.zeros_like(v,dtype=np.uint64) for k,v in targets.items()})[key]+=counts
                    timeline.setdefault(time,{})[key]=metrics(counts,target)
            if tick==last:break
            step=steps[tick//10]
            if tick%10==0:
                changes={}
                for car in road.vehicles:
                    code=step['actions'][str(car.id)]
                    if code in (0,32):
                        car.target=car.lane+(1 if code==0 else -1);car.elapsed=0.;changes[car.id]=(car.lane,car.target)
                actions=[step['accelerations'][str(car.id)] for car in road.vehicles]
            advance(road.vehicles,actions,.1,{**m.config,'lane_change_duration_s':1.})
            for car in road.vehicles:
                if car.id in changes:
                    origin,target=changes[car.id];fraction=(tick%10+1)/10
                    car.y=(origin+(target-origin)*fraction)*m.config['lane_width_m']
            crash=collisions(road.vehicles)
            if crash:extra_collisions.append({'time_s':(tick+1)/10,'pairs':crash})
        runs.append({'seed':seed,'duration_s':saved['elapsed_s'],'lane_change_onsets':saved['diagnostics'].get('lane_change_onsets',0),
            'collection':{k:metrics(v,targets[k]) for k,v in own.items()},'by_time':timeline,'additional_10hz_collision_observations':extra_collisions})
    report={'collision_screen_sha256':sha(root/'collision_summary.json'),'reference_sha256':ref['reference_sha256'],
        'runs':runs,'collection':{k:metrics(v,targets[k]) for k,v in collection.items()},
        'by_time':{t:{k:metrics(v,targets[k]) for k,v in d.items()} for t,d in bytime.items()},
        'scope':'All declared seeds complete the full 60 s collision screen. Reconstruct the same held 1 s actions at 10 Hz for original fixed-bin/legacy-neighbor scoring. Direct constant-acceleration kinematics and linear lateral path; this is not an additional stochastic run or independent evidence. Empirical feasibility exclusions and collision-screen selection remain disclosed.'}
    report['development_metric_gate']=report['collection']['speed']['hellinger']<=.147 and report['collection']['gap']['hellinger']<=.197 and not any(r['additional_10hz_collision_observations'] for r in runs)
    report['natural_law_accepted']=False
    write(output,report);print({'H':{k:v['hellinger'] for k,v in report['collection'].items()},'metric_gate':report['development_metric_gate'],'lane_changes':[r['lane_change_onsets'] for r in runs]})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','reference','initial-bank','rollout','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();score(a.model,a.reference,a.initial_bank,a.rollout,a.output)
