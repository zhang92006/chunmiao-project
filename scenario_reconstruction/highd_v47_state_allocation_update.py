"""One explicit development update; retain the failed state-allocation experiment."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np

from . import highd_v47_state_allocation as parent
from .highd_v46_assets import read, save, digest
from .highd_v46_d2rl import identity
from .highd_v46_collision_coverage import audit_sample

DEFAULT = 'configs/highd_v47_state_allocation_update_v1.json'


def add_observation(table, state_id, sample):
    key = (state_id, sample['seed'])
    value = (bool(sample['event_result']), float(sample['generation_log_weight']))
    if key in table and table[key] != value:
        raise ValueError('Same physical state/seed has inconsistent labels')
    table[key] = value


def inputs(config):
    cfg = read(config)
    if (cfg['version'] != 1 or cfg['state_floor'] != .5 or cfg['M2_regularization'] != .01
            or cfg['draws_per_arm'] != 2112 or cfg['state_rng_seed'] != 490017 or cfg['continuation_seed_start'] != 4900000):
        raise ValueError('Version the development update before changing it')
    pc, base, physical, assets, plans, designs, _, parent_lock = parent.inputs(cfg['parent_config'])
    root = Path(cfg['development_path'])
    if digest(root/'summary.json') != cfg['development_summary_sha256']:
        raise ValueError('Completed development summary changed')
    old = read(root/'summary.json')
    if not old['complete'] or any(old['lock'].get(k) != v for k, v in parent_lock.items()):
        raise ValueError('Development experiment or target differs')
    for name, sha in old['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Development artifact changed')
    table = {}; raw_count = 0
    for i in range(len(plans)):
        for sample in read(Path(pc['teacher_path'])/f'case_{i:03d}_labels.json')['sequences']:
            add_observation(table, i, sample); raw_count += 1
    for arm in pc['arms']:
        for sample in read(root/f'{arm}_sequences.json')['sequences']:
            audit_sample(sample, pc['epsilon'])
            add_observation(table, sample['state_id'], sample); raw_count += 1
    alpha = np.array([p['target_mass'] for p in plans]); m2 = []; counts = []
    for i in range(len(plans)):
        z = [event*np.exp(lw) for (state_id, seed), (event, lw) in table.items() if state_id == i]
        counts.append(len(z)); m2.append(float(np.mean(np.square(z))))
    q, reg = parent.state_allocation(alpha, m2, cfg['state_floor'], cfg['M2_regularization'])
    allocation = {'target_masses': alpha.tolist(), 'state_Q': q.tolist(), 'pilot_M2': m2,
                  'regularized_M2': reg.tolist(), 'fitting_counts': counts, 'raw_rows': raw_count,
                  'unique_state_seed_rows': len(table), 'duplicate_rows_not_counted': raw_count-len(table),
                  'source_Q': {str(s): float(sum(q[i] for i,p in enumerate(plans) if p['source_seed']==s)) for s in sorted({p['source_seed'] for p in plans})},
                  'schedules': parent.schedule(alpha, q, cfg['draws_per_arm'], cfg['state_rng_seed'])}
    lock = {'config_sha256': digest(config), 'implementation_sha256': digest(__file__), 'parent': parent_lock,
            'development_summary_sha256': cfg['development_summary_sha256'], 'allocation_sha256': identity(allocation)}
    return cfg, pc, base, physical, assets, plans, designs, allocation, lock


def freeze(config, output):
    *_, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def run(config, lock_path, output):
    cfg, pc, base, physical, assets, plans, designs, allocation, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k,v in expected.items()):
        raise ValueError('Updated proposal lock differs')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', {'config': cfg, 'lock': lock})
    save(out/'state_allocation.json', dict(allocation, plans=plans))
    model = parent.PairModelQueries(assets/'model', assets/'lateral', **parent.SINGLE_OPTIONS)
    alpha = np.array(allocation['target_masses']); q = np.array(allocation['state_Q'])
    results = {}; started = time.perf_counter()
    for arm in pc['arms']:
        samples = []; tick = time.perf_counter()
        qs = alpha if arm == 'target_iid' else q if arm == 'allocated_iid' else np.full(len(plans), 1/len(plans))
        for j,i in enumerate(allocation['schedules'][arm]):
            plan,d = plans[i],designs[i]
            s = parent.sample_continuation(model, assets, physical, d['state'], {'ids':plan['actor_ids'], 'h':np.array(d['H'])},
                pc['epsilon'], cfg['continuation_seed_start']+j, pc, base['proxy']['clearance_scale_m'])
            audit_sample(s,pc['epsilon']); slw=float(np.log(alpha[i]/qs[i]))
            s.update(draw_id=j,state_id=i,target_state_mass=float(alpha[i]),generation_state_mass=float(qs[i]),
                     state_log_weight=slw,total_log_weight=slw+s['generation_log_weight'])
            samples.append(s)
            if (j+1)%512==0:print({'arm':arm,'draws':j+1},flush=True)
        save(out/f'{arm}_sequences.json',{'arm':arm,'sequences':samples})
        results[arm]=dict(parent.summarize(samples,plans,arm),wall_s=time.perf_counter()-tick)
        print({'arm':arm,**{k:results[arm][k] for k in ('events','mean','estimator_variance','maximum_source_second_moment_share')}},flush=True)
    ratios={a:results['allocated_iid']['estimator_variance']/results[a]['estimator_variance'] if results[a]['estimator_variance']>0 else None for a in ('target_iid','equal_state_stratified')}
    result={'complete':True,'config':cfg,'lock':lock,'arms':results,'allocated_variance_ratio':ratios,
            'total_wall_s':time.perf_counter()-started,'source_generalization_validated':False,'ppo_ready':False,
            'artifact_hashes':{p.name:digest(p) for p in sorted(out.glob('*.json'))}}
    save(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','run']);p.add_argument('--config',default=DEFAULT)
    p.add_argument('--lock');p.add_argument('--output',required=True);a=p.parse_args()
    if a.mode=='freeze':print(freeze(a.config,a.output))
    else:print(run(a.config,a.lock,a.output)['allocated_variance_ratio'])


if __name__=='__main__':main()
