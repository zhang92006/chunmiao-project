"""Fresh held-out native validation of the completed shared-teacher PPO runs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import subprocess
import numpy as np

from . import highd_v47_shared_teacher as training
from . import highd_v47_shared_teacher_recovery as recovery
from . import highd_v47_teacher_expansion as expansion
from . import highd_v47_shared_validation_native as native
from .highd_v46_assets import ROOT,read,save,digest,ensure_assets
from .highd_v46_d2rl import identity
from .highd_v46_contract import FIXED_ALPHA_OBSERVATION,replay_observation
from .highd_v46_native_calibration import discover as natural_discover
from .highd_v46_policy import NumpyPolicy
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
from .highd_v47_fixed_alpha_contract import state_schedule
from .highd_v47_formal_statistics import describe,select_checkpoint,readiness
from .highd_v47_formal_comparison import canonical_path
from d2rl_training.highd_v46_sequence_env import log_weight

DEFAULT='configs/highd_v47_shared_teacher_validation_v1.json'
_WORK=None
_POLICIES={}


def protocol(config):
    cfg=read(config)
    fixed=dict(version=1,source=dict(split='validation',seed_start=1800010000,episodes=1024),
        continuation_seed_start=2000000000,state_seed=2000017,draws_per_checkpoint=4096,
        checkpoint_iterations=list(range(0,201,10)),training=dict(seeds=[7,17,29,41,53]),
        workers=4,batch_size=128,horizon_s=4,state_likelihood_ratio=1.,state_model=None,
        source_claim='outputs/v47_shared_teacher_validation_v1_source_claim.json')
    if any(cfg[k]!=v for k,v in fixed.items()):raise ValueError('Version changed validation design before opening sources')
    training.components(cfg['training_config'])  # pinned Python/NumPy/Ray/Torch
    assets,frozen=ensure_assets()
    deps={str(p.relative_to(ROOT)):digest(p) for folder in ('scenario_reconstruction','d2rl_training')
          for p in sorted((ROOT/folder).glob('highd*.py'))}
    dc=read('configs/highd_v46_discovery_v2.json');base=read(dc['base_discovery_config'])
    configs={p:digest(p) for p in ('configs/highd_v46_discovery_v2.json',dc['base_discovery_config'],
        base['physical_config'],'configs/highd_v47_teacher_expansion_v1.json',cfg['training_config'])}
    return cfg,read(base['physical_config']),assets,dict(config_sha256=digest(config),dependencies=deps,
        configs=configs,natural_target_sha256=frozen['natural_target_sha256'])


def verify_training(cfg):
    tc,tl,ts=training.training_inputs(cfg['training_config'],cfg['training_lock'],cfg['training_data'],cfg['training_audit'])
    if not ts['readiness']['passed']:raise ValueError('Training signal audit failed')
    root=Path(cfg['training_results']);s=read(root/'summary.json');r=read(root/'recovery_protocol.json')
    if (digest(root/'summary.json')!=cfg['training_summary_sha256'] or not s['complete']
            or s['lock_sha256']!=identity(tl) or s['recovery_protocol_sha256']!=digest(root/'recovery_protocol.json')
            or r['recovery_sha256']!=digest(recovery.__file__) or r['lock_sha256']!=identity(tl)):
        raise ValueError('Completed recovery identity differs')
    expected={f'{m}_seed{seed}' for seed in cfg['training']['seeds'] for m in ('single','dual')}
    if set(s['runs'])!=expected:raise ValueError('All ten paired trained runs required')
    for rel,sha in r['previous_files'].items():
        if digest(Path(r['previous'])/rel)!=sha:raise ValueError('Original training evidence changed')
    ppo=read(root/'ppo_config.json')
    if digest(root/'ppo_config.json')!=r['ppo_config_sha256']:raise ValueError('PPO config changed')
    hashes={str(p):digest(p) for p in (root/'summary.json',root/'recovery_protocol.json',root/'ppo_config.json',
        Path(cfg['training_lock']),Path(cfg['training_audit']),Path(cfg['training_data'])/'summary.json')}
    arms=[dict(label='natural',mode='natural',seed=None,iteration=None,policy_path=None)]
    validation=set(range(cfg['source']['seed_start'],cfg['source']['seed_start']+cfg['source']['episodes']))
    for seed in cfg['training']['seeds']:
        for mode in ('single','dual'):
            name=f'{mode}_seed{seed}';path=Path(s['runs'][name])
            recovery.completed_run(path,mode,seed,ppo,cfg['training_data'])
            meta_train=read(Path(cfg['training_data'])/mode/'sequence_manifest.json')
            for p in (path/'training_summary.json',path/'checkpoint_catalog.json'):hashes[str(p)]=digest(p)
            for item in read(path/'checkpoint_catalog.json')['checkpoints']:
                policy=path/item['policy_path'];meta=read(policy/'policy.json');sources=set(meta['training_source_seeds'])
                if (not sources or not sources<=set(range(1800000000,1800004096)) or sources&validation
                        or meta['validation_source_seeds'] or meta['observation_contract']!=FIXED_ALPHA_OBSERVATION
                        or meta['state_likelihood_ratio']!=1 or meta['state_sampling']!='fixed_stratified_alpha'
                        or meta['intervention_mode']!=mode or meta['action_dimension']!=(1 if mode=='single' else 2)
                        or any(meta[k]!=meta_train[k] for k in ('natural_target_sha256','experiment_target_sha256',
                            'conditional_target_sha256','state_law_sha256','proposal_structure_sha256'))):
                    raise ValueError('Model is incompatible with held-out shared-teacher target')
                for p in (policy/'policy.json',policy/'policy_weights.npz'):hashes[str(p)]=digest(p)
                arms.append(dict(label=f'{name}_iter{item["iteration"]}',mode=mode,seed=seed,
                    iteration=item['iteration'],policy_path=str(policy)))
    return hashes,arms


def freeze(config,output):
    cfg,_,_,current=protocol(config)
    if Path(cfg['source_claim']).exists():raise ValueError('Reserved validation already opened; retain existing attempt')
    hashes,arms=verify_training(cfg)
    lock=dict(current,config=cfg,inputs=hashes,arms=arms,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),validation_opened=False)
    save(output,lock);return dict(frozen=True,validation_opened=False,arms=len(arms),maximum_continuations=len(arms)*cfg['draws_per_checkpoint'])


def checked(config,lock_path):
    cfg,physical,assets,current=protocol(config);lock=read(lock_path)
    if lock['config']!=cfg or any(lock[k]!=v for k,v in current.items()):raise ValueError('Frozen validation implementation/config changed')
    for p,sha in lock['inputs'].items():
        if digest(p)!=sha:raise ValueError('Frozen trained evidence changed: '+p)
    return cfg,physical,assets,lock


def initialize(assets,physical,cases=None):
    global _WORK,_POLICIES
    assets=Path(assets);_WORK=(PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS),assets,physical,cases)
    _POLICIES={}


def discovery_worker(task):
    start,output=task;model,assets,physical,_=_WORK
    dc=read('configs/highd_v46_discovery_v2.json');base=read(dc['base_discovery_config'])
    source=dict(split='validation',seed_start=start,episodes=64)
    bank,rows,report=natural_discover(dict(source=source),dc,base,physical,model,assets)
    bank=expansion.json_value(bank);indices,funnel=expansion.selected_rows(rows,list(range(start,start+64)))
    name=f'chunk_{start}.json';save(Path(output)/name,dict(source=source,bank=bank,rows=rows,report=report,
        selected_indices=indices,funnel=funnel,usage='Held-out validation only; never fitting data'))
    return name,digest(Path(output)/name)


def build_bank(chunks,cfg):
    cases=[];sources=[]
    for name,chunk in chunks:
        start=chunk['source']['seed_start'];seeds=list(range(start,start+64))
        if [e['seed'] for e in chunk['report']['episodes']]!=seeds:raise ValueError('Incomplete discovery sources')
        indices,funnel=expansion.selected_rows(chunk['rows'],seeds)
        if indices!=chunk['selected_indices'] or funnel!=chunk['funnel']:raise ValueError('Discovery selection changed')
        sources.extend(funnel)
        for i in indices:
            row=chunk['rows'][i];state=chunk['bank'][row['state_index']]
            if state['seed']!=row['seed']:raise ValueError('Selected state/source mismatch')
            cases.append(dict(case_id=len(cases),source_seed=state['seed'],state=state,state_sha256=identity(state),
                actor_ids=row['actor_ids'],discovery_file=name,row_index=i))
    expected=list(range(cfg['source']['seed_start'],cfg['source']['seed_start']+cfg['source']['episodes']))
    if [s['source_seed'] for s in sources]!=expected:raise ValueError('Complete ordered validation cohort required')
    counts=Counter(c['source_seed'] for c in cases)
    if counts and max(counts.values())>3:raise ValueError('Per-source bank budget exceeded')
    if len({(c['source_seed'],c['state_sha256'],tuple(c['actor_ids'])) for c in cases})!=len(cases):raise ValueError('Duplicate bank case')
    for c in cases:c['alpha']=1/(len(counts)*counts[c['source_seed']])
    return dict(cases=cases,source_records=sources,selected_sources=len(counts),selection_uses_outcomes=False)


def bank(config,lock_path,output):
    cfg,physical,assets,lock=checked(config,lock_path);out=Path(output)
    if out.exists():raise FileExistsError(out)
    save(cfg['source_claim'],dict(lock_sha256=identity(lock),bank_output=str(out.resolve()),source=cfg['source']))
    out.mkdir(parents=True,exist_ok=False);save(out/'opened_protocol.json',dict(lock=lock))
    hashes={};start=cfg['source']['seed_start']
    with ProcessPoolExecutor(cfg['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
                             initargs=(str(assets),physical)) as pool:
        jobs=[pool.submit(discovery_worker,(s,str(out))) for s in range(start,start+cfg['source']['episodes'],64)]
        for future in as_completed(jobs):
            name,sha=future.result();hashes[name]=sha
            print(dict(phase='validation_discovery',chunks=len(hashes),total=cfg['source']['episodes']//64),flush=True)
    b=build_bank(((n,read(out/n)) for n in sorted(hashes)),cfg)
    save(out/'bank.json',dict(b,lock_sha256=identity(lock)))
    hashes.update({n:digest(out/n) for n in ('bank.json','opened_protocol.json')})
    save(out/'summary.json',dict(complete=True,lock_sha256=identity(lock),hashes=hashes,selected_sources=b['selected_sources'],cases=len(b['cases'])))
    return dict(complete=True,selected_sources=b['selected_sources'],cases=len(b['cases']),validation_continuations_opened=False)


def load_bank(root,lock,reconstruct=False):
    root=Path(root);s=read(root/'summary.json');cfg=lock['config'];claim=read(cfg['source_claim'])
    if (not s['complete'] or s['lock_sha256']!=identity(lock) or claim['lock_sha256']!=identity(lock)
            or claim['bank_output']!=str(root.resolve())):raise ValueError('Wrong validation source claim/bank')
    for name,sha in s['hashes'].items():
        if Path(name).name!=name or digest(root/name)!=sha:raise ValueError('Bank artifact changed')
    b=read(root/'bank.json')
    if b['lock_sha256']!=identity(lock):raise ValueError('Wrong bank identity')
    if reconstruct:
        expected=build_bank(((n,read(root/n)) for n in sorted(s['hashes']) if n.startswith('chunk_')),cfg)
        if any(b[k]!=v for k,v in expected.items()):raise ValueError('Outcome-blind bank reconstruction differs')
    if not b['cases']:raise ValueError('Empty eligible bank; no backfill or fabricated zero outcome')
    for c in b['cases']:
        if identity(c['state'])!=c['state_sha256']:raise ValueError('State hash mismatch')
    return b


def batch_worker(task):
    arm,draws=task;model,assets,physical,cases=_WORK;path=arm['policy_path']
    if path and path not in _POLICIES:_POLICIES[path]=NumpyPolicy(path)
    policy=_POLICIES.get(path)
    return [native.rollout(model,assets,physical,arm['mode'],seed,cases[k],policy) for k,seed in draws]


def report_selection(metrics,lock):
    selected=[];cfg=lock['config']
    for seed in cfg['training']['seeds']:
        for mode in ('single','dual'):
            name=f'{mode}_seed{seed}'
            reports={i:metrics[f'{name}_iter{i}'] for i in cfg['checkpoint_iterations']}
            iteration,reason=select_checkpoint(reports)
            arm=next(a for a in lock['arms'] if a['label']==f'{name}_iter{iteration}')
            selected.append(dict(arm,reason=reason,metrics=reports[iteration]))
    return dict(selected=selected,readiness=readiness(selected,cfg),
        scope='Native source-disjoint checkpoint selection only; selected comparisons are selection-biased, not FinalTest',
        final_test_opened=False)


def run(config,lock_path,bank_path,output):
    cfg,physical,assets,lock=checked(config,lock_path);b=load_bank(bank_path,lock,True)
    schedule=state_schedule(b['cases'],cfg['draws_per_checkpoint'],cfg['state_seed'])
    draws=[(k,cfg['continuation_seed_start']+i) for i,k in enumerate(schedule)]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    save(out/'opened_protocol.json',dict(lock_sha256=identity(lock),bank_summary_sha256=digest(Path(bank_path)/'summary.json')))
    save(out/'schedule.json',dict(arms=lock['arms'],draws=draws,selection_uses_outcomes=False))
    hashes={n:digest(out/n) for n in ('opened_protocol.json','schedule.json')};metrics={}
    with ProcessPoolExecutor(cfg['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
                             initargs=(str(assets),physical,b['cases'])) as pool:
        for arm in lock['arms']:
            records=[];tasks=[(arm,draws[i:i+cfg['batch_size']]) for i in range(0,len(draws),cfg['batch_size'])]
            for j,part in enumerate(pool.map(batch_worker,tasks,chunksize=1)):
                name=f'{arm["label"]}/batch_{j:04d}.json';save(out/name,part);hashes[name]=digest(out/name);records.extend(part)
                print(dict(phase='native_validation',arm=arm['label'],completed=len(records),total=len(draws)),flush=True)
            metrics[arm['label']]=describe(records,cfg)
            name=arm['label']+'/metrics.json';save(out/name,metrics[arm['label']]);hashes[name]=digest(out/name)
    selection=report_selection(metrics,lock);save(out/'selection.json',selection);hashes['selection.json']=digest(out/'selection.json')
    save(out/'summary.json',dict(complete=True,lock_sha256=identity(lock),hashes=hashes,
        logical_continuations=len(draws)*len(lock['arms']),readiness=selection['readiness'],validation_is_selection=True,final_test_opened=False))
    return dict(complete=True,readiness=selection['readiness'],audit_pending=True,final_test_opened=False)


def check_record(s,case,seed,policy,mode):
    if (s['seed']!=seed or s['case_id']!=case['case_id'] or s['source_seed']!=case['source_seed']
            or s['state_target_mass']!=case['alpha'] or s['state_proposal_mass']!=case['alpha']
            or s['state_likelihood_ratio']!=1 or s['state_log_weight']!=0
            or s['event_result']!=any(s['cav_id'] in p for p in s['collision_pairs'])):raise ValueError('Native endpoint/schedule/target mismatch')
    if not 0<s['elapsed_s']<=4+1e-9:raise ValueError('Wrong stopped horizon')
    expected_event='cav_first_collision' if s['event_result'] else 'bv_only_competing_collision' if s['collision_pairs'] else 'horizon_no_collision'
    if s['event_type']!=expected_event:raise ValueError('Endpoint type mismatch')
    prefix=0.;last=-1.
    for t in s['steps']:
        dim=1 if mode=='single' else 2
        if (mode=='natural' or not last<t['time_s']<4 or t['time_s']<0
                or t['factor_mask'] not in ([1],[1,0],[1,1]) or len(t['factor_mask'])!=dim
                or len(t['actor_ids'])!=sum(t['factor_mask']) or len(t['factor_pmfs'])!=dim
                or not np.isclose(t['observation'][16],np.clip(prefix/30.,-5,5),atol=1e-12)
                or not np.isclose(t['observation'][17],1-t['time_s']/4,atol=1e-12)
                or t['observation'][18:18+dim]!=t['factor_mask']):raise ValueError('Target time/mask/prefix differs')
        decision=next((d for d in s['native_decisions'] if d['time_s']==t['time_s']),None)
        if decision is None or t['action_indices']!=[decision['actions'][str(i)] for i in t['actor_ids']]:
            raise ValueError('Target action detached from native decision')
        if decision['log_p']!=t['full_log_p'] or decision['log_q']!=t['full_log_q']:
            raise ValueError('Target density detached from native decision')
        eps=policy(replay_observation(t,prefix,FIXED_ALPHA_OBSERVATION))
        if not np.allclose(eps,t['generation_epsilon'],rtol=0,atol=1e-12):raise ValueError('Policy/causal observation mismatch')
        for j,f in enumerate(t['factor_pmfs']):
            p,h=np.asarray(f['p']),np.asarray(f['h']);active=t['factor_mask'][j];a=t['action_indices'][j] if active else 0
            if (not np.isfinite(p).all() or not np.isfinite(h).all() or np.any(p<0) or np.any(h<0) or not np.isclose(p.sum(),1) or not np.isclose(h.sum(),1)
                    or np.any(h[p==0]>0) or p[a]!=t['p_factors'][j] or h[a]!=t['h_factors'][j]
                    or not np.isclose(eps[j]*p[a]+(1-eps[j])*h[a],t['q_factors'][j])):raise ValueError('Target factor PMFs differ')
        inc=log_weight(t,eps)
        if abs(inc-t['log_weight'])>1e-9 or abs(inc-t['full_log_p']+t['full_log_q'])>1e-9:raise ValueError('Full/factor target mismatch')
        prefix+=inc;last=t['time_s']
    full=sum(t['log_p']-t['log_q'] for t in s['native_decisions'])
    if not np.isfinite([full,prefix,s['generation_log_weight'],s['action_log_weight']]).all():
        raise ValueError('Nonfinite stopped likelihood')
    if mode=='natural' and s['steps']:raise ValueError('Natural reference tilted')
    if max(abs(full-prefix),abs(prefix-s['generation_log_weight']),abs(prefix-s['action_log_weight']))>1e-9:
        raise ValueError('Stopped trajectory likelihood mismatch')
    return abs(full-prefix)


def audit(config,lock_path,bank_path,result,output,replay=False):
    cfg,physical,assets,lock=checked(config,lock_path);b=load_bank(bank_path,lock,True);root=Path(result)
    summary=read(root/'summary.json')
    if not summary['complete'] or summary['lock_sha256']!=identity(lock):raise ValueError('Incomplete validation')
    opened=read(root/'opened_protocol.json')
    if opened!=dict(lock_sha256=identity(lock),bank_summary_sha256=digest(Path(bank_path)/'summary.json')):
        raise ValueError('Native validation is detached from frozen bank')
    if summary['logical_continuations']!=cfg['draws_per_checkpoint']*len(lock['arms']):raise ValueError('Declared native budget differs')
    for name,sha in summary['hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(root/name)!=sha:raise ValueError('Native artifact changed')
    expected=[[k,cfg['continuation_seed_start']+i] for i,k in enumerate(state_schedule(b['cases'],cfg['draws_per_checkpoint'],cfg['state_seed']))]
    schedule=read(root/'schedule.json')
    if schedule['draws']!=expected or schedule['arms']!=lock['arms']:raise ValueError('Outcome-blind schedule changed')
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS) if replay else None
    metrics={};max_error=0.;replayed=0
    for arm in lock['arms']:
        policy=NumpyPolicy(arm['policy_path']) if arm['policy_path'] else None
        records=[]
        for i in range((len(expected)+cfg['batch_size']-1)//cfg['batch_size']):
            name=f'{arm["label"]}/batch_{i:04d}.json'
            if name not in summary['hashes']:raise ValueError('Unhashed sample shard')
            records.extend(read(root/name))
        if len(records)!=len(expected):raise ValueError('Native budget mismatch')
        for s,(k,seed) in zip(records,expected):max_error=max(max_error,check_record(s,b['cases'][k],seed,policy,arm['mode']))
        metrics[arm['label']]=describe(records,cfg)
        if metrics[arm['label']]!=read(root/arm['label']/'metrics.json'):raise ValueError('Metrics reconstruction differs')
        if replay:
            k,seed=expected[0];s=native.rollout(model,assets,physical,arm['mode'],seed,b['cases'][k],policy)
            if canonical_path(s)!=canonical_path(records[0]):raise ValueError('Scheduled native replay differs')
            replayed+=1
        print(dict(phase='validation_audit',arm=arm['label'],native_replays=replayed),flush=True)
    selection=report_selection(metrics,lock)
    if selection!=read(root/'selection.json') or selection['readiness']!=summary['readiness']:raise ValueError('Selection reconstruction differs')
    report=dict(verified=True,summary_sha256=digest(root/'summary.json'),auditor_sha256=digest(__file__),
        lock_sha256=identity(lock),native_replays=replayed,maximum_full_factor_log_error=max_error,
        readiness=selection['readiness'],final_test_opened=False)
    save(output,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','bank','run','audit'])
    p.add_argument('--config',default=DEFAULT);p.add_argument('--lock');p.add_argument('--bank');p.add_argument('--result')
    p.add_argument('--replay',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    if a.command=='freeze':r=freeze(a.config,a.output)
    elif a.command=='bank':r=bank(a.config,a.lock,a.output)
    elif a.command=='run':r=run(a.config,a.lock,a.bank,a.output)
    else:r=audit(a.config,a.lock,a.bank,a.result,a.output,a.replay)
    print(r,flush=True)


if __name__=='__main__':main()
