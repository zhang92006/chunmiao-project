"""Prospective matched Single/Dual D2RL: freeze, bank, train, select, final test.

Native jobs are deliberately separate commands for the researcher to run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .highd_v46_assets import read,save,digest,ensure_assets,ROOT
from .highd_v46_d2rl import NativeExperiment,identity,experiment_target,proposal_structure
from .highd_v46_pair_synergy import snapshot
from .highd_v46_contract import FIXED_ALPHA_OBSERVATION
from .highd_v47_fixed_alpha_contract import CONTRACT,SCOPE,state_schedule
from .highd_v46_policy import NumpyPolicy
from . import highd_v47_formal_native as native
from .highd_v47_formal_statistics import describe,select_checkpoint,readiness,paired_comparison,final_readiness
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS

DEFAULT='configs/highd_v47_formal_comparison_v1.json'


def ranges(cfg):
    return {k:set(range(v['seed_start'],v['seed_start']+v['episodes'])) for k,v in cfg['source_splits'].items()}


def historical_inventory(config, cfg):
    """Conservative namespace exclusion plus persisted historical source evidence."""
    files=set(ROOT.glob('configs/*.json'))|set(ROOT.glob('results/*.json'))
    for p in (ROOT/'outputs').rglob('*.json'):
        name=p.name
        if any(s in name for s in ('summary','manifest','lock','protocol','case_plan','state_bank','state_pool','source')) or name=='sequences.json':files.add(p)
    files.discard(Path(config).resolve()); facts={}; used=set()
    def visit(value):
        if isinstance(value,dict):
            if type(value.get('seed_start')) is int and type(value.get('episodes')) is int:
                used.update(range(value['seed_start'],value['seed_start']+value['episodes']))
            for k,v in value.items():
                if k in ('seed','source_seed','seed_start') and type(v) is int:used.add(v)
                elif k in ('training_source_seeds','validation_source_seeds','source_seeds') and isinstance(v,list):used.update(x for x in v if type(x) is int)
                elif k.endswith('source_ranges') and isinstance(v,list):
                    for interval in v:
                        if len(interval)==2 and all(type(x) is int for x in interval):used.update(range(interval[0],interval[1]+1))
                visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for p in sorted(files):
        visit(read(p));facts[str(p.relative_to(ROOT))]=digest(p)
    overlap=set().union(*ranges(cfg).values())&used
    if overlap:raise ValueError('Source range was previously used/reserved: '+str(sorted(overlap)[:8]))
    return dict(metadata_hashes=facts,observed_seed_count=len(used),known_seeds_sha256=identity(sorted(used)),
        excluded_namespace=cfg['historical_namespace_excluded'])


def protocol(config):
    cfg=read(config); rs=ranges(cfg)
    if set(rs)!= {'train','validation','final_test'} or any(not s for s in rs.values()):raise ValueError('Three nonempty source splits required')
    if any(rs[a]&rs[b] for a in rs for b in rs if a!=b):raise ValueError('Whole-source splits overlap')
    if min(set().union(*rs.values()))<=cfg['historical_namespace_excluded'][1]:raise ValueError('Historical source namespace excluded')
    if cfg['state_sampling']!='Q_S_equals_alpha' or cfg['state_likelihood_ratio']!=1 or cfg['state_model'] is not None:raise ValueError('Action-only formal experiment required')
    if cfg['training']['seeds']!=[7,17,29,41,53] or cfg['training']['iterations']!=200 or cfg['training']['checkpoint_interval']!=10:raise ValueError('Version a changed seed/iteration budget')
    if cfg['environment']!={k:sys.version.split()[0] if k=='python' else version(k) for k in cfg['environment']}:raise ValueError('Pinned environment required')
    physical=read(cfg['physical_config']); original=read(cfg['single_reference_config'])
    if physical['cav']!=original['cav'] or physical['horizon_s']!=60:raise ValueError('Frozen physical target required')
    for k,v in original['criticality'].items():
        if physical['criticality'][k]!=v:raise ValueError('Original Single criticality changed')
    ppo=read(cfg['ppo_config'])
    if ppo['iterations']!=200 or ppo['checkpoint_selection']!='external_native_validation' or not ppo['export_all_checkpoints']:raise ValueError('Native validation checkpoint protocol required')
    assets,nde=ensure_assets()
    files=[Path(__file__),Path(native.__file__),Path(__file__).with_name('highd_v47_formal_statistics.py'),
        Path(__file__).with_name('highd_v46_assets.py'),
        Path(__file__).with_name('highd_v46_train.py'),Path(__file__).with_name('highd_v46_d2rl.py'),
        Path(__file__).with_name('highd_v47_formal_training_metrics.py'),
        Path(__file__).with_name('highd_v46_pair_witness.py'),Path(__file__).with_name('highd_v46_pair_synergy.py'),
        Path(__file__).with_name('highd_v46_cross_unit.py'),Path(__file__).with_name('highd_v47_fixed_alpha_native.py'),
        Path(__file__).with_name('highd_v47_fixed_alpha_contract.py'),Path(__file__).with_name('highd_v46_contract.py'),
        Path(__file__).with_name('highd_v46_policy.py'),Path(__file__).with_name('highd_v46_action_dist.py'),
        Path(__file__).with_name('highd_mean_precision_policy.py'),Path(__file__).with_name('highd_v46_statistics.py'),
        ROOT/'d2rl_training/highd_v46_sequence_env.py',Path(cfg['physical_config']),Path(cfg['single_reference_config']),Path(cfg['ppo_config'])]
    lock=dict(config_sha256=digest(config),components={str(p.resolve().relative_to(ROOT)):digest(p) for p in files},
        nde_lock_sha256=digest(ROOT/'configs/highd_paper_v46_freeze.json'),archive_sha256=nde['archive_sha256'],
        natural_target_sha256=nde['natural_target_sha256'],physical_target_sha256=identity(experiment_target(nde,physical)),environment=cfg['environment'])
    return cfg,physical,assets,lock


def freeze(config, output):
    cfg,_,_,lock=protocol(config); history=historical_inventory(config,cfg)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    lock.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,
        historical_inventory_sha256=identity(history),state_model=None,state_likelihood_ratio=1.)
    save(out/'historical_inventory.json',history)
    for split,seeds in ranges(cfg).items():save(out/(split+'_source_manifest.json'),dict(split=split,source_seeds=sorted(seeds),source_episode_is_split_unit=True,opened=False))
    save(out/'lock.json',lock);return dict(frozen=True,protocol_sha256=identity(lock),fresh_sources_opened=False)


def checked(config, frozen):
    cfg,physical,assets,current=protocol(config); lock=read(Path(frozen)/'lock.json')
    if any(lock.get(k)!=v for k,v in current.items()) or lock['config']!=cfg:raise ValueError('Formal frozen rule changed')
    if identity(read(Path(frozen)/'historical_inventory.json'))!=lock['historical_inventory_sha256']:raise ValueError('Historical source audit changed')
    for split,seeds in ranges(cfg).items():
        if read(Path(frozen)/(split+'_source_manifest.json'))['source_seeds']!=sorted(seeds):raise ValueError('Source manifest changed')
    return cfg,physical,assets,lock


def bank(config,frozen,split,output,training=None):
    cfg,physical,assets,lock=checked(config,frozen)
    if split=='validation':verified_training(training,cfg,lock)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);save(out/'opened_protocol.json',dict(split=split,lock=lock))
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS); cases=[];rows=[]
    for source in sorted(ranges(cfg)[split]):
        road=NativeExperiment(model,assets/'pair',assets/'lateral',source,physical);eligible=[]
        while not road.crashes and road.time<physical['horizon_s']-1e-9:
            pmfs=road.action_pmfs();units,actors=road.prepare(pmfs)
            choice=NativeExperiment.choose(road,units,actors,pmfs,'single')
            if choice is not None and road.time<=physical['horizon_s']-cfg['bank']['continuation_horizon_s']:
                state=snapshot(road,units,pmfs,source,dict(original_single_critical=True))
                state.update(cav_acceleration=road.cav_acceleration,last_acceleration=road.last_acceleration.copy())
                eligible.append(state)
            native.advance_prepared(road,pmfs,native.conditional_actions(units,{},road.rng))
        rng=np.random.default_rng(np.random.SeedSequence([cfg['bank']['selection_seed'],source]))
        selected=sorted(rng.choice(len(eligible),size=min(len(eligible),cfg['bank']['maximum_states_per_source']),replace=False).tolist()) if eligible else []
        for index in selected:cases.append(dict(case_id=str(len(cases)),source_seed=source,state=eligible[index],horizon_s=cfg['bank']['continuation_horizon_s']))
        rows.append(dict(source_seed=source,eligible_critical_states=len(eligible),selected_states=len(selected),natural_CAV_event=any(road.cav.id in pair for pair in road.crashes)))
        save(out/'sources'/('source_%d.json'%source),dict(rows[-1],selected_times=[eligible[i]['time_s'] for i in selected],
            elapsed_s=road.time,collision_pairs=road.crashes))
        print(dict(phase='original_single_bank',split=split,completed=len(rows),total=len(ranges(cfg)[split])),flush=True)
    sources={c['source_seed'] for c in cases}
    if not cases:raise ValueError('No original Single critical state; preserve all source records')
    for c in cases:c['alpha']=1/(len(sources)*sum(x['source_seed']==c['source_seed'] for x in cases))
    law=[dict(case_id=c['case_id'],source_seed=c['source_seed'],state_sha256=identity(c['state']),alpha=c['alpha']) for c in cases]
    save(out/'bank.json',dict(cases=cases,state_law=law,source_records=rows,split=split,lock=lock))
    save(out/'summary.json',dict(complete=True,split=split,source_episodes=len(rows),selected_sources=len(sources),states=len(cases),bank_sha256=digest(out/'bank.json'),lock=lock))
    return dict(complete=True,split=split,states=len(cases),selected_sources=len(sources))


def load_bank(path,lock,split):
    root=Path(path);summary=read(root/'summary.json');b=read(root/'bank.json')
    if not summary['complete'] or summary['lock']!=lock or summary['split']!=split or digest(root/'bank.json')!=summary['bank_sha256']:raise ValueError('Bank checksum/split mismatch')
    if b['split']!=split or b['lock']!=lock or len(b['cases'])!=len(b['state_law']) or len(b['cases'])!=summary['states']:
        raise ValueError('Incomplete bank identities')
    if not {c['source_seed'] for c in b['cases']}<=ranges(lock['config'])[split]:raise ValueError('Bank source belongs to another split')
    for c,s in zip(b['cases'],b['state_law']):
        if c['case_id']!=s['case_id'] or identity(c['state'])!=s['state_sha256'] or c['source_seed']!=s['source_seed'] or c['alpha']!=s['alpha']:raise ValueError('State identity/mass mismatch')
    state_schedule(b['state_law'],1,0)
    return b


_WORK=None
_POLICIES={}
def worker_init(assets,physical):
    global _WORK
    assets=Path(assets);_WORK=(PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS),assets,physical)


def worker(task):
    mode,seed,case,policy_path=task;model,assets,physical=_WORK
    if policy_path and policy_path not in _POLICIES:_POLICIES[policy_path]=NumpyPolicy(policy_path)
    policy=_POLICIES[policy_path] if policy_path else None
    return native.rollout(model,assets,physical,mode,seed,case,policy)


def native_tasks(tasks,assets,physical,workers,out,label):
    save(out/(label+'_plan.json'),dict(tasks=[dict(mode=m,seed=s,case_id=c['case_id'] if c else None,policy_path=p) for m,s,c,p in tasks]))
    records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(str(assets),physical)) as pool:
        for i,result in enumerate(pool.map(worker,tasks,chunksize=1)):
            save(out/label/('draw_%05d.json'%i),result);records.append(result)
            if (i+1)%128==0 or i+1==len(tasks):print(dict(phase=label,completed=i+1,total=len(tasks)),flush=True)
    save(out/(label+'_artifacts.json'),dict(hashes={str(p.relative_to(out)):digest(p) for p in sorted((out/label).glob('draw_*.json'))}))
    return records


def dataset(mode,seq,b,cfg,lock):
    target=identity(dict(physical=lock['physical_target_sha256'],horizon=cfg['bank']['continuation_horizon_s'],initialization='original_single_critical_bank'))
    structure=identity(dict(components=lock['components'],mode=mode,discovery='original_single',proposal=proposal_structure(read(cfg['physical_config']),mode)))
    return dict(contract=CONTRACT,observation_contract=FIXED_ALPHA_OBSERVATION,action_dimension=1 if mode=='single' else 2,
        intervention_mode=mode,natural_target_sha256=lock['natural_target_sha256'],experiment_target_sha256=target,
        conditional_target_sha256=identity(dict(target=target,law=b['state_law'])),proposal_structure_sha256=structure,
        proposal_sha256=identity(dict(structure=structure,epsilon=.1)),state_law=b['state_law'],state_law_sha256=identity(b['state_law']),
        state_sampling='Q_S_equals_alpha',state_likelihood_ratio=1.,state_model=None,sampling_scope=SCOPE,horizon_s=cfg['bank']['continuation_horizon_s'],
        state_schedule_seed=cfg['training']['state_seed'],continuation_seed_start=cfg['training']['continuation_seed_start'],sequences=seq)


def collect_train(config,frozen,bank_path,output):
    cfg,physical,assets,lock=checked(config,frozen);b=load_bank(bank_path,lock,'train');out=Path(output);out.mkdir(parents=True,exist_ok=False)
    schedule=state_schedule(b['state_law'],cfg['training']['draws_per_mode'],cfg['training']['state_seed']);summaries={};hashes={}
    save(out/'opened_protocol.json',dict(lock=lock,bank_sha256=digest(Path(bank_path)/'bank.json')))
    save(out/'shared_schedule.json',dict(case_indices=schedule,seeds=[cfg['training']['continuation_seed_start']+i for i in range(len(schedule))]))
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    for mode in ('single','dual'):
        seq=native_tasks([(mode,cfg['training']['continuation_seed_start']+i,b['cases'][k],None) for i,k in enumerate(schedule)],assets,physical,cfg['workers'],out,mode)
        for i,s in enumerate(seq):s['draw_index']=i
        data=dataset(mode,seq,b,cfg,lock);directory=out/mode;save(directory/'sequences.json',data)
        manifest=dict(data_path='sequences.json',sha256=digest(directory/'sequences.json'),**{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256','state_law_sha256','proposal_structure_sha256','proposal_sha256','intervention_mode')})
        save(directory/'sequence_manifest.json',manifest);V46SequenceEnv(dict(sequence_manifest=str(directory/'sequence_manifest.json'),evaluation_only=True))
        hashes[mode]=manifest['sha256'];summaries[mode]=describe(seq,cfg)
    save(out/'summary.json',dict(complete=True,lock=lock,modes=summaries,sequence_hashes=hashes,shared_schedule_sha256=digest(out/'shared_schedule.json')))
    return dict(complete=True,events={m:r['positive_episodes'] for m,r in summaries.items()})


def audit_train(config,frozen,bank_path,data,output,replay=False):
    cfg,physical,assets,lock=checked(config,frozen);b=load_bank(bank_path,lock,'train');root=Path(data);summary=read(root/'summary.json')
    if summary['lock']!=lock or not summary['complete']:raise ValueError('Training collection mismatch')
    schedule=state_schedule(b['state_law'],cfg['training']['draws_per_mode'],cfg['training']['state_seed'])
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS) if replay else None
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    count=0;error=0.
    for mode in ('single','dual'):
        env=V46SequenceEnv(dict(sequence_manifest=str(root/mode/'sequence_manifest.json'),evaluation_only=True))
        if env.source_sha256!=summary['sequence_hashes'][mode] or env.dataset['state_law']!=b['state_law']:raise ValueError('Training law/hash changed')
        if len(env.sequences)!=len(schedule):raise ValueError('Budget changed')
        first={}
        for i,s in enumerate(env.sequences):
            raw=read(root/mode/('draw_%05d.json'%i))
            if identity({k:v for k,v in s.items() if k!='draw_index'})!=identity(raw):raise ValueError('Raw training path changed')
            first.setdefault(schedule[i],i)
            error=max(error,abs(sum(d['log_p']-d['log_q'] for d in s['native_decisions'])-s['generation_log_weight']))
        if describe(env.sequences,cfg)!=summary['modes'][mode]:raise ValueError('Training moment report differs')
        if replay:
            for k,i in sorted(first.items()):
                expected=env.sequences[i];actual=native.rollout(model,assets,physical,mode,expected['seed'],b['cases'][k])
                skip={'cpu_s','wall_time_s','draw_index'}
                def canonical(s):
                    value={k:v for k,v in s.items() if k not in skip}
                    value['steps']=[{k:v for k,v in t.items() if k!='decision_cpu_s'} for t in s['steps']]
                    value['native_decisions']=[{k:v for k,v in t.items() if k!='decision_cpu_s'} for t in s['native_decisions']]
                    return identity(value)
                if canonical(expected)!=canonical(actual):raise ValueError('Native scheduled replay mismatch')
                count+=1
    if error>1e-9:raise ValueError('Full likelihood differs')
    save(output,dict(verified=True,lock=lock,collection_sha256=digest(root/'summary.json'),native_replays=count,full_log_error=error))
    return dict(verified=True,native_replays=count,full_log_error=error)


def train(config,frozen,data,output):
    cfg,_,_,lock=checked(config,frozen);data=Path(data);summary=read(data/'summary.json')
    if not summary['complete'] or summary['lock']!=lock:raise ValueError('Complete frozen training data required')
    audit=read(data/'audit.json')
    if not audit['verified'] or audit['lock']!=lock or audit['collection_sha256']!=digest(data/'summary.json') or audit['native_replays']<2:
        raise ValueError('Run audit-train --replay before PPO')
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    envs={m:V46SequenceEnv(dict(sequence_manifest=str(data/m/'sequence_manifest.json'))) for m in ('single','dual')}
    if envs['single'].dataset['state_law']!=envs['dual'].dataset['state_law']:raise ValueError('Single/Dual state distributions differ')
    for m,env in envs.items():
        if env.source_sha256!=summary['sequence_hashes'][m]:raise ValueError('Training data changed')
    required_replays=2*len(set(state_schedule(envs['single'].dataset['state_law'],cfg['training']['draws_per_mode'],cfg['training']['state_seed'])))
    if audit['native_replays']!=required_replays:raise ValueError('Missing predetermined per-state/mode native replays')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);ppo=read(cfg['ppo_config']);ppo['log_reward_scale']=max(e.log_scale for e in envs.values())
    save(out/'ppo_config.json',ppo);save(out/'opened_protocol.json',dict(lock=lock,data_summary_sha256=digest(data/'summary.json')))
    reports={}
    for seed in cfg['training']['seeds']:
        for mode in ('single','dual'):
            name=mode+'_seed'+str(seed);args=[sys.executable,'-m','scenario_reconstruction.highd_v46_train','--sequence-manifest',str(data/mode/'sequence_manifest.json'),'--config',str(out/'ppo_config.json'),'--seed',str(seed),'--output',str(out/name)]
            # Avoid macOS's long per-user temporary path; no change to model RNGs.
            env=dict(os.environ,TMPDIR='/tmp',RAY_TMPDIR='/tmp/v47formal_'+str(os.getpid())+'_'+name,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            print(dict(phase='ppo_started',run=name,iterations=cfg['training']['iterations'],log=str(out/(name+'.log'))),flush=True)
            with (out/(name+'.log')).open('x') as stream:
                process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env);save(out/(name+'_process.json'),dict(pid=process.pid));code=process.wait()
            if code:raise RuntimeError('Retain failed run and logs: '+name)
            reports[name]=read(out/name/'training_summary.json')
    save(out/'summary.json',dict(complete=True,lock=lock,runs=reports,data_summary_sha256=digest(data/'summary.json'),selection_pending=True))
    return dict(complete=True,runs=len(reports),native_validation_pending=True)


def verified_training(path,cfg,lock):
    if path is None:raise ValueError('Completed formal training required before validation')
    root=Path(path);summary=read(root/'summary.json')
    expected={mode+'_seed'+str(seed) for seed in cfg['training']['seeds'] for mode in ('single','dual')}
    if not summary['complete'] or summary['lock']!=lock or set(summary['runs'])!=expected:raise ValueError('Wrong/incomplete formal training')
    for name,r in summary['runs'].items():
        if r!=read(root/name/'training_summary.json'):raise ValueError('Training summary changed')
        if r['completed_iterations']!=cfg['training']['iterations'] or r['state_likelihood_ratio']!=1:raise ValueError('Training budget/state LR changed')
        catalog=read(root/name/'checkpoint_catalog.json')
        if [c['iteration'] for c in catalog['checkpoints']]!=list(range(0,cfg['training']['iterations']+1,10)):raise ValueError('Missing checkpoint including iteration 0')
        for c in catalog['checkpoints']:
            policy=root/name/c['policy_path']
            if digest(policy/'policy.json')!=c['metadata_sha256'] or digest(policy/'policy_weights.npz')!=c['weights_sha256']:raise ValueError('Checkpoint policy changed')
            meta=read(policy/'policy.json');mode=name.split('_')[0]
            if (meta['state_likelihood_ratio']!=1. or meta['state_sampling']!='Q_S_equals_alpha'
                    or meta['natural_target_sha256']!=lock['natural_target_sha256'] or meta['intervention_mode']!=mode
                    or meta['action_dimension']!=(1 if mode=='single' else 2) or meta['observation_contract']!=FIXED_ALPHA_OBSERVATION
                    or not set(meta['training_source_seeds']) or not set(meta['training_source_seeds'])<=ranges(cfg)['train'] or meta['validation_source_seeds']):
                raise ValueError('Checkpoint source/action/state identity differs')
    return summary


def validate(config,frozen,bank_path,training,output):
    cfg,physical,assets,lock=checked(config,frozen);verified_training(training,cfg,lock);b=load_bank(bank_path,lock,'validation');out=Path(output);out.mkdir(parents=True,exist_ok=False)
    schedule=state_schedule(b['state_law'],cfg['validation']['draws_per_checkpoint'],cfg['validation']['state_seed'])
    save(out/'opened_protocol.json',dict(lock=lock,training_summary_sha256=digest(Path(training)/'summary.json'),bank_sha256=digest(Path(bank_path)/'bank.json')))
    save(out/'shared_schedule.json',dict(case_indices=schedule,seeds=[cfg['validation']['continuation_seed_start']+i for i in range(len(schedule))]))
    selected=[];all_reports={}
    for seed in cfg['training']['seeds']:
        for mode in ('single','dual'):
            name=mode+'_seed'+str(seed);root=Path(training)/name;catalog=read(root/'checkpoint_catalog.json');reports={}
            for item in catalog['checkpoints']:
                iteration=item['iteration'];policy=str(root/item['policy_path']);label=name+'_iter'+str(iteration)
                seq=native_tasks([(mode,cfg['validation']['continuation_seed_start']+i,b['cases'][k],policy) for i,k in enumerate(schedule)],assets,physical,cfg['workers'],out,label)
                reports[iteration]=describe(seq,cfg);save(out/(label+'_metrics.json'),reports[iteration])
            iteration,reason=select_checkpoint(reports);item=next(c for c in catalog['checkpoints'] if c['iteration']==iteration)
            policy=Path(training)/name/item['policy_path'];all_reports[name]=reports
            selected.append(dict(mode=mode,seed=seed,iteration=iteration,reason=reason,policy_path=str(policy),metadata_sha256=digest(policy/'policy.json'),weights_sha256=digest(policy/'policy_weights.npz'),metrics=reports[iteration]))
    screen=readiness(selected,cfg)
    save(out/'selection.json',dict(complete=True,lock=lock,selected=selected,readiness=screen,checkpoint_reports=all_reports,training_summary_sha256=digest(Path(training)/'summary.json')))
    return dict(complete=True,final_test_allowed=screen['passed'],failure_reasons=screen['reasons'])


def seal(config,frozen,validation,output):
    cfg,_,_,lock=checked(config,frozen);p=Path(validation)/'selection.json';report=read(p)
    if report['lock']!=lock or not report['complete']:raise ValueError('Complete native selection required')
    expected={(mode,seed) for mode in ('single','dual') for seed in cfg['training']['seeds']}
    if len(report['selected'])!=len(expected) or {(i['mode'],i['seed']) for i in report['selected']}!=expected:raise ValueError('Every paired training seed must be retained')
    validation=Path(validation)
    # Reconstruct ALL checkpoint metrics and the selection; not just the winners.
    for item in report['selected']:
        name=item['mode']+'_seed'+str(item['seed']);reports={}
        for iteration in range(0,cfg['training']['iterations']+1,cfg['training']['checkpoint_interval']):
            label=name+'_iter'+str(iteration);records=read_native_records(validation,label)
            reports[iteration]=describe(records,cfg)
            if reports[iteration]!=report['checkpoint_reports'][name][str(iteration)]:raise ValueError('Validation metrics changed')
        chosen,reason=select_checkpoint(reports)
        if chosen!=item['iteration'] or reason!=item['reason'] or reports[chosen]!=item['metrics']:raise ValueError('Checkpoint selection differs from native evidence')
    screen=readiness(report['selected'],cfg)
    if not screen['passed'] or screen!=report['readiness']:raise ValueError('Stop condition: '+str(screen['reasons']))
    for item in report['selected']:
        root=Path(item['policy_path'])
        if digest(root/'policy.json')!=item['metadata_sha256'] or digest(root/'policy_weights.npz')!=item['weights_sha256']:raise ValueError('Selected policy changed')
    save(output,dict(lock=lock,selection_sha256=digest(p),selected=report['selected'],rules_frozen=True,final_test_opened=False))
    return dict(sealed=True,policies=len(report['selected']),final_test_opened=False)


def read_native_records(root,label):
    manifest=read(root/(label+'_artifacts.json'));plan=read(root/(label+'_plan.json'))
    records=[]
    for name,sha in sorted(manifest['hashes'].items()):
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(root/name)!=sha:raise ValueError('Native artifact changed')
        records.append(read(root/name))
    if len(records)!=len(plan['tasks']):raise ValueError('Missing native endpoint')
    for s,t in zip(records,plan['tasks']):
        if s['seed']!=t['seed'] or (t['case_id'] is not None and s['case_id']!=t['case_id']):raise ValueError('Native schedule changed')
        if s['state_likelihood_ratio']!=1 or s['state_log_weight']!=0 or abs(sum(d['log_p']-d['log_q'] for d in s['native_decisions'])-s['generation_log_weight'])>1e-9:raise ValueError('Full native likelihood mismatch')
    return records


def final_test(config,frozen,policy_lock,output):
    cfg,physical,assets,lock=checked(config,frozen);sealed=read(policy_lock)
    if sealed['lock']!=lock or not sealed['rules_frozen'] or not readiness(sealed['selected'],cfg)['passed']:raise ValueError('All selected models and stop-condition review must pass before FinalTest')
    for item in sealed['selected']:
        root=Path(item['policy_path'])
        if digest(root/'policy.json')!=item['metadata_sha256'] or digest(root/'policy_weights.npz')!=item['weights_sha256']:raise ValueError('Frozen checkpoint changed')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);seeds=sorted(ranges(cfg)['final_test']);arms={}
    save(out/'opened_protocol.json',dict(lock=lock,policy_lock_sha256=digest(policy_lock),source_seeds=seeds,initialization='original natural initialization'))
    for mode in ('natural','single','dual'):
        arms['fixed_'+mode]=native_tasks([(mode,s,None,None) for s in seeds],assets,physical,cfg['workers'],out,'fixed_'+mode)
    reports={}
    for item in sealed['selected']:
        name=item['mode']+'_seed'+str(item['seed']);arms[name]=native_tasks([(item['mode'],s,None,item['policy_path']) for s in seeds],assets,physical,cfg['workers'],out,name)
    reference=arms['fixed_natural']
    for records in arms.values():
        if [identity(dict(fleet=s['initial_fleet'],cav_id=s['cav_id'])) for s in records]!=[identity(dict(fleet=s['initial_fleet'],cav_id=s['cav_id'])) for s in reference]:raise ValueError('Final reset populations differ')
    for label in arms:
        if identity(read_native_records(out,label))!=identity(arms[label]):raise ValueError('Final records differ')
    for seed in cfg['training']['seeds']:reports[str(seed)]=paired_comparison(arms['single_seed'+str(seed)],arms['dual_seed'+str(seed)],cfg)
    supports=[r['supports_dual_lower_M2'] for r in reports.values()];screen=final_readiness(reports,cfg)
    accepted=all(supports) and screen['passed']
    save(out/'summary.json',dict(complete=True,lock=lock,policy_lock_sha256=digest(policy_lock),paired_seed_comparisons=reports,
        final_readiness=screen,
        fixed_controls={m:describe(arms['fixed_'+m],cfg) for m in ('natural','single','dual')},
        conclusion='Supports Dual lower M2 in every paired seed on this final population' if accepted else 'Does not establish consistent Dual superiority; retain all paired seed results'))
    return dict(complete=True,dual_superiority_supported=accepted,failure_reasons=screen['reasons'])


def canonical_path(s):
    value={k:v for k,v in s.items() if k not in ('cpu_s','wall_time_s','draw_index')}
    for key in ('steps','native_decisions'):
        value[key]=[{k:v for k,v in t.items() if k!='decision_cpu_s'} for t in s[key]]
    return identity(value)


def audit_final(config,frozen,result,output,replay=False):
    cfg,physical,assets,lock=checked(config,frozen);root=Path(result);summary=read(root/'summary.json')
    if not summary['complete'] or summary['lock']!=lock:raise ValueError('Complete frozen FinalTest required')
    labels=['fixed_'+m for m in ('natural','single','dual')]+[m+'_seed'+str(s) for s in cfg['training']['seeds'] for m in ('single','dual')]
    arms={label:read_native_records(root,label) for label in labels};seeds=sorted(ranges(cfg)['final_test'])
    reference=[identity(dict(fleet=s['initial_fleet'],cav_id=s['cav_id'])) for s in arms['fixed_natural']]
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS) if replay else None
    count=0
    for label,records in arms.items():
        if [s['source_seed'] for s in records]!=seeds or [s['seed'] for s in records]!=seeds:raise ValueError('Final reset sources changed')
        if [identity(dict(fleet=s['initial_fleet'],cav_id=s['cav_id'])) for s in records]!=reference:raise ValueError('Matched final populations changed')
        if replay:
            task=read(root/(label+'_plan.json'))['tasks'][0];policy=NumpyPolicy(task['policy_path']) if task['policy_path'] else None
            actual=native.rollout(model,assets,physical,task['mode'],task['seed'],None,policy)
            if canonical_path(actual)!=canonical_path(records[0]):raise ValueError('Predetermined original-reset native replay changed')
            count+=1
    for m in ('natural','single','dual'):
        if describe(arms['fixed_'+m],cfg)!=summary['fixed_controls'][m]:raise ValueError('Final control statistics differ')
    for seed in cfg['training']['seeds']:
        actual=paired_comparison(arms['single_seed'+str(seed)],arms['dual_seed'+str(seed)],cfg)
        if actual!=summary['paired_seed_comparisons'][str(seed)]:raise ValueError('Final paired statistics differ')
    if final_readiness(summary['paired_seed_comparisons'],cfg)!=summary['final_readiness']:raise ValueError('Final evidence guards changed')
    manifest=dict(verified=True,summary_sha256=digest(root/'summary.json'),lock=lock,native_replays=count,
        physical_episodes=sum(len(s) for s in arms.values()),state_likelihood_ratio=1.,
        evidence_scope='Read-only reconstruction and predetermined path replays; no new observations')
    save(output,manifest);return dict(verified=True,native_replays=count)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default=DEFAULT);sub=p.add_subparsers(dest='command',required=True)
    for name in ('freeze','bank','collect-train','audit-train','train','validate','seal','final-test','audit-final'):
        a=sub.add_parser(name);a.add_argument('--output',required=True)
        if name!='freeze':a.add_argument('--frozen',required=True)
        if name=='bank':a.add_argument('--split',choices=['train','validation'],required=True);a.add_argument('--training')
        if name in ('collect-train','validate','audit-train'):a.add_argument('--bank',required=True)
        if name in ('train','audit-train'):a.add_argument('--data',required=True)
        if name=='audit-train':a.add_argument('--replay',action='store_true')
        if name=='validate':a.add_argument('--training',required=True)
        if name=='seal':a.add_argument('--validation',required=True)
        if name=='final-test':a.add_argument('--policy-lock',required=True)
        if name=='audit-final':a.add_argument('--result',required=True);a.add_argument('--replay',action='store_true')
    a=p.parse_args();c=a.config
    if a.command=='freeze':result=freeze(c,a.output)
    elif a.command=='bank':result=bank(c,a.frozen,a.split,a.output,a.training)
    elif a.command=='collect-train':result=collect_train(c,a.frozen,a.bank,a.output)
    elif a.command=='audit-train':result=audit_train(c,a.frozen,a.bank,a.data,a.output,a.replay)
    elif a.command=='train':result=train(c,a.frozen,a.data,a.output)
    elif a.command=='validate':result=validate(c,a.frozen,a.bank,a.training,a.output)
    elif a.command=='seal':result=seal(c,a.frozen,a.validation,a.output)
    elif a.command=='audit-final':result=audit_final(c,a.frozen,a.result,a.output,a.replay)
    else:result=final_test(c,a.frozen,a.policy_lock,a.output)
    print(result,flush=True)


if __name__=='__main__':main()
