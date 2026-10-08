"""Freeze Q_S=alpha, collect real Single/Dual sequences, launch formal PPO."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import highd_v47_source_transfer as source
from . import highd_v47_fixed_alpha_native as native
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import read, save, digest
from .highd_v46_contract import FIXED_ALPHA_OBSERVATION
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import environment
from .highd_v47_fixed_alpha_contract import CONTRACT, SCOPE, state_schedule, validate_dataset

DEFAULT='configs/highd_v47_fixed_alpha_ppo_v1.json'


def protocol(config):
    cfg=read(config)
    required=dict(version=1,state_sampling='Q_S_equals_alpha',state_likelihood_ratio=1.,state_model=None,
        modes=['single','dual'],native_draws_per_mode=4096,state_schedule_seed=100000017,
        continuation_seed_start=100000000,horizon_s=4,decision_interval_s=1,generation_epsilon=.1,
        epsilon_min=.05,risk_edges=[0,.2,.4,.6,.8,1],workers=4,training_seeds=[7,17,29])
    if any(cfg.get(k)!=v for k,v in required.items()) or cfg['environment']!=environment():
        raise ValueError('Explicitly version changed fixed-alpha training design/environment')
    if cfg['training_environment']!={k:version(k) for k in ('ray','torch','gym')}:
        raise ValueError('Pinned PPO environment required')
    training=read(cfg['training_config'])
    if (training['iterations']!=100 or training['checkpoint_selection']!='fixed_iteration'
            or training['require_validation'] or training['initial_epsilon']!=cfg['generation_epsilon']):
        raise ValueError('Use declared fixed-iteration PPO configuration')
    old_cfg,dc,base,physical,assets,old_lock=source.protocol(cfg['parent_config'])
    names=['highd_v47_fixed_alpha_native.py','highd_v47_fixed_alpha_contract.py','highd_v46_train.py',
        'highd_v46_contract.py','highd_v46_action_dist.py','highd_v46_policy.py','highd_mean_precision_policy.py']
    deps={name:digest(Path(__file__).with_name(name)) for name in names}
    deps['sequence_env']=digest(Path(__file__).parents[1]/'d2rl_training/highd_v46_sequence_env.py')
    lock=dict(config_sha256=digest(config),implementation_sha256=digest(__file__),dependencies=deps,
        training_config_sha256=digest(cfg['training_config']),parent_lock=old_lock,
        environment=cfg['environment'],training_environment=cfg['training_environment'],
        natural_target_sha256=old_lock['natural_target_sha256'],physical_target_sha256=old_lock['physical_target_sha256'])
    return cfg,old_cfg,base,physical,assets,lock


def training_cases(cfg, old_cfg, lock, model, physical, proxy):
    """Verify the immutable state bank; NEVER fit or load a transferable Q_S."""
    root=Path(cfg['training_source']); summary=read(root/'summary.json')
    if (digest(root/'summary.json')!=cfg['training_summary_sha256']
            or digest(cfg['training_audit'])!=cfg['training_audit_sha256']
            or not summary['complete'] or summary['split']!='train' or summary['config']!=old_cfg
            or summary['conditional_target_sha256']!=cfg['original_conditional_target_sha256']
            or any(summary['lock'].get(k)!=v for k,v in lock['parent_lock'].items())):
        raise ValueError('Use the complete checksum-locked training cohort')
    for name,sha in summary['artifact_hashes'].items():
        if Path(name).name!=name or digest(root/name)!=sha:raise ValueError('Training artifact changed: '+name)
    source.check_training_audit(cfg['training_audit'],root)
    cases=[]
    for i in range(summary['selected_sources']):
        old=read(root/('case_%03d_design.json'%i)); state=old['state']
        if not 260000<=state['seed']<=260255 or old['plan']['source_seed']!=state['seed']:
            raise ValueError('A consumed validation/test source cannot enter PPO fitting')
        single,audits=native.choose_single(state,model,physical,proxy,cfg['risk_edges'])
        p=np.asarray(old['P']); risk=np.asarray(old['risk']); h=native.teacher(p,risk,cfg['risk_edges'])
        if not np.array_equal(h,np.asarray(old['H'])):raise ValueError('Initial dual teacher changed')
        cases.append(dict(case_id=str(i),state=state,source_seed=state['seed'],alpha=old['plan']['target_mass'],
            original_case_sha256=digest(root/('case_%03d_design.json'%i)),single_candidate_audit=audits,
            choices=dict(single=single,dual=dict(ids=old['plan']['actor_ids'],p=old['P'],risk=old['risk'],h=h.tolist()))))
    if len({c['source_seed'] for c in cases})!=len(cases):raise ValueError('Expected one locked case per source')
    return cases


def law_from_cases(cases):
    return [dict(case_id=c['case_id'],source_seed=c['source_seed'],state_sha256=identity(c['state']),alpha=c['alpha']) for c in cases]


def freeze(config, output):
    cfg,old_cfg,base,physical,assets,lock=protocol(config)
    destination=Path(output); destination.mkdir(parents=True,exist_ok=False)
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    cases=training_cases(cfg,old_cfg,lock,model,physical,base['proxy']); law=law_from_cases(cases)
    schedule=state_schedule(law,cfg['native_draws_per_mode'],cfg['state_schedule_seed'])
    save(destination/'bank.json',dict(cases=cases,state_law=law))
    save(destination/'schedule.json',dict(case_indices=schedule,continuation_seeds=[cfg['continuation_seed_start']+i for i in range(len(schedule))],
        modes=cfg['modes'],state_sampling=cfg['state_sampling'],state_likelihood_ratio=1.))
    lock.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        bank_sha256=digest(destination/'bank.json'),schedule_sha256=digest(destination/'schedule.json'),
        original_training_summary_sha256=cfg['training_summary_sha256'],state_law_sha256=identity(law),
        state_sampling=cfg['state_sampling'],state_likelihood_ratio=1.,state_model=None,new_native_outcomes_opened=False)
    save(destination/'lock.json',lock)
    return dict(frozen=True,cases=len(cases),native_draws_per_mode=len(schedule),state_LR=1.,code_commit=lock['code_commit'])


def checked(config, frozen):
    cfg,_,base,physical,assets,current=protocol(config); root=Path(frozen); lock=read(root/'lock.json')
    if any(lock.get(k)!=v for k,v in current.items()):raise ValueError('Implementation/config changed after fixed-alpha freeze')
    if digest(root/'bank.json')!=lock['bank_sha256'] or digest(root/'schedule.json')!=lock['schedule_sha256']:
        raise ValueError('Outcome-blind bank or complete schedule changed')
    bank=read(root/'bank.json'); plan=read(root/'schedule.json')
    expected=state_schedule(bank['state_law'],cfg['native_draws_per_mode'],cfg['state_schedule_seed'])
    if (plan['case_indices']!=expected or plan['continuation_seeds']!=list(range(cfg['continuation_seed_start'],cfg['continuation_seed_start']+len(expected)))
            or identity(bank['state_law'])!=lock['state_law_sha256'] or bank['state_law']!=law_from_cases(bank['cases'])):
        raise ValueError('Fixed alpha state schedule no longer reproduces')
    return cfg,base,physical,assets,lock,bank,plan


_WORKER=None


def worker_init(bank, cfg, physical, proxy, assets):
    global _WORKER
    assets=Path(assets); model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    _WORKER=(bank['cases'],cfg,physical,proxy,assets,model)


def worker(task):
    mode,draw_index,k,seed=task; cases,cfg,physical,proxy,assets,model=_WORKER
    return mode,draw_index,dict(native.rollout(cases[k],mode,seed,model,assets,physical,proxy,cfg),draw_index=draw_index)


def dataset_for(mode, sequences, bank, cfg, lock):
    target=dict(parent_physical_target_sha256=lock['physical_target_sha256'],horizon_s=cfg['horizon_s'],
        event='CAV_in_first_any_actor_collision; BV_only_competing_negative',initialization='uniform_source_then_locked_state')
    structure=identity(dict(contract=CONTRACT,mode=mode,config_sha256=lock['config_sha256'],
        implementation_sha256=lock['implementation_sha256'],dependencies=lock['dependencies']))
    return dict(contract=CONTRACT,observation_contract=FIXED_ALPHA_OBSERVATION,action_dimension=1 if mode=='single' else 2,
        intervention_mode=mode,natural_target_sha256=lock['natural_target_sha256'],experiment_target_sha256=identity(target),
        conditional_target_sha256=identity(dict(target=target,state_law=bank['state_law'])),
        proposal_structure_sha256=structure,proposal_sha256=identity(dict(structure=structure,epsilon=cfg['generation_epsilon'])),
        state_law=bank['state_law'],state_law_sha256=lock['state_law_sha256'],state_sampling=cfg['state_sampling'],
        state_likelihood_ratio=1.,state_model=None,sampling_scope=SCOPE,horizon_s=cfg['horizon_s'],
        state_schedule_seed=cfg['state_schedule_seed'],continuation_seed_start=cfg['continuation_seed_start'],sequences=sequences)


def collect(config, frozen, output):
    cfg,base,physical,assets,lock,bank,plan=checked(config,frozen)
    out=Path(output); out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    save(out/'opened_protocol.json',dict(config=cfg,lock=lock))
    tasks=[(mode,i,k,seed) for i,(k,seed) in enumerate(zip(plan['case_indices'],plan['continuation_seeds'])) for mode in cfg['modes']]
    sequences={mode:{} for mode in cfg['modes']}
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init,initargs=(bank,cfg,physical,base['proxy'],str(assets))) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for future in as_completed(futures):
            mode,i,result=future.result(); save(out/mode/'raw'/('draw_%05d.json'%i),result); sequences[mode][i]=result
            count=len(sequences[mode])
            if count%128==0 or count==cfg['native_draws_per_mode']:
                print(dict(phase='native_collection',mode=mode,completed=count,total=cfg['native_draws_per_mode'],
                    CAV_events=sum(s['event_result'] for s in sequences[mode].values())),flush=True)
    summaries={}
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    for mode in cfg['modes']:
        seq=[sequences[mode][i] for i in range(cfg['native_draws_per_mode'])]
        data=dataset_for(mode,seq,bank,cfg,lock); directory=out/mode
        save(directory/'sequences.json',data)
        manifest=dict(data_path='sequences.json',sha256=digest(directory/'sequences.json'),
            **{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256',
                'state_law_sha256','proposal_structure_sha256','proposal_sha256','intervention_mode')})
        save(directory/'sequence_manifest.json',manifest)
        checked_env=V46SequenceEnv(dict(sequence_manifest=str(directory/'sequence_manifest.json'),evaluation_only=True))
        weights=np.exp([s['action_log_weight'] for s in seq]); y=np.asarray([s['event_result'] for s in seq],float)
        values=y*weights; m2=y*weights*weights
        events={s['source_seed'] for s in seq if s['event_result']}
        summaries[mode]=dict(episodes=len(seq),critical_decisions=sum(len(s['steps']) for s in seq),
            CAV_events=int(y.sum()),event_sources=sorted(events),event_counts=dict(Counter(s['event_type'] for s in seq)),
            IS_mean=float(values.mean()),M2=float(m2.mean()),state_LR_min=1.,state_LR_max=1.,
            source_draw_counts=dict(Counter(str(s['source_seed']) for s in seq)),
            maximum_full_factor_log_error=max(abs(step['full_log_p']-step['full_log_q']-step['log_weight']) for s in seq for step in s['steps']),
            sequence_sha256=manifest['sha256'],training_signal_available=bool(checked_env.positive))
        save(directory/'collection_summary.json',summaries[mode])
    result=dict(complete=True,config=cfg,lock=lock,modes=summaries,wall_time_s=time.perf_counter()-start,
        state_likelihood_ratio=1.,source_episodes=len(bank['state_law']),transferable_QS_training_stopped=True,
        validation_opened=False,artifact_hashes={str(p.relative_to(out)):digest(p) for p in sorted(out.glob('*/*.json'))})
    save(out/'summary.json',result);return result


def audit_collection(config, frozen, data, output, replay=False):
    cfg,base,physical,assets,lock,bank,plan=checked(config,frozen); root=Path(data); summary=read(root/'summary.json')
    if not summary['complete'] or summary['lock']!=lock or summary['config']!=cfg:raise ValueError('Incomplete/mismatched native collection')
    for name,sha in summary['artifact_hashes'].items():
        if '..' in Path(name).parts or Path(name).is_absolute() or digest(root/name)!=sha:raise ValueError('Collection artifact changed')
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS) if replay else None
    replay_count=0; max_error=0.
    for mode in cfg['modes']:
        env=V46SequenceEnv(dict(sequence_manifest=str(root/mode/'sequence_manifest.json'),evaluation_only=True))
        data=env.dataset
        if data['state_law']!=bank['state_law']:raise ValueError('State law differs from frozen plan')
        first={}
        for i,s in enumerate(data['sequences']):
            k=plan['case_indices'][i]; first.setdefault(k,i)
            if s!=read(root/mode/'raw'/('draw_%05d.json'%i)):raise ValueError('Raw path differs from training sequence')
            if s['actor_ids']!=bank['cases'][k]['choices'][mode]['ids']:raise ValueError('Controlled actor selector changed')
            full=sum(step['full_log_p']-step['full_log_q'] for step in s['steps'])
            max_error=max(max_error,abs(full-s['generation_log_weight']))
            if not np.isclose(full,s['generation_log_weight'],rtol=1e-10,atol=1e-9):raise ValueError('Full path likelihood differs')
        if replay:
            for k,i in sorted(first.items()):
                s=data['sequences'][i]; repeated=native.rollout(bank['cases'][k],mode,s['seed'],model,assets,physical,base['proxy'],cfg)
                expected={key:value for key,value in s.items() if key not in ('wall_time_s','draw_index')}
                if identity(expected)!=identity({key:value for key,value in repeated.items() if key!='wall_time_s'}):
                    raise ValueError('Predetermined native path replay differs')
                replay_count+=1
    result=dict(verified=True,collection_summary_sha256=digest(root/'summary.json'),
        native_sequences=2*cfg['native_draws_per_mode'],predetermined_native_replays=replay_count,
        maximum_full_factor_log_error=max_error,state_likelihood_ratio=1.,validation_opened=False,
        auditor_sha256=digest(__file__))
    save(output,result);return result


def train(config, frozen, data, audit, output):
    cfg,_,_,_,lock,bank,_=checked(config,frozen); root=Path(data); summary=read(root/'summary.json'); record=read(audit)
    if (not record['verified'] or record['collection_summary_sha256']!=digest(root/'summary.json')
            or record['auditor_sha256']!=digest(__file__) or record['predetermined_native_replays']!=2*len(bank['cases'])
            or summary['lock']!=lock or not summary['complete']):
        raise ValueError('Complete fixed-alpha native collection and replay audit required')
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    for mode in cfg['modes']:
        manifest=root/mode/'sequence_manifest.json'
        env=V46SequenceEnv(dict(sequence_manifest=str(manifest)))
        if (env.source_sha256!=summary['modes'][mode]['sequence_sha256']
                or env.dataset['state_law']!=bank['state_law']):
            raise ValueError('Audited PPO data or fixed state law changed')
        if not env.positive:raise ValueError('No event-bearing critical sequence for '+mode+'; retain the declared collection')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    save(out/'training_plan.json',dict(config=cfg,lock=lock,collection_summary_sha256=digest(root/'summary.json'),
        collection_audit_sha256=digest(audit),modes=cfg['modes'],training_seeds=cfg['training_seeds'],
        state_sampling=cfg['state_sampling'],state_likelihood_ratio=1.,state_model=None))
    reports={}
    # Separate subprocesses leave Ray lifecycle and random initialization isolated.
    for seed in cfg['training_seeds']:
        for mode in cfg['modes']:
            name=mode+'_seed'+str(seed); destination=out/name
            args=[sys.executable,'-m','scenario_reconstruction.highd_v46_train',
                '--sequence-manifest',str(root/mode/'sequence_manifest.json'),'--config',cfg['training_config'],
                '--iterations','100','--seed',str(seed),'--output',str(destination)]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                RAY_TMPDIR=tempfile.mkdtemp(prefix='v47ppo-'))
            print(dict(phase='ppo_started',mode=mode,seed=seed,iterations=100),flush=True)
            with (out/(name+'.log')).open('x') as stream:
                process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env)
                save(out/(name+'_process.json'),dict(pid=process.pid,mode=mode,seed=seed,command=args))
                code=process.wait()
            if code:raise RuntimeError('PPO process failed; retain logs and partial artifacts: '+name)
            result=read(destination/'training_summary.json')
            if result['completed_iterations']!=100 or result['state_likelihood_ratio']!=1.:raise ValueError('Incomplete or wrong-state PPO training')
            reports[name]=result
            print(dict(phase='ppo_complete',mode=mode,seed=seed,iterations=result['completed_iterations'],
                training_M2_ratio=result['resubstitution_ratio']),flush=True)
    report=dict(complete=True,config=cfg,collection_summary_sha256=digest(root/'summary.json'),
        runs=reports,state_likelihood_ratio=1.,state_model=None,validation_opened=False,
        transferable_QS_training_stopped=True,checkpoint_selection='fixed_iteration_100')
    save(out/'summary.json',report);return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default=DEFAULT);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--output',required=True)
    for action in ('collect','audit','train'):
        f=sub.add_parser(action);f.add_argument('--frozen',required=True);f.add_argument('--output',required=True)
        if action!='collect':f.add_argument('--data',required=True)
        if action=='audit':f.add_argument('--replay',action='store_true')
        if action=='train':f.add_argument('--audit',required=True)
    args=p.parse_args()
    if args.command=='freeze':result=freeze(args.config,args.output)
    elif args.command=='collect':result=collect(args.config,args.frozen,args.output)
    elif args.command=='audit':result=audit_collection(args.config,args.frozen,args.data,args.output,args.replay)
    else:result=train(args.config,args.frozen,args.data,args.audit,args.output)
    print({k:v for k,v in result.items() if k not in ('config','lock','runs','artifact_hashes')},flush=True)


if __name__=='__main__':main()
