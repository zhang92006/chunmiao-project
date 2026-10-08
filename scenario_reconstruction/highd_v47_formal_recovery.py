"""Versioned training-behavior recovery; never change frozen formal v1 files.

Native collection, native audit and PPO are separate researcher-run commands.
Validation still executes the original target sampler, never this behavior.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from .highd_v46_assets import ROOT,read,save,digest
from .highd_v46_d2rl import identity
from .highd_v47_fixed_alpha_contract import state_schedule
from . import highd_v47_formal_comparison as parent
from . import highd_v47_formal_identity_compat as compat
from . import highd_v47_formal_recovery_native as native
from d2rl_training.highd_v47_offpolicy_sequence_env import OffPolicySequenceEnv,CONTRACT,BEHAVIOR

DEFAULT='configs/highd_v47_formal_recovery_v2.json'


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def implementation(config):
    cfg=read(config)
    expected=read(DEFAULT)
    if (cfg!=expected or cfg['version']!=2 or cfg['state_model'] is not None or cfg['state_likelihood_ratio']!=1
            or cfg['state_sampling']!='Q_S_equals_alpha' or cfg['behavior_H']!=BEHAVIOR or cfg['behavior_epsilon']!=.1
            or cfg['target_H']!='unchanged_original_formal_Single_or_Dual_H' or cfg['validation_or_final_test_during_recovery']):
        raise ValueError('Version changes explicitly; fixed-alpha recovery v2 required')
    from d2rl_training import highd_v47_offpolicy_sequence_env as replay
    files=[Path(config),Path(__file__),Path(native.__file__),Path(replay.__file__),Path(compat.__file__),compat.CONFIG]
    return cfg,{relative(p):digest(p) for p in files}


def original_evidence(frozen,bank,data):
    cfg,_,_,lock=parent.checked(parent.DEFAULT,frozen);b,evidence=compat.inspect_bank(bank,lock,'train')
    root=Path(data);summary=read(root/'summary.json');audit=read(root/'audit.json')
    schedule=state_schedule(b['state_law'],cfg['training']['draws_per_mode'],cfg['training']['state_seed'])
    if (not summary['complete'] or summary['lock']!=lock or not audit['verified'] or audit['lock']!=lock
            or audit['collection_sha256']!=digest(root/'summary.json') or audit['native_replays']!=2*len(set(schedule))):
        raise ValueError('Complete audited formal v1 collection required')
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    modes={}
    for mode in ('single','dual'):
        env=V46SequenceEnv(dict(sequence_manifest=str(root/mode/'sequence_manifest.json'),evaluation_only=True))
        if env.source_sha256!=summary['sequence_hashes'][mode] or env.dataset['state_law']!=b['state_law']:
            raise ValueError('Parent collection law/hash changed')
        labels=[any(s['cav_id'] in pair for pair in s['collision_pairs']) for s in env.sequences]
        if any(label!=s['event_result'] for label,s in zip(labels,env.sequences)):raise ValueError('Native event labels differ')
        modes[mode]=dict(sequences=len(labels),CAV_events=sum(labels),
            competing_BV_endpoints=sum(s['event_type']=='bv_only_competing_collision' for s in env.sequences),
            critical_decisions=sum(len(s['steps']) for s in env.sequences))
        if len(labels)!=cfg['training']['draws_per_mode']:raise ValueError('Incomplete parent budget')
    return dict(stage='training_collection_and_native_audit_complete_PPO_blocked',
        parent_lock_sha256=identity(lock),training_bank=evidence,modes=modes,
        native_replays=audit['native_replays'],maximum_full_log_error=audit['full_log_error'],
        collection_summary_sha256=digest(root/'summary.json'),audit_sha256=digest(root/'audit.json'),
        formal_PPO_started=False,state_likelihood_ratio=1.,
        diagnosis='Zero CAV labels, not an identity/likelihood failure; held-action clearance proxy is not native collision probability')


def freeze(config,parent_frozen,bank,data,output):
    cfg,components=implementation(config);pc,_,_,lock=parent.checked(cfg['parent_config'],parent_frozen)
    if any(cfg['training'][k]!=pc['training'][k] for k in ('draws_per_mode','state_seed','seeds','iterations','checkpoint_interval')):
        raise ValueError('Matched parent state/PPO budget required')
    if cfg['training']['continuation_seed_start']!=1300000000:raise ValueError('Recovery continuation namespace changed')
    report=original_evidence(parent_frozen,bank,data)
    if any(v['CAV_events'] for v in report['modes'].values()):raise ValueError('This declared recovery is for the complete zero-event v1, not a favorable retry')
    old=Path(data);pf=Path(parent_frozen);bp=Path(bank)
    inputs=[pf/'lock.json',pf/'historical_inventory.json',bp/'bank.json',bp/'summary.json',old/'summary.json',old/'audit.json',old/'shared_schedule.json']
    inputs += [pf/(s+'_source_manifest.json') for s in parent.ranges(pc)]
    inputs += [old/m/name for m in ('single','dual') for name in ('sequences.json','sequence_manifest.json')]
    b=compat.load_bank(bank,lock,'train');schedule=state_schedule(b['state_law'],cfg['training']['draws_per_mode'],cfg['training']['state_seed'])
    plan=dict(case_indices=schedule,seeds=[cfg['training']['continuation_seed_start']+i for i in range(len(schedule))])
    recovery=dict(config=cfg,components=components,inputs={relative(p):digest(p) for p in inputs},
        parent_frozen=relative(parent_frozen),training_bank=relative(bank),failed_v1_data=relative(data),
        parent_lock_sha256=identity(lock),state_law_sha256=identity(b['state_law']),schedule_sha256=identity(plan),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        no_validation_or_final_test=True,old_new_data_pooling=False)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    save(out/'failed_v1_evidence.json',report);save(out/'shared_schedule.json',plan);save(out/'lock.json',recovery)
    return dict(frozen=True,states=len(b['cases']),sources=len({c['source_seed'] for c in b['cases']}),
        state_likelihood_ratio=1.,new_native_outcomes_opened=False)


def checked(config,frozen):
    cfg,components=implementation(config);r=read(Path(frozen)/'lock.json')
    if r['config']!=cfg or r['components']!=components:raise ValueError('Recovery implementation/config changed after freeze')
    for path,sha in r['inputs'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Protected input changed: '+path)
    pc,physical,assets,lock=parent.checked(cfg['parent_config'],ROOT/r['parent_frozen'])
    if identity(lock)!=r['parent_lock_sha256']:raise ValueError('Parent frozen target changed')
    b=compat.load_bank(ROOT/r['training_bank'],lock,'train')
    schedule=read(Path(frozen)/'shared_schedule.json')
    if identity(schedule)!=r['schedule_sha256'] or identity(b['state_law'])!=r['state_law_sha256']:raise ValueError('State plan changed')
    if schedule['case_indices']!=state_schedule(b['state_law'],cfg['training']['draws_per_mode'],cfg['training']['state_seed']):raise ValueError('Q_S must equal alpha')
    if schedule['seeds']!=list(range(1300000000,1300000000+cfg['training']['draws_per_mode'])):raise ValueError('Continuation schedule changed')
    return cfg,pc,physical,assets,lock,r,b,schedule


def worker(task):
    mode,seed,case=task;model,assets,physical=parent._WORK
    return native.rollout(model,assets,physical,mode,seed,case)


def training_readiness(envs,cfg):
    reasons=[];reports={}
    for mode,env in envs.items():
        values=env.objective_values(lambda obs:[.1]*env.dimension)
        usable=[i for i in env.positive if np.isfinite(values[i]) and values[i]>0 and
            any(np.any(np.asarray(t['p_factors'])!=np.asarray(t['h_factors'])) for t in env.sequences[i]['steps'])]
        sources={env.sequences[i]['source_seed'] for i in usable}
        second=[i for i in usable if mode=='dual' and any(t['factor_mask'][1] and t['p_factors'][1]!=t['h_factors'][1]
            for t in env.sequences[i]['steps'])]
        total=values.sum();shares=values/total if total>0 else np.zeros_like(values)
        reports[mode]=dict(sequences=len(env.sequences),CAV_events=sum(s['event_result'] for s in env.sequences),usable_events=len(usable),
            usable_event_sources=len(sources),maximum_training_objective_event_share=float(shares.max()),
            effective_training_objective_events=float(1/np.dot(shares,shares)) if total>0 else None,
            second_factor_usable_events=len(second),log_reward_scale=env.log_scale,state_likelihood_ratio=1.)
        if len(usable)<cfg['readiness']['minimum_usable_events_per_mode'] or len(sources)<cfg['readiness']['minimum_usable_event_sources_per_mode']:
            reasons.append(mode+': insufficient nonzero CAV reward/contrast across independent training sources')
        if mode=='dual' and cfg['readiness']['require_dual_second_factor_signal'] and not second:
            reasons.append('dual: no event-weighted active second-factor target contrast')
    if envs['single'].dataset['state_law']!=envs['dual'].dataset['state_law']:reasons.append('Single/Dual state laws differ')
    return dict(passed=not reasons,reasons=reasons,modes=reports,
        scope='Training-signal gate only; not validation readiness, source generalization or efficiency evidence')


def load_data(root,lock,r,b):
    root=Path(root);s=read(root/'summary.json')
    if not s['complete'] or s['lock']!=lock or s['recovery_lock_sha256']!=identity(r):raise ValueError('Wrong recovery collection')
    envs={}
    for mode in ('single','dual'):
        env=OffPolicySequenceEnv(dict(sequence_manifest=str(root/mode/'sequence_manifest.json'),evaluation_only=True))
        expected=parent.dataset(mode,[],b,lock['config'],lock)
        target_keys=('natural_target_sha256','experiment_target_sha256','conditional_target_sha256',
            'proposal_structure_sha256','proposal_sha256','observation_contract','action_dimension',
            'intervention_mode','horizon_s','state_schedule_seed')
        if (env.source_sha256!=s['sequence_hashes'][mode] or env.dataset['state_law']!=b['state_law']
                or env.dataset['recovery_lock_sha256']!=identity(r)
                or any(env.dataset[k]!=expected[k] for k in target_keys)
                or env.dataset['behavior_sha256']!=identity(dict(recovery=identity(r),mode=mode))
                or env.dataset['continuation_seed_start']!=r['config']['training']['continuation_seed_start']
                or len(env.sequences)!=r['config']['training']['draws_per_mode']):raise ValueError('Recovery data/law/budget changed')
        envs[mode]=env
    if training_readiness(envs,r['config'])!=s['training_readiness']:raise ValueError('Training-signal report changed')
    return s,envs


def collect_train(config,frozen,output):
    cfg,pc,physical,assets,lock,r,b,plan=checked(config,frozen)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);save(out/'opened_protocol.json',dict(lock=lock,recovery_lock=r))
    save(out/'shared_schedule.json',plan);envs={};hashes={}
    for mode in ('single','dual'):
        tasks=[(mode,seed,b['cases'][k]) for seed,k in zip(plan['seeds'],plan['case_indices'])]
        save(out/(mode+'_plan.json'),dict(tasks=[dict(mode=m,seed=s,case_id=c['case_id']) for m,s,c in tasks]))
        seq=[]
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=parent.worker_init,initargs=(str(assets),physical)) as pool:
            for i,result in enumerate(pool.map(worker,tasks,chunksize=1)):
                save(out/mode/('draw_%05d.json'%i),result);result['draw_index']=i;seq.append(result)
                if (i+1)%128==0 or i+1==len(tasks):print(dict(phase=mode+'_behavior_collection',completed=i+1,total=len(tasks)),flush=True)
        save(out/(mode+'_artifacts.json'),dict(hashes={str(p.relative_to(out)):digest(p) for p in sorted((out/mode).glob('draw_*.json'))}))
        # Keep TARGET identities byte-identical to formal v1; identify BEHAVIOR separately.
        d=parent.dataset(mode,seq,b,pc,lock)
        d.update(contract=CONTRACT,behavior_H=BEHAVIOR,target_H=cfg['target_H'],
            behavior_sha256=identity(dict(recovery=identity(r),mode=mode)),recovery_lock_sha256=identity(r),
            continuation_seed_start=cfg['training']['continuation_seed_start'])
        save(out/mode/'sequences.json',d)
        manifest=dict(data_path='sequences.json',sha256=digest(out/mode/'sequences.json'),**{k:d[k] for k in
            ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256','state_law_sha256',
             'proposal_structure_sha256','proposal_sha256','intervention_mode','behavior_sha256')})
        save(out/mode/'sequence_manifest.json',manifest);hashes[mode]=manifest['sha256']
        envs[mode]=OffPolicySequenceEnv(dict(sequence_manifest=str(out/mode/'sequence_manifest.json'),evaluation_only=True))
    screen=training_readiness(envs,cfg)
    save(out/'summary.json',dict(complete=True,lock=lock,recovery_lock_sha256=identity(r),sequence_hashes=hashes,
        shared_schedule_sha256=digest(out/'shared_schedule.json'),training_readiness=screen,
        sampling_role='Training behavior only; event rates are not target-Q evaluation results'))
    return dict(complete=True,events={m:v['CAV_events'] for m,v in screen['modes'].items()},training_signal_ready=screen['passed'])


def canonical_path(s):
    skip={'cpu_s','wall_time_s','draw_index'};value={k:v for k,v in s.items() if k not in skip}
    for key in ('steps','native_decisions'):value[key]=[{k:v for k,v in t.items() if k!='decision_cpu_s'} for t in s[key]]
    return identity(value)


def audit_train(config,frozen,data,output,replay=False):
    cfg,pc,physical,assets,lock,r,b,plan=checked(config,frozen);root=Path(data)
    s,envs=load_data(root,lock,r,b);count=0;error=0.
    if read(root/'shared_schedule.json')!=plan or digest(root/'shared_schedule.json')!=s['shared_schedule_sha256']:raise ValueError('Changed shared schedule')
    model=None
    if replay:
        from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
        model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    for mode,env in envs.items():
        manifest=read(root/(mode+'_artifacts.json'))['hashes']
        expected_paths={mode+'/draw_%05d.json'%i for i in range(len(plan['seeds']))}
        if set(manifest)!=expected_paths:raise ValueError('Incomplete raw artifact manifest')
        if read(root/(mode+'_plan.json'))['tasks']!=[dict(mode=mode,seed=seed,case_id=b['cases'][k]['case_id']) for seed,k in zip(plan['seeds'],plan['case_indices'])]:raise ValueError('Changed mode plan')
        first={}
        for i,seq in enumerate(env.sequences):
            path=mode+'/draw_%05d.json'%i
            if digest(root/path)!=manifest[path]:raise ValueError('Raw native path changed')
            raw=read(root/path)
            if identity(raw)!=identity({k:v for k,v in seq.items() if k!='draw_index'}):raise ValueError('Raw/sequence differs')
            first.setdefault(plan['case_indices'][i],i)
            error=max(error,abs(sum(t['log_p']-t['log_q'] for t in seq['native_decisions'])-seq['generation_log_weight']))
        if replay:
            for k,i in sorted(first.items()):
                actual=native.rollout(model,assets,physical,mode,plan['seeds'][i],b['cases'][k])
                if canonical_path(actual)!=canonical_path(env.sequences[i]):raise ValueError('Recovery scheduled native replay differs')
                count+=1
                if count%64==0:print(dict(phase='recovery_native_replay',completed=count,total=2*len(set(plan['case_indices']))),flush=True)
    if error>1e-9:raise ValueError('Full behavior likelihood differs')
    save(output,dict(verified=True,lock=lock,recovery_lock_sha256=identity(r),collection_sha256=digest(root/'summary.json'),
        native_replays=count,full_log_error=error,training_readiness=s['training_readiness']))
    return dict(verified=True,native_replays=count,full_log_error=error,training_signal_ready=s['training_readiness']['passed'])


def train(config,frozen,data,output):
    cfg,pc,_,_,lock,r,b,plan=checked(config,frozen);data=Path(data);s,envs=load_data(data,lock,r,b)
    audit=read(data/'audit.json')
    if (not audit['verified'] or audit['lock']!=lock or audit['recovery_lock_sha256']!=identity(r)
            or audit['collection_sha256']!=digest(data/'summary.json') or audit['native_replays']!=2*len(set(plan['case_indices']))):
        raise ValueError('Complete scheduled native audit required before PPO')
    screen=training_readiness(envs,cfg)
    if not screen['passed']:
        # No output directory or Ray subprocess is started on a scientific stop.
        return dict(phase='training_blocked',PPO_started=False,validation_opened=False,
            reasons=screen['reasons'],readiness=screen,action='Retain this complete attempt; no automatic extension, retry, relabeling or guard removal')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ppo=read(pc['ppo_config']);ppo['log_reward_scale']=max(e.log_scale for e in envs.values())
    save(out/'ppo_config.json',ppo);save(out/'opened_protocol.json',dict(lock=lock,recovery_lock=r,data_summary_sha256=digest(data/'summary.json')))
    reports={};runtime_hashes={}
    for seed in cfg['training']['seeds']:
        for mode in ('single','dual'):
            name=mode+'_seed'+str(seed)
            args=[sys.executable,'-m','scenario_reconstruction.highd_v47_formal_recovery','ppo',
                '--sequence-manifest',str(data/mode/'sequence_manifest.json'),'--ppo-config',str(out/'ppo_config.json'),
                '--seed',str(seed),'--output',str(out/name),'--frozen',str(frozen),'--data',str(data)]
            env=dict(os.environ,TMPDIR='/tmp',RAY_TMPDIR='/tmp/v47recovery_'+str(os.getpid())+'_'+name,
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            print(dict(phase='ppo_started',run=name,iterations=200,log=str(out/(name+'.log'))),flush=True)
            with (out/(name+'.log')).open('x') as stream:
                process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env)
                save(out/(name+'_process.json'),dict(pid=process.pid));code=process.wait()
            if code:raise RuntimeError('Retain failed run and logs: '+name)
            reports[name]=read(out/name/'training_summary.json')
            runtime_hashes[name]=digest(out/(name+'_offpolicy_runtime.json'))
    save(out/'summary.json',dict(complete=True,lock=lock,recovery_lock_sha256=identity(r),runs=reports,
        data_summary_sha256=digest(data/'summary.json'),recovery_data=relative(data),
        audit_sha256=digest(data/'audit.json'),runtime_hashes=runtime_hashes,selection_pending=True))
    return dict(complete=True,runs=len(reports),native_validation_pending=True)


@contextmanager
def replay_adapter():
    from d2rl_training import highd_v46_sequence_env as base
    old=base.V46SequenceEnv;base.V46SequenceEnv=OffPolicySequenceEnv
    try:yield
    finally:base.V46SequenceEnv=old


def ppo(frozen,data,manifest,ppo_config,seed,output,config=DEFAULT):
    cfg,pc,_,_,lock,r,b,plan=checked(config,frozen);s,envs=load_data(data,lock,r,b)
    if not s['training_readiness']['passed']:raise ValueError('Training readiness failed')
    audit=read(Path(data)/'audit.json')
    if not audit['verified'] or audit['recovery_lock_sha256']!=identity(r) or audit['collection_sha256']!=digest(Path(data)/'summary.json') or audit['native_replays']!=2*len(set(plan['case_indices'])):
        raise ValueError('Native replay audit required')
    p=Path(manifest).resolve();allowed={Path(data).resolve()/m/'sequence_manifest.json' for m in ('single','dual')}
    expected=read(pc['ppo_config']);expected['log_reward_scale']=max(e.log_scale for e in envs.values())
    if p not in allowed or seed not in cfg['training']['seeds'] or read(ppo_config)!=expected:
        raise ValueError('PPO must use the frozen paired budget/config/collection')
    # Sidecar documents the actual replay subclass; old trainer provenance's
    # environment_sha256 continues to name the inherited BASE implementation.
    runtime=dict(recovery_lock_sha256=identity(r),actual_replay_sha256=r['components']['d2rl_training/highd_v47_offpolicy_sequence_env.py'],
        base_environment_sha256=lock['components']['d2rl_training/highd_v46_sequence_env.py'],
        behavior_role=BEHAVIOR,target_role=cfg['target_H'],state_likelihood_ratio=1.,
        sequence_manifest_sha256=digest(manifest),ppo_config_sha256=digest(ppo_config),seed=seed)
    out=Path(output);save(out.with_name(out.name+'_offpolicy_runtime.json'),runtime)
    from . import highd_v46_train as trainer
    previous=sys.argv
    try:
        sys.argv=[previous[0],'--sequence-manifest',str(manifest),'--config',str(ppo_config),'--seed',str(seed),'--output',str(output)]
        with replay_adapter():trainer.main()
    finally:sys.argv=previous


def verified_training(path,pc,lock,r):
    summary=parent.verified_training(path,pc,lock)
    if summary.get('recovery_lock_sha256')!=identity(r):raise ValueError('Recovery training provenance missing')
    data=ROOT/summary['recovery_data'];collection,envs=load_data(data,lock,r,compat.load_bank(ROOT/r['training_bank'],lock,'train'))
    if (not collection['training_readiness']['passed'] or digest(data/'summary.json')!=summary['data_summary_sha256']
            or digest(data/'audit.json')!=summary['audit_sha256']):raise ValueError('Training collection/audit changed')
    for name,report in summary['runs'].items():
        runtime_path=Path(path)/(name+'_offpolicy_runtime.json');runtime=read(runtime_path)
        mode=name.split('_')[0]
        if (runtime['recovery_lock_sha256']!=identity(r)
                or digest(runtime_path)!=summary['runtime_hashes'][name]
                or runtime['actual_replay_sha256']!=r['components']['d2rl_training/highd_v47_offpolicy_sequence_env.py']
                or runtime['behavior_role']!=BEHAVIOR or runtime['target_role']!=r['config']['target_H']
                or runtime['state_likelihood_ratio']!=1. or runtime['seed']!=report['seed']
                or report['sequence_sha256']!=envs[mode].source_sha256):raise ValueError('Off-policy runtime identity changed')
    return summary


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default=DEFAULT);sub=p.add_subparsers(dest='command',required=True)
    for name in ('status','freeze','collect-train','audit-train','train','ppo','bank','validate','seal','final-test','audit-final'):
        a=sub.add_parser(name)
        if name!='status':a.add_argument('--output',required=True)
        if name in ('status','freeze'):
            a.add_argument('--parent-frozen',required=True);a.add_argument('--bank',required=True);a.add_argument('--data',required=True)
        else:a.add_argument('--frozen',required=True)
        if name in ('audit-train','train','ppo'):a.add_argument('--data',required=True)
        if name in ('audit-train','audit-final'):a.add_argument('--replay',action='store_true')
        if name=='ppo':
            a.add_argument('--sequence-manifest',required=True);a.add_argument('--ppo-config',required=True);a.add_argument('--seed',required=True,type=int)
        if name=='bank':a.add_argument('--split',choices=['validation'],required=True);a.add_argument('--training',required=True)
        if name=='validate':a.add_argument('--bank',required=True);a.add_argument('--training',required=True)
        if name=='seal':a.add_argument('--validation',required=True);a.add_argument('--training',required=True)
        if name=='final-test':a.add_argument('--policy-lock',required=True);a.add_argument('--training',required=True)
        if name=='audit-final':a.add_argument('--result',required=True)
    a=p.parse_args()
    if a.command=='status':result=original_evidence(a.parent_frozen,a.bank,a.data)
    elif a.command=='freeze':result=freeze(a.config,a.parent_frozen,a.bank,a.data,a.output)
    elif a.command=='collect-train':result=collect_train(a.config,a.frozen,a.output)
    elif a.command=='audit-train':result=audit_train(a.config,a.frozen,a.data,a.output,a.replay)
    elif a.command=='train':
        result=train(a.config,a.frozen,a.data,a.output)
        print(result,flush=True)
        if not result.get('complete'):raise SystemExit(2)
        return
    elif a.command=='ppo':
        ppo(a.frozen,a.data,a.sequence_manifest,a.ppo_config,a.seed,a.output,a.config);return
    else:
        # Recovery guards first; downstream native jobs retain ORIGINAL target H.
        _,pc,_,_,lock,r,_,_=checked(a.config,a.frozen);pf=ROOT/r['parent_frozen']
        if a.command in ('bank','validate','seal','final-test'):verified_training(a.training,pc,lock,r)
        with compat.bank_reader_adapter():
            if a.command=='bank':result=parent.bank(parent.DEFAULT,pf,'validation',a.output,a.training)
            elif a.command=='validate':result=parent.validate(parent.DEFAULT,pf,a.bank,a.training,a.output)
            elif a.command=='seal':
                if read(Path(a.validation)/'selection.json')['training_summary_sha256']!=digest(Path(a.training)/'summary.json'):
                    raise ValueError('Validation belongs to another training run')
                result=parent.seal(parent.DEFAULT,pf,a.validation,a.output)
                save(Path(a.output).with_name(Path(a.output).name+'_recovery.json'),dict(
                    recovery_lock_sha256=identity(r),training_summary_sha256=digest(Path(a.training)/'summary.json'),
                    policy_lock_sha256=digest(a.output)))
            elif a.command=='final-test':
                sealed=Path(a.policy_lock);sidecar=read(sealed.with_name(sealed.name+'_recovery.json'))
                if (sidecar['recovery_lock_sha256']!=identity(r) or sidecar['training_summary_sha256']!=digest(Path(a.training)/'summary.json')
                        or sidecar['policy_lock_sha256']!=digest(sealed)):raise ValueError('Wrong recovery policy seal')
                result=parent.final_test(parent.DEFAULT,pf,a.policy_lock,a.output)
            else:result=parent.audit_final(parent.DEFAULT,pf,a.result,a.output,a.replay)
    print(result,flush=True)


if __name__=='__main__':main()
