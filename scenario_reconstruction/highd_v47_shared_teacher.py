"""Freeze/export/audit/train Single and Dual on identical fixed teacher paths."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from importlib.metadata import version
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from . import highd_v47_shared_teacher_native as native
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
from .highd_v46_assets import ROOT,read,save,digest,ensure_assets
from .highd_v46_d2rl import identity,experiment_target
from .highd_v46_contract import FIXED_ALPHA_OBSERVATION
from .highd_v46_collision_coverage import audit_sample
from .highd_v47_teacher_restore import physical_record
from d2rl_training.highd_v46_sequence_env import log_weight
from d2rl_training.highd_v47_shared_teacher_env import SharedTeacherEnv,CONTRACT,diagnostics

DEFAULT='configs/highd_v47_shared_teacher_ppo_v1.json'
_WORK=None


def components(config):
    cfg=read(config)
    if (cfg['version']!=1 or cfg['training_seeds']!=[7,17,29,41,53]
            or cfg['state_likelihood_ratio']!=1. or cfg['state_model'] is not None
            or cfg['horizon_s']!=4 or cfg['workers']!=4
            or cfg['environment']!={k:sys.version.split()[0] if k=='python' else version(k) for k in cfg['environment']}):
        raise ValueError('Explicitly version changed shared-teacher protocol/environment')
    ppo=read(cfg['ppo_config'])
    if ppo['iterations']!=200 or not ppo['export_all_checkpoints'] or ppo['initial_epsilon']!=.1:
        raise ValueError('Retain paired formal PPO budget and initialization')
    files=[Path(__file__),Path(native.__file__),ROOT/'d2rl_training/highd_v47_shared_teacher_env.py',
        ROOT/'d2rl_training/highd_v46_sequence_env.py',ROOT/'scenario_reconstruction/highd_v46_train.py',
        ROOT/cfg['ppo_config'],ROOT/cfg['physical_config'],ROOT/'scenario_reconstruction/highd_v46_action_dist.py',
        ROOT/'scenario_reconstruction/highd_v46_policy.py',ROOT/'scenario_reconstruction/highd_v47_formal_training_metrics.py']
    return cfg,dict(config_sha256=digest(config),components={str(p.relative_to(ROOT)):digest(p) for p in files})


def completed_teacher(cfg):
    """Honor the parent's RECORDED dependency set, not a new glob including us.

    Adding a new consumer does not rewrite any historical freeze/summary. Every
    dependency that produced the teacher still has to match byte-for-byte.
    """
    parent=read(cfg['teacher_lock'])
    for name,sha in parent['dependencies'].items():
        if digest(ROOT/'scenario_reconstruction'/name)!=sha:raise ValueError('Teacher dependency changed: '+name)
    for path,sha in parent['configs'].items():
        if digest(path)!=sha:raise ValueError('Teacher configuration changed: '+path)
    if digest('configs/highd_v47_teacher_expansion_v1.json')!=parent['config_sha256']:
        raise ValueError('Teacher acquisition protocol changed')
    assets,frozen=ensure_assets();physical=read(cfg['physical_config'])
    if (frozen['natural_target_sha256']!=parent['natural_target_sha256']
            or identity(experiment_target(frozen,physical))!=parent['physical_target_sha256']):
        raise ValueError('Natural/physical target changed')
    bank,data=Path(cfg['bank']),Path(cfg['data'])
    for root,key in ((bank,'bank_summary_sha256'),(data,'data_summary_sha256')):
        if digest(root/'summary.json')!=cfg[key]:raise ValueError('Teacher summary changed')
        s=read(root/'summary.json')
        if not s['complete'] or s['lock_sha256']!=identity(parent):raise ValueError('Incomplete teacher evidence')
        for name,sha in s['artifact_hashes'].items():
            if Path(name).name!=name or digest(root/name)!=sha:raise ValueError('Teacher artifact changed: '+name)
    summary=read(data/'summary.json');audit=read(cfg['audit'])
    if (not audit['verified'] or audit['summary_sha256']!=cfg['data_summary_sha256']
            or audit['auditor_sha256']!=parent['implementation_sha256']
            or audit['native_replays']!=1043 or not audit['collision_count_gate_passed']
            or summary['draws']!=66752 or summary['events']!=2009 or summary['collision_sources']!=209):
        raise ValueError('Complete audited collision-producing teacher required')
    return physical,assets,parent,read(bank/'case_plan.json')['cases']


def freeze(config,output):
    cfg,current=components(config);physical,assets,parent,cases=completed_teacher(cfg)
    result=dict(current,config=cfg,parent_lock_sha256=identity(parent),
        inputs={p:digest(p) for p in (cfg['teacher_lock'],cfg['audit'],
            str(Path(cfg['bank'])/'summary.json'),str(Path(cfg['data'])/'summary.json'),str(Path(cfg['bank'])/'case_plan.json'))},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cases=len(cases),draws=64*len(cases),new_outcomes_opened=False,validation_opened=False)
    save(output,result)
    return dict(frozen=True,shared_draws=result['draws'],new_native_samples=0,PPO_started=False)


def checked(config,lock_path,full=False):
    cfg,current=components(config);lock=read(lock_path)
    if any(lock.get(k)!=v for k,v in current.items()) or cfg!=lock['config']:raise ValueError('Shared replay lock changed')
    for path,sha in lock['inputs'].items():
        if digest(path)!=sha:raise ValueError('Protected input changed: '+path)
    if full:physical,assets,parent,cases=completed_teacher(cfg)
    else:
        assets,_=ensure_assets();physical=read(cfg['physical_config']);parent=read(cfg['teacher_lock'])
        for name,sha in parent['dependencies'].items():
            if digest(ROOT/'scenario_reconstruction'/name)!=sha:raise ValueError('Parent runtime changed: '+name)
        for path,sha in parent['configs'].items():
            if digest(path)!=sha:raise ValueError('Parent configuration changed: '+path)
        cases=read(Path(cfg['bank'])/'case_plan.json')['cases']
    return cfg,lock,physical,assets,parent,cases


def worker_init(assets,physical,cfg):
    global _WORK
    assets=Path(assets);_WORK=(PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS),assets,physical,cfg)


def export_worker(task):
    case,output=task;model,assets,physical,cfg=_WORK
    design=read(Path(cfg['bank'])/case['design_file'])
    originals=read(Path(cfg['data'])/f'case_{case["case_id"]:05d}.json')['sequences']
    paths=[native.replay(design,s,model,assets,physical) for s in originals]
    name=f'case_{case["case_id"]:05d}.json';save(Path(output)/name,dict(plan=case,paths=paths))
    return name,digest(Path(output)/name)


def law(cases):
    return [{k:c[k] for k in ('case_id','source_seed','alpha','state_sha256','actor_ids','continuation_seed_start')}
            for c in cases]


def dataset(mode,seq,cases,parent,lock,shared_hash):
    state_law=law(cases)
    target=identity(dict(natural=parent['natural_target_sha256'],physical=parent['physical_target_sha256'],
        state_law=state_law,horizon_s=4,event='CAV_in_first_any_actor_collision'))
    return dict(contract=CONTRACT,observation_contract=FIXED_ALPHA_OBSERVATION,
        sampling_scope='conditional_state_bank',state_sampling='fixed_stratified_alpha',state_likelihood_ratio=1.,state_model=None,
        action_dimension=1 if mode=='single' else 2,intervention_mode=mode,horizon_s=4,state_law=state_law,
        state_law_sha256=identity(state_law),conditional_target_sha256=target,
        natural_target_sha256=parent['natural_target_sha256'],experiment_target_sha256=parent['physical_target_sha256'],
        behavior_sha256=lock['config']['data_summary_sha256'],shared_paths_sha256=shared_hash,
        proposal_structure_sha256=identity(dict(mode=mode,rule=lock['config']['target_rule'],
            target_native_sha256=lock['components']['scenario_reconstruction/highd_v47_shared_teacher_native.py'],
            physical=parent['physical_target_sha256'])),shared_lock_sha256=identity(lock),sequences=seq)


def readiness(envs,cfg):
    reports={};reasons=[]
    for mode,env in envs.items():
        values=env.objective_values(lambda _:[.1]*env.dimension)
        usable=[];second=[];source_values=defaultdict(float);source_gradient=defaultdict(float)
        for i,s in enumerate(env.sequences):
            source_values[s['source_seed']]+=float(values[i])
            if not s['event_result']:continue
            derivatives=[]
            for t in s['steps']:
                p,h=np.asarray(t['p_factors']),np.asarray(t['h_factors'])
                derivatives.append(np.abs((p-h)/(.1*p+.9*h)))
            useful=bool(derivatives) and max(float(x.max()) for x in derivatives)>=cfg['minimum_relative_derivative']
            if useful and np.isfinite(values[i]) and values[i]>0:usable.append(i)
            if mode=='dual' and useful and any(x[1]>=cfg['minimum_relative_derivative'] for x in derivatives):second.append(i)
            source_gradient[s['source_seed']]+=float(values[i])*sum(float(x.sum()) for x in derivatives)
        sources={env.sequences[i]['source_seed'] for i in usable}
        second_sources={env.sequences[i]['source_seed'] for i in second}
        total=float(values.sum());gtotal=sum(source_gradient.values())
        shares=np.asarray(list(source_values.values()))/total if total else np.array([])
        reports[mode]=dict(sequences=len(env.sequences),CAV_events=sum(s['event_result'] for s in env.sequences),
            target_active_sequences=len(env.eligible),usable_events=len(usable),usable_event_sources=len(sources),
            second_factor_usable_events=len(second),second_factor_usable_sources=len(second_sources),
            target_inactive_event_constants=sum(s['event_result'] and not s['steps'] for s in env.sequences),
            reference_empirical_M2=float(values.mean()*np.exp(env.log_scale)),
            maximum_objective_event_share=float(values.max()/total) if total else None,
            maximum_objective_source_share=float(shares.max()) if total else None,
            effective_objective_sources=float(1/np.sum(shares**2)) if total else None,
            maximum_absolute_epsilon_gradient_source_share=max(source_gradient.values())/gtotal if gtotal else None,
            log_reward_scale=env.log_scale)
        if len(usable)<cfg['minimum_usable_events'] or len(sources)<cfg['minimum_usable_sources']:
            reasons.append(mode+': insufficient target-dependent events/sources')
        if mode=='dual' and (len(second)<cfg['minimum_second_factor_events'] or len(second_sources)<cfg['minimum_second_factor_sources']):
            reasons.append('dual: insufficient independent second-factor signal')
        if not np.isfinite(values).all() or not gtotal>0:reasons.append(mode+': invalid or absent gradient signal')
    a,b=envs['single'],envs['dual']
    keys=('seed','source_seed','case_id','event_result','collision_pairs','generation_log_weight','target_mass')
    if (a.dataset['state_law']!=b.dataset['state_law'] or a.dataset['shared_paths_sha256']!=b.dataset['shared_paths_sha256']
            or len(a.sequences)!=len(b.sequences)
            or any(any(x[k]!=y[k] for k in keys) for x,y in zip(a.sequences,b.sequences))):
        raise ValueError('Single/Dual do not share identical physical data/target masses')
    return dict(passed=not reasons,reasons=reasons,modes=reports,
        scope='Training signal permission only, not validation readiness or variance superiority. Concentration remains reported, not clipped.')


def load_envs(root):
    return {m:SharedTeacherEnv(dict(sequence_manifest=str(Path(root)/m/'sequence_manifest.json'),evaluation_only=True))
            for m in ('single','dual')}


def export(config,lock_path,output):
    cfg,lock,physical,assets,parent,cases=checked(config,lock_path,True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);save(out/'opened_protocol.json',dict(lock=lock))
    shards=out/'paths';shards.mkdir();hashes={}
    with ProcessPoolExecutor(cfg['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init,initargs=(str(assets),physical,cfg)) as pool:
        jobs=[pool.submit(export_worker,(c,str(shards))) for c in cases]
        for job in as_completed(jobs):
            name,sha=job.result();hashes[name]=sha
            if len(hashes)%16==0 or len(hashes)==len(cases):
                print(dict(phase='shared_teacher_target_replay',completed_cases=len(hashes),total_cases=len(cases)),flush=True)
    hashes=dict(sorted(hashes.items()));save(out/'paths_manifest.json',dict(hashes=hashes))
    shared_hash=identity(hashes)
    for mode in ('single','dual'):
        seq=[]
        for c in cases:
            raw=read(shards/f'case_{c["case_id"]:05d}.json')
            for s in raw['paths']:
                seq.append(dict(seed=s['seed'],source_seed=c['source_seed'],case_id=c['case_id'],cav_id=s['cav_id'],
                    event_result=s['event_result'],collision_pairs=s['collision_pairs'],generation_log_weight=s['generation_log_weight'],
                    state_likelihood_ratio=1.,state_log_weight=0.,target_mass=c['alpha']/64,
                    steps=[native.compact_step(t) for t in s['views'][mode]]))
        d=dataset(mode,seq,cases,parent,lock,shared_hash);save(out/mode/'sequences.json',d)
        save(out/mode/'sequence_manifest.json',dict(data_path='sequences.json',sha256=digest(out/mode/'sequences.json'),
            **{k:d[k] for k in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256',
                'state_law_sha256','proposal_structure_sha256','behavior_sha256','shared_paths_sha256','intervention_mode')}))
        del seq,d
    envs=load_envs(out);screen=readiness(envs,cfg)
    files=['opened_protocol.json','paths_manifest.json']+[m+'/'+n for m in envs for n in ('sequences.json','sequence_manifest.json')]
    save(out/'summary.json',dict(complete=True,lock_sha256=identity(lock),shared_paths_sha256=shared_hash,
        files={p:digest(out/p) for p in files},readiness=screen,physical_paths=64*len(cases),new_samples=0,PPO_started=False))
    return dict(complete=True,physical_paths=64*len(cases),new_samples=0,training_signal_ready=screen['passed'],readiness=screen)


def verified_export(root,lock):
    root=Path(root);summary=read(root/'summary.json')
    if not summary['complete'] or summary['lock_sha256']!=identity(lock):raise ValueError('Incomplete/wrong shared export')
    for name,sha in summary['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Invalid export artifact path')
        if digest(root/name)!=sha:raise ValueError('Export changed: '+name)
    hashes=read(root/'paths_manifest.json')['hashes']
    if identity(hashes)!=summary['shared_paths_sha256']:raise ValueError('Shared path identity changed')
    for name,sha in hashes.items():
        if Path(name).name!=name or digest(root/'paths'/name)!=sha:raise ValueError('Shared trajectory changed: '+name)
    return summary


def audit(config,lock_path,data,output,replay=False):
    cfg,lock,physical,assets,parent,cases=checked(config,lock_path,True);root=Path(data)
    summary=verified_export(root,lock);envs=load_envs(root);screen=readiness(envs,cfg)
    if screen!=summary['readiness']:raise ValueError('Readiness reconstruction differs')
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS) if replay else None
    replay_count=0;max_error=0.
    for c in cases:
        raw=read(root/'paths'/f'case_{c["case_id"]:05d}.json')
        originals=read(Path(cfg['data'])/f'case_{c["case_id"]:05d}.json')['sequences']
        if raw['plan']!=c or len(raw['paths'])!=64:raise ValueError('Raw case plan differs')
        for j,(s,original) in enumerate(zip(raw['paths'],originals)):
            audit_sample(original,[.1,.1])
            if (s['seed']!=original['seed'] or s['event_result']!=original['event_result']
                    or s['collision_pairs']!=original['collision_pairs']
                    or s['generation_log_weight']!=original['generation_log_weight']
                    or s['behavior_initial_audit']!=original['initial_likelihood_audit']):
                raise ValueError('Teacher label/path/behavior overwritten')
            decisions=s['native_decisions'];full=sum(t['behavior_full_log_p']-t['behavior_full_log_q'] for t in decisions)
            max_error=max(max_error,abs(full-s['generation_log_weight']))
            for mode,env in envs.items():
                actual=env.sequences[c['case_id']*64+j]
                if actual['steps']!=[native.compact_step(t) for t in s['views'][mode]]:
                    raise ValueError('Compact target view detached from raw replay')
                times=[t['time_s'] for t in s['views'][mode]]
                if times!=[t['time_s'] for t in decisions if t['target_active'][mode]]:
                    raise ValueError('Densification lost a target-active decision')
                for t in s['views'][mode]:
                    decision=next(x for x in decisions if x['time_s']==t['time_s'])
                    if t['full_log_p']!=decision['behavior_full_log_p']:raise ValueError('Natural path density differs by target')
                    if t['action_indices']!=[decision['actions'][str(i)] for i in t['actor_ids']]:
                        raise ValueError('Target actions were resampled')
                    for k,dist in enumerate(t['factor_pmfs']):
                        p,h=np.asarray(dist['p']),np.asarray(dist['h']);active=t['factor_mask'][k]
                        a=t['action_indices'][k] if active else 0
                        if (not np.isfinite(p).all() or not np.isfinite(h).all() or np.any(p<0) or np.any(h<0)
                                or not np.isclose(p.sum(),1) or not np.isclose(h.sum(),1) or np.any(h[p==0]>0)
                                or p[a]!=t['p_factors'][k] or h[a]!=t['h_factors'][k]
                                or not np.isclose(.5*np.abs(p-h).sum(),t['factor_tv'][k])):
                            raise ValueError('Target PMF/support/TV differs')
        if replay:
            actual=native.replay(read(Path(cfg['bank'])/c['design_file']),originals[0],model,assets,physical)
            if physical_record(actual)!=physical_record(raw['paths'][0]):raise ValueError('Independent scheduled replay differs')
            replay_count+=1
        if (c['case_id']+1)%16==0 or c==cases[-1]:
            print(dict(phase='shared_target_audit',cases=c['case_id']+1,total=len(cases),native_replays=replay_count),flush=True)
    if max_error>1e-9:raise ValueError('Full behavior likelihood lost')
    result=dict(verified=True,summary_sha256=digest(root/'summary.json'),lock_sha256=identity(lock),
        auditor_sha256=digest(__file__),native_replays=replay_count,physical_paths=64*len(cases),
        maximum_behavior_log_error=max_error,readiness=screen,PPO_started=False,validation_opened=False)
    save(output,result);return dict(verified=True,native_replays=replay_count,training_signal_ready=screen['passed'])


def training_inputs(config,lock_path,data,audit_path):
    cfg,lock,*_=checked(config,lock_path);summary=verified_export(data,lock);a=read(audit_path)
    if (not a['verified'] or a['summary_sha256']!=digest(Path(data)/'summary.json') or a['lock_sha256']!=identity(lock)
            or a['auditor_sha256']!=digest(__file__) or a['native_replays']!=lock['cases']
            or a['readiness']!=summary['readiness']):raise ValueError('Complete independent replay audit required before PPO')
    return cfg,lock,summary


@contextmanager
def adapter():
    from d2rl_training import highd_v46_sequence_env as base
    from . import highd_v47_formal_training_metrics as metrics
    old,old_metrics=base.V46SequenceEnv,metrics.diagnostics
    base.V46SequenceEnv=SharedTeacherEnv;metrics.diagnostics=diagnostics
    try:yield
    finally:base.V46SequenceEnv=old;metrics.diagnostics=old_metrics


def train(config,lock_path,data,audit_path,output):
    cfg,lock,summary=training_inputs(config,lock_path,data,audit_path)
    if not summary['readiness']['passed']:
        return dict(phase='training_blocked',PPO_started=False,readiness=summary['readiness'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ppo=read(cfg['ppo_config']);ppo['log_reward_scale']=max(r['log_reward_scale'] for r in summary['readiness']['modes'].values())
    save(out/'ppo_config.json',ppo);save(out/'opened_protocol.json',dict(lock=lock,data_summary_sha256=digest(Path(data)/'summary.json'),audit_sha256=digest(audit_path)))
    reports={}
    for seed in cfg['training_seeds']:
        for mode in ('single','dual'):
            name=mode+'_seed'+str(seed)
            args=[sys.executable,'-u','-m','scenario_reconstruction.highd_v47_shared_teacher','ppo',
                '--config',config,'--lock',str(lock_path),'--data',str(data),'--audit',str(audit_path),
                '--mode',mode,'--seed',str(seed),'--ppo-config',str(out/'ppo_config.json'),'--output',str(out/name)]
            env=dict(os.environ,TMPDIR='/tmp',RAY_TMPDIR=tempfile.mkdtemp(prefix='v47shared-'),
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            print(dict(phase='ppo_started',mode=mode,seed=seed,iterations=200,log=str(out/(name+'.log'))),flush=True)
            with (out/(name+'.log')).open('x') as stream:
                process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env)
                save(out/(name+'_process.json'),dict(pid=process.pid));code=process.wait()
            if code:raise RuntimeError('Retain failed PPO and logs: '+name)
            reports[name]=read(out/name/'training_summary.json')
    save(out/'summary.json',dict(complete=True,lock_sha256=identity(lock),runs=reports,
        data_summary_sha256=digest(Path(data)/'summary.json'),audit_sha256=digest(audit_path),validation_pending=True))
    return dict(complete=True,runs=len(reports),native_validation_pending=True)


def ppo(config,lock_path,data,audit_path,mode,seed,ppo_config,output):
    cfg,lock,summary=training_inputs(config,lock_path,data,audit_path)
    expected=read(cfg['ppo_config']);expected['log_reward_scale']=max(r['log_reward_scale'] for r in summary['readiness']['modes'].values())
    if (not summary['readiness']['passed'] or mode not in ('single','dual') or seed not in cfg['training_seeds']
            or read(ppo_config)!=expected):raise ValueError('PPO budget/signal/config differs')
    save(Path(output).with_name(Path(output).name+'_shared_runtime.json'),dict(lock_sha256=identity(lock),
        actual_replay_sha256=lock['components']['d2rl_training/highd_v47_shared_teacher_env.py'],
        shared_behavior_sha256=cfg['data_summary_sha256'],mode=mode,seed=seed,state_likelihood_ratio=1.))
    from . import highd_v46_train as trainer
    previous=sys.argv
    try:
        sys.argv=[previous[0],'--sequence-manifest',str(Path(data)/mode/'sequence_manifest.json'),
            '--config',str(ppo_config),'--seed',str(seed),'--output',str(output)]
        with adapter():trainer.main()
    finally:sys.argv=previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','export','audit','train','ppo'])
    p.add_argument('--config',default=DEFAULT);p.add_argument('--lock');p.add_argument('--data');p.add_argument('--audit')
    p.add_argument('--replay',action='store_true');p.add_argument('--mode');p.add_argument('--seed',type=int)
    p.add_argument('--ppo-config');p.add_argument('--output',required=True);a=p.parse_args()
    if a.command=='freeze':result=freeze(a.config,a.output)
    elif a.command=='export':result=export(a.config,a.lock,a.output)
    elif a.command=='audit':result=audit(a.config,a.lock,a.data,a.output,a.replay)
    elif a.command=='train':result=train(a.config,a.lock,a.data,a.audit,a.output)
    else:result=ppo(a.config,a.lock,a.data,a.audit,a.mode,a.seed,a.ppo_config,a.output)
    print(result,flush=True)


if __name__=='__main__':main()
