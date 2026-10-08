"""Startup-only Ray recovery; preserve completed runs and frozen training inputs."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import FunctionType, SimpleNamespace

from . import highd_v47_shared_teacher as shared
from .highd_v46_assets import read, save, digest
from .highd_v46_d2rl import identity


def compact_forward(model):
    """Copy ONE closure cell; never replace the trainer's live check variable.

    Frozen NativeHead.forward uses check.observation_contract only. Capturing
    the complete environment unnecessarily serializes its entire dataset into
    Ray's registry. Keep bytecode, globals, class/super cell and all other cells.
    """
    method=model.forward
    names=method.__code__.co_freevars
    if 'check' not in names or 'observation_contract' not in method.__code__.co_names:
        raise ValueError('Unexpected frozen head closure')
    cells=list(method.__closure__)
    old=cells[names.index('check')].cell_contents
    lightweight=SimpleNamespace(observation_contract=old.observation_contract)
    def capture():return lightweight
    cells[names.index('check')]=capture.__closure__[0]
    new=FunctionType(method.__code__,method.__globals__,method.__name__,method.__defaults__,tuple(cells))
    new.__kwdefaults__=method.__kwdefaults__
    new.__annotations__=method.__annotations__
    new.__dict__.update(method.__dict__)
    new.__qualname__=method.__qualname__
    model.forward=new
    return model


@contextmanager
def small_model_registration():
    from ray.rllib.models import ModelCatalog
    import ray.cloudpickle as pickle
    original=ModelCatalog.register_custom_model
    def register(name,model):
        if name=='highd_v46_mean_precision':
            model=compact_forward(model)
            size=len(pickle.dumps(model))
            if size>=1_000_000:raise ValueError('Unexpectedly large compact model registration')
            print(dict(phase='compact_model_registration',serialized_bytes=size),flush=True)
        return original(name,model)
    ModelCatalog.register_custom_model=staticmethod(register)
    try:yield
    finally:ModelCatalog.register_custom_model=staticmethod(original)


def completed_run(path,mode,seed,config,data):
    """Require all scheduled iterations/exports, not merely a directory name."""
    path=Path(path);s=read(path/'training_summary.json')
    if (s['completed_iterations']!=200 or s['seed']!=seed or s['intervention_mode']!=mode
            or s['config']!=config or s['sequence_sha256']!=digest(Path(data)/mode/'sequences.json')
            or s['state_likelihood_ratio']!=1. or s['validation_episodes']!=0):
        raise ValueError('Completed run differs from locked schedule: '+str(path))
    for i in range(1,201):
        if read(path/f'iteration_{i:04d}.json')['iteration']!=i:raise ValueError('Missing/mismatched iteration')
    catalog=read(path/'checkpoint_catalog.json')['checkpoints']
    if [x['iteration'] for x in catalog]!=list(range(0,201,10)):raise ValueError('Checkpoint schedule differs')
    for c in catalog:
        directory=path/c['policy_path']
        if (digest(directory/'policy.json')!=c['metadata_sha256']
                or digest(directory/'policy_weights.npz')!=c['weights_sha256']
                or not (path/c['checkpoint']).exists()):raise ValueError('Completed checkpoint differs')
    final=read(path/'policy.json')
    if digest(path/'policy_weights.npz')!=final['weights_sha256']:raise ValueError('Final policy changed')
    return s


def plan(previous,cfg,lock,data,audit):
    previous=Path(previous);opened=read(previous/'opened_protocol.json')
    if (opened['lock']!=lock or opened['data_summary_sha256']!=digest(Path(data)/'summary.json')
            or opened['audit_sha256']!=digest(audit)):
        raise ValueError('Interrupted run provenance differs')
    config=read(cfg['ppo_config'])
    screen=read(Path(data)/'summary.json')['readiness']
    config['log_reward_scale']=max(v['log_reward_scale'] for v in screen['modes'].values())
    if read(previous/'ppo_config.json')!=config:raise ValueError('Interrupted PPO config differs')
    done={};pending=[]
    for seed in cfg['training_seeds']:
        for mode in ('single','dual'):
            name=mode+'_seed'+str(seed);p=previous/name
            if (p/'training_summary.json').exists():
                completed_run(p,mode,seed,config,data);done[name]=str(p.resolve())
            else:
                # A run with any checkpoint/iteration must use explicit trainer
                # restoration, never silently start again from initialization.
                if p.exists() and any(p.iterdir()):raise ValueError('Partial trained run needs checkpoint recovery: '+name)
                if p.exists():
                    log=(previous/(name+'.log')).read_text()
                    if 'Received message larger than max' not in log or 'trainer.train' in log:
                        raise ValueError('Not the documented pre-PPO registration failure: '+name)
                pending.append(dict(name=name,mode=mode,seed=seed))
    hashes={str(p.relative_to(previous)):digest(p) for p in sorted(previous.rglob('*')) if p.is_file()}
    return config,done,pending,hashes


def run(config,lock_path,data,audit,previous,output):
    cfg,lock,summary=shared.training_inputs(config,lock_path,data,audit)
    if not summary['readiness']['passed']:raise ValueError('Training signal gate failed')
    ppo,done,pending,hashes=plan(previous,cfg,lock,data,audit)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    save(out/'ppo_config.json',ppo)
    save(out/'recovery_protocol.json',dict(lock_sha256=identity(lock),config=config,
        lock_path=str(lock_path),data=str(data),audit=str(audit),previous=str(previous),
        previous_files=hashes,completed=done,pending=pending,
        recovery_sha256=digest(__file__),ppo_config_sha256=digest(out/'ppo_config.json'),
        reason='Ray model closure serialized training dataset; startup-only retries, no partial-training restart',
        data_summary_sha256=digest(Path(data)/'summary.json'),audit_sha256=digest(audit),validation_opened=False))
    paths=dict(done)
    print(dict(phase='recovery_plan',retained=list(done),remaining=len(pending)),flush=True)
    for task in pending:
        name=task['name'];log=out/(name+'.log')
        args=[sys.executable,'-u','-m','scenario_reconstruction.highd_v47_shared_teacher_recovery','ppo',
            '--recovery',str(out),'--mode',task['mode'],'--seed',str(task['seed'])]
        env=dict(os.environ,TMPDIR='/tmp',RAY_TMPDIR=tempfile.mkdtemp(prefix='v47recover-'),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        print(dict(phase='ppo_started',**task,iterations=200,log=str(log)),flush=True)
        with log.open('x') as stream:
            process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env)
            save(out/(name+'_process.json'),dict(pid=process.pid));code=process.wait()
        if code:raise RuntimeError('Retain recovery logs and artifacts: '+name)
        completed_run(out/name,task['mode'],task['seed'],ppo,data);paths[name]=str((out/name).resolve())
    for name,sha in hashes.items():
        if digest(Path(previous)/name)!=sha:raise ValueError('Original run changed during recovery')
    save(out/'summary.json',dict(complete=True,runs=paths,recovery_protocol_sha256=digest(out/'recovery_protocol.json'),
        lock_sha256=identity(lock),validation_pending=True))
    return dict(complete=True,runs=len(paths),validation_pending=True)


def ppo(recovery,mode,seed):
    root=Path(recovery);p=read(root/'recovery_protocol.json');name=mode+'_seed'+str(seed)
    if (p['recovery_sha256']!=digest(__file__) or p['ppo_config_sha256']!=digest(root/'ppo_config.json')
            or dict(name=name,mode=mode,seed=seed) not in p['pending']):raise ValueError('Recovery identity/schedule changed')
    with small_model_registration():
        return shared.ppo(p['config'],p['lock_path'],p['data'],p['audit'],mode,seed,root/'ppo_config.json',root/name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','ppo'])
    parser.add_argument('--config',default=shared.DEFAULT);parser.add_argument('--lock');parser.add_argument('--data')
    parser.add_argument('--audit');parser.add_argument('--previous');parser.add_argument('--output')
    parser.add_argument('--recovery');parser.add_argument('--mode');parser.add_argument('--seed',type=int)
    a=parser.parse_args()
    result=run(a.config,a.lock,a.data,a.audit,a.previous,a.output) if a.command=='run' else ppo(a.recovery,a.mode,a.seed)
    print(result,flush=True)


if __name__=='__main__':main()
