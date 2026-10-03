"""One-command V46 freeze verification, data collection, training and evaluation."""
import argparse
from pathlib import Path
import sys
from .highd_v46_assets import ROOT, ensure_assets, read, save, digest


def merge_collections(sources,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    datasets=[read(Path(s)/'sequences.json') for s in sources]
    result={k:v for k,v in datasets[0].items() if k!='sequences'};sequences=[];seen=set()
    for d in datasets:
        for key in ('contract','observation_contract','action_dimension','intervention_mode','natural_target_sha256','experiment_target_sha256'):
            if d[key]!=result[key]:raise ValueError('Mixed collection targets: '+key)
        for seq in d['sequences']:
            if seq['seed'] in seen:raise ValueError('Duplicate collection seed')
            seen.add(seq['seed']);sequences.append(seq)
    result['sequences']=sequences
    save(output/'sequences.json',result)
    manifest={'data_path':'sequences.json','sha256':digest(output/'sequences.json'),
        'natural_target_sha256':result['natural_target_sha256'],'experiment_target_sha256':result['experiment_target_sha256'],
        'intervention_mode':result['intervention_mode'],'episodes':len(sequences),'positive_episodes':sum(s['event_result'] for s in sequences)}
    save(output/'sequence_manifest.json',manifest);return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify')
    merge=sub.add_parser('merge');merge.add_argument('--sources',nargs='+',required=True);merge.add_argument('--output',required=True)
    for command in ('start','evaluate'):
        p=sub.add_parser(command);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20000 if command=='evaluate' else 1000)
        p.add_argument('--episodes',type=int,default=128 if command=='evaluate' else 512);p.add_argument('--config',default=str(ROOT/'configs/highd_v46_d2rl_experiment.json'))
        if command=='start':
            p.add_argument('--mode',choices=['single','dual'],default='single');p.add_argument('--iterations',type=int,default=100)
            p.add_argument('--sequence-manifest');p.add_argument('--training-seed',type=int,default=7)
        else:p.add_argument('--policy',required=True)
    args=parser.parse_args()
    if args.command=='merge':print(merge_collections(args.sources,args.output));return
    assets,lock=ensure_assets()
    if args.command=='verify':print({'verified':True,'natural_target_sha256':lock['natural_target_sha256'],'assets':str(assets)});return
    from .highd_v46_d2rl import collect,identity
    from . import highd_v46_d2rl as runtime
    expected={'natural_target_sha256':lock['natural_target_sha256'],'experiment':read(args.config),
        'cav_role':'uniform_initialized_actor_with_observed_current_leader',
        'event':'CAV_in_first_any_actor_collision_by_horizon; BV_only_is_competing_negative','runtime_sha256':digest(runtime.__file__)}
    expected_hash=identity(expected)
    if args.command=='evaluate':
        from .highd_v46_policy import NumpyPolicy
        policy=NumpyPolicy(args.policy)
        if policy.metadata['experiment_target_sha256']!=expected_hash:raise ValueError('Policy belongs to another CAV/initial/event/runtime target')
        print(collect(args.config,args.output,policy.metadata['intervention_mode'],args.episodes,args.seed,assets,policy));return
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=args.sequence_manifest
    if manifest is None:
        collect(args.config,out/'collection',args.mode,args.episodes,args.seed,assets)
        manifest=str(out/'collection/sequence_manifest.json')
    m=read(manifest)
    if m['natural_target_sha256']!=lock['natural_target_sha256'] or m['intervention_mode']!=args.mode:
        raise ValueError('Training data belongs to another mode or frozen NDE')
    if m['experiment_target_sha256']!=expected_hash:
        raise ValueError('Training data belongs to another CAV/initial/event/runtime target')
    from .highd_v46_train import main as train
    sys.argv=[sys.argv[0],'--sequence-manifest',manifest,'--output',str(out/'training'),'--iterations',str(args.iterations),'--seed',str(args.training_seed)]
    train()


if __name__=='__main__':main()
