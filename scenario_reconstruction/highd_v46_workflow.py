"""One-command V46 freeze verification, data collection, training and evaluation."""
import argparse
from pathlib import Path
import sys
from .highd_v46_assets import ROOT, ensure_assets, read, save, digest


def merge_collections(sources,output):
    output=Path(output)
    from .highd_v46_statistics import load_collection,summarize_sequences
    from .highd_v46_d2rl import identity
    datasets=[load_collection(s) for s in sources]
    result={k:v for k,v in datasets[0].items() if k!='sequences'};sequences=[];seen=set()
    for d in datasets:
        for key in ('contract','observation_contract','action_dimension','intervention_mode','natural_target_sha256','experiment_target_sha256'):
            if d[key]!=result[key]:raise ValueError('Mixed collection targets: '+key)
        if d.get('proposal_structure_sha256')!=result.get('proposal_structure_sha256'):
            raise ValueError('Mixed candidate/risk-proxy structures')
        for seq in d['sequences']:
            if seq['seed'] in seen:raise ValueError('Duplicate collection seed')
            seen.add(seq['seed']);sequences.append(seq)
    result['sequences']=sequences
    if 'proposal' in result:
        result['proposal']={'kind':'logged_behavior_mixture','sources':[d['proposal'] for d in datasets]}
        result['proposal_sha256']=identity(result['proposal'])
    output.mkdir(parents=True,exist_ok=False)
    save(output/'sequences.json',result)
    manifest={'data_path':'sequences.json','sha256':digest(output/'sequences.json'),
        'natural_target_sha256':result['natural_target_sha256'],'experiment_target_sha256':result['experiment_target_sha256'],
        'intervention_mode':result['intervention_mode'],'episodes':len(sequences),'positive_episodes':sum(s['event_result'] for s in sequences),
        'proposal_structure_sha256':result.get('proposal_structure_sha256'),'proposal_sha256':result.get('proposal_sha256')}
    save(output/'sequence_manifest.json',manifest)
    save(output/'collection_summary.json',summarize_sequences(sequences))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify')
    merge=sub.add_parser('merge');merge.add_argument('--sources',nargs='+',required=True);merge.add_argument('--output',required=True)
    for command in ('start','evaluate','collect'):
        p=sub.add_parser(command);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20000 if command=='evaluate' else 1000)
        p.add_argument('--episodes',type=int,default=128 if command=='evaluate' else 512);p.add_argument('--config',default=str(ROOT/'configs/highd_v46_d2rl_experiment.json'))
        if command in ('start','collect'):
            p.add_argument('--mode',choices=['single','dual','natural'] if command=='collect' else ['single','dual'],default='single')
            p.add_argument('--epsilon',type=float)
        if command=='start':
            p.add_argument('--iterations',type=int,default=100)
            p.add_argument('--sequence-manifest');p.add_argument('--training-seed',type=int,default=7)
            p.add_argument('--training-config',default=str(ROOT/'configs/highd_v46_d2rl_ppo.json'))
            p.add_argument('--validation-manifest');p.add_argument('--validation-episodes',type=int,default=0)
            p.add_argument('--validation-seed',type=int,default=50000)
        elif command=='evaluate':p.add_argument('--policy',required=True)
    args=parser.parse_args()
    if args.command=='merge':print(merge_collections(args.sources,args.output));return
    assets,lock=ensure_assets()
    if args.command=='verify':print({'verified':True,'natural_target_sha256':lock['natural_target_sha256'],'assets':str(assets)});return
    from .highd_v46_d2rl import collect,identity,experiment_target,proposal_structure
    config=read(args.config);expected=experiment_target(lock,config)
    expected_hash=identity(expected)
    if args.command=='collect':
        print(collect(args.config,args.output,args.mode,args.episodes,args.seed,assets,epsilon=args.epsilon));return
    if args.command=='evaluate':
        from .highd_v46_policy import NumpyPolicy
        policy=NumpyPolicy(args.policy)
        if policy.metadata['experiment_target_sha256']!=expected_hash:raise ValueError('Policy belongs to another CAV/initial/event/runtime target')
        print(collect(args.config,args.output,policy.metadata['intervention_mode'],args.episodes,args.seed,assets,policy));return
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if read(args.training_config).get('require_validation') and not (args.validation_manifest or args.validation_episodes>0):
        raise ValueError('Specify --validation-manifest or --validation-episodes for independent selection')
    if args.validation_episodes>0 and args.sequence_manifest is None:
        if set(range(args.seed,args.seed+args.episodes)) & set(range(args.validation_seed,args.validation_seed+args.validation_episodes)):
            raise ValueError('Training/validation seed ranges overlap')
    manifest=args.sequence_manifest
    if manifest is None:
        collect(args.config,out/'collection',args.mode,args.episodes,args.seed,assets,epsilon=args.epsilon)
        manifest=str(out/'collection/sequence_manifest.json')
    m=read(manifest)
    if m['natural_target_sha256']!=lock['natural_target_sha256'] or m['intervention_mode']!=args.mode:
        raise ValueError('Training data belongs to another mode or frozen NDE')
    if m['experiment_target_sha256']!=expected_hash:
        raise ValueError('Training data belongs to another CAV/initial/event/runtime target')
    if m.get('proposal_structure_sha256')!=identity(proposal_structure(config,args.mode)):
        raise ValueError('Training data has another candidate/risk-proxy/observation structure')
    validation=args.validation_manifest
    if validation is None and args.validation_episodes>0:
        collect(args.config,out/'validation',args.mode,args.validation_episodes,args.validation_seed,assets,epsilon=args.epsilon)
        validation=str(out/'validation/sequence_manifest.json')
    from .highd_v46_train import main as train
    sys.argv=[sys.argv[0],'--sequence-manifest',manifest,'--output',str(out/'training'),'--iterations',str(args.iterations),
              '--seed',str(args.training_seed),'--config',args.training_config]
    if validation:sys.argv+=['--validation-manifest',validation]
    train()


if __name__=='__main__':main()
