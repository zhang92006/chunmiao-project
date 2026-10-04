"""Select constant epsilon on validation data, then freeze it for online tests."""
import argparse
from itertools import product
from pathlib import Path
import numpy as np
from .highd_v46_assets import save,digest


def select_fixed(validation_manifest,output,grid=(.0501,.1,.5,.99)):
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    validation=V46SequenceEnv({'sequence_manifest':str(validation_manifest),'evaluation_only':True})
    if not any(s['event_result'] for s in validation.sequences):
        raise ValueError('No validation CAV events; a fixed baseline cannot be selected from all-zero objectives')
    grid=tuple(float(e) for e in grid)
    if not grid or not np.isfinite(grid).all() or any(e<.05 or e>1 for e in grid):raise ValueError('Invalid fixed epsilon grid')
    results=[];best=None
    for epsilon in product(grid,repeat=validation.dimension):
        try:score=validation.empirical_objective(lambda obs:epsilon)
        except FloatingPointError:
            results.append({'epsilon':list(epsilon),'scaled_second_moment':None,'status':'numeric_overflow'})
            continue
        results.append({'epsilon':list(epsilon),'scaled_second_moment':score,'status':'finite'})
        if best is None or score<best['score']:best={'epsilon':epsilon,'score':score}
    if best is None:raise FloatingPointError('All fixed candidates overflow')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'policy_weights.npz',epsilon=np.asarray(best['epsilon']))
    data=validation.dataset
    metadata={'policy_type':'fixed_epsilon','observation_contract':data['observation_contract'],
              'action_dimension':validation.dimension,'hidden_layers':0,
              'weights_sha256':digest(output/'policy_weights.npz'),
              'natural_target_sha256':data['natural_target_sha256'],'experiment_target_sha256':data['experiment_target_sha256'],
              'proposal_structure_sha256':data.get('proposal_structure_sha256'),'intervention_mode':data['intervention_mode'],
              'training_seeds':[],'validation_seeds':[s['seed'] for s in validation.sequences],
              'sampling_scope':data.get('sampling_scope','natural_initialization'),
              'conditional_target_sha256':data.get('conditional_target_sha256'),
              'training_source_seeds':[],
              'validation_source_seeds':sorted({s['source_seed'] for s in validation.sequences if 'source_seed' in s})}
    save(output/'policy.json',metadata)
    report={'selection':'minimum_validation_second_moment','epsilon':list(best['epsilon']),
            'validation_sequence_sha256':validation.source_sha256,'validation_episodes':len(validation.sequences),
            'validation_positive_episodes':sum(s['event_result'] for s in validation.sequences),
            'grid':list(grid),'candidates':results,'scope':'Frozen validation-selected baseline; not independent online efficiency evidence'}
    save(output/'selection.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--grid',type=float,nargs='+',default=[.0501,.1,.5,.99])
    a=p.parse_args();print(select_fixed(a.validation_manifest,a.output,a.grid))


if __name__=='__main__':main()
