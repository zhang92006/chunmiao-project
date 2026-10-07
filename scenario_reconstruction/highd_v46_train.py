"""PPO single-BV / following-pair D2RL on frozen native V46 critical sequences."""
import argparse
from pathlib import Path
import json
import os
import sys
from itertools import islice


def export_policy(trainer,output,dataset,training_seeds,validation_seeds,validation_dataset=None):
    from .highd_v46_assets import save,digest
    import numpy as np
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'policy_weights.npz').exists() or (output/'policy.json').exists():
        raise FileExistsError('Policy export already exists: '+str(output))
    model=trainer.get_policy().model;arrays={}
    layers=[m for m in model._hidden_layers.modules() if isinstance(m,torch.nn.Linear)]
    for i,layer in enumerate(layers):arrays[f'w{i}']=layer.weight.detach().cpu().numpy();arrays[f'b{i}']=layer.bias.detach().cpu().numpy()
    head=[m for m in model._logits.modules() if isinstance(m,torch.nn.Linear)][0]
    arrays['head_w']=head.weight.detach().cpu().numpy();arrays['head_b']=head.bias.detach().cpu().numpy()
    np.savez_compressed(output/'policy_weights.npz',**arrays)
    save(output/'policy.json',{'observation_contract':dataset['observation_contract'],
        'action_dimension':dataset['action_dimension'],'hidden_layers':len(layers),
        'weights_sha256':digest(output/'policy_weights.npz'),'natural_target_sha256':dataset['natural_target_sha256'],
        'experiment_target_sha256':dataset['experiment_target_sha256'],'intervention_mode':dataset['intervention_mode'],
        'proposal_structure_sha256':dataset.get('proposal_structure_sha256'),
        'training_seeds':training_seeds,'validation_seeds':validation_seeds,
        'sampling_scope':dataset.get('sampling_scope','natural_initialization'),
        'conditional_target_sha256':dataset.get('conditional_target_sha256'),
        'training_source_seeds':sorted({s['source_seed'] for s in dataset['sequences'] if 'source_seed' in s}),
        'validation_source_seeds':sorted({s['source_seed'] for s in (validation_dataset or {}).get('sequences',[]) if 'source_seed' in s}),
        **{k:dataset[k] for k in ('state_sampling','state_likelihood_ratio','state_law_sha256') if k in dataset}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--validation-manifest')
    p.add_argument('--config',default='configs/highd_v46_d2rl_ppo.json');p.add_argument('--iterations',type=int);p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,9):raise RuntimeError('Use Python 3.9 / Ray 1.11 training environment')
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
    from .highd_v46_assets import read,save,digest
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=read(a.config);iterations=cfg['iterations'] if a.iterations is None else a.iterations
    fixed_iteration=cfg.get('checkpoint_selection')=='fixed_iteration'
    if fixed_iteration and (a.validation_manifest or cfg.get('require_validation')):
        raise ValueError('Fixed-iteration training cannot select on validation')
    if iterations<1 or cfg.get('validation_interval',10)<1:raise ValueError('Iterations and validation interval must be positive')
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    env_config={'sequence_manifest':str(Path(a.sequence_manifest).resolve()),'seed':a.seed,
                'replay_positive_fraction':cfg.get('replay_positive_fraction')}
    check=V46SequenceEnv(env_config)
    validation=None
    from .highd_v46_contract import independent_split,CONTEXT_CONTRACTS
    if a.validation_manifest:
        validation=V46SequenceEnv({'sequence_manifest':a.validation_manifest,'evaluation_only':True})
        independent_split(check.dataset,validation.dataset)
        count=sum(s['event_result'] for s in validation.sequences)
        if count<cfg.get('minimum_validation_positive_episodes',1):
            raise ValueError('Independent validation has insufficient CAV events: '+str(count)+'; retain all episodes and collect a predeclared additional seed range')
        validation.log_scale=check.log_scale
    elif cfg.get('require_validation',False):raise ValueError('This training configuration requires an independent validation manifest')
    import numpy as np
    import torch
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
    from .highd_v46_action_dist import V46MaskedBeta
    class NativeHead(FullyConnectedNetwork):
        def __init__(self,obs_space,action_space,num_outputs,model_config,name,**kw):
            super().__init__(obs_space,action_space,num_outputs,model_config,name)
            n=int(np.prod(action_space.shape))
            if num_outputs!=2*n:raise ValueError('Wrong epsilon head dimension')
            mean=(cfg['initial_epsilon']-.05)/.95
            s=(mean-.0001)/.9998;excess=cfg['initial_precision']-2/min(mean,1-mean)
            if excess<=0:raise ValueError('Invalid initial precision')
            bias=np.r_[np.full(n,np.log(s)-np.log1p(-s)),np.full(n,excess+np.log(-np.expm1(-excess)))]
            head=[m for m in self._logits.modules() if isinstance(m,torch.nn.Linear)][0]
            with torch.no_grad():head.weight.zero_();head.bias.copy_(torch.as_tensor(bias,dtype=head.bias.dtype))
        def forward(self,input_dict,state,seq_lens):
            logits,state=super().forward(input_dict,state,seq_lens)
            n=int(np.prod(self.action_space.shape))
            if check.observation_contract in CONTEXT_CONTRACTS:
                self.active_factor_mask=input_dict['obs_flat'].float().reshape(-1,22)[:,18:18+n]
            else:self.active_factor_mask=torch.ones_like(logits[:,:n])
            return logits,state
    register_env('highd_v46_sequence',lambda c:V46SequenceEnv(c))
    ModelCatalog.register_custom_model('highd_v46_mean_precision',NativeHead)
    ModelCatalog.register_custom_action_dist('highd_v46_beta',V46MaskedBeta)
    config={'env':'highd_v46_sequence','env_config':env_config,'framework':'torch','num_workers':0,'num_gpus':0,
        'seed':a.seed,'gamma':1.,'lambda':1.,'batch_mode':'complete_episodes','rollout_fragment_length':128,
        'normalize_actions':True,'model':{'custom_model':'highd_v46_mean_precision','custom_action_dist':'highd_v46_beta',
        'fcnet_hiddens':cfg.get('fcnet_hiddens',[64,64]),'fcnet_activation':'tanh','vf_share_layers':False},**cfg['ppo']}
    ray.init(num_cpus=1,num_gpus=0,include_dashboard=False,ignore_reinit_error=True)
    trainer=None
    try:
        trainer=PPOTrainer(config=config)
        deterministic=lambda obs:trainer.compute_single_action(np.asarray(obs,np.float32),explore=False)
        baseline=check.empirical_objective(lambda obs:[cfg['initial_epsilon']]*check.dimension)
        initial=check.empirical_objective(deterministic)
        if not np.isclose(initial,baseline,rtol=1e-4,atol=1e-10):raise RuntimeError('Policy initialization differs from fixed epsilon baseline')
        provenance={'natural_target_sha256':check.dataset['natural_target_sha256'],'experiment_target_sha256':check.dataset['experiment_target_sha256'],
            'sequence_sha256':check.source_sha256,'action_dimension':check.dimension,
            'intervention_mode':check.dataset['intervention_mode'],'iterations':iterations,'seed':a.seed,'config':cfg,
            'trainer_sha256':digest(__file__),'environment_sha256':digest(Path(__file__).parents[1]/'d2rl_training/highd_v46_sequence_env.py'),
            'action_distribution_sha256':digest(Path(__file__).with_name('highd_v46_action_dist.py')),
            'inactive_factor_gradient':'Masked log-probability, entropy and KL for padded inactive factors',
            'objective':'E_Qb[I_first_CAV_collision * P/Qb * P/Q_epsilon], full critical sequences, fixed positive scaling, no reward clipping',
            'replay_sampling':'Stratified positive/nonpositive replay with exact 1/(N*q_episode) correction; same full-collection objective',
            'episodes':len(check.sequences),'positive_episodes':sum(s['event_result'] for s in check.sequences),'log_reward_scale':check.log_scale,
            'validation_sequence_sha256':validation.source_sha256 if validation else None,
            'validation_episodes':len(validation.sequences) if validation else 0,
            'validation_positive_episodes':sum(s['event_result'] for s in validation.sequences) if validation else 0,
            'observation_contract':check.dataset['observation_contract'],
            'sampling_scope':check.dataset.get('sampling_scope','natural_initialization'),
            **{k:check.dataset[k] for k in ('state_sampling','state_likelihood_ratio','state_law_sha256') if k in check.dataset},
            'conditional_target_sha256':check.dataset.get('conditional_target_sha256'),
            'training_source_episodes':len({s['source_seed'] for s in check.sequences if 'source_seed' in s})}
        save(output/'training_protocol.json',provenance);records=[]
        validation_baseline=validation.empirical_objective(lambda obs:[cfg['initial_epsilon']]*check.dimension) if validation else None
        best=validation.empirical_objective(deterministic) if validation else None
        selected_iteration=0 if validation else iterations
        selected_checkpoint=trainer.save(str(output/'checkpoints')) if validation else None
        print(json.dumps({'initial_scaled_second_moment':initial,'fixed_epsilon_scaled_second_moment':baseline}),flush=True)
        for i in range(1,iterations+1):
            result=trainer.train();stats=result.get('info',{}).get('learner',{}).get('default_policy',{}).get('learner_stats',{})
            row={'iteration':i,'timesteps_total':result['timesteps_total'],'reward_mean':result['episode_reward_mean'],
                'learner':{k:float(stats[k]) for k in ('policy_loss','vf_loss','entropy','kl') if k in stats}}
            if not np.isfinite([row['reward_mean'],*row['learner'].values()]).all():raise FloatingPointError('Nonfinite training metrics')
            if i%cfg.get('validation_interval',10)==0 or i==iterations:
                row['empirical_scaled_second_moment']=check.empirical_objective(deterministic)
                checkpoint=trainer.save(str(output/'checkpoints'))
                if validation:
                    values=validation.objective_values(deterministic);score=float(values.mean())
                    row['validation_scaled_second_moment']=score
                    row['validation_objective_standard_error']=validation.objective_standard_error(values)
                    if score<best:
                        best=score;selected_iteration=i;selected_checkpoint=checkpoint
            save(output/f'iteration_{i:04d}.json',row);records.append(row);print(json.dumps(row),flush=True)
        training_seeds=[s['seed'] for s in check.sequences]
        validation_seeds=[s['seed'] for s in validation.sequences] if validation else []
        if validation:
            export_policy(trainer,output/'last_iteration',check.dataset,training_seeds,validation_seeds,validation.dataset)
            trainer.restore(selected_checkpoint)
        else:selected_checkpoint=checkpoint
        export_policy(trainer,output,check.dataset,training_seeds,validation_seeds,validation.dataset if validation else None)
        from .highd_v46_policy import NumpyPolicy
        portable=NumpyPolicy(output)
        observations=islice((s['observation'] for seq in check.sequences for s in seq['steps']),64)
        error=max(float(np.max(np.abs(np.asarray(portable(obs))-deterministic(obs)))) for obs in observations)
        if not np.isfinite(error) or error>1e-6:raise RuntimeError('Portable policy differs from training policy')
        final=check.empirical_objective(deterministic)
        save(output/'training_summary.json',{**provenance,'completed_iterations':len(records),'timesteps_total':records[-1]['timesteps_total'],
            'fixed_epsilon_scaled_second_moment':baseline,'trained_empirical_scaled_second_moment':final,'resubstitution_ratio':final/baseline,
            'portable_policy_max_error':error,'checkpoint':str(Path(selected_checkpoint).relative_to(output)),
            'selected_iteration':selected_iteration,'selection':'independent_validation_second_moment' if validation else 'last_iteration_no_validation',
            'validation_fixed_scaled_second_moment':validation_baseline,'validation_selected_scaled_second_moment':best,
            'validation_ratio':best/validation_baseline if validation else None,
            'scope':('Validation selects a checkpoint including the initial fixed policy. Independent online testing is still required.' if validation
                     else 'Fixed predeclared iteration PPO training completed; independent native evaluation remains separate.' if fixed_iteration
                     else 'Unvalidated training/interface check only; no generalization or acceleration claim.')})
    finally:
        if trainer is not None:trainer.stop()
        ray.shutdown()


if __name__=='__main__':main()
