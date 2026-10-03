"""PPO single-BV / following-pair D2RL on frozen native V46 critical sequences."""
import argparse
from pathlib import Path
import json
import os
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/highd_v46_d2rl_ppo.json');p.add_argument('--iterations',type=int);p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,9):raise RuntimeError('Use Python 3.9 / Ray 1.11 training environment')
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
    from .highd_v46_assets import read,save,digest
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=read(a.config);iterations=a.iterations or cfg['iterations']
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    env_config={'sequence_manifest':str(Path(a.sequence_manifest).resolve()),'seed':a.seed,
                'replay_positive_fraction':cfg.get('replay_positive_fraction')}
    check=V46SequenceEnv(env_config)
    import numpy as np
    import torch
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
    from .highd_mean_precision_policy import HighDMeanPrecisionBeta
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
    register_env('highd_v46_sequence',lambda c:V46SequenceEnv(c))
    ModelCatalog.register_custom_model('highd_v46_mean_precision',NativeHead)
    ModelCatalog.register_custom_action_dist('highd_v46_beta',HighDMeanPrecisionBeta)
    config={'env':'highd_v46_sequence','env_config':env_config,'framework':'torch','num_workers':0,'num_gpus':0,
        'seed':a.seed,'gamma':1.,'lambda':1.,'batch_mode':'complete_episodes','rollout_fragment_length':128,
        'normalize_actions':True,'model':{'custom_model':'highd_v46_mean_precision','custom_action_dist':'highd_v46_beta',
        'fcnet_hiddens':[64,64],'fcnet_activation':'tanh','vf_share_layers':False},**cfg['ppo']}
    ray.init(num_cpus=1,num_gpus=0,include_dashboard=False,ignore_reinit_error=True)
    trainer=None
    try:
        trainer=PPOTrainer(config=config)
        deterministic=lambda obs:trainer.compute_single_action(np.asarray(obs,np.float32),explore=False)
        baseline=check.empirical_objective(lambda obs:[cfg['initial_epsilon']]*check.dimension)
        initial=check.empirical_objective(deterministic)
        if not np.isclose(initial,baseline,rtol=1e-4,atol=1e-10):raise RuntimeError('Policy initialization differs from fixed epsilon baseline')
        provenance={'natural_target_sha256':check.dataset['natural_target_sha256'],'experiment_target_sha256':check.dataset['experiment_target_sha256'],
            'sequence_sha256':digest(Path(a.sequence_manifest).parent/'sequences.json'),'action_dimension':check.dimension,
            'intervention_mode':check.dataset['intervention_mode'],'iterations':iterations,'seed':a.seed,'config':cfg,
            'trainer_sha256':digest(__file__),'environment_sha256':digest(Path(__file__).parents[1]/'d2rl_training/highd_v46_sequence_env.py'),
            'objective':'E_Qb[I_first_CAV_collision * P/Qb * P/Q_epsilon], full critical sequences, fixed positive scaling, no reward clipping',
            'replay_sampling':'Stratified positive/nonpositive replay with exact 1/(N*q_episode) correction; same full-collection objective',
            'episodes':len(check.sequences),'positive_episodes':sum(s['event_result'] for s in check.sequences),'log_reward_scale':check.log_scale}
        save(output/'training_protocol.json',provenance);records=[]
        print(json.dumps({'initial_scaled_second_moment':initial,'fixed_epsilon_scaled_second_moment':baseline}),flush=True)
        for i in range(1,iterations+1):
            result=trainer.train();stats=result.get('info',{}).get('learner',{}).get('default_policy',{}).get('learner_stats',{})
            row={'iteration':i,'timesteps_total':result['timesteps_total'],'reward_mean':result['episode_reward_mean'],
                'learner':{k:float(stats[k]) for k in ('policy_loss','vf_loss','entropy','kl') if k in stats}}
            if not np.isfinite([row['reward_mean'],*row['learner'].values()]).all():raise FloatingPointError('Nonfinite training metrics')
            if i%10==0 or i==iterations:row['empirical_scaled_second_moment']=check.empirical_objective(deterministic)
            save(output/f'iteration_{i:04d}.json',row);records.append(row);print(json.dumps(row),flush=True)
        checkpoint=trainer.save(str(output/'checkpoint'))
        model=trainer.get_policy().model;arrays={}
        layers=[m for m in model._hidden_layers.modules() if isinstance(m,torch.nn.Linear)]
        for i,layer in enumerate(layers):arrays[f'w{i}']=layer.weight.detach().cpu().numpy();arrays[f'b{i}']=layer.bias.detach().cpu().numpy()
        head=[m for m in model._logits.modules() if isinstance(m,torch.nn.Linear)][0]
        arrays['head_w']=head.weight.detach().cpu().numpy();arrays['head_b']=head.bias.detach().cpu().numpy()
        np.savez_compressed(output/'policy_weights.npz',**arrays)
        save(output/'policy.json',{'observation_contract':'native_cav6_bv4x2_v1','action_dimension':check.dimension,'hidden_layers':len(layers),
            'weights_sha256':digest(output/'policy_weights.npz'),'natural_target_sha256':check.dataset['natural_target_sha256'],
            'experiment_target_sha256':check.dataset['experiment_target_sha256'],'intervention_mode':check.dataset['intervention_mode']})
        from .highd_v46_policy import NumpyPolicy
        portable=NumpyPolicy(output);error=max(float(np.max(np.abs(np.asarray(portable(s['observation']))-deterministic(s['observation'])))) for seq in check.sequences[:8] for s in seq['steps'])
        if error>1e-6:raise RuntimeError('Portable policy differs from training policy')
        final=records[-1]['empirical_scaled_second_moment']
        save(output/'training_summary.json',{**provenance,'completed_iterations':len(records),'timesteps_total':records[-1]['timesteps_total'],
            'fixed_epsilon_scaled_second_moment':baseline,'trained_empirical_scaled_second_moment':final,'resubstitution_ratio':final/baseline,
            'portable_policy_max_error':error,'checkpoint':str(Path(checkpoint).relative_to(output)),
            'scope':'Actual PPO training on native V46 single/following intervention data. Resubstitution objective is not independent online efficiency evidence.'})
    finally:
        if trainer is not None:trainer.stop()
        ray.shutdown()


if __name__=='__main__':main()
