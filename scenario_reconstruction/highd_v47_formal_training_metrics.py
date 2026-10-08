"""Training diagnostics; these are replay estimates, not native evaluation."""
import numpy as np
from .highd_v46_contract import replay_observation
from .highd_v46_policy import NumpyPolicy
from d2rl_training.highd_v46_sequence_env import log_weight


def policy_from_model(model, dataset):
    import torch
    policy=object.__new__(NumpyPolicy)
    layers=[m for m in model._hidden_layers.modules() if isinstance(m,torch.nn.Linear)]
    head=[m for m in model._logits.modules() if isinstance(m,torch.nn.Linear)][0]
    policy.metadata=dict(observation_contract=dataset['observation_contract'],action_dimension=dataset['action_dimension'],hidden_layers=len(layers))
    policy.weights={}
    for i,layer in enumerate(layers):policy.weights['w'+str(i)]=layer.weight.detach().cpu().numpy();policy.weights['b'+str(i)]=layer.bias.detach().cpu().numpy()
    policy.weights.update(head_w=head.weight.detach().cpu().numpy(),head_b=head.bias.detach().cpu().numpy())
    return policy


def diagnostics(env,policy):
    epsilons=[[] for _ in range(env.dimension)]; TVs=[[] for _ in range(env.dimension)]; logs=[]; values=[]
    for sequence in env.sequences:
        prefix=0.
        for step in sequence['steps']:
            eps=policy(replay_observation(step,prefix,env.observation_contract))
            for i,active in enumerate(step['factor_mask']):
                if active:
                    epsilons[i].append(eps[i]);p=np.asarray(step['factor_pmfs'][i]['p']);h=np.asarray(step['factor_pmfs'][i]['h'])
                    TVs[i].append(float(.5*np.abs(eps[i]*p+(1-eps[i])*h-p).sum()))
            prefix+=log_weight(step,eps)
        logs.append(prefix);values.append(env.loss(sequence,prefix))
    return dict(empirical_scaled_second_moment=float(np.mean(values)),
        empirical_second_moment=float(np.mean(values)*np.exp(env.log_scale)),
        dataset_event_count=sum(s['event_result'] for s in env.sequences),
        dataset_event_source_count=len({s['source_seed'] for s in env.sequences if s['event_result']}),
        likelihood_log_quantiles=np.quantile(logs,[.5,.95,.99,1]).tolist(),
        epsilon_quantiles_by_active_factor=[np.quantile(x,[.05,.5,.95]).tolist() if x else None for x in epsilons],
        conditional_QP_TV_at_logged_action_rows=[float(np.mean(x)) if x else None for x in TVs],
        diagnostic_scope='Full training replay; event counts are fixed collection labels, TV uses logged conditional rows')
