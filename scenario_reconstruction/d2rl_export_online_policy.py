"""Export a bounded-Beta RLlib checkpoint and verify deterministic action parity."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml_conf', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Override the YAML training seed for checkpoint provenance.',
    )
    parser.add_argument(
        '--experiment_name', default=None,
        help='Override the YAML experiment name for checkpoint provenance.',
    )
    args = parser.parse_args()
    import numpy as np
    import torch
    import yaml
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.registry import register_env
    from d2rl_training.d2rl_training_env import D2RLTrainingEnv
    from .d2rl_bounded_action_dist import register_bounded_action_distribution
    from .d2rl_checkpoint_evaluate import _checkpoint_file, discover_crash_episodes
    from .d2rl_smoke_train import build_rllib_config

    config = yaml.safe_load(Path(args.yaml_conf).read_text(encoding='utf-8'))
    if args.seed is not None:
        config['seed'] = args.seed
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if config.get('action_distribution') != 'bounded_beta' or config.get('multi_bv_num') != 2:
        raise ValueError('Only the two-BV bounded-Beta policy is supported')
    output = Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new export path to preserve existing artifacts')
    register_bounded_action_distribution()
    register_env('online_export', lambda _: D2RLTrainingEnv(config))
    rlconfig = build_rllib_config(config, 'online_export')
    rlconfig.update(num_workers=0, explore=False)
    checkpoint = _checkpoint_file(args.checkpoint).resolve()
    ray.init(num_gpus=0, include_dashboard=False)
    trainer = None
    try:
        trainer = PPOTrainer(config=rlconfig)
        trainer.restore(str(checkpoint))
        policy = trainer.get_policy()
        if policy.config.get('observation_filter', 'NoFilter') != 'NoFilter':
            raise ValueError('Observation filters must be exported explicitly')
        if policy.is_recurrent() or not policy.config['normalize_actions']:
            raise ValueError('Only nonrecurrent normalized-action policies are supported')

        class EpsilonModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, obs):
                logits, _ = self.model({'obs': obs, 'obs_flat': obs}, [], None)
                # Match Ray 1.11 TorchBeta exactly, including its concentration clamp.
                from ray.rllib.utils.numpy import SMALL_NUMBER
                logits = torch.clamp(logits, np.log(SMALL_NUMBER), -np.log(SMALL_NUMBER))
                concentration = torch.log(torch.exp(logits) + 1.0) + 1.0
                alpha, beta = torch.chunk(concentration, 2, dim=-1)
                normalized = 2.0 * (alpha / (alpha + beta)) - 1.0
                return (normalized + 1.0) * .998 / 2.0 + .001

        wrapper = EpsilonModel(policy.model.cpu().eval()).eval()
        traced = torch.jit.trace(wrapper, torch.zeros(1, 14))
        env = D2RLTrainingEnv(config)
        observations = []
        for folder in config['data_folders']:
            for path in discover_crash_episodes(Path(config['root_folder']) / folder, 'train'):
                observations.append(env.reset(str(path)))
        observations.extend(np.random.RandomState(912).uniform(-5, 5, (32, 14)).astype(np.float32))
        max_error = 0.0
        with torch.no_grad():
            for obs in observations:
                expected = trainer.compute_single_action(obs, explore=False)
                actual = traced(torch.as_tensor(obs).float().unsqueeze(0)).numpy()[0]
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        if max_error > 1e-6:
            raise ValueError(f'Export parity failed: {max_error}')
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced, str(output))
        metadata = {
            'schema_version': 1, 'observation_size': 14, 'action_size': 2,
            'model_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'checkpoint': str(checkpoint), 'training_iteration': trainer.iteration,
            'distribution': 'bounded_beta_mean', 'online_decision_mode': 'all_causal_critical_steps',
            'actor_order': 'selected_subset_in_controlled_candidate_order',
            'parity_observations': len(observations), 'max_abs_action_error': max_error,
            'training_config': config,
        }
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata, indent=2))
    finally:
        if trainer is not None:
            trainer.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
