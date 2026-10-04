"""Causal observation contracts and independent collection split checks."""
import numpy as np

LEGACY_OBSERVATION = 'native_cav6_bv4x2_v1'
CONTEXT_OBSERVATION = 'native_cav6_bv4x2_context_v2'
CONDITIONAL_OBSERVATION = 'native_conditional_geometry22_v1'
CONTEXT_CONTRACTS = (CONTEXT_OBSERVATION, CONDITIONAL_OBSERVATION)
OBSERVATION_DIMENSIONS = {LEGACY_OBSERVATION: 14, CONTEXT_OBSERVATION: 22, CONDITIONAL_OBSERVATION: 22}
LOG_WEIGHT_INDEX = 16
LOG_WEIGHT_SCALE = 30.


def enrich_observation(base, mass, contrast, log_weight, time, horizon,
                       factor_mask, pair_gap=0., pair_relative_speed=0.):
    """Append pre-draw risk, prefix weight, time and active-factor information."""
    context = [np.log10(max(mass, 1e-30))/16., contrast,
               log_weight/LOG_WEIGHT_SCALE, max(0., 1-time/horizon),
               *factor_mask, pair_gap/115., pair_relative_speed/20.]
    return np.clip(np.r_[base, context], -5., 5.).tolist()


def replay_observation(step, prefix_log_weight, contract=LEGACY_OBSERVATION):
    obs = np.asarray(step['observation'], dtype=np.float32).copy()
    if obs.shape != (OBSERVATION_DIMENSIONS[contract],) or not np.isfinite(obs).all():
        raise ValueError('Invalid observation for '+contract)
    if contract in CONTEXT_CONTRACTS:
        # The replay policy changes its own likelihood prefix. Never supply the
        # behavior-policy prefix or the future terminal weight as this feature.
        obs[LOG_WEIGHT_INDEX] = np.clip(prefix_log_weight/LOG_WEIGHT_SCALE, -5., 5.)
    return obs


def independent_split(training, validation):
    for key in ('contract', 'observation_contract', 'action_dimension',
                'intervention_mode', 'natural_target_sha256', 'experiment_target_sha256'):
        if training[key] != validation[key]:
            raise ValueError('Validation contract differs: '+key)
    if training.get('proposal_structure_sha256')!=validation.get('proposal_structure_sha256'):
        raise ValueError('Validation candidate/risk-proxy structure differs')
    for dataset in (training, validation):
        seeds = [s['seed'] for s in dataset['sequences']]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate seed inside a collection split')
    overlap = {s['seed'] for s in training['sequences']} & {s['seed'] for s in validation['sequences']}
    if overlap:
        raise ValueError('Training/validation seeds overlap: '+str(sorted(overlap)[:8]))
    if training.get('sampling_scope') == 'conditional_state_bank':
        if validation.get('sampling_scope') != 'conditional_state_bank':
            raise ValueError('Conditional and population collections cannot be mixed')
        overlap = {s['source_seed'] for s in training['sequences']} & {s['source_seed'] for s in validation['sequences']}
        if overlap:
            raise ValueError('Training/validation source episodes overlap')
