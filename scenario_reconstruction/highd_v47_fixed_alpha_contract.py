"""Fixed Q_S=alpha: explicit IID state law, no learned state likelihood."""
import numpy as np

from .highd_v46_d2rl import identity

CONTRACT = 'highd_v47_fixed_alpha_d2rl_sequence_v1'
SCOPE = 'fixed_alpha_state_bank_iid'


def state_schedule(law, count, seed):
    alpha=np.asarray([s['alpha'] for s in law],float)
    if (not law or count<1 or not np.isfinite(alpha).all() or np.any(alpha<=0)
            or not np.isclose(alpha.sum(),1.) or len({s['case_id'] for s in law})!=len(law)):
        raise ValueError('Invalid fixed alpha state law')
    sources={s['source_seed'] for s in law}
    for s in law:
        n=sum(r['source_seed']==s['source_seed'] for r in law)
        if not np.isclose(s['alpha'],1/(len(sources)*n),rtol=1e-12,atol=0):
            raise ValueError('Alpha must remain uniform source then state')
    return np.random.default_rng(seed).choice(len(law),size=count,p=alpha).tolist()


def validate_dataset(data, manifest):
    if (data['contract']!=CONTRACT or data.get('sampling_scope')!=SCOPE
            or data.get('state_sampling')!='Q_S_equals_alpha' or data.get('state_likelihood_ratio')!=1.
            or data.get('state_model') is not None or any('target_mass' in s for s in data['sequences'])):
        raise ValueError('Fixed-alpha contract forbids learned state proposals or rebalanced weights')
    law=data['state_law']; law_hash=identity(law)
    for key in ('conditional_target_sha256','state_law_sha256','proposal_structure_sha256'):
        if not data.get(key) or manifest.get(key)!=data[key]:raise ValueError('Fixed-alpha manifest identity mismatch: '+key)
    if law_hash!=data['state_law_sha256']:raise ValueError('State law checksum differs')
    seq=data['sequences']; schedule=state_schedule(law,len(seq),data['state_schedule_seed'])
    for i,(s,k) in enumerate(zip(seq,schedule)):
        state=law[k]
        if (s['draw_index']!=i or s['case_id']!=state['case_id'] or s['source_seed']!=state['source_seed']
                or s['seed']!=data['continuation_seed_start']+i
                or s['state_target_mass']!=state['alpha'] or s['state_proposal_mass']!=state['alpha']
                or s['state_log_weight']!=0. or s['state_likelihood_ratio']!=1.
                or s['generation_log_weight']!=s['action_log_weight']):
            raise ValueError('Recorded state draw must have Q_S=alpha and state LR=1')
        if len(s['steps'])>data['horizon_s'] or any(step['time_s']<0 or step['time_s']>=data['horizon_s'] for step in s['steps']):
            raise ValueError('Critical sequence exceeds declared conditional horizon')
