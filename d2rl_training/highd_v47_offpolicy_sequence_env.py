"""Explicit Q_behavior != Q_target replay, retaining the original PPO objective.

Only initialization/validation changes. Reset, episode correction, target LR,
reward, policy observation reconstruction and PPO optimizer are inherited.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from gym import spaces

from .highd_v46_sequence_env import V46SequenceEnv,log_weight
from scenario_reconstruction.highd_v46_contract import (
    FIXED_ALPHA_OBSERVATION,LOG_WEIGHT_INDEX,LOG_WEIGHT_SCALE,replay_observation,
)
from scenario_reconstruction.highd_v47_fixed_alpha_contract import CONTRACT as STATE_CONTRACT,validate_dataset

CONTRACT='highd_v47_offpolicy_fixed_alpha_d2rl_sequence_v2'
BEHAVIOR='uniform_over_naturally_supported_selected_action_cells'


class OffPolicySequenceEnv(V46SequenceEnv):
    def __init__(self,config):
        path=Path(config['sequence_manifest']);manifest=json.loads(path.read_text(encoding='utf-8'))
        self.source=path.parent/manifest['data_path'];raw=self.source.read_bytes()
        self.source_sha256=hashlib.sha256(raw).hexdigest()
        if self.source_sha256!=manifest['sha256']:raise ValueError('Dataset checksum mismatch')
        self.dataset=json.loads(raw);d=self.dataset
        if (d.get('contract')!=CONTRACT or d.get('behavior_H')!=BEHAVIOR
                or d.get('observation_contract')!=FIXED_ALPHA_OBSERVATION
                or d.get('target_H')!='unchanged_original_formal_Single_or_Dual_H'):
            raise ValueError('Explicit off-policy fixed-alpha role required')
        self.observation_contract=FIXED_ALPHA_OBSERVATION;self.dimension=d['action_dimension'];self.sequences=d['sequences']
        if not self.sequences or self.dimension not in (1,2) or self.dimension!=(1 if d['intervention_mode']=='single' else 2):
            raise ValueError('Empty collection or mode/action mismatch')
        for key in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256',
                    'intervention_mode','proposal_structure_sha256','behavior_sha256'):
            if not d.get(key) or manifest.get(key)!=d[key]:raise ValueError('Manifest identity mismatch: '+key)
        # Reuse the STATE invariant checker only; action roles are validated below.
        validate_dataset(dict(d,contract=STATE_CONTRACT),manifest)
        if len({s['seed'] for s in self.sequences})!=len(self.sequences):raise ValueError('Duplicate collection seed')
        for seq in self.sequences:
            total=0.
            if seq['event_result']!=any(seq['cav_id'] in pair for pair in seq['collision_pairs']):
                raise ValueError('Label is not an actual CAV first collision')
            for step in seq['steps']:
                p,h,g,q,e=[np.asarray(step[k],float) for k in
                    ('p_factors','h_factors','generation_h_factors','q_factors','generation_epsilon')]
                if any(x.shape!=(self.dimension,) or not np.isfinite(x).all() for x in (p,h,g,q,e)):
                    raise ValueError('Invalid logged factors')
                if (np.any(p<=0) or np.any(q<=0) or np.any(e!=.1)
                        or any(np.any(x<0) or np.any(x>1+1e-12) for x in (p,h,g,q))):
                    raise ValueError('Invalid probability support')
                if not np.allclose(q,e*p+(1-e)*g,rtol=1e-10,atol=0):raise ValueError('Behavior Q mismatch')
                mask=np.asarray(step['factor_mask']);indices=step['action_indices']
                if mask.shape!=(self.dimension,) or mask[0]!=1 or not np.isin(mask,[0,1]).all():raise ValueError('Invalid mask')
                if len(step['actor_ids'])!=int(mask.sum()) or len(indices)!=int(mask.sum()):raise ValueError('Actor/action/mask mismatch')
                if not np.array_equal(np.asarray(step['observation'])[18:18+self.dimension],mask):raise ValueError('Observation mask differs')
                if any(np.any(x[mask==0]!=1.) for x in (p,h,g,q)):raise ValueError('Inactive factor must have unit LR')
                for i,(target,behavior) in enumerate(zip(step['factor_pmfs'],step['generation_factor_pmfs'])):
                    a=indices[i] if mask[i] else 0
                    pp,hh,gg=[np.asarray(x,float) for x in (target['p'],target['h'],behavior['h'])]
                    if (pp.ndim!=1 or pp.shape!=hh.shape or pp.shape!=gg.shape
                            or not all(np.isfinite(x).all() and np.all(x>=0) and np.isclose(x.sum(),1.) for x in (pp,hh,gg))
                            or not np.array_equal(pp,np.asarray(behavior['p']))
                            or np.any(hh[pp==0]!=0) or np.any(gg[pp==0]!=0)):
                        raise ValueError('Invalid supported target/behavior PMFs')
                    if a<0 or a>=len(pp) or not np.allclose([pp[a],hh[a],gg[a]],[p[i],h[i],g[i]],rtol=1e-10,atol=0):
                        raise ValueError('Draw factors differ from target/behavior PMFs')
                if len(step['factor_pmfs'])!=self.dimension or len(step['generation_factor_pmfs'])!=self.dimension:
                    raise ValueError('Missing factor distributions')
                replay_observation(step,total,self.observation_contract)
                if not np.isclose(step['observation'][LOG_WEIGHT_INDEX],np.clip(total/LOG_WEIGHT_SCALE,-5,5),atol=1e-7):
                    raise ValueError('Noncausal behavior observation prefix')
                increment=float(np.log(p/q).sum())
                if not np.isclose(step['log_weight'],increment,atol=1e-10):raise ValueError('Behavior factor LR differs')
                if not np.isclose(step['full_log_p']-step['full_log_q'],increment,atol=1e-9):raise ValueError('Full behavior LR differs')
                total+=increment
            full=sum(t['log_p']-t['log_q'] for t in seq['native_decisions'])
            if not np.isclose(total,seq['generation_log_weight'],atol=1e-9) or not np.isclose(full,total,atol=1e-9):
                raise ValueError('Lost full behavior likelihood factors')
        self.eligible=[i for i,s in enumerate(self.sequences) if s['steps']]
        self.positive=[i for i in self.eligible if self.sequences[i]['event_result']]
        self.negative=[i for i in self.eligible if not self.sequences[i]['event_result']]
        if not config.get('evaluation_only') and not self.positive:raise ValueError('No usable CAV training signal; run recovery readiness, not PPO')
        # With distinct behavior and target laws, 2*behavior_log_weight is WRONG.
        self.log_scale=max((s['generation_log_weight']+sum(log_weight(t,[.1]*self.dimension) for t in s['steps'])
                            for s in self.sequences if s['event_result']),default=0.)
        if 'log_reward_scale' in config:
            if not np.isfinite(config['log_reward_scale']):raise ValueError('Nonfinite common reward scale')
            self.log_scale=float(config['log_reward_scale'])
        self.sampling_correction=len(self.eligible)/len(self.sequences);self.positive_fraction=config.get('replay_positive_fraction')
        if self.positive_fraction is not None and not 0<self.positive_fraction<1:raise ValueError('Invalid positive replay fraction')
        self.action_space=spaces.Box(low=.05,high=1.,shape=(self.dimension,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-5.,high=5.,shape=(22,),dtype=np.float32)
        self.rng=np.random.default_rng(config.get('seed',7));self.done=True
