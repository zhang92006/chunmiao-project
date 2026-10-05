"""Full critical-sequence D2RL objective for one or two native BV factors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from gym import Env, spaces


def log_weight(step, epsilon):
    p=np.asarray(step['p_factors'],float);h=np.asarray(step['h_factors'],float);e=np.asarray(epsilon,float)
    if p.shape!=e.shape or not np.isfinite(e).all() or np.any(e<=0) or np.any(e>1):raise ValueError('Epsilon dimension/support mismatch')
    return float(np.log(p/(e*p+(1-e)*h)).sum())


class V46SequenceEnv(Env):
    def __init__(self,config):
        manifest_path=Path(config['sequence_manifest']);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        source=manifest_path.parent/manifest['data_path'];raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:raise ValueError('Dataset checksum mismatch')
        self.dataset=json.loads(raw)
        if self.dataset['contract']!='highd_v46_native_d2rl_sequence_v1':raise ValueError('Wrong native event/observation contract')
        self.sequences=self.dataset['sequences'];self.dimension=self.dataset['action_dimension']
        if self.dimension not in (1,2):raise ValueError('Expected single-BV or following-pair intervention')
        self.eligible=[i for i,s in enumerate(self.sequences) if s['steps']]
        if not self.eligible or not any(s['event_result'] for s in self.sequences):raise ValueError('Need critical sequences and actual CAV first collisions; collect more episodes')
        for seq in self.sequences:
            total=0.
            for step in seq['steps']:
                p,h,q=[np.asarray(step[k],float) for k in ('p_factors','h_factors','q_factors')]
                eps=np.asarray(step['generation_epsilon'],float)
                if any(x.shape!=(self.dimension,) or not np.isfinite(x).all() for x in (p,h,q,eps)):raise ValueError('Invalid logged factors')
                if np.any(p<=0) or np.any(q<=0) or np.any(h<0) or np.any(p>1+1e-12) or np.any(h>1+1e-12):raise ValueError('Invalid probability support')
                if not np.allclose(q,eps*p+(1-eps)*h,rtol=1e-10,atol=0):raise ValueError('Generation Q mismatch')
                total+=log_weight(step,eps)
            if not np.isclose(total,seq['generation_log_weight'],rtol=1e-10,atol=1e-9):raise ValueError('Densification lost likelihood factors')
        self.log_scale=max(2*s['generation_log_weight'] for s in self.sequences if s['event_result'])
        self.sampling_correction=len(self.eligible)/len(self.sequences)
        self.positive=[i for i in self.eligible if self.sequences[i]['event_result']]
        self.negative=[i for i in self.eligible if not self.sequences[i]['event_result']]
        self.positive_fraction=config.get('replay_positive_fraction')
        if self.positive_fraction is not None and not 0<self.positive_fraction<1:raise ValueError('Replay positive fraction must be inside (0,1)')
        self.action_space=spaces.Box(low=.05,high=1.,shape=(self.dimension,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-5.,high=5.,shape=(14,),dtype=np.float32)
        self.rng=np.random.default_rng(config.get('seed',7));self.done=True

    def reset(self):
        group=self.eligible;probability=1.
        if self.positive_fraction is not None and self.positive and self.negative:
            positive=self.rng.random()<self.positive_fraction
            group=self.positive if positive else self.negative
            probability=self.positive_fraction if positive else 1-self.positive_fraction
        self.episode_correction=len(group)/(len(self.sequences)*probability)
        self.sequence=self.sequences[int(self.rng.choice(group))]
        self.position=0;self.log_weight=0.;self.done=False
        return np.asarray(self.sequence['steps'][0]['observation'],np.float32)

    def loss(self,seq,weight):
        if not seq['event_result']:return 0.
        exponent=seq['generation_log_weight']+weight-self.log_scale
        if exponent>700:raise FloatingPointError('Second-moment overflow; no reward clipping')
        return float(np.exp(exponent))

    def step(self,action):
        if self.done:raise RuntimeError('Reset a completed episode')
        if np.any(np.asarray(action)<.05-1e-7):raise ValueError('Epsilon below support floor')
        self.log_weight+=log_weight(self.sequence['steps'][self.position],action)
        self.position+=1;self.done=self.position==len(self.sequence['steps'])
        reward=-self.episode_correction*self.loss(self.sequence,self.log_weight) if self.done else 0.
        obs=self.sequence['steps'][min(self.position,len(self.sequence['steps'])-1)]['observation']
        return np.asarray(obs,np.float32),reward,self.done,{}

    def empirical_objective(self,policy):
        return float(np.mean([self.loss(s,sum(log_weight(step,policy(step['observation'])) for step in s['steps'])) for s in self.sequences]))
