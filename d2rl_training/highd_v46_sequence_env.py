"""Full critical-sequence D2RL objective for one or two native BV factors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from gym import Env, spaces
from scenario_reconstruction.highd_v46_contract import (
    LEGACY_OBSERVATION, CONTEXT_OBSERVATION, OBSERVATION_DIMENSIONS, replay_observation,
    LOG_WEIGHT_INDEX, LOG_WEIGHT_SCALE,
    CONDITIONAL_OBSERVATION, FIXED_ALPHA_OBSERVATION, CONTEXT_CONTRACTS,
)


def log_weight(step, epsilon):
    p=np.asarray(step['p_factors'],float);h=np.asarray(step['h_factors'],float);e=np.asarray(epsilon,float)
    if p.shape!=e.shape or h.shape!=p.shape or not all(np.isfinite(x).all() for x in (p,h,e)) or np.any(e<=0) or np.any(e>1):raise ValueError('Epsilon dimension/support mismatch')
    if np.any(p<=0) or np.any(p>1+1e-12) or np.any(h<0) or np.any(h>1+1e-12):raise ValueError('Invalid likelihood factors')
    return float(np.log(p/(e*p+(1-e)*h)).sum())


class V46SequenceEnv(Env):
    def __init__(self,config):
        manifest_path=Path(config['sequence_manifest']);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        source=manifest_path.parent/manifest['data_path'];raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:raise ValueError('Dataset checksum mismatch')
        self.source=source;self.source_sha256=manifest['sha256']
        self.dataset=json.loads(raw)
        conditional = self.dataset['contract'] == 'highd_v46_conditional_d2rl_sequence_v1'
        fixed_alpha = self.dataset['contract'] == 'highd_v47_fixed_alpha_d2rl_sequence_v1'
        if self.dataset['contract'] not in ('highd_v46_native_d2rl_sequence_v1','highd_v46_native_d2rl_sequence_v2','highd_v46_conditional_d2rl_sequence_v1','highd_v47_fixed_alpha_d2rl_sequence_v1'):raise ValueError('Wrong native event/observation contract')
        self.observation_contract=self.dataset['observation_contract']
        if self.observation_contract not in OBSERVATION_DIMENSIONS:raise ValueError('Wrong observation contract')
        expected=('highd_v47_fixed_alpha_d2rl_sequence_v1' if self.observation_contract==FIXED_ALPHA_OBSERVATION else
                  'highd_v46_conditional_d2rl_sequence_v1' if self.observation_contract==CONDITIONAL_OBSERVATION else
                  'highd_v46_native_d2rl_sequence_v2' if self.observation_contract==CONTEXT_OBSERVATION else 'highd_v46_native_d2rl_sequence_v1')
        if self.dataset['contract']!=expected:raise ValueError('Sequence and observation versions differ')
        self.sequences=self.dataset['sequences'];self.dimension=self.dataset['action_dimension']
        if not self.sequences:raise ValueError('Empty collection')
        if len({s['seed'] for s in self.sequences})!=len(self.sequences):raise ValueError('Duplicate collection seed')
        if fixed_alpha:
            from scenario_reconstruction.highd_v47_fixed_alpha_contract import validate_dataset
            validate_dataset(self.dataset,manifest)
        if conditional:
            if self.dataset.get('sampling_scope') != 'conditional_state_bank':raise ValueError('Missing conditional scope')
            if manifest.get('conditional_target_sha256')!=self.dataset.get('conditional_target_sha256') or not self.dataset.get('conditional_target_sha256'):raise ValueError('Missing or mismatched conditional target')
            masses=np.asarray([s['target_mass'] for s in self.sequences],float)
            if not np.isfinite(masses).all() or np.any(masses<=0) or not np.isclose(masses.sum(),1.):raise ValueError('Invalid conditional target masses')
            sources={s['source_seed'] for s in self.sequences}
            owners={}
            for s in self.sequences:
                if len(s['steps'])!=1:raise ValueError('Conditional v1 requires exactly one intervention')
                if owners.setdefault(s['case_id'],s['source_seed'])!=s['source_seed']:raise ValueError('Case ID crosses source episodes')
            for source_seed in sources:
                group=[s for s in self.sequences if s['source_seed']==source_seed]
                cases={s['case_id'] for s in group}
                for case in cases:
                    strata=[s for s in group if s['case_id']==case]
                    expected_mass=1/(len(sources)*len(cases)*len(strata))
                    if not all(np.isclose(s['target_mass'],expected_mass,rtol=1e-10,atol=0) for s in strata):raise ValueError('Conditional source/state law differs')
        elif any('target_mass' in s for s in self.sequences):raise ValueError('Conditional masses require conditional contract')
        if self.dimension not in (1,2):raise ValueError('Expected single-BV or following-pair intervention')
        if self.dimension!=(2 if self.dataset['intervention_mode']=='dual' else 1):raise ValueError('Mode and policy dimension differ')
        for key in ('natural_target_sha256','experiment_target_sha256','intervention_mode'):
            if manifest[key]!=self.dataset[key]:raise ValueError('Manifest identity mismatch: '+key)
        self.eligible=[i for i,s in enumerate(self.sequences) if s['steps']]
        if not config.get('evaluation_only') and (not self.eligible or not any(s['event_result'] for s in self.sequences)):
            raise ValueError('Need critical sequences and actual CAV first collisions; collect more episodes')
        for seq in self.sequences:
            total=0.
            for step in seq['steps']:
                p,h,q=[np.asarray(step[k],float) for k in ('p_factors','h_factors','q_factors')]
                eps=np.asarray(step['generation_epsilon'],float)
                if any(x.shape!=(self.dimension,) or not np.isfinite(x).all() for x in (p,h,q,eps)):raise ValueError('Invalid logged factors')
                if np.any(eps<.05) or np.any(eps>1):raise ValueError('Generation epsilon outside support bounds')
                if np.any(p<=0) or np.any(q<=0) or np.any(h<0) or np.any(p>1+1e-12) or np.any(h>1+1e-12):raise ValueError('Invalid probability support')
                if not np.allclose(q,eps*p+(1-eps)*h,rtol=1e-10,atol=0):raise ValueError('Generation Q mismatch')
                replay_observation(step,total,self.observation_contract)
                if self.observation_contract in CONTEXT_CONTRACTS:
                    mask=np.asarray(step['factor_mask'])
                    if mask.shape!=(self.dimension,) or not np.isin(mask,[0,1]).all() or mask[0]!=1:raise ValueError('Invalid active factor mask')
                    if len(step['actor_ids'])!=int(mask.sum()):raise ValueError('Active actors and factor mask differ')
                    inactive=mask==0
                    if any(np.any(x[inactive]!=1.) for x in (p,h,q)):raise ValueError('Inactive factor must have unit likelihood')
                    if not np.array_equal(np.asarray(step['observation'])[18:18+self.dimension],mask):raise ValueError('Observation and likelihood masks differ')
                    if not np.isclose(step['observation'][LOG_WEIGHT_INDEX],np.clip(total/LOG_WEIGHT_SCALE,-5.,5.),atol=1e-7):
                        raise ValueError('Observation contains a wrong or noncausal behavior weight prefix')
                total+=log_weight(step,eps)
            if not np.isclose(total,seq['generation_log_weight'],rtol=1e-10,atol=1e-9):raise ValueError('Densification lost likelihood factors')
        self.log_scale=max((2*s['generation_log_weight'] for s in self.sequences if s['event_result']),default=0.)
        self.sampling_correction=len(self.eligible)/len(self.sequences)
        self.positive=[i for i in self.eligible if self.sequences[i]['event_result']]
        self.negative=[i for i in self.eligible if not self.sequences[i]['event_result']]
        self.positive_fraction=config.get('replay_positive_fraction')
        if self.positive_fraction is not None and not 0<self.positive_fraction<1:raise ValueError('Replay positive fraction must be inside (0,1)')
        self.action_space=spaces.Box(low=.05,high=1.,shape=(self.dimension,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-5.,high=5.,shape=(OBSERVATION_DIMENSIONS[self.observation_contract],),dtype=np.float32)
        self.rng=np.random.default_rng(config.get('seed',7));self.done=True

    def reset(self):
        if not self.eligible:raise ValueError('No critical sequences to replay')
        group=self.eligible;probability=1.
        if self.positive_fraction is not None and self.positive and self.negative:
            positive=self.rng.random()<self.positive_fraction
            group=self.positive if positive else self.negative
            probability=self.positive_fraction if positive else 1-self.positive_fraction
        self.episode_correction=len(group)/(len(self.sequences)*probability)
        self.sequence=self.sequences[int(self.rng.choice(group))]
        self.position=0;self.log_weight=0.;self.done=False
        return self.current_observation(self.sequence['steps'][0],0.)

    def current_observation(self,step,prefix):
        return replay_observation(step,prefix,getattr(self,'observation_contract',LEGACY_OBSERVATION))

    def loss(self,seq,weight):
        if not seq['event_result']:return 0.
        exponent=seq['generation_log_weight']+weight-self.log_scale
        if exponent>700:raise FloatingPointError('Second-moment overflow; no reward clipping')
        return float(np.exp(exponent))*seq.get('target_mass',1/len(self.sequences))*len(self.sequences)

    def step(self,action):
        if self.done:raise RuntimeError('Reset a completed episode')
        if np.any(np.asarray(action)<.05-1e-7):raise ValueError('Epsilon below support floor')
        self.log_weight+=log_weight(self.sequence['steps'][self.position],action)
        self.position+=1;self.done=self.position==len(self.sequence['steps'])
        reward=-self.episode_correction*self.loss(self.sequence,self.log_weight) if self.done else 0.
        step=self.sequence['steps'][min(self.position,len(self.sequence['steps'])-1)]
        return self.current_observation(step,self.log_weight),reward,self.done,{}

    def objective_values(self,policy):
        values=[]
        for sequence in self.sequences:
            prefix=0.
            for step in sequence['steps']:
                obs=self.current_observation(step,prefix)
                prefix+=log_weight(step,policy(obs))
            values.append(self.loss(sequence,prefix))
        return np.asarray(values)

    def empirical_objective(self,policy):
        return float(np.mean(self.objective_values(policy)))

    def objective_standard_error(self,values):
        """Fixed-state stratified SE for conditional banks, iid SE otherwise."""
        values=np.asarray(values,float)
        if self.dataset.get('sampling_scope')!='conditional_state_bank':
            return float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        groups={s['case_id'] for s in self.sequences};variance=0.
        for case in groups:
            x=values[[s['case_id']==case for s in self.sequences]]
            if len(x)<2:return None
            variance+=(len(x)/len(values))**2*x.var(ddof=1)/len(x)
        return float(np.sqrt(variance))
