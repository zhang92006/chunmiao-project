"""Shared fixed behavior, independent target actors, exact stratified reward."""
import hashlib
import json
from pathlib import Path
import numpy as np
from gym import spaces

from .highd_v46_sequence_env import V46SequenceEnv,log_weight
from scenario_reconstruction.highd_v46_contract import FIXED_ALPHA_OBSERVATION,replay_observation
from scenario_reconstruction.highd_v46_d2rl import identity

CONTRACT='highd_v47_shared_teacher_sequence_v1'


class SharedTeacherEnv(V46SequenceEnv):
    def __init__(self,config):
        path=Path(config['sequence_manifest']);manifest=json.loads(path.read_text())
        self.source=path.parent/manifest['data_path'];raw=self.source.read_bytes()
        self.source_sha256=hashlib.sha256(raw).hexdigest()
        if self.source_sha256!=manifest['sha256']:raise ValueError('Dataset checksum differs')
        self.dataset=json.loads(raw);del raw
        d=self.dataset;self.sequences=d['sequences'];self.dimension=d['action_dimension']
        self.observation_contract=d['observation_contract']
        if (d['contract']!=CONTRACT or self.observation_contract!=FIXED_ALPHA_OBSERVATION
                or d['sampling_scope']!='conditional_state_bank' or d['state_likelihood_ratio']!=1.
                or d['state_model'] is not None or d['state_sampling']!='fixed_stratified_alpha'
                or self.dimension!=(1 if d['intervention_mode']=='single' else 2)):
            raise ValueError('Explicit shared-teacher stratified contract required')
        for key in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256',
                    'state_law_sha256','proposal_structure_sha256','behavior_sha256','shared_paths_sha256','intervention_mode'):
            if not d.get(key) or manifest.get(key)!=d[key]:raise ValueError('Manifest identity differs: '+key)
        law=d['state_law'];sources={c['source_seed'] for c in law};by_case={c['case_id']:c for c in law}
        if (not law or len(by_case)!=len(law) or identity(law)!=d['state_law_sha256']
                or len(self.sequences)!=64*len(law)):
            raise ValueError('Complete fixed-stratum data required')
        for c in law:
            expected=1/(len(sources)*sum(r['source_seed']==c['source_seed'] for r in law))
            if c['alpha']!=expected:raise ValueError('Source/state target mass changed')
        for i,s in enumerate(self.sequences):
            c=law[i//64]
            if (s['case_id']!=c['case_id'] or s['source_seed']!=c['source_seed']
                    or s['seed']!=c['continuation_seed_start']+i%64 or s['target_mass']!=c['alpha']/64
                    or s['state_likelihood_ratio']!=1. or s['state_log_weight']!=0.
                    or s['event_result']!=any(s['cav_id'] in pair for pair in s['collision_pairs'])
                    or not np.isfinite(s['generation_log_weight']) or len(s['steps'])>4):
                raise ValueError('Shared path/label/schedule or target mass differs')
            prefix=0.;last=-1.
            for step in s['steps']:
                p,h=np.asarray(step['p_factors']),np.asarray(step['h_factors']);mask=np.asarray(step['factor_mask'])
                tv=np.asarray(step['factor_tv'])
                if (p.shape!=(self.dimension,) or h.shape!=p.shape or mask.shape!=p.shape or tv.shape!=p.shape
                        or not np.isin(mask,[0,1]).all() or mask[0]!=1
                        or not np.isfinite(tv).all() or np.any(tv<0) or np.any(tv>1+1e-12)
                        or len(step['actor_ids'])!=int(mask.sum()) or len(step['action_indices'])!=int(mask.sum())
                        or np.any(p[mask==0]!=1) or np.any(h[mask==0]!=1)
                        or np.any(tv[mask==0]!=0)
                        or not last<step['time_s']<4 or step['time_s']<0):
                    raise ValueError('Invalid target factors/critical timeline')
                obs=replay_observation(step,prefix,self.observation_contract)
                if (not np.array_equal(obs[18:18+self.dimension],mask)
                        or not np.isclose(step['observation'][16],np.clip(prefix/30.,-5,5),atol=1e-7)):
                    raise ValueError('Target prefix or active observation mask differs')
                increment=log_weight(step,[.1]*self.dimension)
                if (not np.isclose(increment,step['reference_log_weight'],atol=1e-10)
                        or not np.isclose(increment,step['full_log_p']-step['reference_full_log_q'],atol=1e-9)):
                    raise ValueError('Target full/factor likelihood differs')
                prefix+=increment;last=step['time_s']
        self.eligible=[i for i,s in enumerate(self.sequences) if s['steps']]
        self.positive=[i for i in self.eligible if self.sequences[i]['event_result']]
        self.negative=[i for i in self.eligible if not self.sequences[i]['event_result']]
        if not config.get('evaluation_only') and not self.positive:raise ValueError('No target-dependent CAV signal')
        self.log_scale=max((s['generation_log_weight']+sum(log_weight(t,[.1]*self.dimension) for t in s['steps'])
            for s in self.sequences if s['event_result']),default=0.)
        if 'log_reward_scale' in config:self.log_scale=float(config['log_reward_scale'])
        if not np.isfinite(self.log_scale):raise ValueError('Invalid common scale')
        self.sampling_correction=len(self.eligible)/len(self.sequences)
        self.positive_fraction=config.get('replay_positive_fraction')
        if self.positive_fraction is not None and not 0<self.positive_fraction<1:raise ValueError('Invalid replay probability')
        self.action_space=spaces.Box(low=.05,high=1.,shape=(self.dimension,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-5.,high=5.,shape=(22,),dtype=np.float32)
        self.rng=np.random.default_rng(config.get('seed',7));self.done=True


def diagnostics(env,policy):
    epsilons=[[] for _ in range(env.dimension)];tvs=[[] for _ in range(env.dimension)]
    logs=[];values=[]
    for s in env.sequences:
        prefix=0.
        for step in s['steps']:
            eps=policy(env.current_observation(step,prefix))
            for j,active in enumerate(step['factor_mask']):
                if active:epsilons[j].append(eps[j]);tvs[j].append((1-eps[j])*step['factor_tv'][j])
            prefix+=log_weight(step,eps)
        logs.append(prefix);values.append(env.loss(s,prefix))
    return dict(empirical_scaled_second_moment=float(np.mean(values)),
        empirical_second_moment=float(np.mean(values)*np.exp(env.log_scale)),
        dataset_event_count=sum(s['event_result'] for s in env.sequences),
        dataset_event_source_count=len({s['source_seed'] for s in env.sequences if s['event_result']}),
        likelihood_log_quantiles=np.quantile(logs,[.5,.95,.99,1]).tolist(),
        epsilon_quantiles_by_active_factor=[np.quantile(v,[.05,.5,.95]).tolist() if v else None for v in epsilons],
        conditional_QP_TV_at_logged_action_rows=[float(np.mean(v)) if v else None for v in tvs],
        diagnostic_scope='Same fixed teacher labels; exact source/state and replay correction; training only')
