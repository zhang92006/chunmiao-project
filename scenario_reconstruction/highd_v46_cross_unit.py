"""Opt-in cross-unit D2RL Q with exact natural conditional partners.

Q(A)=Qij(ai,aj)*P(A_rest|ai,aj); the unchanged conditional factor cancels
in P/Q. Matching and physics are inherited from the frozen native experiment.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from .highd_v46_assets import read,save,digest,ensure_assets
from .highd_v46_d2rl import NativeExperiment,experiment_target,identity,observation,factor_influence
from .highd_v46_contract import CONTEXT_OBSERVATION,enrich_observation
from .highd_v46_pair_synergy import candidate_pairs,pair_marginal,pair_metrics,response_risk
from .highd_v46_pair_witness import conditional_actions
from .highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS,inverse_cdf


def positive_interaction(p,r):
    p=np.asarray(p,float);r=np.asarray(r,float)
    a,b=np.unravel_index(np.argmax(p),p.shape)
    support=(p>0)&(p[:,b,None]>0)&(p[a,None,:]>0)
    return np.where(support,np.maximum(r-r[:,b,None]-r[a,None,:]+r[a,b],0.),0.)


def proposal_factors(p,h,epsilon):
    p=np.asarray(p,float);h=np.asarray(h,float);e=np.asarray(epsilon,float)
    if (p.ndim!=2 or h.shape!=p.shape or e.shape!=(2,) or
        not all(np.isfinite(a).all() for a in (p,h,e)) or np.any(p<0) or np.any(h<0)
        or not np.isclose(p.sum(),1.) or not np.isclose(h.sum(),1.)
        or np.any(h[p==0]>0) or np.any(e<.05) or np.any(e>1)):
        raise ValueError('Invalid joint P/H, support or epsilon')
    p1=p.sum(1);h1=h.sum(1)
    p2=np.divide(p,p1[:,None],out=np.zeros_like(p),where=p1[:,None]>0)
    h2=np.divide(h,h1[:,None],out=p2.copy(),where=h1[:,None]>0)
    q1=e[0]*p1+(1-e[0])*h1;q2=e[1]*p2+(1-e[1])*h2
    return p1,p2,h1,h2,q1,q2


def natural_log_probability(units,actions):
    result=0.
    for unit in units:
        p=np.asarray(unit['p']);value=float(p[tuple(actions[k] for k in unit['ids'])])
        if value<=0:raise ValueError('Draw outside natural support')
        result+=np.log(value)
    return float(result)


def draw_intervention(units,ids,h,epsilon,rng):
    if len(ids)!=2 or len(set(ids))!=2:raise ValueError('Two distinct controlled actors required')
    p,same=pair_marginal(units,ids)
    p1,p2,h1,h2,q1,q2=proposal_factors(p,h,epsilon)
    a=inverse_cdf(q1,float(rng.random()));b=inverse_cdf(q2[a],float(rng.random()))
    actions=conditional_actions(units,dict(zip(ids,(a,b))),rng)
    pf=[float(p1[a]),float(p2[a,b])];hf=[float(h1[a]),float(h2[a,b])];qf=[float(q1[a]),float(q2[a,b])]
    log_ratio=float(np.log(np.asarray(pf)/qf).sum())
    full_p=natural_log_probability(units,actions)
    conditional=full_p-float(np.log(p[a,b]))
    full_q=float(np.log(qf).sum())+conditional
    if not np.isclose(full_p-full_q,log_ratio,atol=1e-10,rtol=1e-10):
        raise ArithmeticError('Full likelihood and two-factor likelihood differ')
    return actions,{'p_factors':pf,'h_factors':hf,'q_factors':qf,
        'generation_epsilon':np.asarray(epsilon,float).tolist(),'log_weight':log_ratio,
        'full_log_p':full_p,'full_log_q':full_q,'cancelled_log_conditional':conditional,
        'same_natural_unit':same,'affected_natural_units':[list(u['ids']) for u in units if set(ids).intersection(u['ids'])],
        'action_indices':[int(a),int(b)]}


def proposal_structure(config,mode):
    if mode not in ('natural','dual'):raise ValueError('Expected natural or dual mode')
    return {'contract':'cross_unit_positive_interaction_v1','mode':mode,
        'settings':{k:v for k,v in config.items() if k not in ('epsilon','physical_config','scope')},
        'observation_contract':CONTEXT_OBSERVATION,
        'implementation_sha256':digest(__file__),
        'proxy_sha256':digest(Path(__file__).with_name('highd_v46_pair_synergy.py')),
        'conditional_sampler_sha256':digest(Path(__file__).with_name('highd_v46_pair_witness.py'))}


class CrossUnitExperiment(NativeExperiment):
    def __init__(self,*args,proposal_config,**kwargs):
        super().__init__(*args,**kwargs);self.proposal_config=proposal_config

    def select_pair(self,units,actors,pmfs):
        cfg=self.proposal_config
        if round(self.time)%cfg['selection_interval_s']:return None
        best=None
        for unordered in candidate_pairs(self,units,cfg['candidate']):
            ids=sorted(unordered,key=lambda i:(actors[i].x,i))
            p,same=pair_marginal(units,ids)
            risk=response_risk(self,ids,pmfs,cfg['proxy'],reactive=True)
            metrics=pair_metrics(p,risk,cfg['selection'])
            self.diagnostics['evaluated_pair_candidates']+=1
            if not metrics['synergy_eligible']:continue
            score=positive_interaction(p,risk);mass=float(np.sum(p*score))
            h=p*score/mass;tv=factor_influence(p,h)
            if min(tv)<cfg['selection']['minimum_factor_tv']:continue
            self.diagnostics['eligible_synergy_pairs']+=1
            if best is None or mass>best['mass']:
                best={'ids':ids,'p':p,'h':h,'mass':mass,'metrics':metrics,'same_natural_unit':same,
                      'factor_influence':tv,'contrast':float(np.ptp(score[p>0])/max(np.max(score[p>0]),1e-30))}
        return best

    def sample(self,pmfs,mode,epsilon,policy=None):
        if mode not in ('natural','dual'):raise ValueError('Cross-unit sampler supports natural/dual only')
        units,actors=self.prepare(pmfs)
        chosen=self.select_pair(units,actors,pmfs) if mode=='dual' else None
        self.diagnostics['decision_steps']+=1
        self.diagnostics['pair_available_steps']+=int(any(len(u['ids'])==2 for u in units))
        if chosen is None:return conditional_actions(units,{},self.rng)
        ids=chosen['ids'];rear,front=[actors[i] for i in ids]
        obs=enrich_observation(observation(self,ids),chosen['mass'],chosen['contrast'],self.log_weight,
            self.time,self.config['horizon_s'],[1,1],front.x-front.length-rear.x,front.v-rear.v)
        eps=policy(obs) if policy is not None else epsilon
        actions,logged=draw_intervention(units,ids,chosen['h'],eps,self.rng)
        self.steps.append({**logged,'time_s':self.time,'observation':obs,'actor_ids':ids,'factor_mask':[1,1],
            'critical_mass':chosen['mass'],'relative_action_contrast':chosen['contrast'],
            'factor_influence_tv':chosen['factor_influence'],'synergy_metrics':chosen['metrics']})
        biased=int(np.count_nonzero(np.asarray(eps)<1.))
        self.log_weight+=logged['log_weight'];self.max_intervened=max(self.max_intervened,biased)
        self.diagnostics['critical_steps']+=1
        self.diagnostics['dual_steps']+=int(biased==2);self.diagnostics['single_steps']+=int(biased==1)
        self.diagnostics['cross_unit_steps']+=int(not chosen['same_natural_unit'])
        return actions


def validate_config(config,physical):
    if (type(config['selection_interval_s']) is not int or config['selection_interval_s']<1
        or config['adversarial_score']!='positive_four_corner_interaction'
        or config['no_eligible_pair']!='natural_sampling' or config['pair_order']!='rear_to_front_then_actor_id'
        or physical['observation_contract']!=CONTEXT_OBSERVATION or physical['epsilon_min']!=.05):
        raise ValueError('Unsupported cross-unit proposal configuration')


def collect(config_path,output,mode,episodes,seed,epsilon=None,policy=None):
    cfg=read(config_path);physical=read(cfg['physical_config']);validate_config(cfg,physical)
    if mode not in ('natural','dual') or episodes<1:raise ValueError('Invalid collection mode/count')
    eps=np.asarray(cfg['epsilon'] if epsilon is None else epsilon,float)
    if eps.shape!=(2,) or not np.isfinite(eps).all() or np.any(eps<.05) or np.any(eps>1):raise ValueError('Two supported epsilons required')
    assets,lock=ensure_assets();target=experiment_target(lock,physical);target_hash=identity(target)
    structure=proposal_structure(cfg,mode);structure_hash=identity(structure)
    if policy is not None:
        metadata=policy.metadata
        for key,value in [('observation_contract',CONTEXT_OBSERVATION),('experiment_target_sha256',target_hash),
                          ('proposal_structure_sha256',structure_hash),('action_dimension',2),('intervention_mode','dual')]:
            if metadata.get(key)!=value:raise ValueError('Policy mismatch: '+key)
        seen=set(metadata.get('training_seeds',[]))|set(metadata.get('validation_seeds',[]))
        if seen.intersection(range(seed,seed+episodes)):raise ValueError('Evaluation seed overlaps training or validation')
    destination=Path(output);destination.mkdir(parents=True,exist_ok=False)
    model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    proposal={'structure':structure,'epsilon':eps.tolist(),'policy_weights_sha256':policy.metadata['weights_sha256'] if policy else None}
    sequences=[];started=time.perf_counter()
    for run_seed in range(seed,seed+episodes):
        tick=time.perf_counter()
        road=CrossUnitExperiment(model,assets/'pair',assets/'lateral',run_seed,physical,proposal_config=cfg)
        initial=[vars(c).copy() for c in road.vehicles]
        while road.time<physical['horizon_s']-1e-9:
            if not road.step_policy(mode,eps,policy):break
        event=any(road.cav.id in pair for pair in road.crashes)
        seq={'seed':run_seed,'cav_id':road.cav.id,'event_result':event,
            'event_type':'cav_first_collision' if event else 'bv_only_competing_collision' if road.crashes else 'horizon_no_collision',
            'elapsed_s':road.time,'collision_pairs':road.crashes,'generation_log_weight':road.log_weight,'steps':road.steps,
            'maximum_intervened_bvs':road.max_intervened,'diagnostics':dict(road.diagnostics),
            'minimum_ttc_s':float(road.minimum_ttc) if np.isfinite(road.minimum_ttc) else None,
            'minimum_bumper_gap_m':float(road.minimum_gap) if np.isfinite(road.minimum_gap) else None,
            'collision_context':road.collision_context,'wall_time_s':time.perf_counter()-tick}
        save(destination/f'episode_{run_seed}.json',dict(seq,initial_fleet=initial));sequences.append(seq)
        print({'seed':run_seed,'event':seq['event_type'],'critical_steps':len(road.steps),'log_weight':road.log_weight},flush=True)
    data={'contract':'highd_v46_native_d2rl_sequence_v2','observation_contract':CONTEXT_OBSERVATION,
        'action_dimension':2 if mode=='dual' else 1,'intervention_mode':mode,'natural_target_sha256':lock['natural_target_sha256'],
        'experiment_target':target,'experiment_target_sha256':target_hash,'proposal':proposal,
        'proposal_sha256':identity(proposal),'proposal_structure_sha256':structure_hash,'sequences':sequences}
    save(destination/'sequences.json',data)
    save(destination/'sequence_manifest.json',{'data_path':'sequences.json','sha256':digest(destination/'sequences.json'),
        **{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','proposal_sha256','proposal_structure_sha256','intervention_mode')}})
    from .highd_v46_statistics import summarize_sequences
    totals=Counter()
    for seq in sequences:totals.update(seq['diagnostics'])
    summary={**summarize_sequences(sequences),'event_counts':dict(Counter(s['event_type'] for s in sequences)),
        'diagnostics':dict(totals),'experiment_target_sha256':target_hash,'proposal_structure_sha256':structure_hash,
        'wall_time_s':time.perf_counter()-started,'mode':mode,'epsilon':eps.tolist(),
        'scope':'Natural-initialization development collection. No validation selection or acceleration claim.'}
    save(destination/'collection_summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/highd_v46_cross_unit_v1.json');p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['natural','dual'],default='dual');p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--seed',type=int,default=140000);p.add_argument('--epsilon',type=float,nargs=2);p.add_argument('--policy')
    a=p.parse_args();policy=None
    if a.policy:
        from .highd_v46_policy import NumpyPolicy
        policy=NumpyPolicy(a.policy)
    print(collect(a.config,a.output,a.mode,a.episodes,a.seed,a.epsilon,policy))


if __name__=='__main__':main()
