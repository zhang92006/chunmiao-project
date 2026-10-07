"""Two target likelihood views of ONE immutable teacher physical trajectory.

Targets use original formal Single selection / following-pair Dual with common
Single gate. They never sample actions or consume the behavior RNG in replay.
"""
import numpy as np

from .highd_v46_contract import enrich_observation
from .highd_v46_cross_unit import draw_intervention, natural_log_probability, proposal_factors
from .highd_v46_d2rl import NativeExperiment, observation
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_pair_witness import restore_prepared_state
from .highd_v47_fixed_alpha_native import single_marginal
from .highd_v47_teacher_restore import physical_record


def choices(road, units, actors, pmfs):
    single = NativeExperiment.choose(road, units, actors, pmfs, 'single')
    dual = NativeExperiment.choose(road, units, actors, pmfs, 'dual') if single is not None else None
    if dual is None or len(dual['ids']) != 2: dual = single
    return {'single': single, 'dual': dual}


def describe(road, units, selected, mode, initial_time, horizon):
    if selected is None: return None
    ids = selected['ids']; actors = {c.id:c for c in road.vehicles}; cars = [actors[i] for i in ids]
    dim = 1 if mode == 'single' else 2; mask = [1, int(len(ids)==2)]
    obs = enrich_observation(observation(road, ids), selected['mass'], selected['relative_contrast'],
        0., road.time-initial_time, horizon, mask,
        cars[1].x-cars[1].length-cars[0].x if len(cars)==2 else 0.,
        cars[1].v-cars[0].v if len(cars)==2 else 0.)
    p = single_marginal(units,ids[0]) if len(ids)==1 else pair_marginal(units,ids)[0]
    return dict(actor_ids=ids, dimension=dim, factor_mask=mask[:dim], observation=obs,
        time_s=road.time-initial_time, source_time_s=road.time, P=p, H=np.asarray(selected['h']))


def target_step(description, actions, full_log_p, reference_prefix):
    if description is None: return None
    d = description; ids=d['actor_ids']; p,h=d['P'],d['H']; indices=[actions[i] for i in ids]
    if len(ids)==2:
        p1,p2,h1,h2,_,_ = proposal_factors(p,h,[.1,.1]); a,b=indices
        pf=[float(p1[a]),float(p2[a,b])];hf=[float(h1[a]),float(h2[a,b])]
        pmfs=[dict(p=p1.tolist(),h=h1.tolist()),dict(p=p2[a].tolist(),h=h2[a].tolist())]
    else:
        a=indices[0];pf=[float(p[a])];hf=[float(h[a])];pmfs=[dict(p=p.tolist(),h=h.tolist())]
        if d['dimension']==2:
            pf.append(1.);hf.append(1.);pmfs.append(dict(p=[1.],h=[1.]))
    pf,hf=np.asarray(pf),np.asarray(hf);q=.1*pf+.9*hf
    lr=float(np.log(pf/q).sum());obs=list(d['observation']);obs[16]=float(np.clip(reference_prefix/30.,-5,5))
    return dict(actor_ids=ids,action_indices=indices,factor_mask=d['factor_mask'],observation=obs,
        time_s=d['time_s'],source_time_s=d['source_time_s'],p_factors=pf.tolist(),h_factors=hf.tolist(),
        reference_epsilon=[.1]*d['dimension'],reference_q_factors=q.tolist(),reference_log_weight=lr,
        full_log_p=full_log_p,reference_full_log_q=full_log_p-lr,factor_pmfs=pmfs,
        factor_tv=[float(.5*np.abs(np.asarray(x['p'])-x['h']).sum()) for x in pmfs])


def replay(design, original, model, assets, physical, horizon=4, scale=2.):
    state=design['state'];seed=original['seed'];start=state['time_s']
    road=restore_prepared_state(model,assets,physical,state,seed)
    views={'single':[], 'dual':[]};prefix={'single':0.,'dual':0.};decisions=[]
    units=[dict(ids=u['ids'],p=np.asarray(u['p'])) for u in state['units']]
    pmfs={int(k):np.asarray(v) for k,v in state['pmfs'].items()}
    actors={c.id:c for c in road.vehicles}
    selected=choices(road,units,actors,pmfs)
    desc={m:describe(road,units,selected[m],m,start,horizon) for m in views}
    actions,behavior=draw_intervention(units,design['row']['actor_ids'],np.asarray(design['H']),[.1,.1],road.rng)
    if physical_record(behavior)!=physical_record(original['initial_likelihood_audit']):
        raise ValueError('Teacher initial actions/likelihood changed')

    def record(actions, desc, full_p, full_q):
        steps={}
        for mode in views:
            step=target_step(desc[mode],actions,full_p,prefix[mode])
            steps[mode]=step
            if step is not None:
                views[mode].append(step);prefix[mode]+=step['reference_log_weight']
        decisions.append(dict(time_s=road.time-start,actions={str(k):int(v) for k,v in actions.items()},
            behavior_full_log_p=full_p,behavior_full_log_q=full_q,
            target_active={m:steps[m] is not None for m in views}))

    record(actions,desc,behavior['full_log_p'],behavior['full_log_q'])
    road.action_pmfs=lambda:pmfs;road.sample=lambda *args:actions
    road.step_policy('natural',1.)
    del road.action_pmfs;del road.sample
    original_sample=road.sample
    prepared={}

    def observe_choose(units, actors, pmfs, mode):
        if mode!='natural':raise ValueError('Replay must retain the original natural future')
        selected=choices(road,units,actors,pmfs)
        prepared['units']=units
        prepared['views']={m:describe(road,units,selected[m],m,start,horizon) for m in views}
        return None

    def observe_sample(pmfs, mode, epsilon, policy=None):
        # Invoke EXACT native natural sampler, not a distribution-equivalent
        # sampler with a different RNG consumption order.
        actions=original_sample(pmfs,mode,epsilon,policy)
        full=natural_log_probability(prepared['units'],actions)
        record(actions,prepared['views'],full,full)
        return actions

    road.choose=observe_choose;road.sample=observe_sample
    while not road.crashes and road.time < start+horizon-1e-9:road.step_policy('natural',1.)
    event=any(road.cav.id in pair for pair in road.crashes)
    reproduced=dict(seed=seed,event_result=event,bv_only_collision=bool(road.crashes) and not event,
        generation_log_weight=behavior['log_weight'],initial_likelihood_audit=behavior,
        collision_pairs=road.crashes,clearance_risk=float(np.exp(-max(road.minimum_gap,0.)/scale)))
    if physical_record(reproduced)!=physical_record(original):
        expected=physical_record(original);actual=physical_record(reproduced)
        different={k:(actual.get(k),expected.get(k)) for k in actual.keys()|expected.keys()
                   if actual.get(k)!=expected.get(k)}
        raise ValueError('Immutable teacher endpoint/likelihood no longer reproduces: '+repr(different))
    return dict(seed=seed,cav_id=road.cav.id,event_result=event,collision_pairs=road.crashes,
        generation_log_weight=original['generation_log_weight'],behavior_initial_audit=behavior,
        native_decisions=decisions,views=views,elapsed_s=road.time-start)


def compact_step(step):
    # Full distributions and likelihood evidence remain in audited case shards.
    return {k:step[k] for k in ('actor_ids','action_indices','factor_mask','observation','time_s',
        'p_factors','h_factors','factor_tv','reference_log_weight','full_log_p','reference_full_log_q')}
