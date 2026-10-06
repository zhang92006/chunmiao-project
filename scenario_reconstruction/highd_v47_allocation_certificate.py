"""Prospective benefit/loss certification of a protected fixed-bank Q_S.

Fit and bets use completed TRAINING branching data only. New independent roots
measure the allocation OBJECTIVE, not precise J in every state. Production and
PPO are untouched. No consumed validation or new natural source is opened.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from . import highd_v47_decision_branching as parent
from . import highd_v47_objective_bounds as bounds
from .highd_v46_assets import digest, read, save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_cross_unit import natural_log_probability
from .highd_v46_d2rl import identity
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state
from .highd_v47_allocation import waterfill
from . import highd_v47_state_allocation as state_sampler

DEFAULT = 'configs/highd_v47_allocation_certificate_v2.json'


def fit_and_design(old, pilot, cfg):
    alpha = np.asarray(old['target_masses'], float); js = []
    for i, state in enumerate(old['states']):
        value = 0.
        for b in range(cfg['batches']):
            rows = pilot[(b, i)]['rows']
            for h in state['strata']:
                z = [r['branching_value'] for r in rows if r['stratum_id'] == h['stratum_id']]
                if (len(z) != h['draws'] or not np.isfinite(z).all()
                        or np.any(np.asarray(z) < 0) or np.any(np.asarray(z) > 2)):
                    raise ValueError('Incomplete or invalid training roots')
                value += h['coefficient_mass']*np.mean(z)/cfg['batches']
        js.append(float(value))
    q = waterfill(alpha, js, 1/cfg['allocation_loss_cap']); d = alpha*(alpha/q-1)
    states = []; schedules = []; pilot_benefit = [[] for _ in range(cfg['batches'])]
    for i, state in enumerate(old['states']):
        side = 'benefit' if d[i] < 0 else 'loss' if d[i] > 0 else 'neutral'
        hs = [dict(h, draws=(h['draws'] if side == 'benefit' else
            max(cfg['minimum_roots_per_stratum'], int(np.ceil(abs(d[i])*h['coefficient_mass']/cfg['maximum_loss_root_bound'])))))
              for h in state['strata']]
        states.append(dict(state_id=i, plan=state['plan'], side=side,
                           signed_objective_coefficient=float(d[i]), strata=hs))
        if side == 'benefit':
            for b in range(cfg['batches']):
                coefficient = {h['stratum_id']: -d[i]*h['coefficient_mass']/h['draws'] for h in hs}
                pilot_benefit[b].extend(coefficient[r['stratum_id']]*r['branching_value'] for r in pilot[(b, i)]['rows'])
    loss_counts = sum(h['draws'] for s in states if s['side'] != 'benefit' for h in s['strata'])
    if loss_counts > cfg['maximum_natural_roots_per_batch']:
        raise ValueError('Prospective uncertainty budget exceeds protocol cap; version a different design')
    loss_limits = [max(0., s['signed_objective_coefficient'])*h['coefficient_mass']/h['draws']
                   for s in states if s['side'] == 'loss' for h in s['strata']]
    maximum = max(loss_limits, default=0.)
    loss_rate = cfg['loss_bet_support_fraction']/maximum if maximum > 0 else 1.
    error = (1-cfg['confidence'])/(2*cfg['batches'])
    grid = np.logspace(*cfg['benefit_bet_grid'])
    bet = bounds.choose_pilot_bet(pilot_benefit, grid, error)
    for b, start in enumerate(cfg['seed_starts']):
        natural_index = 0
        for i, state in enumerate(states):
            rng = np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'], b, i])); rows = []
            for h in state['strata']:
                cells = rng.choice(h['cells'], h['draws'], p=h['measurement_probabilities'])
                for within, cell in enumerate(cells):
                    row = dict(stratum_id=h['stratum_id'], within_stratum_draw=within,
                               action_indices=[int(a) for a in np.unravel_index(cell, (33, 33))])
                    if state['side'] == 'benefit':
                        base = start+64*(i*256+len(rows))
                        row.update(initial_seed=base, step_seeds=list(range(base+1, base+21)),
                                   terminal_seeds=list(range(base+21, base+37)), control_uniforms=rng.random(21).tolist())
                    else:
                        row.update(native_seed=start+cfg['natural_seed_offset']+natural_index)
                        natural_index += 1
                    rows.append(row)
            schedules.append(dict(batch=b, state_id=i, side=state['side'], rows=rows))
    # Schedule the real Q_S/Q_A smoke comparison BEFORE any certificate outcome.
    sampling = state_sampler.schedule(alpha, q, cfg['iid_draws_per_arm'], cfg['iid_state_rng_seed'])
    return dict(states=states, schedules=schedules, target_masses=alpha.tolist(), candidate_q=q.tolist(),
        training_J=js, signed_objective_coefficients=d.tolist(), benefit_bet=bet, loss_bet_rate=loss_rate,
        error_per_side_per_batch=error, natural_roots_per_batch=loss_counts,
        branching_roots_per_batch=sum(len(s['rows']) for s in schedules if s['batch']==0 and s['side']=='benefit'),
        maximum_loss_root_bound=maximum, iid_schedules={k:sampling[k] for k in ('target_iid','allocated_iid')},
        scope='Same fixed-bank conditional target; point fit is not trusted without prospective objective certification.')


def natural_root(model, assets, physical, state, ids, schedule):
    """Cheap native P-future observation, with initial c/C action conditioned.

    No risk-proxy or T calculation is necessary: the future measurement LR is
    exactly one. Record each natural unit's drawn probability and all actions.
    Initial correlated partners remain P(rest | forced initial pair).
    """
    road = restore_prepared_state(model, assets, physical, state, schedule['native_seed']); steps = []
    initial_pmfs = {int(k): np.array(p) for k,p in state['pmfs'].items()}
    initial_log = None
    def record(units, actions):
        factors = [dict(ids=u['ids'], actions=[actions[i] for i in u['ids']],
                        probability=float(np.asarray(u['p'])[tuple(actions[i] for i in u['ids'])])) for u in units]
        return dict(time_s=road.time, all_action_indices=actions, units=factors,
                    full_log_p=natural_log_probability(units, actions))
    def first(*unused):
        nonlocal initial_log
        actions = conditional_actions(state['units'], dict(zip(ids,schedule['action_indices'])), road.rng)
        p,_=pair_marginal(state['units'],ids)
        initial_log=natural_log_probability(state['units'],actions)-float(np.log(p[tuple(schedule['action_indices'])]))
        steps.append(dict(record(state['units'],actions), initial=True)); return actions
    road.action_pmfs=lambda:initial_pmfs; road.sample=first
    road.step_policy('natural',1.); del road.action_pmfs; del road.sample
    def later(pmfs,*unused):
        units,_=road.prepare(pmfs); actions=conditional_actions(units,{},road.rng)
        steps.append(dict(record(units,actions),initial=False)); return actions
    road.sample=later
    while not road.crashes and road.time < state['time_s']+4-1e-9:
        road.step_policy('natural',1.)
    event=any(road.cav.id in pair for pair in road.crashes)
    return dict(event_result=event, competing_result=bool(road.crashes) and not event,
                collision_pairs=road.crashes, elapsed_s=road.time-state['time_s'], steps=steps,
                initial_cancelled_log_conditional=initial_log, future_weight=1., measurement_value=float(event))


def inputs(config):
    cfg=read(config)
    required=dict(version=2,batches=2,confidence=.95,allocation_loss_cap=1.1,
        minimum_roots_per_stratum=2, maximum_loss_root_bound=1e-5,
        maximum_natural_roots_per_batch=100000,loss_bet_support_fraction=.8,
        benefit_bet_grid=[1,9,81],seed_starts=[11000000,13000000],natural_seed_offset=1000000,
        schedule_seed=110017,workers=4,iid_draws_per_arm=2112,iid_state_rng_seed=150017,
        iid_continuation_seed_start=15000000)
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise ValueError('Version changed objective-certificate protocol before new outcomes')
    old_cfg,physical,assets,plans,ads,old_design,old_lock=parent.inputs(cfg['parent_config'])
    root=Path(cfg['parent_path']); previous=read(root/'summary.json')
    if (digest(root/'summary.json')!=cfg['parent_summary_sha256'] or not previous['complete']
            or previous['config']!=old_cfg or any(previous['lock'].get(k)!=v for k,v in old_lock.items())):
        raise ValueError('Completed training branching measurement changed')
    for name,sha in previous['artifact_hashes'].items():
        if Path(name).name!=name or digest(root/name)!=sha: raise ValueError('Training branching artifact changed')
    if read(root/'measurement_plan.json')!=old_design:
        raise ValueError('Inherited full-bank action measurement design changed')
    # Locks verified ALL tree bytes above. Fitting needs only independent root
    # values, so do not keep multi-GB nested native trees alive in the loader.
    pilot={}
    for b in range(2):
        for i in range(len(plans)):
            rows=read(root/f'batch_{b}_state_{i:03d}.json')['rows']
            pilot[(b,i)]=dict(rows=[{k:r[k] for k in ('stratum_id','branching_value')} for r in rows])
    design=fit_and_design(old_design,pilot,cfg); cfg=dict(cfg,continuation=old_cfg['continuation'])
    lock=dict(config_sha256=digest(config),implementation_sha256=digest(__file__),
        bounds_implementation_sha256=digest(Path(__file__).with_name('highd_v47_objective_bounds.py')),
        state_sampler_sha256=digest(Path(__file__).with_name('highd_v47_state_allocation.py')),
        allocator_sha256=digest(Path(__file__).with_name('highd_v47_allocation.py')),
        parent_lock=old_lock,parent_summary_sha256=cfg['parent_summary_sha256'],design_sha256=identity(design))
    return cfg,physical,assets,plans,ads,design,lock


def _collect(task):
    schedule,state,plan,assets,physical,cfg=task; model=parent.parent.parent.parent._MODEL
    rows=[]
    for r in schedule['rows']:
        if schedule['side']=='benefit':
            value=parent.root_continuation(model,Path(assets),physical,state,plan['actor_ids'],r,cfg)
            value['measurement_value']=value['branching_value']
        else:
            value=natural_root(model,Path(assets),physical,state,plan['actor_ids'],r)
        rows.append(dict(r,**value))
    return dict(schedule,rows=rows)


def summarize(design,records):
    reports=[]
    for b in (0,1):
        benefit=[]; loss=[]; support=[]; states=[]
        for s in design['states']:
            i=s['state_id']; rows=records[(b,i)]['rows']; j=v=0.
            for h in s['strata']:
                zs=[r['measurement_value'] for r in rows if r['stratum_id']==h['stratum_id']]
                if len(zs)!=h['draws'] or len(zs)<2: raise ValueError('Incomplete objective stratum')
                z=np.array(zs); bound=2. if s['side']=='benefit' else 1.
                if not np.isfinite(z).all() or np.any(z<0) or np.any(z>bound): raise ValueError('Invalid bounded root')
                c=h['coefficient_mass']; j+=c*z.mean(); v+=c*c*z.var(ddof=1)/len(z)
                w=abs(s['signed_objective_coefficient'])*c/len(z)
                if s['side']=='benefit': benefit.extend(w*z)
                elif s['side']=='loss': loss.extend(w*z); support.extend([w]*len(z))
            states.append(dict(state_id=i,source_seed=s['plan']['source_seed'],side=s['side'],
                J=float(j),measurement_variance=float(v),roots=len(rows),positive_roots=sum(r['measurement_value']>0 for r in rows)))
        lower=bounds.benefit_lower(benefit,design['benefit_bet']['rate'],design['error_per_side_per_batch'])
        upper=bounds.loss_upper(loss,support,design['loss_bet_rate'],design['error_per_side_per_batch'])
        report=bounds.objective_difference(lower,upper); a=np.array(design['target_masses']);q=np.array(design['candidate_q'])
        js=np.array([s['J'] for s in states]); vs=np.array([s['measurement_variance'] for s in states]);d=a*(a/q-1)
        report.update(batch=b,states=states,target_M2=float(a@js),candidate_M2=float((a*a/q)@js),
            difference_empirical_MCSE=float(np.sqrt((d*d)@vs)))
        reports.append(report)
    certified=all(r['improvement_certified'] for r in reports)
    return dict(batches=reports,allocation_improvement_certified=certified,
        family_confidence=1-sum(r['family_error'] for r in reports),
        candidate_q=design['candidate_q'],recommended_q=design['candidate_q'] if certified else design['target_masses'],
        universal_true_M2_loss_cap=1.1,
        decision='certified_fixed_bank_candidate' if certified else 'retain_alpha_objective_uncertain',
        source_generalization_validated=False,production_changed=False,ppo_ready=False)


def iid_comparison(cfg,assets,physical,plans,ads,design):
    results={}; records={}; cache={}
    a=np.array(design['target_masses']); q=np.array(design['candidate_q']); model=parent.parent.parent.parent._MODEL
    sample_cfg=dict(horizon_s=4.)
    for arm in ('target_iid','allocated_iid'):
        qs=a if arm=='target_iid' else q; rows=[]
        for draw,i in enumerate(design['iid_schedules'][arm]):
            key=(draw,i); seed=cfg['iid_continuation_seed_start']+draw
            if key not in cache:
                cache[key]=state_sampler.sample_continuation(model,assets,physical,ads[i]['state'],
                    dict(ids=plans[i]['actor_ids'],h=np.array(ads[i]['H'])),[.1,.1],seed,sample_cfg,10.)
            row=copy.deepcopy(cache[key]); audit_sample(row,[.1,.1]); lw=float(np.log(a[i]/qs[i]))
            row.update(draw_id=draw,state_id=i,target_state_mass=float(a[i]),generation_state_mass=float(qs[i]),
                state_log_weight=lw,total_log_weight=lw+row['generation_log_weight'])
            rows.append(row)
        records[arm]=dict(arm=arm,sequences=rows); results[arm]=state_sampler.summarize(rows,plans,arm)
    return records,dict(arms=results,physical_paths=len(cache),
        interpretation='Finite IID smoke comparison, not the objective confidence certificate; CRN arms are not independent data.')


def freeze(config,output):
    *_,design,lock=inputs(config)
    lock['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();save(output,lock)
    return dict(branching_roots_per_batch=design['branching_roots_per_batch'],
                natural_roots_per_batch=design['natural_roots_per_batch'],candidate_q=design['candidate_q'])


def run(config,lock_path,output):
    cfg,physical,assets,plans,ads,design,expected=inputs(config);lock=read(lock_path)
    if any(lock.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen objective design changed')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    save(out/'opened_protocol.json',dict(config=cfg,lock=lock));save(out/'measurement_plan.json',design)
    records={}
    for b in (0,1):
        tasks=[(s,ads[s['state_id']]['state'],plans[s['state_id']],str(assets),physical,cfg)
               for s in design['schedules'] if s['batch']==b]
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=parent.parent.parent.parent._init_worker,initargs=(str(assets),)) as pool:
            for future in as_completed([pool.submit(_collect,t) for t in tasks]):
                r=future.result();i=r['state_id'];records[(b,i)]=r;save(out/f'batch_{b}_state_{i:03d}.json',r)
                print(dict(completed_cases=len(records),total=66),flush=True)
    # Run the predeclared Q_S/Q_A smoke arms even if certification fails.
    parent.parent.parent.parent._init_worker(str(assets))
    iid,smoke=iid_comparison(cfg,assets,physical,plans,ads,design)
    for arm,r in iid.items():save(out/f'{arm}_sequences.json',r)
    result=dict(complete=True,config=cfg,lock=lock,**summarize(design,records),iid_smoke=smoke,
        total_wall_s=time.perf_counter()-started,
        artifact_hashes={p.name:digest(p) for p in sorted(out.glob('*.json'))})
    save(out/'summary.json',result)
    return {k:result[k] for k in ('allocation_improvement_certified','decision','family_confidence')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','run'])
    p.add_argument('--config',default=DEFAULT);p.add_argument('--lock');p.add_argument('--output',required=True)
    x=p.parse_args();print(freeze(x.config,x.output) if x.mode=='freeze' else run(x.config,x.lock,x.output))
