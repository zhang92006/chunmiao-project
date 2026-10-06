"""Nested conditional branching and protected allocation on the fixed bank.

Each root shares a native prefix and branches the remaining future. Independent
roots, not leaves, define measurement variance. P_future is always the target.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from . import highd_v47_fixed_mixture_moments as parent
from .highd_v46_assets import digest, read, save
from .highd_v46_cross_unit import draw_intervention, natural_log_probability, proposal_factors
from .highd_v46_d2rl import identity
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state
from .highd_v47_allocation import waterfill

DEFAULT = 'configs/highd_v47_branching_moments_v1.json'
ARMS = ['single', 'branching']


def clone_native(road, seed):
    """Copy all dynamic state, including history/physics, but not readonly assets.

    Never reuse a bound sample callback capturing the parent road. The clone is
    at an unprepared decision boundary; each suffix prepares exactly once.
    """
    clone = copy.copy(road)
    excluded = {'model', 'config', 'natural', 'rng', 'sample', 'action_pmfs'}
    dynamic = {k: copy.deepcopy(v) for k, v in road.__dict__.items() if k not in excluded}
    clone.__dict__.clear(); clone.__dict__.update(dynamic)
    clone.model, clone.config = road.model, road.config
    clone.natural = copy.copy(road.natural)
    clone.natural.histories = copy.deepcopy(road.natural.histories)
    clone.natural.previous = set(road.natural.previous)
    clone.rng = np.random.default_rng(seed)
    clone.cav = next(c for c in clone.vehicles if c.id == road.cav.id)
    return clone


def dynamic_identity(road):
    return identity(dict(time_s=road.time, vehicles=[vars(c) for c in road.vehicles],
                         cav_id=road.cav.id, histories={str(k): v for k, v in road.natural.histories.items()},
                         previous=[list(x) for x in sorted(road.natural.previous)],
                         minimum_gap=road.minimum_gap if np.isfinite(road.minimum_gap) else None,
                         minimum_ttc=road.minimum_ttc if np.isfinite(road.minimum_ttc) else None,
                         crashes=road.crashes, last_acceleration=road.last_acceleration))


def advance_future(road, tilted, end, cfg):
    records = []
    def sample(pmfs, *unused):
        units, actors = road.prepare(pmfs)
        choice = parent.parent.pair_choice(road, units, actors, pmfs, end-road.time, cfg)
        if choice is None:
            actions = conditional_actions(units, {}, road.rng)
            lp = natural_log_probability(units, actions)
            record = dict(actor_ids=[], p_factors=[], t_factors=[], full_log_p=lp, full_log_t=lp, log_t_over_p=0.)
        else:
            ids, p, h, q1, q2 = choice
            actions = draw_intervention(units, ids, h, cfg['epsilon'], road.rng)[0] if tilted else conditional_actions(units, {}, road.rng)
            a, b = [actions[i] for i in ids]
            p1, p2, *_ = proposal_factors(p, h, cfg['epsilon'])
            pf, tf = [float(p1[a]), float(p2[a, b])], [float(q1[a]), float(q2[a, b])]
            lp = natural_log_probability(units, actions); conditional = lp-float(np.log(p[a, b]))
            lt = conditional+float(np.log(tf).sum())
            record = dict(actor_ids=ids, action_indices=[a, b], p_factors=pf, t_factors=tf,
                          full_log_p=lp, full_log_t=lt, cancelled_log_conditional=conditional,
                          log_t_over_p=float(np.log(np.array(tf)/pf).sum()),
                          proposal_table_sha256=identity({'p': p.tolist(), 'h': h.tolist()}))
        record.update(time_s=road.time, all_action_indices=actions, natural_unit_ids=[u['ids'] for u in units])
        records.append(record); return actions
    road.sample = sample
    road.step_policy('natural', 1.)
    del road.sample
    return records[0]


def pack_path(road, steps, initial_conditional, origin):
    ratio = float(sum(s['log_t_over_p'] for s in steps))
    weight = parent.parent.path_weight(ratio, .5)
    event = any(road.cav.id in p for p in road.crashes)
    return dict(steps=steps, full_log_p=float(sum(s['full_log_p'] for s in steps)),
                full_log_t=float(sum(s['full_log_t'] for s in steps)), log_t_over_p=ratio,
                initial_cancelled_log_conditional=initial_conditional,
                event_result=event, bv_only_collision=bool(road.crashes) and not event,
                collision_pairs=road.crashes, elapsed_s=road.time-origin,
                future_weight=weight, weighted_event=float(event)*weight)


def branch_statistics(pz, tz, eta):
    pz, tz = np.asarray(pz, float), np.asarray(tz, float)
    if (pz.shape != (2,) or tz.shape != (2,) or not 0 < eta < 1
            or not all(np.isfinite(z).all() and np.all((z >= 0) & (z <= 2+1e-12)) for z in (pz, tz))):
        raise ValueError('Invalid posterior-component branches')
    mp, mt = pz.mean(), tz.mean(); sp, st = pz.var(ddof=1), tz.var(ddof=1)
    vb = eta**2*sp/2+(1-eta)**2*st/2
    contrast = (mp-mt)**2-sp/2-st/2
    vs = eta*sp+(1-eta)*st+eta*(1-eta)*contrast
    return dict(value=float(eta*mp+(1-eta)*mt),
                conditional_branch_variance_unbiased=float(vb),
                conditional_single_variance_unbiased=float(vs),
                removed_conditional_variance_unbiased=float(vs-vb))


def root_continuation(model, assets, physical, state, ids, schedule, cfg):
    road = restore_prepared_state(model, assets, physical, state, schedule['prefix_seed'])
    pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}; initial = None
    def first_sample(*unused):
        nonlocal initial
        a = conditional_actions(state['units'], dict(zip(ids, schedule['action_indices'])), road.rng)
        p, _ = pair_marginal(state['units'], ids)
        initial = natural_log_probability(state['units'], a)-float(np.log(p[tuple(schedule['action_indices'])]))
        return a
    road.action_pmfs = lambda: pmfs; road.sample = first_sample
    road.step_policy('natural', 1.); del road.action_pmfs; del road.sample
    origin = state['time_s']; end = origin+cfg['horizon_s']; fork = origin+cfg['fork_s']; steps = []
    while not road.crashes and road.time < fork-1e-9:
        steps.append(advance_future(road, schedule['prefix_tilted_component'], end, cfg['continuation']))
    prefix = pack_path(road, steps, initial, origin); signature = dynamic_identity(road)
    if road.crashes or road.time >= end-1e-9:
        return dict(prefix=prefix, fork_state_sha256=signature, terminal_prefix=True,
                    branches=[], posterior_natural_fraction=None, selected_branch=None,
                    single_value=prefix['weighted_event'], branching_value=prefix['weighted_event'],
                    removed_conditional_variance_unbiased=0., future_decisions=len(steps))
    # Bayes update for the WHOLE-path G=(P+T)/2. Half/half suffix weights after
    # an informative prefix are wrong. The full terminal P/G weight includes it.
    eta = prefix['future_weight']/2
    if not 0 < eta < 1:
        raise FloatingPointError('Degenerate mixture posterior')
    branches = []
    for index, seed in enumerate(schedule['suffix_seeds']):
        tilted = index >= 2; child = clone_native(road, seed)
        if dynamic_identity(child) != signature:
            raise ValueError('Branch clone changed native prefix')
        suffix = []
        while not child.crashes and child.time < end-1e-9:
            suffix.append(advance_future(child, tilted, end, cfg['continuation']))
        path = pack_path(child, steps+suffix, initial, origin)
        branches.append(dict(component='T' if tilted else 'P', seed=seed,
                             fork_state_sha256=signature, suffix_steps=len(suffix), path=path))
    e = branch_statistics([b['path']['weighted_event'] for b in branches[:2]],
                          [b['path']['weighted_event'] for b in branches[2:]], eta)
    selected = 0 if schedule['control_uniform'] < eta else 2
    return dict(prefix=prefix, fork_state_sha256=signature, terminal_prefix=False, branches=branches,
                posterior_natural_fraction=eta, selected_branch=selected,
                single_value=branches[selected]['path']['weighted_event'], branching_value=e['value'],
                removed_conditional_variance_unbiased=e['removed_conditional_variance_unbiased'],
                conditional_single_variance_unbiased=e['conditional_single_variance_unbiased'],
                conditional_branch_variance_unbiased=e['conditional_branch_variance_unbiased'],
                future_decisions=len(steps)+sum(b['suffix_steps'] for b in branches))


def protected_allocation(alpha, lower, upper, cap=1.1):
    """Optimize a moment upper envelope, never plug zeros into an unsafe q.

    q>=alpha/cap guarantees J(q)<=cap*J(alpha) for EVERY true J>=0,
    independently of estimated intervals. Strict promotion needs an objective
    certificate, not all-state relative precision; otherwise recommend alpha.
    """
    a, lo, hi = [np.asarray(x, float) for x in (alpha, lower, upper)]
    if (a.ndim != 1 or a.shape != lo.shape or lo.shape != hi.shape or not len(a)
            or not all(np.isfinite(x).all() for x in (a, lo, hi)) or np.any(a <= 0)
            or not np.isclose(a.sum(), 1) or np.any(lo < 0) or np.any(hi < lo) or not 1 <= cap <= 2):
        raise ValueError('Invalid allocation uncertainty set')
    q = waterfill(a, hi, 1/cap) if np.any(hi > 0) else a.copy()
    d = a*(a/q-1)
    gap = float(np.where(d >= 0, d*hi, d*lo).sum())
    tol = 64*np.finfo(float).eps*max(float(a@hi), np.finfo(float).tiny)
    certified = gap < -tol
    if np.any(q < a/cap-1e-14):
        raise ValueError('Allocation violated uncertainty protection')
    return dict(upper_envelope_candidate=q.tolist(), recommended_q=q.tolist() if certified else a.tolist(),
                worst_case_M2_difference_upper=gap, strict_improvement_certified=bool(certified),
                state_LR_max=float(np.max(a/q)), true_M2_loss_cap=cap,
                decision='eligible_for_separate_evaluation' if certified else 'retain_target_uncertain_objective',
                interpretation='Cap bounds true uncentered M2, not centered variance or realized finite-sample estimates.')


def design_for(old, ads, cfg):
    states = []
    for s in old['states']:
        c = np.array([h['coefficient_mass'] for h in s['strata']])
        counts = parent.parent.parent.integer_budget((c/c.max())**2, cfg['roots_per_state'], minimum=2)
        states.append(dict(s, strata=[dict(h, draws=n) for h, n in zip(s['strata'], counts)],
                           minimax_variance_bound=float(np.sum(c*c/np.array(counts))),
                           former_equal_variance_bound=float(np.sum(c*c/np.array([h['draws'] for h in s['strata']])))))
    schedules = []
    for b, start in enumerate(cfg['seed_starts']):
        for i, s in enumerate(states):
            rows = []
            for h in s['strata']:
                rng = np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'], b, i, h['stratum_id']]))
                cells = rng.choice(h['cells'], h['draws'], p=h['measurement_probabilities'])
                components = rng.random(h['draws']) >= .5; controls = rng.random(h['draws'])
                for k, (cell, component, control) in enumerate(zip(cells, components, controls)):
                    seed = start+8*(i*cfg['roots_per_state']+len(rows))
                    rows.append(dict(stratum_id=h['stratum_id'], within_stratum_draw=k,
                                     action_indices=[int(a) for a in np.unravel_index(cell, np.array(ads[i]['P']).shape)],
                                     prefix_seed=seed, suffix_seeds=list(range(seed+1, seed+5)),
                                     prefix_tilted_component=bool(component), control_uniform=float(control)))
            schedules.append(dict(batch=b, state_id=i, rows=rows))
    return dict(states=states, schedules=schedules, target_masses=old['target_masses'],
                state_proposals=old['state_proposals'], independent_roots=2*len(states)*cfg['roots_per_state'],
                max_suffixes=8*len(states)*cfg['roots_per_state'])


def inputs(config):
    cfg = read(config)
    required = dict(version=1, roots_per_state=256, batches=2, fork_s=2, horizon_s=4,
                    suffixes_per_component=2, seed_starts=[6200000, 6300000], schedule_seed=620017,
                    workers=4, confidence=.95, allocation_loss_cap=1.1)
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Explicitly version changed branching design')
    old_cfg, physical, assets, plans, ads, old_design, old_lock = parent.inputs(cfg['parent_config'])
    root = Path(cfg['parent_path']); summary = read(root/'summary.json')
    if (digest(root/'summary.json') != cfg['parent_summary_sha256'] or not summary['complete']
            or summary['config'] != old_cfg or any(summary['lock'].get(k) != v for k, v in old_lock.items())):
        raise ValueError('Completed parent training artifacts changed')
    for name, sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Parent artifact changed')
    design = design_for(old_design, ads, cfg)
    cfg = dict(cfg, continuation=old_cfg['continuation'])
    lock = dict(config_sha256=digest(config), implementation_sha256=digest(__file__), parent_lock=old_lock,
                parent_summary_sha256=cfg['parent_summary_sha256'], design_sha256=identity(design))
    return cfg, physical, assets, plans, ads, design, lock


def _collect(task):
    schedule, state, plan, assets, physical, cfg = task
    rows = [dict(r, **root_continuation(parent.parent._MODEL, Path(assets), physical, state, plan['actor_ids'], r, cfg))
            for r in schedule['rows']]
    return dict(batch=schedule['batch'], state_id=schedule['state_id'], rows=rows)


def estimate(strata, rows, arm, error):
    j = v = lo = hi = removed = 0.; details = []
    for h in strata:
        g = [r for r in rows if r['stratum_id'] == h['stratum_id']]
        if len(g) != h['draws'] or {r['within_stratum_draw'] for r in g} != set(range(h['draws'])):
            raise ValueError('Incomplete independent-root stratum')
        z = [r[arm+'_value'] for r in g]
        mean, variance, bounds = parent.parent.bounded_interval(z, 2., error)
        c = h['coefficient_mass']; j += c*mean; v += c*c*variance; lo += c*bounds[0]; hi += c*bounds[1]
        removed += c*c*np.mean([r['removed_conditional_variance_unbiased'] for r in g])/len(g)
        details.append(dict(stratum_id=h['stratum_id'], roots=len(g), mean=mean, sample_variance=variance*len(g)))
    return dict(J=float(j), estimator_variance=float(v), bank_simultaneous_interval=[lo, hi], strata=details,
                removed_conditional_variance_unbiased=float(removed))


def summarize(cfg, design, records):
    error = (1-cfg['confidence'])/(4*sum(len(s['strata']) for s in design['states']))
    es = {(r['batch'], r['state_id'], arm): estimate(design['states'][r['state_id']]['strata'], r['rows'], arm, error)
          for r in records for arm in ARMS}
    alpha = np.array(design['target_masses']); arms = {}
    for arm in ARMS:
        states = []
        for i, s in enumerate(design['states']):
            a, b = es[(0, i, arm)], es[(1, i, arm)]; j = (a['J']+b['J'])/2
            variance = (a['estimator_variance']+b['estimator_variance'])/4
            bounds = (np.array(a['bank_simultaneous_interval'])+b['bank_simultaneous_interval'])/2
            states.append(dict(state_id=i, source_seed=s['plan']['source_seed'], J=j, estimator_variance=variance,
                               batch_J=[a['J'], b['J']], bank_simultaneous_interval=bounds.tolist(),
                               precision_supported=bool(bounds[0]>0 and (bounds[1]-bounds[0])/2 <= .5*j),
                               relative_empirical_MCSE=float(np.sqrt(variance)/j) if j>0 else None))
        total = float(alpha@np.array([s['J'] for s in states])); v = float((alpha*alpha)@np.array([s['estimator_variance'] for s in states]))
        arms[arm] = dict(states=states, target_M2=total, target_measurement_variance=v, empirical_MCSE=float(np.sqrt(v)))
    pilot = [es[(0, i, 'branching')] for i in range(len(alpha))]
    allocation = protected_allocation(alpha, [x['bank_simultaneous_interval'][0] for x in pilot],
                                     [x['bank_simultaneous_interval'][1] for x in pilot], cfg['allocation_loss_cap'])
    q = np.array(allocation['upper_envelope_candidate']); evaluation = [es[(1, i, 'branching')] for i in range(len(alpha))]
    common = parent.parent.parent.compare_state_proposal(alpha, q, evaluation)
    vb = arms['branching']['target_measurement_variance']
    removed = float(sum(.25*alpha[i]**2*es[(b, i, 'branching')]['removed_conditional_variance_unbiased']
                        for i in range(len(alpha)) for b in (0, 1)))
    return dict(arms=arms, state_batch_estimates=[dict(x, batch=b, state_id=i, arm=a) for (b, i, a), x in sorted(es.items())],
                branching_over_single_variance_point_only=vb/arms['single']['target_measurement_variance'] if arms['single']['target_measurement_variance']>0 else None,
                same_root_removed_conditional_variance_unbiased=removed,
                iid_single_variance_inferred_same_roots_unbiased=vb+removed,
                uncertainty_aware_allocation=allocation, independent_batch_allocation_comparison=common,
                allocation_improvement_validated=bool(allocation['strict_improvement_certified'] and common['certified_improvement']))


def freeze(config, output):
    *_, design, lock = inputs(config); lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return dict(independent_roots=design['independent_roots'], max_suffixes=design['max_suffixes'])


def run(config, lock_path, output):
    cfg, physical, assets, plans, ads, design, expected = inputs(config); lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen branching design changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', dict(config=cfg, lock=lock)); save(out/'measurement_plan.json', design)
    tasks = [(s, ads[s['state_id']]['state'], plans[s['state_id']], str(assets), physical, cfg) for s in design['schedules']]
    records = []; start = time.perf_counter()
    # Batches are opened in order. Save the batch-0 allocation before ANY batch-1
    # native outcome; batch 1 is never used to choose this candidate.
    for b in (0, 1):
        with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
                                 initializer=parent.parent._init_worker, initargs=(str(assets),)) as pool:
            futures = [pool.submit(_collect, t) for t in tasks if t[0]['batch'] == b]
            for f in as_completed(futures):
                r = f.result(); records.append(r); save(out/f'batch_{b}_state_{r["state_id"]:03d}.json', r)
                print(dict(completed_cases=len(records), total=len(tasks)), flush=True)
        if b == 0:
            error = (1-cfg['confidence'])/(4*sum(len(s['strata']) for s in design['states']))
            es = {r['state_id']: estimate(design['states'][r['state_id']]['strata'], r['rows'], 'branching', error) for r in records}
            allocation = protected_allocation(design['target_masses'], [es[i]['bank_simultaneous_interval'][0] for i in range(len(plans))],
                [es[i]['bank_simultaneous_interval'][1] for i in range(len(plans))], cfg['allocation_loss_cap'])
            save(out/'batch0_allocation_before_batch1.json', allocation)
    records.sort(key=lambda r: (r['batch'], r['state_id']))
    result = dict(complete=True, config=cfg, lock=lock, **summarize(cfg, design, records),
                  independent_roots=design['independent_roots'], suffixes_simulated=sum(len(r['branches']) for x in records for r in x['rows']),
                  future_native_decisions=sum(r['future_decisions'] for x in records for r in x['rows']), total_wall_s=time.perf_counter()-start,
                  production_changed=False, ppo_ready=False, source_generalization_validated=False,
                  artifact_hashes={p.name: digest(p) for p in sorted(out.glob('*.json'))})
    if result['uncertainty_aware_allocation'] != read(out/'batch0_allocation_before_batch1.json'):
        raise ValueError('Allocation changed after evaluation outcomes')
    save(out/'summary.json', result)
    return {a: {k: x[k] for k in ('target_M2', 'empirical_MCSE')} for a, x in result['arms'].items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['freeze', 'run'])
    p.add_argument('--config', default=DEFAULT); p.add_argument('--lock'); p.add_argument('--output', required=True)
    a = p.parse_args(); print(freeze(a.config, a.output) if a.mode == 'freeze' else run(a.config, a.lock, a.output))
