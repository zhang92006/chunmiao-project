"""Bounded path-mixture measurement of fixed-teacher state second moments.

The estimand Q_A is untouched. G is a measurement law on later native actions,
not a production testing proposal. Its LR occurs once, not squared, in J.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy

from . import highd_v47_state_moments_v2 as parent
from .highd_v46_assets import digest, read, save
from .highd_v46_cross_unit import draw_intervention, natural_log_probability, proposal_factors
from .highd_v46_d2rl import challenge, identity
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state

DEFAULT = 'configs/highd_v47_continuation_moments_v1.json'


def path_weight(log_t_over_p, natural_fraction):
    """Marginal path-mixture density, even when the natural component was drawn."""
    if not np.isfinite(log_t_over_p) or not 0 < natural_fraction <= 1:
        raise ValueError('Invalid path mixture')
    if natural_fraction == 1:
        return 1.
    denominator = np.logaddexp(np.log(natural_fraction),
                              np.log1p(-natural_fraction)+log_t_over_p)
    weight = float(np.exp(-denominator))
    if weight == 0:
        raise FloatingPointError('Future likelihood underflow; cannot drop supported event mass')
    return weight


def pair_choice(road, units, actors, pmfs, remaining, cfg):
    nearby = sorted((abs(c.x-road.cav.x), c.id) for c in road.vehicles
                    if c.id != road.cav.id and abs(c.x-road.cav.x) <= cfg['radius_m'])
    if len(nearby) < 2:
        return None
    ids = sorted([i for _, i in nearby[:2]], key=lambda i: (actors[i].x, i))
    p, _ = pair_marginal(units, ids)
    risk = challenge(road, ids, pmfs, dict(horizon_s=remaining,
                     step_s=cfg['proxy_step_s'], clearance_scale_m=cfg['clearance_scale_m']))
    score = risk**cfg['risk_power']; mass = float(np.sum(p*score))
    h = p*score/mass if mass > 0 else p.copy()
    *_, q1, q2 = proposal_factors(p, h, cfg['epsilon'])
    return ids, p, h, q1, q2


def continue_path(model, assets, physical, state, ids, actions, seed, horizon, cfg, tilted):
    """Draw a stopped P or T path and evaluate BOTH densities along that path.

    Initial forced actions condition P/T equally. Their unforced correlated
    partners stay natural. A future state chooses its pair before action draw.
    """
    road = restore_prepared_state(model, assets, physical, state, seed)
    first_pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}
    steps = []; end = state['time_s']+horizon
    initial_conditional = None

    def first_sample(*unused):
        nonlocal initial_conditional
        result = conditional_actions(state['units'], dict(zip(ids, actions)), road.rng)
        p, _ = pair_marginal(state['units'], ids)
        initial_conditional = natural_log_probability(state['units'], result)-float(np.log(p[tuple(actions)]))
        return result

    road.action_pmfs = lambda: first_pmfs
    road.sample = first_sample
    road.step_policy('natural', 1.)
    del road.action_pmfs

    def later_sample(pmfs, *unused):
        units, actors = road.prepare(pmfs)
        choice = pair_choice(road, units, actors, pmfs, end-road.time, cfg)
        if choice is None:
            result = conditional_actions(units, {}, road.rng)
            lp = natural_log_probability(units, result)
            record = dict(actor_ids=[], p_factors=[], t_factors=[],
                          full_log_p=lp, full_log_t=lp, log_t_over_p=0.)
        else:
            keys, p, h, q1, q2 = choice
            if tilted:
                result, _ = draw_intervention(units, keys, h, cfg['epsilon'], road.rng)
            else:
                result = conditional_actions(units, {}, road.rng)
            a, b = [result[k] for k in keys]
            p1, p2, *_ = proposal_factors(p, h, cfg['epsilon'])
            pf, tf = [float(p1[a]), float(p2[a, b])], [float(q1[a]), float(q2[a, b])]
            lp = natural_log_probability(units, result)
            conditional = lp-float(np.log(p[a, b]))
            lt = conditional+float(np.log(tf).sum())
            record = dict(actor_ids=keys, action_indices=[a, b], p_factors=pf, t_factors=tf,
                          full_log_p=lp, full_log_t=lt, cancelled_log_conditional=conditional,
                          log_t_over_p=float(np.log(np.array(tf)/pf).sum()),
                          proposal_table_sha256=identity({'p': p.tolist(), 'h': h.tolist()}))
        record.update(time_s=road.time, all_action_indices=result,
                      natural_unit_ids=[u['ids'] for u in units])
        steps.append(record)
        return result

    road.sample = later_sample
    while not road.crashes and road.time < end-1e-9:
        road.step_policy('natural', 1.)
    log_ratio = float(sum(r['log_t_over_p'] for r in steps))
    weight = path_weight(log_ratio, cfg['natural_fraction'])
    event = any(road.cav.id in pair for pair in road.crashes)
    return dict(event_result=event, bv_only_collision=bool(road.crashes) and not event,
                collision_pairs=road.crashes, elapsed_s=road.time-state['time_s'],
                initial_cancelled_log_conditional=initial_conditional,
                full_log_p=float(sum(r['full_log_p'] for r in steps)),
                full_log_t=float(sum(r['full_log_t'] for r in steps)),
                log_t_over_p=log_ratio, future_weight=weight,
                weighted_event=float(event)*weight, steps=steps)


def bounded_interval(values, bound, error):
    """Two-sided empirical Bernstein, Maurer/Pontil Thm 4, scaled range.

    Union both one-sided bounds (log(4/error)); valid for IID bounded roots.
    Zero empirical variance leaves a strictly positive uncertainty radius.
    """
    x = np.asarray(values, float)
    if (x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or bound <= 0
            or np.any(x < 0) or np.any(x > bound+1e-12) or not 0 < error < 1):
        raise ValueError('Invalid bounded observations')
    mean = float(x.mean()); variance = float(x.var(ddof=1)); log = np.log(4/error)
    radius = np.sqrt(2*variance*log/len(x))+7*bound*log/(3*(len(x)-1))
    # The known expectation is a native probability, <=1, even if Z can be 2.
    return mean, variance/len(x), [min(1., max(0., mean-radius)), min(1., mean+radius)]


def estimate(strata, rows, arm, error):
    j = variance = low = high = 0.; details = []
    bound = 1. if arm == 'natural' else 2.
    for h in strata:
        group = [r for r in rows if r['stratum_id'] == h['stratum_id']]
        if len(group) != h['draws'] or {r['within_stratum_draw'] for r in group} != set(range(h['draws'])):
            raise ValueError('Incomplete or duplicate measurement stratum')
        z = [float(r['natural']['event_result']) if arm == 'natural'
             else r['defensive']['weighted_event'] for r in group]
        mean, v, bounds = bounded_interval(z, bound, error)
        c = h['coefficient_mass']; j += c*mean; variance += c*c*v
        low += c*bounds[0]; high += c*bounds[1]
        details.append(dict(stratum_id=h['stratum_id'], draws=len(z), mean=mean,
                            sample_variance=v*len(z), probability_interval=bounds))
    return dict(J=j, estimator_variance=variance, bank_simultaneous_interval=[low, high],
                strata=details)


def inputs(config):
    cfg = read(config)
    required = dict(version=1, arms=['natural', 'defensive'], batches=2,
                    draws_per_state_per_arm=256, continuation_seed_starts=[5800000, 5900000],
                    schedule_seed=580017, horizon_s=4, workers=4, confidence=.95,
                    continuation=dict(natural_fraction=.5, epsilon=[.1, .1], radius_m=115,
                                      risk_power=4, proxy_step_s=1/3, clearance_scale_m=10.))
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Explicitly version changed continuation measurement settings')
    old_cfg, base, physical, assets, plans, designs, old_design, parent_lock = parent.inputs(cfg['parent_config'])
    root = Path(cfg['parent_path']); old = read(root/'summary.json')
    if (digest(root/'summary.json') != cfg['parent_summary_sha256'] or not old['complete']
            or old['config'] != old_cfg or any(old['lock'].get(k) != v for k, v in parent_lock.items())):
        raise ValueError('Completed parent training measurement changed')
    for name, sha in old['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Parent artifact differs')
    if read(root/'measurement_plan.json') != old_design:
        raise ValueError('Parent design differs')
    schedules = []; states = []
    for i, s in enumerate(old_design['states']):
        strata = [dict(h, draws=n) for h, n in zip(s['strata'], s['counts']['equal'])]
        states.append(dict(state_id=i, plan=plans[i], strata=strata))
    for batch, start in enumerate(cfg['continuation_seed_starts']):
        for i, s in enumerate(states):
            rows = []
            for h in s['strata']:
                rng = np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'], batch, i, h['stratum_id']]))
                cells = rng.choice(h['cells'], size=h['draws'], p=h['measurement_probabilities'])
                coins = rng.random(h['draws']) >= cfg['continuation']['natural_fraction']
                for n, (cell, coin) in enumerate(zip(cells, coins)):
                    actions = [int(a) for a in np.unravel_index(cell, np.array(designs[i]['P']).shape)]
                    rows.append(dict(stratum_id=h['stratum_id'], within_stratum_draw=n,
                                     native_seed=start+i*cfg['draws_per_state_per_arm']+len(rows),
                                     action_indices=actions, tilted_component=bool(coin),
                                     constant_moment_weight=h['coefficient_mass']))
            schedules.append(dict(batch=batch, state_id=i, rows=rows))
    design = dict(states=states, schedules=schedules, state_proposals=old_design['state_proposals'],
                  target_masses=[p['target_mass'] for p in plans],
                  logical_draws=cfg['batches']*len(states)*2*cfg['draws_per_state_per_arm'],
                  planned_native_paths_upper_bound=sum(1+int(r['tilted_component']) for s in schedules for r in s['rows']))
    deps = ['highd_v46_d2rl.py', 'highd_v46_pair_witness.py', 'highd_v46_cross_unit.py',
            'highd_v46_pair_synergy.py', 'highd_paper_discrete.py', 'highd_paper_discrete_pairs.py',
            'highd_paper_dynamics.py']
    environment = dict(python=__import__('sys').version.split()[0], numpy=np.__version__, scipy=scipy.__version__)
    if environment != dict(python='3.9.19', numpy='1.23.1', scipy='1.13.1'):
        raise ValueError('Use the inherited pinned environment')
    lock = dict(config_sha256=digest(config), implementation_sha256=digest(__file__),
                parent_lock=parent_lock, parent_summary_sha256=cfg['parent_summary_sha256'],
                design_sha256=identity(design), environment=environment,
                dependency_sha256={n: digest(Path(__file__).with_name(n)) for n in deps})
    return cfg, physical, assets, plans, designs, design, lock


def _init_worker(assets):
    global _MODEL
    _MODEL = parent.parent.bank.PairModelQueries(Path(assets)/'model', Path(assets)/'lateral',
                                                **parent.parent.bank.SINGLE_OPTIONS)


def _collect(task):
    schedule, state, plan, assets, physical, cfg = task
    rows = []
    for r in schedule['rows']:
        args = (_MODEL, Path(assets), physical, state, plan['actor_ids'], r['action_indices'],
                r['native_seed'], cfg['horizon_s'], cfg['continuation'])
        natural = continue_path(*args, tilted=False)
        # A first-decision endpoint has identical P/T law. Simulate it once,
        # even when the predeclared latent component was T.
        separate = r['tilted_component'] and bool(natural['steps'])
        proposed = continue_path(*args, tilted=True) if separate else natural
        rows.append(dict(r, natural=natural, defensive=proposed,
                         shared_physical_path=not separate))
    return dict(batch=schedule['batch'], state_id=schedule['state_id'], rows=rows)


def summarize(cfg, design, records):
    family = cfg['batches']*len(cfg['arms'])*sum(len(s['strata']) for s in design['states'])
    error = (1-cfg['confidence'])/family
    estimates = {}; paired = []
    for record in records:
        b, i = record['batch'], record['state_id']; hs = design['states'][i]['strata']
        for arm in cfg['arms']:
            estimates[(b, i, arm)] = estimate(hs, record['rows'], arm, error)
        diff = variance = 0.
        for h in hs:
            group = [r for r in record['rows'] if r['stratum_id'] == h['stratum_id']]
            dz = np.array([r['defensive']['weighted_event']-r['natural']['event_result'] for r in group])
            diff += h['coefficient_mass']*dz.mean()
            variance += h['coefficient_mass']**2*dz.var(ddof=1)/len(dz)
        paired.append(dict(batch=b, state_id=i, J_difference=float(diff),
                           difference_variance=float(variance)))
    alpha = np.array(design['target_masses']); arms = {}
    for arm in cfg['arms']:
        states = []
        for i, s in enumerate(design['states']):
            a, b = estimates[(0, i, arm)], estimates[(1, i, arm)]
            j = (a['J']+b['J'])/2; v = (a['estimator_variance']+b['estimator_variance'])/4
            bounds = ((np.array(a['bank_simultaneous_interval'])+b['bank_simultaneous_interval'])/2).tolist()
            groups = [r for x in records if x['state_id'] == i for r in x['rows']]
            key = 'natural' if arm == 'natural' else 'defensive'
            states.append(dict(state_id=i, source_seed=s['plan']['source_seed'], J=j,
                       estimator_variance=v, bank_simultaneous_interval=bounds,
                       batch_J=[a['J'], b['J']], relative_batch_gap=abs(a['J']-b['J'])/j if j>0 else None,
                       relative_empirical_MCSE=float(np.sqrt(v)/j) if j>0 else None,
                       events=sum(r[key]['event_result'] for r in groups),
                       precision_supported=bool(bounds[0]>0 and (bounds[1]-bounds[0])/2<=.5*j)))
        arms[arm] = dict(states=states, target_M2=float(alpha@np.array([s['J'] for s in states])),
                        target_M2_empirical_MCSE=float(np.sqrt((alpha*alpha)@np.array([s['estimator_variance'] for s in states]))),
                        state_proposal_comparisons={k: parent.compare_state_proposal(alpha, q, states)
                                                    for k, q in design['state_proposals'].items()})
    ve = sum(alpha[i]**2*arms['natural']['states'][i]['estimator_variance'] for i in range(len(alpha)))
    vp = sum(alpha[i]**2*arms['defensive']['states'][i]['estimator_variance'] for i in range(len(alpha)))
    return dict(arms=arms, state_batch_estimates=[dict(e, batch=b, state_id=i, arm=a)
                    for (b, i, a), e in sorted(estimates.items())], paired_differences=paired,
                target_measurement_variance_ratio_point_only=vp/ve if ve>0 else None)


def freeze(config, output):
    *_, design, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return dict(design_sha256=lock['design_sha256'], logical_draws=design['logical_draws'],
                planned_native_paths_upper_bound=design['planned_native_paths_upper_bound'])


def run(config, lock_path, output):
    cfg, physical, assets, plans, designs, design, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen continuation measurement changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', dict(config=cfg, lock=lock))
    save(out/'measurement_plan.json', design)
    tasks = [(s, designs[s['state_id']]['state'], plans[s['state_id']], str(assets), physical, cfg)
             for s in design['schedules']]
    records = []; started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=(str(assets),)) as pool:
        futures = [pool.submit(_collect, t) for t in tasks]
        for f in as_completed(futures):
            record = f.result(); records.append(record)
            save(out/f'batch_{record["batch"]}_state_{record["state_id"]:03d}.json', record)
            print(dict(completed_cases=len(records), total=len(tasks)), flush=True)
    records.sort(key=lambda r: (r['batch'], r['state_id']))
    result = dict(complete=True, config=cfg, lock=lock, **summarize(cfg, design, records),
                  logical_draws=design['logical_draws'],
                  planned_native_paths_upper_bound=design['planned_native_paths_upper_bound'],
                  native_paths_simulated=sum(1+int(not r['shared_physical_path']) for x in records for r in x['rows']),
                  total_wall_s=time.perf_counter()-started,
                  production_changed=False, ppo_ready=False, source_generalization_validated=False,
                  artifact_hashes={p.name: digest(p) for p in sorted(out.glob('*.json'))})
    save(out/'summary.json', result)
    return {a: dict(target_M2=r['target_M2'], target_M2_empirical_MCSE=r['target_M2_empirical_MCSE'])
            for a, r in result['arms'].items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run']); parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--lock'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(freeze(args.config, args.output) if args.mode == 'freeze' else run(args.config, args.lock, args.output))
