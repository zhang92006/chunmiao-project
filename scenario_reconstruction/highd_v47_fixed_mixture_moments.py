"""Fixed-component balance-heuristic measurement of the locked state moments.

Only the measurement design changes. P_future, T_future, teacher Q_A and the
conditional target are inherited unchanged. Never treat fixed components as IID.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from . import highd_v47_continuation_moments as parent
from .highd_v46_assets import digest, read, save
from .highd_v46_d2rl import identity

DEFAULT = 'configs/highd_v47_fixed_mixture_moments_v1.json'
ARMS = ['natural', 'iid_defensive', 'fixed_defensive']


def even_counts(number, requested=256):
    """Equal even counts, >=2 per component, without dropping a stratum."""
    if int(number) != number or number < 1 or int(requested) != requested or requested < 4:
        raise ValueError('Invalid fixed-component budget')
    budget = max(2*((requested+1)//2), 4*number)
    pairs, extra = divmod(budget//2, number)
    return [2*(pairs+int(i < extra)) for i in range(number)]


def mean_interval(x, error, bound=2.):
    """Two-sided scaled empirical Bernstein for a component mean, not a probability."""
    x = np.asarray(x, float)
    if (x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or bound <= 0
            or np.any(x < 0) or np.any(x > bound+1e-12) or not 0 < error < 1):
        raise ValueError('Invalid bounded component observations')
    mean, s2 = float(x.mean()), float(x.var(ddof=1))
    log = np.log(4/error)
    radius = np.sqrt(2*s2*log/len(x))+7*bound*log/(3*(len(x)-1))
    return mean, s2, [max(0., mean-radius), min(bound, mean+radius)]


def fixed_statistics(p_values, t_values, error):
    """Independent component strata, each with the same c/C initial-action law.

    P/G balance weights occur once. Component means can exceed one; only their
    equally weighted sum is a native probability. Negative finite unbiased
    estimates of the removed variance term are retained, not clipped.
    """
    if len(p_values) != len(t_values) or len(p_values) < 2:
        raise ValueError('Need equal component counts >=2')
    mp, sp, ip = mean_interval(p_values, error)
    mt, st, it = mean_interval(t_values, error)
    nc = len(p_values); n = 2*nc
    variance = .25*(sp+st)/nc
    delta2 = (mp-mt)**2-sp/nc-st/nc
    removed = delta2/(4*n)
    delta = [ip[0]-it[1], ip[1]-it[0]]
    square_lower = 0. if delta[0] <= 0 <= delta[1] else min(d*d for d in delta)
    square_upper = max(d*d for d in delta)
    return dict(mean=.5*(mp+mt), estimator_variance=variance,
                probability_interval=[min(1., .5*(ip[0]+it[0])), min(1., .5*(ip[1]+it[1]))],
                component_means=[mp, mt], component_sample_variances=[sp, st],
                component_mean_intervals=[ip, it],
                removed_between_component_variance_unbiased=removed,
                removed_between_component_variance_interval=[square_lower/(4*n), square_upper/(4*n)],
                iid_variance_inferred_same_draws_unbiased=variance+removed)


def estimate(strata, rows, arm, error):
    j = variance = low = high = removed = inferred = 0.
    removed_bounds = np.zeros(2); details = []
    for h in strata:
        group = [r for r in rows if r['stratum_id'] == h['stratum_id']]
        n = h['draws']
        if len(group) != n or {r['within_stratum_draw'] for r in group} != set(range(n)):
            raise ValueError('Incomplete/duplicate fixed-mixture stratum')
        if arm == 'fixed_defensive':
            pz = [r[arm]['weighted_event'] for r in group if not r['fixed_tilted_component']]
            tz = [r[arm]['weighted_event'] for r in group if r['fixed_tilted_component']]
            e = fixed_statistics(pz, tz, error)
            removed += h['coefficient_mass']**2*e['removed_between_component_variance_unbiased']
            inferred += h['coefficient_mass']**2*e['iid_variance_inferred_same_draws_unbiased']
            removed_bounds += h['coefficient_mass']**2*np.array(e['removed_between_component_variance_interval'])
        elif arm in ARMS[:2]:
            x = [float(r[arm]['event_result']) if arm == 'natural' else r[arm]['weighted_event'] for r in group]
            mean, s2, bounds = mean_interval(x, error, 1. if arm == 'natural' else 2.)
            e = dict(mean=mean, estimator_variance=s2/n,
                     probability_interval=[min(1., b) for b in bounds], sample_variance=s2)
        else:
            raise ValueError('Unknown measurement arm')
        c = h['coefficient_mass']; j += c*e['mean']; variance += c*c*e['estimator_variance']
        low += c*e['probability_interval'][0]; high += c*e['probability_interval'][1]
        details.append(dict(e, stratum_id=h['stratum_id'], draws=n))
    result = dict(J=float(j), estimator_variance=float(variance), bank_simultaneous_interval=[low, high], strata=details)
    if arm == 'fixed_defensive':
        result.update(removed_between_component_variance_unbiased=float(removed),
                      removed_between_component_variance_interval=removed_bounds.tolist(),
                      iid_variance_inferred_same_draws_unbiased=float(inferred))
    return result


def make_design(old, action_designs, cfg):
    states = []
    for s in old['states']:
        counts = even_counts(len(s['strata']), cfg['minimum_draws_per_state_per_arm'])
        strata = [dict(h, draws=n) for h, n in zip(s['strata'], counts)]
        states.append(dict(s, strata=strata, draws=sum(counts)))
    schedules = []
    for batch, start in enumerate(cfg['continuation_seed_starts']):
        offset = 0
        for i, s in enumerate(states):
            rows = []
            for h in s['strata']:
                rng = np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'], batch, i, h['stratum_id']]))
                n = h['draws']; cells = rng.choice(h['cells'], size=n, p=h['measurement_probabilities'])
                iid_coins = rng.random(n) >= .5
                fixed_coins = rng.permutation(np.array([False]*(n//2)+[True]*(n//2)))
                for k, (cell, iid, fixed) in enumerate(zip(cells, iid_coins, fixed_coins)):
                    a = [int(x) for x in np.unravel_index(cell, np.array(action_designs[i]['P']).shape)]
                    rows.append(dict(stratum_id=h['stratum_id'], within_stratum_draw=k,
                                     native_seed=start+offset+len(rows), action_indices=a,
                                     iid_tilted_component=bool(iid), fixed_tilted_component=bool(fixed),
                                     constant_moment_weight=h['coefficient_mass']))
            schedules.append(dict(batch=batch, state_id=i, rows=rows)); offset += len(rows)
    if len({r['native_seed'] for s in schedules for r in s['rows']}) != sum(len(s['rows']) for s in schedules):
        raise ValueError('Overlapping native schedules')
    return dict(states=states, schedules=schedules, state_proposals=old['state_proposals'],
                target_masses=old['target_masses'],
                logical_draws=len(ARMS)*sum(len(s['rows']) for s in schedules),
                planned_native_paths_upper_bound=sum(1+int(r['iid_tilted_component'] or r['fixed_tilted_component'])
                                                     for s in schedules for r in s['rows']))


def inputs(config):
    cfg = read(config)
    required = dict(version=1, arms=ARMS, batches=2, minimum_draws_per_state_per_arm=256,
                    continuation_seed_starts=[6000000, 6100000], schedule_seed=600017,
                    workers=4, confidence=.95)
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Explicitly version changed fixed-mixture settings')
    old_cfg, physical, assets, plans, action_designs, old_design, old_lock = parent.inputs(cfg['parent_config'])
    root = Path(cfg['parent_path']); summary = read(root/'summary.json')
    if (digest(root/'summary.json') != cfg['parent_summary_sha256'] or not summary['complete']
            or summary['config'] != old_cfg or any(summary['lock'].get(k) != v for k, v in old_lock.items())):
        raise ValueError('Completed parent measurement changed')
    for name, sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Parent artifact changed')
    if read(root/'measurement_plan.json') != old_design:
        raise ValueError('Parent design changed')
    design = make_design(old_design, action_designs, cfg)
    cfg = dict(cfg, horizon_s=old_cfg['horizon_s'], continuation=old_cfg['continuation'])
    lock = dict(config_sha256=digest(config), implementation_sha256=digest(__file__),
                parent_lock=old_lock, parent_summary_sha256=digest(root/'summary.json'),
                design_sha256=identity(design))
    return cfg, physical, assets, plans, action_designs, design, lock


def _collect(task):
    schedule, state, plan, assets, physical, cfg = task; rows = []
    for r in schedule['rows']:
        args = (parent._MODEL, Path(assets), physical, state, plan['actor_ids'], r['action_indices'],
                r['native_seed'], cfg['horizon_s'], cfg['continuation'])
        natural = parent.continue_path(*args, tilted=False)
        separate = (r['iid_tilted_component'] or r['fixed_tilted_component']) and bool(natural['steps'])
        tilted = parent.continue_path(*args, tilted=True) if separate else natural
        rows.append(dict(r, natural=natural,
                         iid_defensive=tilted if r['iid_tilted_component'] else natural,
                         fixed_defensive=tilted if r['fixed_tilted_component'] else natural,
                         simulated_tilted_path=bool(separate)))
    return dict(batch=schedule['batch'], state_id=schedule['state_id'], rows=rows)


def summarize(cfg, design, records):
    # Natural + IID + the two fixed components, unioned before summing intervals.
    family = cfg['batches']*4*sum(len(s['strata']) for s in design['states'])
    error = (1-cfg['confidence'])/family
    estimates = {}; paired = []
    for record in records:
        b, i = record['batch'], record['state_id']; hs = design['states'][i]['strata']
        for arm in ARMS:
            estimates[(b, i, arm)] = estimate(hs, record['rows'], arm, error)
        diff = v = 0.
        for h in hs:
            g = [r for r in record['rows'] if r['stratum_id'] == h['stratum_id']]
            ds = [np.array([r['fixed_defensive']['weighted_event']-r['iid_defensive']['weighted_event']
                            for r in g if r['fixed_tilted_component'] == t]) for t in (False, True)]
            c = h['coefficient_mass']
            diff += c*.5*sum(x.mean() for x in ds)
            v += c*c*.25*sum(x.var(ddof=1)/len(x) for x in ds)
        paired.append(dict(batch=b, state_id=i, fixed_minus_iid_J=float(diff), difference_variance=float(v)))
    alpha = np.array(design['target_masses']); arms = {}
    for arm in ARMS:
        states = []
        for i, s in enumerate(design['states']):
            e = [estimates[(b, i, arm)] for b in range(2)]
            j = sum(x['J'] for x in e)/2; v = sum(x['estimator_variance'] for x in e)/4
            bounds = np.mean([x['bank_simultaneous_interval'] for x in e], axis=0).tolist()
            rows = [r for rec in records if rec['state_id'] == i for r in rec['rows']]
            states.append(dict(state_id=i, source_seed=s['plan']['source_seed'], J=j,
                          estimator_variance=v, batch_J=[x['J'] for x in e],
                          bank_simultaneous_interval=bounds,
                          relative_batch_gap=abs(e[0]['J']-e[1]['J'])/j if j>0 else None,
                          relative_empirical_MCSE=float(np.sqrt(v)/j) if j>0 else None,
                          events=sum(r[arm]['event_result'] for r in rows),
                          precision_supported=bool(bounds[0]>0 and (bounds[1]-bounds[0])/2<=.5*j)))
        arms[arm] = dict(states=states, target_M2=float(alpha@np.array([s['J'] for s in states])),
                        target_measurement_variance=float((alpha*alpha)@np.array([s['estimator_variance'] for s in states])),
                        state_proposal_comparisons={k: parent.parent.compare_state_proposal(alpha, q, states)
                                                    for k, q in design['state_proposals'].items()})
        arms[arm]['target_M2_empirical_MCSE'] = float(np.sqrt(arms[arm]['target_measurement_variance']))
    fixed_v = arms['fixed_defensive']['target_measurement_variance']
    removed = sum(.25*alpha[i]**2*estimates[(b, i, 'fixed_defensive')]['removed_between_component_variance_unbiased']
                  for b in range(2) for i in range(len(alpha)))
    removed_bounds = [sum(.25*alpha[i]**2*estimates[(b, i, 'fixed_defensive')]['removed_between_component_variance_interval'][j]
                         for b in range(2) for i in range(len(alpha))) for j in (0, 1)]
    inferred = fixed_v+removed
    return dict(arms=arms, state_batch_estimates=[dict(e, batch=b, state_id=i, arm=a)
                                                for (b, i, a), e in sorted(estimates.items())],
                paired_differences=paired,
                fixed_over_iid_variance_point_only=fixed_v/arms['iid_defensive']['target_measurement_variance']
                    if arms['iid_defensive']['target_measurement_variance']>0 else None,
                same_draw_structural_comparison=dict(fixed_variance=fixed_v,
                    removed_between_component_variance_unbiased=float(removed),
                    removed_between_component_variance_interval=removed_bounds,
                    iid_variance_inferred_unbiased=float(inferred),
                    fixed_over_inferred_iid_point_only=fixed_v/inferred if inferred>0 else None,
                    strictly_positive_removed_variance_certified=bool(removed_bounds[0]>0)))


def freeze(config, output):
    *_, design, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return {k: design[k] for k in ('logical_draws', 'planned_native_paths_upper_bound')}


def run(config, lock_path, output):
    cfg, physical, assets, plans, action_designs, design, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen fixed-mixture design changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', dict(config=cfg, lock=lock)); save(out/'measurement_plan.json', design)
    tasks = [(s, action_designs[s['state_id']]['state'], plans[s['state_id']], str(assets), physical, cfg)
             for s in design['schedules']]
    records = []; started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=parent._init_worker, initargs=(str(assets),)) as pool:
        futures = [pool.submit(_collect, t) for t in tasks]
        for f in as_completed(futures):
            record = f.result(); records.append(record)
            save(out/f'batch_{record["batch"]}_state_{record["state_id"]:03d}.json', record)
            print(dict(completed_cases=len(records), total=len(tasks)), flush=True)
    records.sort(key=lambda r: (r['batch'], r['state_id']))
    result = dict(complete=True, config=cfg, lock=lock, **summarize(cfg, design, records),
                  logical_draws=design['logical_draws'], planned_native_paths_upper_bound=design['planned_native_paths_upper_bound'],
                  native_paths_simulated=sum(1+int(r['simulated_tilted_path']) for x in records for r in x['rows']),
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
