"""Coefficient action coverage and uncertainty-aware fixed moment budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import beta

from . import highd_v47_state_moments as parent
from .highd_v46_assets import read, save, digest
from .highd_v46_d2rl import identity
from .highd_v46_pair_witness import continue_state

DEFAULT = 'configs/highd_v47_state_moments_v2.json'


def integer_budget(scores, budget, minimum=2, floors=None):
    """Exact integer minimizer of sum(score_h/n_h), with a support floor."""
    scores = np.asarray(scores, float)
    if (scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores))
            or np.any(scores < 0) or minimum < 2 or int(minimum) != minimum
            or int(budget) != budget or budget < minimum*len(scores)):
        raise ValueError('Invalid measurement budget')
    raw = np.full(len(scores), minimum, dtype=int) if floors is None else np.asarray(floors)
    if (raw.shape != scores.shape or not np.all(np.isfinite(raw)) or np.any(raw < minimum)
            or np.any(raw != np.floor(raw)) or raw.sum() > budget):
        raise ValueError('Invalid per-stratum floors')
    counts = raw.astype(int).copy()
    heap = [(-float(s/(counts[i]*(counts[i]+1))), i) for i, s in enumerate(scores)]
    heapq.heapify(heap)
    for _ in range(int(budget)-int(counts.sum())):
        _, i = heapq.heappop(heap); counts[i] += 1
        heapq.heappush(heap, (-float(scores[i]/(counts[i]*(counts[i]+1))), i))
    return counts.tolist()


def bernoulli_variance_range(lower, upper):
    if not 0 <= lower <= upper <= 1:
        raise ValueError('Invalid probability range')
    return min(lower*(1-lower), upper*(1-upper)), (min(max(.5, lower), upper))*(1-min(max(.5, lower), upper))


def partition(p, qa, risk, historical_events, cfg):
    p, qa = np.asarray(p, float), np.asarray(qa, float)
    c = np.divide(p*p, qa, out=np.zeros_like(p), where=p > 0)
    # Parent computes P*(P/Q), checks support/underflow, and normalizes G.
    c[p > 0] = p[p > 0]*(p[p > 0]/qa[p > 0])
    mass = float(c.sum())
    heads = set(np.flatnonzero(c.ravel() >= cfg['coefficient_singleton_fraction']*mass))
    history = {int(a) for a in historical_events if c.ravel()[a] >= cfg['historical_singleton_fraction']*mass}
    strata = parent.action_strata(p, qa, risk, [.2, .4, .6, .8], [-6, -3, -1, 0],
                                 cfg['draws_per_state_per_arm'], sorted(heads | history))
    for h in strata:
        if len(h['cells']) == 1 and h['cells'][0] in heads:
            h['kind'] = 'coefficient_action'
        elif len(h['cells']) == 1 and h['cells'][0] in history:
            h['kind'] = 'historical_event_action'
    return strata


def planning_ranges(strata, counts, events, cfg):
    a0, b0 = cfg['planning_beta_prior']; error = 1-cfg['planning_probability_level']
    a, b = a0+events, b0+counts-events
    lower = beta.ppf(error/2, a, b); upper = beta.ppf(1-error/2, a, b)
    result = []
    for h in strata:
        cells = h['cells']; g = np.array(h['measurement_probabilities'])
        lo, hi = float(g@lower[cells]), float(g@upper[cells])
        _, vu = bernoulli_variance_range(lo, hi)
        result.append({'probability_range': [lo, hi], 'variance_upper': vu,
                       'objective_coefficient': h['coefficient_mass']**2*vu})
    return result


def training_action_counts(cfg, old_cfg, designs):
    """All prior training rows fit planning scores only; repeated QA CRN draws deduplicate."""
    rows = {}; total = 0
    def add(i, seed, actions, event, kernel):
        nonlocal total
        cell = int(np.ravel_multi_index(actions, np.array(designs[i]['P']).shape))
        key = (kernel, i, seed); value = (cell, bool(event))
        if key in rows and rows[key] != value:
            raise ValueError('Conflicting historical physical draw')
        rows[key] = value; total += 1
    bank_cfg = read(old_cfg['bank_config']); updated_cfg = read(old_cfg['updated_config'])
    for i in range(len(designs)):
        for r in read(Path(bank_cfg['teacher_path'])/f'case_{i:03d}_labels.json')['sequences']:
            add(i, r['seed'], r['initial_likelihood_audit']['action_indices'], r['event_result'], 'QA')
    for root in (Path(updated_cfg['development_path']), Path(old_cfg['completed_update_path'])):
        for arm in ('target_iid', 'allocated_iid', 'equal_state_stratified'):
            for r in read(root/f'{arm}_sequences.json')['sequences']:
                add(r['state_id'], r['seed'], r['initial_likelihood_audit']['action_indices'], r['event_result'], 'QA')
    for i in range(len(designs)):
        data = read(Path(cfg['parent_path'])/f'state_{i:03d}.json')
        for r in data['direct']:
            add(i, r['seed'], r['initial_likelihood_audit']['action_indices'], r['event_result'], 'QA')
        for r in data['probes']:
            add(i, r['native_seed'], r['action_indices'], r['event_result'], 'conditional_probe')
    counts = [np.zeros(np.array(d['P']).size, int) for d in designs]
    events = [np.zeros_like(c) for c in counts]
    for (_, i, _), (cell, event) in rows.items():
        counts[i][cell] += 1; events[i][cell] += event
    return counts, events, {'raw_rows': total, 'unique_physical_rows': len(rows),
                            'duplicates_not_counted': total-len(rows)}


def make_schedules(states, designs, cfg):
    schedules = []; seed_ends = []
    for batch, seed_start in enumerate(cfg['continuation_seed_starts']):
        seed = seed_start
        for s in states:
            probes = []; p = np.array(designs[s['state_id']]['P']); qa = np.array(s['target_QA'])
            for j, h in enumerate(s['strata']):
                rng = np.random.default_rng(np.random.SeedSequence([cfg['action_schedule_seed'], batch, s['state_id'], h['stratum_id']]))
                n = max(s['counts'][arm][j] for arm in cfg['arms'])
                selected = rng.choice(h['cells'], size=n, p=h['measurement_probabilities'])
                for draw, cell in enumerate(selected):
                    a, b = np.unravel_index(cell, p.shape)
                    g = h['measurement_probabilities'][h['cells'].index(int(cell))]
                    probes.append({'stratum_id': h['stratum_id'], 'within_stratum_draw': draw,
                        'native_seed': seed, 'action_indices': [int(a), int(b)],
                        'P_action': float(p[a, b]), 'target_QA_action': float(qa[a, b]),
                        'measurement_G_conditional': g, 'constant_moment_weight': h['coefficient_mass'],
                        'arms': [arm for arm in cfg['arms'] if draw < s['counts'][arm][j]]})
                    seed += 1
            schedules.append({'batch': batch, 'state_id': s['state_id'], 'probes': probes})
        seed_ends.append(seed)
    if any(end >= following for end, following in zip(seed_ends, cfg['continuation_seed_starts'][1:])):
        raise ValueError('Overlapping continuation schedules')
    return schedules


def inputs(config):
    cfg = read(config)
    if (cfg['version'] != 2 or cfg['draws_per_state_per_arm'] != 256 or cfg['minimum_stratum_draws'] != 2
            or cfg['equal_budget_floor_fraction'] != .5 or cfg['protect_observed_event_strata'] is not True
            or cfg['coefficient_singleton_fraction'] != .001 or cfg['historical_singleton_fraction'] != .000001
            or cfg['planning_beta_prior'] != [.5, .5] or cfg['planning_probability_level'] != .95
            or cfg['confidence'] != .95 or cfg['batches'] != 2 or cfg['arms'] != ['equal', 'uncertainty_allocated']
            or cfg['continuation_seed_starts'] != [5300000, 5400000] or cfg['action_schedule_seed'] != 570017
            or cfg['epsilon'] != [.1, .1] or cfg['horizon_s'] != 4 or cfg['workers'] != 4):
        raise ValueError('Version the measurement protocol before changing it')
    old_cfg, base, physical, assets, plans, designs, old_designs, proposals, parent_lock = parent.inputs(cfg['parent_config'])
    root = Path(cfg['parent_path']); complete = read(root/'summary.json')
    if digest(root/'summary.json') != cfg['parent_summary_sha256'] or not complete['complete']:
        raise ValueError('Incomplete or changed training measurement')
    if complete['config'] != old_cfg or any(complete['lock'].get(k) != v for k, v in parent_lock.items()):
        raise ValueError('Parent measurement lock mismatch')
    for name, sha in complete['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Parent measurement artifact changed')
    if read(root/'measurement_plan.json') != {'states': old_designs, 'state_proposals': proposals}:
        raise ValueError('Parent design changed')
    counts, events, history = training_action_counts(cfg, old_cfg, designs)
    states = []
    for i, (d, old) in enumerate(zip(designs, old_designs)):
        historical = np.flatnonzero(events[i] > 0).tolist()
        strata = partition(d['P'], old['target_QA'], d['risk'], historical, cfg)
        ranges = planning_ranges(strata, counts[i], events[i], cfg)
        equal = [h['draws'] for h in strata]
        protected = [bool(events[i][h['cells']].sum()) for h in strata]
        floors = [n if active else max(cfg['minimum_stratum_draws'], int(np.ceil(cfg['equal_budget_floor_fraction']*n)))
                  for n, active in zip(equal, protected)]
        allocated = integer_budget([r['objective_coefficient'] for r in ranges], cfg['draws_per_state_per_arm'],
                                   cfg['minimum_stratum_draws'], floors)
        states.append({'state_id': i, 'plan': plans[i], 'strata': strata, 'planning': ranges,
                       'allocation_floors': floors, 'historically_event_bearing': protected,
                       'counts': {'equal': equal, 'uncertainty_allocated': allocated},
                       'target_QA': old['target_QA'], 'observed_training_action_counts': counts[i].tolist(),
                       'observed_training_action_events': events[i].tolist()})
    schedules = make_schedules(states, designs, cfg)
    design = {'states': states, 'schedules': schedules, 'state_proposals': proposals,
              'logical_draws': cfg['batches']*len(states)*len(cfg['arms'])*cfg['draws_per_state_per_arm'],
              'unique_native_draws': sum(len(s['probes']) for s in schedules)}
    lock = {'config_sha256': digest(config), 'implementation_sha256': digest(__file__),
            'parent_lock': parent_lock, 'parent_summary_sha256': cfg['parent_summary_sha256'],
            'design_sha256': identity(design), 'history': history}
    return cfg, base, physical, assets, plans, designs, design, lock


def paired_variance(strata, counts, probes, error):
    """Same Bernoulli observations estimate both designs' variance and covariance."""
    ve = vu = covariance = difference_lo = difference_hi = 0.
    for j, h in enumerate(strata):
        rows = [r for r in probes if r['stratum_id'] == h['stratum_id']]
        n = len(rows); k = int(sum(r['event_result'] for r in rows))
        phat = k/n; sample_variance = phat*(1-phat)*n/(n-1)
        ne, nu = counts['equal'][j], counts['uncertainty_allocated'][j]
        c2 = h['coefficient_mass']**2
        ve += c2*sample_variance/ne; vu += c2*sample_variance/nu
        covariance += c2*sample_variance/max(ne, nu)
        lower, upper = parent.binomial_bounds(k, n, error)
        vl, vh = bernoulli_variance_range(lower, upper)
        coefficient = c2*(1/nu-1/ne)
        difference_lo += coefficient*(vl if coefficient >= 0 else vh)
        difference_hi += coefficient*(vh if coefficient >= 0 else vl)
    tolerance = 64*np.finfo(float).eps*max(abs(difference_lo), abs(difference_hi), np.finfo(float).tiny)
    return {'equal_variance_same_probes': ve, 'allocated_variance_same_probes': vu,
            'paired_difference_variance': max(0., ve+vu-2*covariance),
            'variance_ratio_point_only': vu/ve if ve > 0 else None,
            'variance_difference_interval': [difference_lo, difference_hi],
            'certified_variance_improvement': bool(difference_hi < -tolerance)}


def compare_state_proposal(alpha, q, estimates):
    """Stable signed moment bounds, including exact identity for q=alpha."""
    alpha, q = np.asarray(alpha, float), np.asarray(q, float)
    if (alpha.shape != q.shape or alpha.ndim != 1 or len(estimates) != len(alpha)
            or not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(q))
            or np.any(alpha <= 0) or np.any(q <= 0) or not np.isclose(alpha.sum(), 1) or not np.isclose(q.sum(), 1)):
        raise ValueError('Invalid state distributions')
    j = np.array([e['J'] for e in estimates]); v = np.array([e['estimator_variance'] for e in estimates])
    bounds = np.array([e['bank_simultaneous_interval'] for e in estimates])
    if (bounds.shape != (len(alpha), 2) or not np.all(np.isfinite(bounds)) or np.any(bounds < 0)
            or np.any(bounds[:, 0] > bounds[:, 1]) or not np.all(np.isfinite(j))
            or not np.all(np.isfinite(v)) or np.any(j < 0) or np.any(v < 0)):
        raise ValueError('Invalid state moment estimates')
    d = alpha*(alpha/q-1)
    lo = float(np.sum(np.where(d >= 0, d*bounds[:, 0], d*bounds[:, 1])))
    hi = float(np.sum(np.where(d >= 0, d*bounds[:, 1], d*bounds[:, 0])))
    tolerance = 64*np.finfo(float).eps*max(abs(lo), abs(hi), np.finfo(float).tiny)
    target = float(alpha@j)
    return {'target_M2': target, 'candidate_M2': float(target+d@j),
            'M2_ratio_point_only': float((target+d@j)/target) if target > 0 else None,
            'difference_empirical_MCSE': float(np.sqrt((d*d)@v)),
            'M2_difference_interval': [lo, hi], 'certified_improvement': bool(hi < -tolerance)}


def _init_worker(assets):
    global _MODEL
    _MODEL = parent.bank.PairModelQueries(Path(assets)/'model', Path(assets)/'lateral', **parent.bank.SINGLE_OPTIONS)


def _collect(task):
    schedule, state, plan, assets, physical, horizon, scale = task
    rows = []
    for probe in schedule['probes']:
        native = continue_state(_MODEL, Path(assets), physical, state,
            dict(zip(plan['actor_ids'], probe['action_indices'])), probe['native_seed'], horizon, scale)
        rows.append(dict(probe, native=native, event_result=bool(native['cav_first_collision'])))
    return dict(schedule, probes=rows)


def summarize(cfg, plans, design, records):
    total_strata = sum(len(s['strata']) for s in design['states'])
    family = cfg['batches']*len(cfg['arms'])*total_strata
    estimates = {}; paired = {}
    for record in records:
        batch, i = record['batch'], record['state_id']; s = design['states'][i]
        for arm in cfg['arms']:
            hs = [dict(h, draws=n) for h, n in zip(s['strata'], s['counts'][arm])]
            rows = [r for r in record['probes'] if arm in r['arms']]
            estimate = parent.estimate_state(hs, rows, cfg['confidence'], family)
            estimates[(batch, i, arm)] = dict(estimate, batch=batch, state_id=i, arm=arm, source_seed=plans[i]['source_seed'])
        paired[(batch, i)] = dict(paired_variance(s['strata'], s['counts'], record['probes'], (1-cfg['confidence'])/(cfg['batches']*total_strata)), batch=batch, state_id=i)
    arms = {}
    alpha = np.array([p['target_mass'] for p in plans])
    for arm in cfg['arms']:
        means = []; stability = []
        for i in range(len(plans)):
            a, b = estimates[(0, i, arm)], estimates[(1, i, arm)]
            mean = {'J': (a['J']+b['J'])/2, 'estimator_variance': (a['estimator_variance']+b['estimator_variance'])/4,
                    'bank_simultaneous_interval': ((np.array(a['bank_simultaneous_interval'])+b['bank_simultaneous_interval'])/2).tolist()}
            means.append(mean)
            stability.append({'state_id': i, 'source_seed': plans[i]['source_seed'], **mean,
                              'batch_J': [a['J'], b['J']],
                              'relative_empirical_MCSE': float(np.sqrt(mean['estimator_variance'])/mean['J']) if mean['J'] > 0 else None,
                              'relative_batch_gap': abs(a['J']-b['J'])/mean['J'] if mean['J'] > 0 else None,
                              'events': a['events']+b['events']})
        arms[arm] = {'states': stability,
                     'target_M2': float(alpha@np.array([m['J'] for m in means])),
                     'target_M2_empirical_MCSE': float(np.sqrt((alpha*alpha)@np.array([m['estimator_variance'] for m in means]))),
                     'state_proposal_comparisons': {name: compare_state_proposal(alpha, q, means) for name, q in design['state_proposals'].items()}}
    return {'state_estimates': [estimates[k] for k in sorted(estimates)],
            'paired_variance': [paired[k] for k in sorted(paired)], 'arms': arms}


def freeze(config, output):
    *_, design, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return {'design_sha256': lock['design_sha256'], 'logical_draws': design['logical_draws'], 'unique_native_draws': design['unique_native_draws']}


def run(config, lock_path, output):
    cfg, base, physical, assets, plans, designs, design, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Measurement lock changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', {'config': cfg, 'lock': lock})
    save(out/'measurement_plan.json', design)
    tasks = [(s, designs[s['state_id']]['state'], plans[s['state_id']], str(assets), physical,
              cfg['horizon_s'], base['proxy']['clearance_scale_m']) for s in design['schedules']]
    records = []; started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=(str(assets),)) as pool:
        futures = [pool.submit(_collect, t) for t in tasks]
        for future in as_completed(futures):
            result = future.result(); records.append(result)
            save(out/f'batch_{result["batch"]}_state_{result["state_id"]:03d}.json', result)
            print({'completed_cases': len(records), 'total': len(tasks)}, flush=True)
    records.sort(key=lambda r: (r['batch'], r['state_id']))
    result = dict(summarize(cfg, plans, design, records), complete=True, config=cfg, lock=lock,
                  logical_draws=design['logical_draws'], unique_native_draws=design['unique_native_draws'],
                  total_wall_s=time.perf_counter()-started, ppo_ready=False, production_changed=False,
                  source_generalization_validated=False,
                  artifact_hashes={p.name: digest(p) for p in sorted(out.glob('*.json'))})
    save(out/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run']); parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--lock'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode == 'freeze': print(freeze(args.config, args.output))
    else:
        result = run(args.config, args.lock, args.output)
        print({arm: {k: r[k] for k in ('target_M2', 'target_M2_empirical_MCSE')} for arm, r in result['arms'].items()})
