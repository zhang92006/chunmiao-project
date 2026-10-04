"""Real Q samples on a locked multi-state conditional distribution.

The state bank is an offline, episode-wide discovery artifact, not a causal
online selector. No initial-density correction or population manifest exists.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import os
from pathlib import Path
import platform
from statistics import NormalDist
from types import SimpleNamespace
import time

import numpy as np

from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_cross_unit import draw_intervention, natural_log_probability, positive_interaction, proposal_factors
from .highd_v46_d2rl import NativeExperiment, experiment_target, identity
from .highd_v46_discovery_v2 import choose_pairs, choose_states
from .highd_v46_pair_synergy import pair_marginal, response_risk, snapshot
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state
from .highd_v46_state_discovery import audit_gates, screen_pairs
from .highd_v46_statistics import episode_values, summarize_values


def environment():
    return {'python': platform.python_version(), 'numpy': version('numpy'), 'scipy': version('scipy'),
            **{key: int(os.environ.get(key, '0')) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}}


def dependencies(cfg, dc, base):
    names = ('highd_v46_assets.py', 'highd_v46_discovery_v2.py', 'highd_v46_state_discovery.py',
             'highd_v46_pair_synergy.py', 'highd_v46_cross_unit.py', 'highd_v46_pair_witness.py',
             'highd_v46_d2rl.py', 'highd_v46_statistics.py', 'highd_paper_discrete_pairs.py')
    result = {name: digest(Path(__file__).with_name(name)) for name in names}
    result.update(discovery_config=digest(cfg['discovery_config']),
                  base_config=digest(dc['base_discovery_config']), physical_config=digest(base['physical_config']))
    return result


def check_protocol(cfg, split, implementation, deps, lock=None):
    sets = {key: set(range(cfg[key]['seed_start'], cfg[key]['seed_start'] + cfg[key]['episodes']))
            for key in ('train', 'validation')}
    if not all(sets.values()) or sets['train'] & sets['validation']:
        raise ValueError('Source episodes must be nonempty and disjoint')
    if any(sets['validation'] & set(range(lo, hi + 1)) for lo, hi in cfg['forbidden_validation_ranges']):
        raise ValueError('Validation overlaps consumed or reserved source episodes')
    if (sets['train'] | sets['validation']) & set(range(100000, 101024)):
        raise ValueError('Reserved final-test episodes cannot be opened')
    expected = {'natural': [1., 1.], 'first_factor_only': [.1, 1.],
                'second_factor_only': [1., .1], 'dual': [.1, .1]}
    if (cfg['arms'] != expected or cfg['discovery_method'] != 'geometry5_top2' or cfg['horizon_s'] != 4
            or cfg['later_actions'] != 'Unchanged natural NDE; only initial decision is tilted'):
        raise ValueError('Unsupported fixed four-arm protocol')
    if any(cfg[k]['replicates_per_state_arm'] < 2 for k in sets):
        raise ValueError('Need at least two continuations per stratum')
    ranges = [set(range(cfg[k]['continuation_seed_start'], cfg[k]['continuation_seed_start']
                        + sum(cfg['cases'][f] for f in ('maximum_positive_source_episodes', 'negative_control_source_episodes'))
                        * cfg[k]['replicates_per_state_arm'])) for k in sets]
    if ranges[0] & ranges[1] or min(cfg['cases'][f] for f in ('maximum_positive_source_episodes', 'negative_control_source_episodes')) < 1:
        raise ValueError('Invalid continuation split or case budget')
    if split == 'validation':
        if not lock or not lock.get('complete') or lock.get('split') != 'train':
            raise ValueError('Validation requires a completed training lock')
        for field, value in [('config_identity', identity(cfg)), ('implementation_sha256', implementation),
                             ('dependencies', deps), ('environment', cfg['environment'])]:
            if lock[field] != value:
                raise ValueError('Protocol changed after training: ' + field)


def audit_state(state, model, physical, base, bank_index, dc):
    cars = [SimpleNamespace(**car) for car in state['vehicles']]
    road = SimpleNamespace(vehicles=cars, cav=next(car for car in cars if car.id == state['cav_id']),
                           model=model, config=physical)
    units = [{'ids': unit['ids'], 'p': np.asarray(unit['p'])} for unit in state['units']]
    pmfs = {int(key): np.asarray(p) for key, p in state['pmfs'].items()}
    rng = np.random.default_rng(np.random.SeedSequence([dc['pairs']['random_seed'], state['seed'], round(state['time_s'])]))
    rows = []
    for candidate in choose_pairs(state['screen_pairs'], 'geometry5_top2', rng):
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        p, same = pair_marginal(units, candidate['actor_ids'])
        risk = response_risk(road, candidate['actor_ids'], pmfs, base['proxy'], reactive=True)
        rows.append({'state_index': bank_index, 'seed': state['seed'], 'time_s': state['time_s'],
                     **candidate, 'same_natural_unit': same, **audit_gates(p, risk, base['selection']),
                     'proxy_cpu_s': time.process_time() - start_cpu,
                     'proxy_wall_s': time.perf_counter() - start_wall})
    return rows


def discover(cfg, dc, base, physical, model, assets, natural_target):
    """Only the frozen geometry5 arm; at most three expensive pairs per state."""
    bank, episodes, rows = [], [], []
    feature_cpu = 0.
    start = time.perf_counter()

    class Observed(NativeExperiment):
        def choose(self, units, actors, pmfs, mode):
            nonlocal feature_cpu
            if mode != 'natural':
                raise ValueError('Discovery source must remain natural')
            tick = time.process_time()
            screened = screen_pairs(self, pmfs, base['screen'])
            feature_cpu += time.process_time() - tick
            state = snapshot(self, units, pmfs, seed, {'conditional_discovery_only': True})
            states.append(dict(state, screen_pairs=screened, screen_score=screened[0]['score'] if screened else 0.))
            return None

    for seed in range(cfg['validation']['seed_start'], cfg['validation']['seed_start'] + cfg['validation']['episodes']):
        states = []
        road = Observed(model, assets / 'pair', assets / 'lateral', seed, physical)
        while road.time < physical['horizon_s'] - 1e-9:
            if not road.step_policy('natural', 1.):
                break
        selected = choose_states(states, 'geometry5_top2', dc['bank'], physical['horizon_s'], base['proxy']['horizon_s'],
                                 np.random.default_rng(np.random.SeedSequence([dc['bank']['random_seed'], seed])))
        bank.extend(states[i] for i in sorted(selected))
        event = 'cav_first_collision' if any(road.cav.id in pair for pair in road.crashes) else 'bv_only' if road.crashes else 'horizon'
        episodes.append({'seed': seed, 'natural_states': len(states), 'candidate_states': len(selected), 'event': event})
    scan_wall = time.perf_counter() - start
    for i, state in enumerate(bank):
        rows.extend(audit_state(state, model, physical, base, i, dc))
        if (i + 1) % 10 == 0 or i + 1 == len(bank):
            print({'phase': 'fresh_discovery', 'states': i + 1, 'total': len(bank), 'pairs': len(rows)}, flush=True)
    positives = [row for row in rows if row['views']['production_strict']]
    report = {'source_episodes': episodes, 'natural_states': sum(e['natural_states'] for e in episodes),
              'candidate_states': len(bank), 'evaluated_pairs': len(rows), 'strict_positive_pairs': len(positives),
              'strict_positive_states': len({r['state_index'] for r in positives}),
              'positive_source_episodes': len({r['seed'] for r in positives}),
              'first_failure_counts': dict(Counter(r['first_failed_gate'] for r in rows)),
              'feature_cpu_s': feature_cpu, 'proxy_cpu_s': sum(r['proxy_cpu_s'] for r in rows),
              'scan_wall_s': scan_wall, 'discovery_wall_s': time.perf_counter() - start}
    return bank, rows, report


def training_source(cfg, natural_target, dc, deps):
    path = Path(cfg['training_source']['directory'])
    if digest(path / 'summary.json') != cfg['training_source']['summary_sha256']:
        raise ValueError('Checksum-locked training discovery summary differs')
    summary = read(path / 'summary.json')
    if not summary['complete'] or summary['split'] != 'train' or summary['natural_target_sha256'] != natural_target:
        raise ValueError('Expected frozen-P training-only discovery source')
    if summary['config'] != dc or summary['implementation_sha256'] != deps['highd_v46_discovery_v2.py']:
        raise ValueError('Discovery config or implementation differs from training source')
    for key, value in summary['dependencies'].items():
        if deps.get(key) != value:
            raise ValueError('Discovery training dependency differs: ' + key)
    for name, expected in summary['artifacts'].items():
        if digest(path / name) != expected:
            raise ValueError('Discovery training artifact differs: ' + name)
    if [e['seed'] for e in summary['source_episodes']] != list(range(cfg['train']['seed_start'], cfg['train']['seed_start'] + cfg['train']['episodes'])):
        raise ValueError('Wrong training source episodes')
    bank = read(path / 'state_bank.json')['states']
    rows = [r for r in read(path / 'pair_audit.json')['rows'] if cfg['discovery_method'] in r['methods']]
    return bank, rows, {'source_summary_sha256': digest(path / 'summary.json'),
                        'source_episodes': summary['source_episodes'],
                        'method_report': summary['methods'][cfg['discovery_method']]}


def select_cases(bank, rows, cfg):
    """No native labels: one highest proxy-mass eligible pair per source episode."""
    best = {}
    for i, row in enumerate(rows):
        if row['views']['production_strict'] and (row['seed'] not in best or
                row['metrics']['positive_interaction_mass'] > rows[best[row['seed']]]['metrics']['positive_interaction_mass']):
            best[row['seed']] = i
    rng = np.random.default_rng(cfg['selection_seed'])
    positive_seeds = sorted(int(s) for s in rng.permutation(sorted(best))[:cfg['maximum_positive_source_episodes']])
    cases = [{'state_index': rows[best[seed]]['state_index'], 'row_index': best[seed], 'role': 'eligible'} for seed in positive_seeds]
    # Controls come only from episodes with no eligible pair in the whole bank.
    all_seeds = {r['seed'] for r in rows} | {s['seed'] for s in bank if 'methods' not in s or 'geometry5_top2' in s['methods']}
    control_seeds = sorted(int(s) for s in rng.permutation(sorted(all_seeds - set(best)))[:cfg['negative_control_source_episodes']])
    for seed in control_seeds:
        indices = [i for i, state in enumerate(bank) if state['seed'] == seed
                   and ('methods' not in state or 'geometry5_top2' in state['methods'])]
        cases.append({'state_index': int(rng.choice(indices)), 'row_index': None, 'role': 'abstention_control'})
    return cases


def prepare_case(state, row, model, physical, base):
    if row is None:
        return None
    cars = [SimpleNamespace(**car) for car in state['vehicles']]
    road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']), model=model, config=physical)
    units = [{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']]
    p, _ = pair_marginal(units, row['actor_ids'])
    risk = response_risk(road, row['actor_ids'], {int(k): np.asarray(v) for k, v in state['pmfs'].items()}, base['proxy'], reactive=True)
    audit = audit_gates(p, risk, base['selection'])
    if any(identity(audit[key]) != identity(row[key]) for key in audit) or not audit['views']['production_strict']:
        raise ValueError('Selected pair no longer reproduces its locked proxy audit')
    score = p * positive_interaction(p, risk)
    h = score / score.sum()
    _, p2, _, _, q1, q2 = proposal_factors(p, h, [.1, .1])
    return {'ids': row['actor_ids'], 'h': h, 'factor_tv': audit['actual_H_tv'],
            'q_second_tv': float(np.sum(q1 * (.5 * np.abs(p2 - q2).sum(1)))),
            'mass': float(score.sum()), 'mechanism': row['mechanism']}


def sample_continuation(model, assets, physical, state, chosen, epsilon, seed, cfg, scale):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    road = restore_prepared_state(model, assets, physical, state, seed)
    units = [{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']]
    if chosen is None:
        actions = conditional_actions(units, {}, road.rng)
        log_p = natural_log_probability(units, actions)
        logged = {'log_weight': 0., 'full_log_p': log_p, 'full_log_q': log_p, 'abstained': True}
    else:
        actions, logged = draw_intervention(units, chosen['ids'], chosen['h'], epsilon, road.rng)
    pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}
    # Snapshot is already prepared: no second history update at the initial step.
    road.action_pmfs = lambda: pmfs
    road.sample = lambda *args: actions
    road.step_policy('natural', 1.)
    del road.action_pmfs
    del road.sample
    while not road.crashes and road.time < state['time_s'] + cfg['horizon_s'] - 1e-9:
        road.step_policy('natural', 1.)
    event = any(road.cav.id in pair for pair in road.crashes)
    return {'seed': seed, 'event_result': event, 'bv_only_collision': bool(road.crashes) and not event,
            'generation_log_weight': logged['log_weight'], 'initial_likelihood_audit': logged,
            'collision_pairs': road.crashes, 'clearance_risk': float(np.exp(-max(road.minimum_gap, 0.) / scale)),
            'cpu_s': time.process_time() - start_cpu, 'wall_s': time.perf_counter() - start_wall}


def stratum_summary(sequences, stats):
    values = episode_values(sequences)
    weights = np.exp([s['generation_log_weight'] for s in sequences])
    result = summarize_values(values, **{k: stats[k] for k in ('confidence', 'rhw_target', 'minimum_events')})
    result.update(mean_likelihood_ratio=float(weights.mean()), weight_mean_mcse=float(weights.std(ddof=1) / np.sqrt(len(weights))),
                  maximum_weight=float(weights.max()), weight_ess=float(weights.sum()**2 / (weights @ weights)),
                  top_event_contribution_fraction=float(values.max() / values.sum()) if values.sum() else None,
                  bv_only_events=sum(s['bv_only_collision'] for s in sequences),
                  mean_clearance_risk=float(np.mean([s['clearance_risk'] for s in sequences])),
                  cpu_s=sum(s['cpu_s'] for s in sequences), wall_s=sum(s['wall_s'] for s in sequences),
                  maximum_full_vs_factor_log_weight_error=max(abs(s['initial_likelihood_audit']['full_log_p']
                       - s['initial_likelihood_audit']['full_log_q'] - s['generation_log_weight']) for s in sequences))
    return result


def mixture_summary(strata, stats):
    """Equal state mass, fixed stratification; not pooled iid standard errors."""
    if not strata:
        return {'states': 0, 'precision_status': 'no_eligible_states'}
    n, k = strata[0].size, len(strata)
    if any(x.ndim != 1 or len(x) != n or not np.isfinite(x).all() or np.any(x < 0) for x in strata) or n < 2:
        raise ValueError('Need equal-size finite nonnegative strata with >=2 draws')
    mean = float(np.mean([x.mean() for x in strata]))
    second = float(np.mean([np.mean(x*x) for x in strata]))
    within = float(np.mean([x.var(ddof=1) for x in strata]))
    se = float(np.sqrt(within / (k*n)))
    z = NormalDist().inv_cdf((1 + stats['confidence']) / 2)
    count = sum(np.count_nonzero(x) for x in strata)
    event_states = sum(np.any(x > 0) for x in strata)
    guard = count >= stats['minimum_events'] and event_states >= stats['minimum_event_source_episodes']
    rhw = z*se / mean if mean > 0 else None
    pooled = np.concatenate(strata)
    return {'states': k, 'replicates_per_state': n, 'continuations': k*n, 'raw_cav_events': int(count),
            'event_source_episodes': int(event_states), 'raw_event_fraction': count / (k*n),
            'conditional_event_mean': mean, 'weighted_event_second_moment': second,
            'within_state_sample_variance_mean': within, 'stratified_estimator_variance': se*se,
            'stratified_standard_error': se, 'confidence_interval_normal': [mean-z*se, mean+z*se] if mean > 0 else None,
            'relative_half_width': rhw, 'evidence_guard_passed': bool(guard),
            'precision_status': 'insufficient_multistate_events' if not guard else
                'precision_reached' if rhw <= stats['rhw_target'] else 'precision_not_reached',
            'event_contribution_ess': float(pooled.sum()**2 / (pooled @ pooled)) if second > 0 else 0.,
            'top_event_contribution_fraction': float(pooled.max() / pooled.sum()) if mean > 0 else None}


def compare_strata(values, summaries, stats):
    comparisons = {}
    if not values['dual']:
        return comparisons
    rng = np.random.default_rng(stats['bootstrap_seed'])
    dual = np.asarray(values['dual'])
    for arm in ('natural', 'first_factor_only', 'second_factor_only'):
        reference = np.asarray(values[arm])
        ratios, differences = [], []
        for _ in range(stats['bootstrap_repeats']):
            indices = rng.integers(dual.shape[1], size=dual.shape)
            d = np.mean(np.take_along_axis(dual, indices, axis=1)**2)
            r = np.mean(np.take_along_axis(reference, indices, axis=1)**2)
            differences.append(float(d-r))
            if r > 0:
                ratios.append(float(d/r))
        guard = summaries['dual']['evidence_guard_passed'] and summaries[arm]['evidence_guard_passed']
        m2 = summaries[arm]['weighted_event_second_moment']
        q = [(1-stats['confidence'])/2, (1+stats['confidence'])/2]
        interval = np.quantile(differences, q).tolist() if guard and differences else None
        comparisons[arm] = {'dual_over_reference_second_moment_descriptive': summaries['dual']['weighted_event_second_moment'] / m2 if m2 > 0 else None,
            'guard_passed_both_arms': guard, 'paired_bootstrap_repeats': stats['bootstrap_repeats'],
            'zero_reference_second_moment_repeats': stats['bootstrap_repeats'] - len(ratios),
            'dual_minus_reference_second_moment_interval': interval,
            'second_moment_ratio_interval': np.quantile(ratios, q).tolist() if guard and len(ratios) == len(differences) and ratios else None,
            'lower_dual_second_moment_supported': bool(interval is not None and interval[1] < 0),
            'scope': 'Paired continuation resampling within fixed states; no source-population or original-initialization inference'}
    return comparisons


def run(config_path, split, output, training_lock=None):
    cfg = read(config_path)
    dc = read(cfg['discovery_config'])
    base = read(dc['base_discovery_config'])
    physical = read(base['physical_config'])
    deps, implementation = dependencies(cfg, dc, base), digest(__file__)
    lock = read(training_lock) if training_lock else None
    check_protocol(cfg, split, implementation, deps, lock)
    if environment() != cfg['environment']:
        raise ValueError('Runtime differs from the predeclared environment')
    if base['proxy']['horizon_s'] != cfg['horizon_s']:
        raise ValueError('Proxy and conditional continuation horizons differ')
    if lock is not None:
        for name, expected in lock['artifacts'].items():
            if digest(Path(training_lock).parent / name) != expected:
                raise ValueError('Completed training artifact differs: ' + name)
    assets, frozen = ensure_assets()
    natural_target = frozen['natural_target_sha256']
    target_hash = identity(experiment_target(frozen, physical))
    if lock and (lock['natural_target_sha256'] != natural_target or lock['experiment_target_sha256'] != target_hash):
        raise ValueError('Frozen P or physical event target changed after training')
    model = PairModelQueries(assets / 'model', assets / 'lateral', **SINGLE_OPTIONS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if split == 'train':
        bank, rows, discovery = training_source(cfg, natural_target, dc, deps)
    else:
        bank, rows, discovery = discover(cfg, dc, base, physical, model, assets, natural_target)
    save(output / 'discovery_bank.json', {'states': bank, 'natural_target_sha256': natural_target})
    save(output / 'discovery_pairs.json', {'rows': rows})
    cases = select_cases(bank, rows, cfg['cases'])
    plans = []
    prepared = []
    for case in cases:
        state = bank[case['state_index']]
        row = rows[case['row_index']] if case['row_index'] is not None else None
        chosen = prepare_case(state, row, model, physical, base)
        prepared.append(chosen)
        plans.append({**case, 'source_seed': state['seed'], 'time_s': state['time_s'],
                      'state_sha256': identity(state), 'selected_pair': chosen['ids'] if chosen else None,
                      **({key: chosen[key] for key in ('mass', 'factor_tv', 'q_second_tv', 'mechanism')} if chosen else {})})
    if len({p['source_seed'] for p in plans}) != len(plans):
        raise ValueError('Conditional cases must be source-episode disjoint')
    # Declare mu before seeing any continuation outcome. Eligible-only is primary.
    save(output / 'conditional_distribution.json', {'cases': plans, 'law': cfg['cases']['law'],
         'primary_state_mass': 1 / len([p for p in plans if p['role'] == 'eligible']) if any(p['role'] == 'eligible' for p in plans) else None,
         'secondary_state_mass': 1 / len(plans) if plans else None,
         'selection_uses_native_outcomes': False})
    print({'split': split, 'selected_cases': plans}, flush=True)
    all_values = {arm: [] for arm in cfg['arms']}
    eligible_values = {arm: [] for arm in cfg['arms']}
    records, artifacts = [], ['discovery_bank.json', 'discovery_pairs.json', 'conditional_distribution.json']
    n = cfg[split]['replicates_per_state_arm']
    for i, (case, chosen) in enumerate(zip(cases, prepared)):
        arm_reports = {}
        for arm, epsilon in cfg['arms'].items():
            sequences = [sample_continuation(model, assets, physical, bank[case['state_index']], chosen,
                epsilon, cfg[split]['continuation_seed_start'] + i*n + j, cfg, base['proxy']['clearance_scale_m']) for j in range(n)]
            name = f'case_{i:02d}_{arm}.json'
            save(output / name, {'conditional_case': plans[i], 'epsilon': epsilon, 'sequences': sequences})
            artifacts.append(name)
            arm_reports[arm] = stratum_summary(sequences, cfg['statistics'])
            values = episode_values(sequences)
            all_values[arm].append(values)
            if case['role'] == 'eligible':
                eligible_values[arm].append(values)
            print({'split': split, 'case': i, 'source_seed': plans[i]['source_seed'], 'arm': arm,
                   'cav_events': arm_reports[arm]['positive_episodes'], 'second_moment': arm_reports[arm]['weighted_second_moment'],
                   'rhw': arm_reports[arm]['relative_half_width']}, flush=True)
        records.append({**plans[i], 'arms': arm_reports})
    groups = {}
    for name, values in [('eligible_only_primary', eligible_values), ('all_cases_secondary', all_values)]:
        summaries = {arm: mixture_summary(strata, cfg['statistics']) for arm, strata in values.items()}
        groups[name] = {'arms': summaries, 'dual_comparisons': compare_strata(values, summaries, cfg['statistics'])}
    result = {'complete': True, 'split': split, 'config': cfg, 'config_identity': identity(cfg),
              'config_sha256': digest(config_path), 'implementation_sha256': implementation, 'dependencies': deps,
              'environment': environment(), 'natural_target_sha256': natural_target, 'experiment_target_sha256': target_hash,
              'discovery': discovery, 'cases': records, 'groups': groups,
              'arm_cpu_s': {a: sum(c['arms'][a]['cpu_s'] for c in records) for a in cfg['arms']},
              'arm_wall_s': {a: sum(c['arms'][a]['wall_s'] for c in records) for a in cfg['arms']},
              'total_wall_s': time.perf_counter() - start, 'artifacts': {name: digest(output / name) for name in artifacts},
              'training_lock_sha256': digest(training_lock) if training_lock else None,
              'scope': cfg['scope'], 'single_factor_scope': 'Factor ablations on the same selected ordered pair, not the paper single-BV agent',
              'population_acceleration': None, 'checkpoint_exported': False}
    save(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v46_multistate_q_v1.json')
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--training-lock')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.config, args.split, args.output, args.training_lock)
    print({'split': result['split'], 'groups': result['groups'], 'total_wall_s': result['total_wall_s']})


if __name__ == '__main__':
    main()
