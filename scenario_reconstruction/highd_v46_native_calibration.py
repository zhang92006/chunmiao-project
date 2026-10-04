"""Training-only real-Q endpoint calibration for bounded state/pair discovery.

Mixed discovery and native cases are selected before continuation outcomes.
Only the native-yield ranker sees labels; no production proposal is changed.
"""
import argparse
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
import time

import numpy as np

from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_cross_unit import positive_interaction, proposal_factors
from .highd_v46_d2rl import NativeExperiment, experiment_target, identity
from .highd_v46_discovery_v2 import choose_pairs, choose_states
from .highd_v46_multistate_q import (
    compare_strata, dependencies, environment, prepare_case, sample_continuation, stratum_summary,
)
from .highd_v46_pair_synergy import pair_marginal, response_risk, snapshot
from .highd_v46_state_discovery import audit_gates, screen_pairs
from .highd_v46_statistics import episode_values


def validate_config(cfg, dc):
    source = cfg['source']
    seeds = set(range(source['seed_start'], source['seed_start'] + source['episodes']))
    if source['split'] != 'train' or not seeds or any(seeds & set(range(lo, hi + 1)) for lo, hi in cfg['forbidden_source_ranges']):
        raise ValueError('Calibration accepts only fresh training source episodes')
    arms = {'natural': [1., 1.], 'first_factor_only': [.1, 1.], 'second_factor_only': [1., .1], 'dual': [.1, .1]}
    if cfg['arms'] != arms or cfg['bank_method'] != 'mixed5_top2' or cfg['horizon_s'] != 4:
        raise ValueError('Unsupported fixed calibration experiment')
    if cfg['later_actions'] != 'Unchanged natural NDE; only initial decision is tilted':
        raise ValueError('Later dynamics must remain natural')
    if dc['pairs']['ranked'] != 2 or dc['pairs']['audit'] != 1 or [dc['bank'][k] for k in ('geometry_states', 'uniform_states', 'phase_states')] != [2, 2, 1]:
        raise ValueError('Keep five states and rank 1 + rank 2 + audit budgets')
    if (cfg['cases']['maximum_states_per_source'] != 2 or cfg['replicates_per_case_arm'] < 2
            or cfg['cases']['maximum_source_episodes'] < 1 or cfg['cases']['abstention_control_episodes'] < 1):
        raise ValueError('Invalid calibration budget')
    if cfg['ranker']['features'] != ['intercept', 'dual_Q_expected_proxy_risk', 'maximum_supported_proxy_risk', 'is_lateral']:
        raise ValueError('Unsupported ranker feature contract')
    if (cfg['ranker']['ridge'] <= 0 or cfg['ranker']['maximum_iterations'] < 1
            or min(cfg['ranker'][k] for k in ('minimum_dual_events', 'minimum_event_source_episodes', 'minimum_zero_event_source_episodes')) < 1):
        raise ValueError('Invalid ranker regularization or label guards')


def audit_state(state, model, physical, base, index, dc):
    cars = [SimpleNamespace(**car) for car in state['vehicles']]
    road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']), model=model, config=physical)
    units = [{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']]
    pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}
    rng = np.random.default_rng(np.random.SeedSequence([dc['pairs']['random_seed'], state['seed'], round(state['time_s'])]))
    rows = []
    for candidate in choose_pairs(state['screen_pairs'], 'mixed5_top2', rng):
        cpu, wall = time.process_time(), time.perf_counter()
        p, same = pair_marginal(units, candidate['actor_ids'])
        risk = response_risk(road, candidate['actor_ids'], pmfs, base['proxy'], reactive=True)
        audit = audit_gates(p, risk, base['selection'])
        score = p * positive_interaction(p, risk)
        h = score / score.sum() if score.sum() > 0 else p.copy()
        q1, q2 = proposal_factors(p, h, [.1, .1])[-2:]
        expected = float(np.sum(q1[:, None] * q2 * risk))
        maximum = float(risk[p > 0].max())
        rows.append({'state_index': index, 'seed': state['seed'], 'time_s': state['time_s'],
                     **candidate, 'same_natural_unit': same, **audit,
                     'ranker_features': [1., expected, maximum, float(candidate['mechanism'] == 'lateral')],
                     'proxy_cpu_s': time.process_time() - cpu, 'proxy_wall_s': time.perf_counter() - wall})
    return rows


def discover(cfg, dc, base, physical, model, assets):
    bank, rows, episodes = [], [], []
    feature_cpu = 0.
    start = time.perf_counter()

    class Observed(NativeExperiment):
        def choose(self, units, actors, pmfs, mode):
            nonlocal feature_cpu
            if mode != 'natural':
                raise ValueError('Only natural source rollouts permitted')
            tick = time.process_time()
            screened = screen_pairs(self, pmfs, base['screen'])
            feature_cpu += time.process_time() - tick
            state = snapshot(self, units, pmfs, seed, {'training_calibration_only': True})
            states.append(dict(state, screen_pairs=screened, screen_score=screened[0]['score'] if screened else 0.))
            return None

    cohort = cfg['source']
    for seed in range(cohort['seed_start'], cohort['seed_start'] + cohort['episodes']):
        states = []
        road = Observed(model, assets / 'pair', assets / 'lateral', seed, physical)
        while road.time < physical['horizon_s'] - 1e-9:
            if not road.step_policy('natural', 1.):
                break
        chosen = choose_states(states, 'mixed5_top2', dc['bank'], physical['horizon_s'], base['proxy']['horizon_s'],
                               np.random.default_rng(np.random.SeedSequence([dc['bank']['random_seed'], seed])))
        bank.extend(dict(states[i], discovery_role=chosen[i]) for i in sorted(chosen))
        event = 'cav_first_collision' if any(road.cav.id in pair for pair in road.crashes) else 'bv_only' if road.crashes else 'horizon'
        episodes.append({'seed': seed, 'natural_states': len(states), 'bank_states': len(chosen), 'event': event})
    scan_wall = time.perf_counter() - start
    for i, state in enumerate(bank):
        rows.extend(audit_state(state, model, physical, base, i, dc))
        if (i + 1) % 10 == 0 or i + 1 == len(bank):
            print({'phase': 'training_discovery', 'states': i + 1, 'total': len(bank), 'pairs': len(rows)}, flush=True)
    positives = [r for r in rows if r['views']['production_strict']]
    report = {'episodes': episodes, 'natural_states': sum(e['natural_states'] for e in episodes),
              'bank_states': len(bank), 'bank_roles': dict(Counter(s['discovery_role'] for s in bank)),
              'evaluated_pairs': len(rows), 'strict_positive_pairs': len(positives),
              'strict_positive_states': len({r['state_index'] for r in positives}),
              'strict_positive_source_episodes': len({r['seed'] for r in positives}),
              'strict_positive_mechanisms': dict(Counter(r['mechanism'] for r in positives)),
              'first_failure_counts': dict(Counter(r['first_failed_gate'] for r in rows)),
              'scan_wall_s': scan_wall, 'discovery_wall_s': time.perf_counter() - start,
              'feature_cpu_s': feature_cpu, 'proxy_cpu_s': sum(r['proxy_cpu_s'] for r in rows)}
    return bank, rows, report


def select_cases(bank, rows, cfg):
    eligible = {}
    for i, row in enumerate(rows):
        if row['views']['production_strict']:
            eligible.setdefault(row['seed'], []).append(i)
    rng = np.random.default_rng(cfg['selection_seed'])
    seeds = sorted(int(s) for s in rng.permutation(sorted(eligible))[:cfg['maximum_source_episodes']])
    cases = []
    for seed in seeds:
        first = max(eligible[seed], key=lambda i: rows[i]['metrics']['positive_interaction_mass'])
        cases.append({'row_index': first, 'state_index': rows[first]['state_index'], 'role': 'eligible', 'selection': 'best_mass'})
        remaining = [i for i in eligible[seed] if rows[i]['state_index'] != rows[first]['state_index']]
        if remaining:
            second = max(remaining, key=lambda i: (rows[i]['mechanism'] != rows[first]['mechanism'], rows[i]['metrics']['positive_interaction_mass']))
            cases.append({'row_index': second, 'state_index': rows[second]['state_index'], 'role': 'eligible', 'selection': 'different_state_mechanism'})
    for seed in sorted(int(s) for s in rng.permutation(sorted({s['seed'] for s in bank} - set(eligible)))[:cfg['abstention_control_episodes']]):
        indices = [i for i, state in enumerate(bank) if state['seed'] == seed]
        cases.append({'row_index': None, 'state_index': int(rng.choice(indices)), 'role': 'abstention_control', 'selection': 'uniform_control'})
    return cases


def case_masses(records):
    counts = Counter(r['source_seed'] for r in records)
    return np.asarray([1 / (len(counts) * counts[r['source_seed']]) for r in records])


def aggregate(records, values, stats):
    if not records:
        return {'cases': 0, 'precision_status': 'no_eligible_cases'}
    masses = case_masses(records)
    result = {}
    n = len(values['dual'][0])
    z = NormalDist().inv_cdf((1 + stats['confidence']) / 2)
    for arm, strata in values.items():
        if len(strata) != len(records) or any(len(x) != n for x in strata) or n < 2:
            raise ValueError('Strata do not match the fixed calibration distribution')
        mean = float(sum(a * x.mean() for a, x in zip(masses, strata)))
        second = float(sum(a * np.mean(x*x) for a, x in zip(masses, strata)))
        se = float(np.sqrt(sum(a*a * x.var(ddof=1) / n for a, x in zip(masses, strata))))
        events = int(sum(np.count_nonzero(x) for x in strata))
        sources = len({r['source_seed'] for r, x in zip(records, strata) if np.any(x > 0)})
        guard = events >= stats['minimum_events'] and sources >= stats['minimum_event_source_episodes']
        contributions = np.concatenate([a*x/n for a, x in zip(masses, strata)])
        rhw = z*se / mean if mean > 0 else None
        result[arm] = {'cases': len(records), 'source_episodes': len({r['source_seed'] for r in records}),
                      'continuations': n*len(records), 'raw_cav_events': events, 'event_source_episodes': sources,
                      'conditional_event_mean': mean, 'weighted_event_second_moment': second,
                      'stratified_standard_error': se, 'relative_half_width': rhw,
                      'event_contribution_ess': float(contributions.sum()**2 / (contributions @ contributions)) if mean > 0 else 0.,
                      'top_event_contribution_fraction': float(contributions.max() / contributions.sum()) if mean > 0 else None,
                      'evidence_guard_passed': guard, 'precision_status': 'insufficient_multisource_events' if not guard else
                          'precision_reached' if rhw <= stats['rhw_target'] else 'precision_not_reached'}
    # sqrt(K*alpha) preserves weighted second moments in the shared bootstrap.
    transformed = {a: [np.sqrt(len(records)*mass)*x for mass, x in zip(masses, strata)] for a, strata in values.items()}
    return {'arms': result, 'dual_comparisons': compare_strata(transformed, result, stats), 'case_masses': masses.tolist()}


def fit_binomial(features, events, trials, source_seeds, cfg):
    x, k, n = np.asarray(features, float), np.asarray(events, float), np.asarray(trials, float)
    if x.ndim != 2 or x.shape[1] != 4 or len(k) != len(x) or n.shape != k.shape or len(source_seeds) != len(k):
        raise ValueError('Invalid native-yield regression data')
    if not all(np.isfinite(a).all() for a in (x, k, n)) or np.any(k < 0) or np.any(k > n) or np.any(n <= 0) or k.sum() == 0 or np.all(k == n):
        raise ValueError('Need finite binomial observations with some events and negatives')
    counts = Counter(source_seeds)
    w = np.asarray([1 / counts[s] for s in source_seeds])
    beta = np.zeros(4)
    rate = float(np.sum(w*k) / np.sum(w*n))
    beta[0] = np.log(rate / (1-rate))
    penalty = np.diag([0., cfg['ridge'], cfg['ridge'], cfg['ridge']])
    loss = lambda b: float(np.sum(w*(n*np.logaddexp(0., x@b)-k*(x@b))) + .5*b@penalty@b)
    converged = False
    for iteration in range(cfg['maximum_iterations']):
        probability = np.exp(-np.logaddexp(0., -(x@beta)))
        gradient = x.T @ (w*(n*probability-k)) + penalty@beta
        hessian = x.T @ ((w*n*probability*(1-probability))[:, None]*x) + penalty + np.eye(4)*1e-10
        step = np.linalg.solve(hessian, gradient)
        scale = 1.
        before = loss(beta)
        for _ in range(24):
            if loss(beta-scale*step) <= before:
                break
            scale *= .5
        beta -= scale*step
        if np.max(np.abs(scale*step)) < 1e-8:
            converged = True
            break
    return {'features': cfg['features'], 'coefficients': beta.tolist(), 'ridge': cfg['ridge'],
            'iterations': iteration + 1, 'converged': converged, 'target': cfg['target'],
            'source_weighting': 'Each source episode has equal weight; cases split that source weight',
            'deployment_status': 'training_only_requires_fresh_validation'}


def predict(model, features):
    return np.exp(-np.logaddexp(0., -(np.asarray(features) @ np.asarray(model['coefficients']))))


def ranker_report(records, cfg):
    eligible = [r for r in records if r['role'] == 'eligible']
    sources = {r['source_seed'] for r in eligible}
    event_sources = {r['source_seed'] for r in eligible if r['arms']['dual']['positive_episodes'] > 0}
    count = sum(r['arms']['dual']['positive_episodes'] for r in eligible)
    guard = count >= cfg['minimum_dual_events'] and len(event_sources) >= cfg['minimum_event_source_episodes'] and len(sources-event_sources) >= cfg['minimum_zero_event_source_episodes']
    report = {'dual_events': count, 'event_source_episodes': len(event_sources), 'zero_event_source_episodes': len(sources-event_sources),
              'fit_guard_passed': guard, 'model': None, 'leave_source_out_predictions': [],
              'scope': 'Training calibration/replay only, within preselected cases. No deployment or fresh-validation success.'}
    if not guard:
        report['status'] = 'insufficient_native_training_labels'
        return report
    arguments = lambda group: ([r['ranker_features'] for r in group], [r['arms']['dual']['positive_episodes'] for r in group],
                              [r['arms']['dual']['episodes'] for r in group], [r['source_seed'] for r in group], cfg)
    model = fit_binomial(*arguments(eligible))
    if not model['converged']:
        report['status'] = 'numerical_fit_incomplete'
        return report
    report.update(model=model, status='fitted_training_only')
    for seed in sorted(sources):
        training = [r for r in eligible if r['source_seed'] != seed]
        checking = [r for r in eligible if r['source_seed'] == seed]
        assert not {r['source_seed'] for r in checking}.intersection(r['source_seed'] for r in training)
        fold = fit_binomial(*arguments(training))
        if not fold['converged']:
            report['leave_source_out_predictions'].append({'source_seed': seed, 'status': 'numerical_fit_incomplete'})
            continue
        for record, probability in zip(checking, predict(fold, [r['ranker_features'] for r in checking])):
            report['leave_source_out_predictions'].append({'source_seed': seed, 'state_index': record['state_index'],
                'predicted_dual_event_rate': float(probability), 'observed_dual_events': record['arms']['dual']['positive_episodes'],
                'trials': record['arms']['dual']['episodes'], 'training_source_seeds': sorted({r['source_seed'] for r in training})})
    return report


def selection_replay(records, report):
    eligible = [r for r in records if r['role'] == 'eligible']
    groups = {s: [r for r in eligible if r['source_seed'] == s] for s in sorted({r['source_seed'] for r in eligible})}
    scores = {(r['source_seed'], r['state_index']): r['predicted_dual_event_rate']
              for r in report['leave_source_out_predictions'] if 'predicted_dual_event_rate' in r}
    output = {}
    methods = ['interaction_mass', 'dual_expected_proxy_risk', 'maximum_proxy_risk'] + (['native_leave_source_out'] if scores else [])
    for name in methods:
        selected = []
        for seed, group in groups.items():
            if name == 'native_leave_source_out' and any((seed, r['state_index']) not in scores for r in group):
                continue
            score = lambda r: r['mass'] if name == 'interaction_mass' else r['ranker_features'][1] if name == 'dual_expected_proxy_risk' else r['ranker_features'][2] if name == 'maximum_proxy_risk' else scores[(seed, r['state_index'])]
            selected.append(max(group, key=score))
        output[name] = {'selected_cases': [{'source_seed': r['source_seed'], 'time_s': r['time_s'], 'state_index': r['state_index']} for r in selected],
                        'raw_dual_events': sum(r['arms']['dual']['positive_episodes'] for r in selected),
                        'event_source_episodes': len({r['source_seed'] for r in selected if r['arms']['dual']['positive_episodes']}),
                        'trials': sum(r['arms']['dual']['episodes'] for r in selected),
                        'scope': 'Training-side selection replay among at most two preselected cases per source; not whole-bank recall or validation'}
    return output


def run(config_path, output):
    cfg = read(config_path)
    dc = read(cfg['discovery_config'])
    base = read(dc['base_discovery_config'])
    physical = read(base['physical_config'])
    validate_config(cfg, dc)
    if environment() != cfg['environment'] or base['proxy']['horizon_s'] != cfg['horizon_s']:
        raise ValueError('Environment or horizon differs from predeclared protocol')
    deps = dependencies(cfg, dc, base)
    deps['highd_v46_multistate_q.py'] = digest(Path(__file__).with_name('highd_v46_multistate_q.py'))
    implementation = digest(__file__)
    assets, frozen = ensure_assets()
    model = PairModelQueries(assets / 'model', assets / 'lateral', **SINGLE_OPTIONS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    bank, rows, discovery = discover(cfg, dc, base, physical, model, assets)
    save(output / 'state_bank.json', {'split': 'train', 'states': bank, 'natural_target_sha256': frozen['natural_target_sha256']})
    save(output / 'pair_audit.json', {'rows': rows})
    plans, chosen_cases = [], []
    for case in select_cases(bank, rows, cfg['cases']):
        state = bank[case['state_index']]
        row = rows[case['row_index']] if case['row_index'] is not None else None
        chosen = prepare_case(state, row, model, physical, base)
        chosen_cases.append(chosen)
        plans.append({**case, 'source_seed': state['seed'], 'time_s': state['time_s'], 'state_sha256': identity(state),
                      'ranker_features': row['ranker_features'] if row else None,
                      'selected_pair': chosen['ids'] if chosen else None,
                      **({k: chosen[k] for k in ('mass', 'factor_tv', 'q_second_tv', 'mechanism')} if chosen else {})})
    save(output / 'case_plan.json', {'selection_uses_native_outcomes': False, 'conditional_law': cfg['conditional_law'], 'cases': plans,
         'eligible_case_masses': case_masses([p for p in plans if p['role'] == 'eligible']).tolist()})
    print({'phase': 'case_plan', 'cases': plans}, flush=True)
    records, values = [], {a: [] for a in cfg['arms']}
    artifacts = ['state_bank.json', 'pair_audit.json', 'case_plan.json']
    n = cfg['replicates_per_case_arm']
    for i, (plan, chosen) in enumerate(zip(plans, chosen_cases)):
        arms = {}
        for arm, epsilon in cfg['arms'].items():
            sequences = [sample_continuation(model, assets, physical, bank[plan['state_index']], chosen,
                epsilon, cfg['continuation_seed_start']+i*n+j, cfg, base['proxy']['clearance_scale_m']) for j in range(n)]
            name = f'case_{i:02d}_{arm}.json'
            save(output / name, {'conditional_case': plan, 'epsilon': epsilon, 'sequences': sequences})
            artifacts.append(name)
            arms[arm] = stratum_summary(sequences, cfg['statistics'])
            if plan['role'] == 'eligible':
                values[arm].append(episode_values(sequences))
            print({'phase': 'native_calibration', 'case': i, 'total': len(plans), 'source': plan['source_seed'],
                   'arm': arm, 'cav_events': arms[arm]['positive_episodes'], 'bv_only': arms[arm]['bv_only_events']}, flush=True)
        records.append({**plan, 'arms': arms, 'first_minus_dual_second_moment_per_cpu_descriptive':
            (arms['first_factor_only']['weighted_second_moment']-arms['dual']['weighted_second_moment']) /
            max(arms['first_factor_only']['cpu_s']+arms['dual']['cpu_s'], 1e-12)})
    ranker = ranker_report(records, cfg['ranker'])
    save(output / 'native_calibration.json', {'records': records})
    save(output / 'ranker_training_report.json', ranker)
    artifacts += ['native_calibration.json', 'ranker_training_report.json']
    if ranker['model'] is not None:
        save(output / 'candidate_ranker.json', {**ranker['model'], 'natural_target_sha256': frozen['natural_target_sha256'],
             'implementation_sha256': implementation, 'config_identity': identity(cfg), 'dependencies': deps})
        artifacts.append('candidate_ranker.json')
    result = {'complete': True, 'split': 'train', 'config': cfg, 'config_identity': identity(cfg), 'config_sha256': digest(config_path),
              'implementation_sha256': implementation, 'dependencies': deps, 'environment': environment(),
              'natural_target_sha256': frozen['natural_target_sha256'], 'experiment_target_sha256': identity(experiment_target(frozen, physical)),
              'discovery': discovery, 'cases': records, 'primary_conditional_comparison': aggregate([r for r in records if r['role'] == 'eligible'], values, cfg['statistics']),
              'ranker': ranker, 'training_selection_replay': selection_replay(records, ranker),
              'total_wall_s': time.perf_counter()-start, 'artifacts': {name: digest(output / name) for name in artifacts},
              'scope': cfg['scope'], 'fresh_validation_performed': False, 'population_acceleration': None, 'ppo_trained': False}
    save(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v46_native_calibration_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({'ranker_status': result['ranker']['status'], 'primary': result['primary_conditional_comparison'], 'total_wall_s': result['total_wall_s']})


if __name__ == '__main__':
    main()
