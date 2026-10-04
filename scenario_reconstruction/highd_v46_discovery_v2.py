"""Locked mixed-state discovery with bounded pair screening and fresh validation.

The bank uses episode-wide ranking, not an online causal selection policy.
Shadow Q draws audit conditional activation; natural rollouts never use them.
"""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_v46_d2rl import NativeExperiment, identity
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_pair_synergy import pair_marginal, response_risk, snapshot
from .highd_v46_pair_witness import continue_state
from .highd_v46_cross_unit import draw_intervention, positive_interaction, proposal_factors
from .highd_v46_state_discovery import audit_candidates, audit_gates
from .highd_v46_state_discovery import screen_pairs


def choose_states(states, method, cfg, horizon, proxy_horizon, rng):
    eligible = [i for i, s in enumerate(states) if s['time_s'] + proxy_horizon <= horizon]
    order = sorted(eligible, key=lambda i: (-states[i]['screen_score'], states[i]['time_s']))
    count = cfg['geometry_control_states'] if method == 'geometry5_top2' else cfg['geometry_states']
    chosen = {}
    for i in order:
        if all(abs(states[i]['time_s'] - states[j]['time_s']) >= cfg['minimum_geometry_separation_s'] for j in chosen):
            chosen[i] = 'geometry'
        if len(chosen) == count:
            break
    if method != 'mixed5_top2':
        return chosen
    remaining = [i for i in eligible if i not in chosen]
    for i in rng.choice(remaining, min(cfg['uniform_states'], len(remaining)), replace=False):
        chosen[int(i)] = 'uniform'
    for _ in range(cfg['phase_states']):
        remaining = [i for i in eligible if i not in chosen]
        if not remaining:
            break
        phase = lambda i: min(cfg['phase_bins'] - 1, int(states[i]['time_s'] / horizon * cfg['phase_bins']))
        seen = {phase(i) for i in chosen}
        lateral = {i: max((p['score'] for p in states[i]['screen_pairs'] if p['mechanism'] == 'lateral'), default=0.) for i in remaining}
        active = [i for i in remaining if lateral[i] > 0]
        pool = active or remaining
        i = max(pool, key=lambda j: (phase(j) not in seen, lateral[j] if active else states[j]['screen_score'], -states[j]['time_s']))
        chosen[i] = 'lateral_phase' if active else 'phase_fallback'
    return chosen


def choose_pairs(rows, method, rng):
    if method == 'geometry2_diverse':
        return audit_candidates(rows, {'ranked_pairs': 2, 'random_remaining_pairs': 1}, rng)
    chosen = [dict(r, selection='ranked') for r in rows[:2]]
    if len(rows) > 2:
        chosen.append(dict(rows[int(rng.integers(2, len(rows)))], selection='random_audit'))
    return chosen


def check_lock(cfg, split, implementation, dependencies, lock=None):
    sets = {k: set(range(cfg[k]['seed_start'], cfg[k]['seed_start'] + cfg[k]['episodes'])) for k in ('train', 'validation')}
    if not all(sets.values()) or sets['train'] & sets['validation']:
        raise ValueError('Source episodes must be nonempty and disjoint')
    for lo, hi in cfg['forbidden_validation_ranges']:
        if sets['validation'] & set(range(lo, hi + 1)):
            raise ValueError('Validation cohort overlaps consumed/reserved episodes')
    if (sets['train'] | sets['validation']) & set(range(100000, 101024)):
        raise ValueError('Reserved final-test episodes are forbidden')
    if cfg['pairs']['ranked'] != 2 or cfg['pairs']['audit'] != 1:
        raise ValueError('This protocol requires rank 1 + rank 2 + one audit slot')
    if cfg['methods'] != ['geometry2_diverse', 'geometry5_top2', 'mixed5_top2']:
        raise ValueError('Unsupported comparison arms')
    if split == 'validation':
        if not lock or lock.get('split') != 'train' or not lock.get('complete'):
            raise ValueError('Fresh validation requires a completed training lock')
        if (lock['config_identity'] != identity(cfg) or lock['implementation_sha256'] != implementation
                or lock['dependencies'] != dependencies):
            raise ValueError('Rule or dependency changed after training')


def describe(values):
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'mean': None, 'max': None}
    a = np.asarray(values, float)
    return {'n': len(values), 'min': float(a.min()), 'median': float(np.median(a)),
            'mean': float(a.mean()), 'max': float(a.max())}


def calibration_candidates(rows, methods, cfg):
    """Episode-stratified positive cases and seeded negatives, before outcomes."""
    selected = {}
    for method in methods:
        best = {}
        for i, row in enumerate(rows):
            if method not in row['methods'] or not row['views']['production_strict']:
                continue
            if row['seed'] not in best or row['metrics']['positive_interaction_mass'] > rows[best[row['seed']]]['metrics']['positive_interaction_mass']:
                best[row['seed']] = i
        keys = sorted(best)
        rng = np.random.default_rng(cfg['selection_seed'])
        for seed in rng.permutation(keys)[:cfg['maximum_positive_episodes_per_method']]:
            selected.setdefault(best[int(seed)], []).append(method)
    # A zero interaction matrix has an arbitrary argmax that may be outside
    # natural support. Controls use positive supported contrasts failing gates.
    negatives = [i for i, r in enumerate(rows) if not r['views']['production_strict']
                 and r['metrics']['max_positive_interaction'] > 0]
    rng = np.random.default_rng(cfg['selection_seed'] + 1)
    seen = set()
    for i in rng.permutation(negatives):
        if rows[int(i)]['seed'] in seen:
            continue
        selected[int(i)] = ['negative_control']
        seen.add(rows[int(i)]['seed'])
        if len(seen) == cfg['negative_controls']:
            break
    return sorted(selected.items())


def calibrate(rows, bank, model, assets, physical, cfg, methods, split):
    records = []
    for index, selected_for in calibration_candidates(rows, methods, cfg):
        row = rows[index]
        state = bank[row['state_index']]
        a0, b0 = row['metrics']['baseline_actions']
        a, b = row['metrics']['max_interaction_actions']
        corners = [(a0, b0), (a, b0), (a0, b), (a, b)]
        risks, events = [], []
        for seed in range(cfg[split + '_seed_start'], cfg[split + '_seed_start'] + cfg['replicates_per_corner']):
            samples = [continue_state(model, assets, physical, state, dict(zip(row['actor_ids'], corner)),
                                      seed, cfg['horizon_s'], cfg['clearance_scale_m']) for corner in corners]
            risks.append([s['clearance_risk'] for s in samples])
            events.append([int(s['cav_first_collision']) for s in samples])
        values = np.asarray(risks)
        contrast = values[:, 3] - values[:, 1] - values[:, 2] + values[:, 0]
        records.append({'row_index': index, 'seed': row['seed'], 'time_s': row['time_s'],
                        'actor_ids': row['actor_ids'], 'mechanism': row['mechanism'], 'selected_for': selected_for,
                        'strict_positive': row['views']['production_strict'], 'corners': corners,
                        'proxy_interaction': row['metrics']['max_positive_interaction'],
                        'native_interaction': float(contrast.mean()),
                        'native_interaction_se': float(contrast.std(ddof=1) / np.sqrt(len(contrast))),
                        'native_interaction_ge_threshold': bool(contrast.mean() >= cfg['minimum_native_interaction']),
                        'native_mean_risk': values.mean(0).tolist(), 'cav_events': np.sum(events, axis=0).tolist(),
                        'replicate_risk': risks, 'replicate_events': events})
    return records


def method_report(method, rows, bank, episodes, scanned, costs, calibration):
    selected = [r for r in rows if method in r['methods']]
    positives = [r for r in selected if r['views']['production_strict']]
    state_ids = {r['state_index'] for r in positives}
    count = sum(method in s['methods'] for s in bank)
    best = [max((r for r in positives if r['state_index'] == i), key=lambda r: r['metrics']['positive_interaction_mass']) for i in sorted(state_ids)]
    checked = [r for r in calibration if method in r['selected_for']]
    return {
        'candidate_states': count, 'candidate_states_per_episode': [e['bank_counts'][method] for e in episodes],
        'candidate_roles': dict(Counter(s['roles'][method] for s in bank if method in s['methods'])),
        'evaluated_pairs': len(selected), 'strict_positive_pairs': len(positives),
        'strict_positive_pair_rate': len(positives) / len(selected) if selected else None,
        'strict_positive_states': len(state_ids), 'strict_positive_state_rate': len(state_ids) / count if count else None,
        'positive_source_episodes': len({r['seed'] for r in positives}),
        'positive_source_episode_coverage': len({r['seed'] for r in positives}) / len(episodes),
        'strict_positive_mechanisms': dict(Counter(r['mechanism'] for r in positives)),
        'second_factor_H_vs_P_tv_all_pairs': describe([r['actual_H_tv'][1] for r in selected]),
        'second_factor_H_vs_P_tv_chosen': describe([r['actual_H_tv'][1] for r in best]),
        'second_factor_Q_vs_P_tv_chosen': describe([r['q_second_tv'] for r in best]),
        'shadow_intervention_count': len(best),
        'shadow_intervention_rate_selected_states': len(best) / count if count else None,
        'shadow_intervention_coverage_natural_decisions': len(best) / scanned,
        'online_intervention_rate': None,
        'selector_cpu_s': costs['features_cpu_s'] + costs['bank_cpu_s'][method] + costs['pair_selection_cpu_s'][method] + sum(r['proxy_cpu_s'] for r in selected),
        'proxy_cpu_s': sum(r['proxy_cpu_s'] for r in selected),
        'proxy_wall_s': sum(r['proxy_wall_s'] for r in selected),
        'native_checked_positive_cases': len(checked),
        'native_interaction_ge_threshold_cases': sum(r['native_interaction_ge_threshold'] for r in checked),
        'native_minus_proxy_interaction': describe([r['native_interaction'] - r['proxy_interaction'] for r in checked]),
        'first_failure_counts': dict(Counter(r['first_failed_gate'] for r in selected)),
    }


def run(config_path, split, output, training_lock=None):
    cfg = read(config_path)
    dc = read(cfg['base_discovery_config'])
    physical = read(dc['physical_config'])
    dependencies = {name: digest(Path(__file__).with_name(name)) for name in (
        'highd_v46_state_discovery.py', 'highd_v46_pair_synergy.py', 'highd_v46_cross_unit.py',
        'highd_v46_pair_witness.py', 'highd_v46_d2rl.py')}
    dependencies.update(base_config=digest(cfg['base_discovery_config']), physical_config=digest(dc['physical_config']))
    implementation = digest(__file__)
    check_lock(cfg, split, implementation, dependencies, read(training_lock) if training_lock else None)
    if cfg['calibration']['horizon_s'] != dc['proxy']['horizon_s']:
        raise ValueError('Native and proxy horizons must agree')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets, lock = ensure_assets()
    model = PairModelQueries(assets / 'model', assets / 'lateral', **SINGLE_OPTIONS)
    bank, episodes, rows, shadow = [], [], [], []
    costs = {'features_cpu_s': 0., 'bank_cpu_s': {m: 0. for m in cfg['methods']},
             'pair_selection_cpu_s': {m: 0. for m in cfg['methods']}}
    started = time.perf_counter()

    class Observed(NativeExperiment):
        def choose(self, units, actors, pmfs, mode):
            if mode != 'natural':
                raise ValueError('Only natural rollouts are permitted')
            start = time.process_time()
            fast = screen_pairs(self, pmfs, dc['screen'])
            costs['features_cpu_s'] += time.process_time() - start
            state = snapshot(self, units, pmfs, seed, {'discovery_only': True})
            state.update(screen_pairs=fast, screen_score=fast[0]['score'] if fast else 0.)
            states.append(state)
            return None

    cohort = cfg[split]
    for seed in range(cohort['seed_start'], cohort['seed_start'] + cohort['episodes']):
        states = []
        road = Observed(model, assets / 'pair', assets / 'lateral', seed, physical)
        while road.time < physical['horizon_s'] - 1e-9:
            if not road.step_policy('natural', 1.):
                break
        union, counts = {}, {}
        for method in cfg['methods']:
            start = time.process_time()
            rng = np.random.default_rng(np.random.SeedSequence([cfg['bank']['random_seed'], seed]))
            chosen = choose_states(states, method, cfg['bank'], physical['horizon_s'], dc['proxy']['horizon_s'], rng)
            costs['bank_cpu_s'][method] += time.process_time() - start
            counts[method] = len(chosen)
            for i, role in chosen.items():
                union.setdefault(i, {})[method] = role
        for i, roles in sorted(union.items()):
            bank.append(dict(states[i], methods=list(roles), roles=roles))
        event = 'cav_first_collision' if any(road.cav.id in pair for pair in road.crashes) else 'bv_only' if road.crashes else 'horizon'
        episodes.append({'seed': seed, 'natural_states': len(states), 'bank_counts': counts, 'event': event})
    scan_time = time.perf_counter() - started
    save(output / 'state_bank.json', {'split': split, 'config': cfg, 'natural_target_sha256': lock['natural_target_sha256'], 'states': bank})
    for index, state in enumerate(bank):
        candidates = {}
        for method in state['methods']:
            start = time.process_time()
            rng = np.random.default_rng(np.random.SeedSequence([cfg['pairs']['random_seed'], state['seed'], round(state['time_s'])]))
            for candidate in choose_pairs(state['screen_pairs'], method, rng):
                candidates.setdefault(tuple(candidate['actor_ids']), {})[method] = candidate['selection']
            costs['pair_selection_cpu_s'][method] += time.process_time() - start
        cars = [SimpleNamespace(**c) for c in state['vehicles']]
        road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']), model=model, config=physical)
        units = [{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']]
        pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}
        positive = {}
        for ids, selections in candidates.items():
            start_cpu, start_wall = time.process_time(), time.perf_counter()
            p, same = pair_marginal(units, ids)
            risk = response_risk(road, ids, pmfs, dc['proxy'], reactive=True)
            audit = audit_gates(p, risk, dc['selection'])
            cpu, wall = time.process_time() - start_cpu, time.perf_counter() - start_wall
            candidate = next(c for c in state['screen_pairs'] if tuple(c['actor_ids']) == ids)
            row = {'state_index': index, 'seed': state['seed'], 'time_s': state['time_s'], **candidate,
                   'methods': list(selections), 'selections': selections, 'same_natural_unit': same,
                   'proxy_cpu_s': cpu, 'proxy_wall_s': wall, 'q_second_tv': None, **audit}
            if audit['views']['production_strict']:
                score = p * positive_interaction(p, risk)
                h = score / score.sum()
                _, p2, _, _, q1, q2 = proposal_factors(p, h, cfg['shadow_epsilon'])
                row['q_second_tv'] = float(np.sum(q1 * (.5 * np.abs(p2 - q2).sum(1))))
                positive[ids] = (row, h)
            rows.append(row)
        for method in state['methods']:
            available = [(ids, row, h) for ids, (row, h) in positive.items() if method in row['methods']]
            if not available:
                continue
            ids, row, h = max(available, key=lambda item: item[1]['metrics']['positive_interaction_mass'])
            rng = np.random.default_rng(np.random.SeedSequence([cfg['shadow_seed'], state['seed'], round(state['time_s'])]))
            _, logged = draw_intervention(units, ids, h, cfg['shadow_epsilon'], rng)
            shadow.append({'state_index': index, 'method': method, 'actor_ids': list(ids), **logged})
        print({'split': split, 'states': index + 1, 'total': len(bank), 'unique_pairs': len(rows)}, flush=True)
    save(output / 'pair_audit.json', {'rows': rows})
    save(output / 'shadow_interventions.json', {'scope': 'Conditional draws only; never applied to natural source rollouts', 'rows': shadow})
    discovery_time = time.perf_counter() - started
    calibration = calibrate(rows, bank, model, assets, physical, cfg['calibration'], cfg['methods'], split)
    save(output / 'native_calibration.json', {'records': calibration, 'scope': 'Selected conditional action corners, not Q samples or population event rates'})
    scanned = sum(e['natural_states'] for e in episodes)
    result = {'complete': True, 'split': split, 'config': cfg, 'config_identity': identity(cfg),
              'implementation_sha256': implementation, 'dependencies': dependencies,
              'natural_target_sha256': lock['natural_target_sha256'], 'source_episodes': episodes,
              'natural_states': scanned, 'unique_bank_states': len(bank), 'unique_evaluated_pairs': len(rows),
              'costs': costs, 'scan_wall_s': scan_time, 'discovery_wall_s': discovery_time,
              'total_wall_s': time.perf_counter() - started,
              'methods': {m: method_report(m, rows, bank, episodes, scanned, costs, calibration) for m in cfg['methods']},
              'calibration_cases': len(calibration),
              'maximum_shadow_likelihood_error': max((abs(r['full_log_p'] - r['full_log_q'] - r['log_weight']) for r in shadow), default=0.),
              'artifacts': {name: digest(output / name) for name in ('state_bank.json', 'pair_audit.json', 'shadow_interventions.json', 'native_calibration.json')},
              'scope': 'Offline bank discovery. Shadow activation is not online intervention rate. No population risk, PPO or significance claim.'}
    save(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v46_discovery_v2.json')
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--training-lock')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(args.config, args.split, args.output, args.training_lock))


if __name__ == '__main__':
    main()
