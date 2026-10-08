"""Opt-in H-score comparison on the entire locked eligible training bank.

The state law, selected pairs, natural dynamics and exact factor sampler are
unchanged. This does not export PPO data or a population acceleration estimate.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_v46_collision_coverage import audit_sample, support_category
from .highd_v46_cross_unit import positive_interaction, proposal_factors
from .highd_v46_d2rl import experiment_target, identity
from .highd_v46_multistate_q import dependencies, environment, sample_continuation
from .highd_v46_native_calibration import case_masses
from .highd_v46_pair_synergy import pair_marginal, response_risk
from .highd_v46_state_discovery import audit_gates
from .highd_v46_training_expansion import load_calibration

DEFAULT_CONFIG = 'configs/highd_v46_proposal_comparison_v1.json'
CONTRACT = 'normalized_risk_interaction_H_v1'


def build_h(p, risk, interaction_fraction):
    """Mix normalized components, not unnormalized scores; zero mass uses P."""
    p, risk = np.asarray(p, float), np.asarray(risk, float)
    if (p.ndim != 2 or risk.shape != p.shape or not np.isfinite(p).all()
            or not np.isfinite(risk).all() or np.any(p < 0) or np.any(risk < 0)
            or not np.isclose(p.sum(), 1.) or not np.isfinite(interaction_fraction)
            or not 0 <= interaction_fraction <= 1):
        raise ValueError('Invalid supported P, nonnegative risk or component balance')
    scores = {'risk': p * risk, 'interaction': p * positive_interaction(p, risk)}
    masses = {key: float(value.sum()) for key, value in scores.items()}
    components = {key: value / masses[key] if masses[key] > 0 else p.copy()
                  for key, value in scores.items()}
    h = ((1-interaction_fraction) * components['risk']
         + interaction_fraction * components['interaction'])
    return h, {'component_masses': masses,
               'component_natural_fallback': {k: v == 0 for k, v in masses.items()}}


def factor_tv(p, h, epsilon):
    p1, p2, _, _, q1, q2 = proposal_factors(p, h, epsilon)
    return [float(.5 * np.abs(p1-q1).sum()),
            float(np.sum(q1 * (.5 * np.abs(p2-q2).sum(axis=1))))]


def load_inputs(config):
    cfg = read(config)
    if (cfg['contract'] != CONTRACT or cfg['split'] != 'train'
            or cfg['candidates'] != {'interaction': 1., 'absolute_risk': 0., 'mixed': .5}
            or cfg['epsilon'] != [.1, .1] or cfg['horizon_s'] != 4
            or cfg['zero_component_mass'] != 'Use P for that component before mixing'
            or cfg['replicates_per_case_candidate'] != 64
            or cfg['later_actions'] != 'Unchanged natural NDE; only initial decision is tilted'
            or cfg['conditional_law'] != 'Uniform source, then uniform locked eligible cases within source'
            or cfg['expected_cases'] != 31 or cfg['expected_sources'] != 24):
        raise ValueError('Unsupported fixed training proposal experiment')
    if environment() != cfg['environment']:
        raise ValueError('Use the pinned Python/dependency/thread environment')
    stats = cfg['statistics']
    if (not 0 < stats['confidence'] < 1 or stats['bootstrap_repeats'] < 2
            or stats['minimum_events'] < 20 or stats['minimum_event_sources'] < 5
            or not 0 < stats['minimum_effective_factor_tv'] < 1
            or not 0 < stats['maximum_source_share'] < 1
            or stats['minimum_effective_second_moment_sources'] < 3):
        raise ValueError('Invalid statistical evidence guards')
    cases, seen, summaries = [], set(), []
    n = cfg['replicates_per_case_candidate']
    fresh = set(range(cfg['continuation_seed_start'], cfg['continuation_seed_start']+cfg['expected_cases']*n))
    if min(fresh) < 0 or any(fresh & set(range(a, b+1)) for a, b in cfg['forbidden_continuation_ranges']):
        raise ValueError('Continuation seeds overlap a consumed schedule')
    for entry in cfg['training_locks']:
        cal = read(entry['config'])
        summary, seeds, eligible, _, plans, _ = load_calibration(entry['path'], cal, entry['summary_sha256'])
        if seen & seeds:
            raise ValueError('Training source cohorts overlap')
        seen.update(seeds); summaries.append(summary)
        old_count = len(summary['cases']) * cal['replicates_per_case_arm']
        if fresh & set(range(cal['continuation_seed_start'], cal['continuation_seed_start']+old_count)):
            raise ValueError('Continuation seeds overlap locked training draws')
        bank = read(Path(entry['path'])/'state_bank.json')['states']
        for case, plan in zip(eligible, plans):
            state = bank[plan['state_index']]
            if identity(state) != case['state_sha256'] or state['seed'] in range(100000, 101024):
                raise ValueError('State identity or reserved final-test source differs')
            cases.append({**case, 'state': state, 'original_plan': plan,
                          'origin_directory': entry['path'], 'origin_config': cal})
    if len(cases) != cfg['expected_cases'] or len({c['source_seed'] for c in cases}) != cfg['expected_sources']:
        raise ValueError('Do not drop negatives or substitute training cases')
    dc = read(cfg['discovery_config']); base = read(dc['base_discovery_config'])
    physical = read(base['physical_config']); assets, frozen = ensure_assets()
    target = identity(experiment_target(frozen, physical))
    if any(s['natural_target_sha256'] != frozen['natural_target_sha256']
           or s['experiment_target_sha256'] != target for s in summaries):
        raise ValueError('Training and current natural/physical targets differ')
    deps = dependencies(cfg, dc, base)
    deps.update({name: digest(Path(__file__).with_name(name)) for name in (
        'highd_v46_multistate_q.py', 'highd_v46_native_calibration.py',
        'highd_v46_training_expansion.py', 'highd_v46_conditional_baseline.py',
        'highd_v46_collision_coverage.py')})
    plans = [{k: c[k] for k in ('source_seed', 'time_s', 'state_sha256',
                               'origin_summary_sha256', 'original_case_index')} for c in cases]
    for i, (plan, case, mass) in enumerate(zip(plans, cases, case_masses(cases))):
        plan.update(case_id=i, actor_ids=case['row']['actor_ids'], row_sha256=identity(case['row']),
                    target_mass=float(mass), continuation_seed_start=cfg['continuation_seed_start']+i*n)
    lock = {'contract': CONTRACT, 'config_identity': identity(cfg), 'config_sha256': digest(config),
            'implementation_sha256': digest(__file__), 'dependencies': deps, 'environment': environment(),
            'natural_target_sha256': frozen['natural_target_sha256'], 'physical_target_sha256': target,
            'conditional_target_sha256': identity([(p['state_sha256'], p['actor_ids'], p['target_mass']) for p in plans]),
            'case_plan_sha256': identity(plans), 'training_summaries': [s['config_sha256'] for s in summaries],
            'validation_sources_opened': False}
    lock['proposal_structure_sha256'] = identity({'contract': CONTRACT, 'config': cfg,
        'implementation': lock['implementation_sha256'], 'dependencies': deps})
    return cfg, cases, plans, base, physical, assets, lock


def freeze(config, output):
    *_, lock = load_inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def prepare(case, model, physical, base, candidates, epsilon):
    state, row = case['state'], case['row']
    cars = [SimpleNamespace(**c) for c in state['vehicles']]
    road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']),
                           model=model, config=physical)
    p, _ = pair_marginal([{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']], row['actor_ids'])
    risk = response_risk(road, row['actor_ids'], {int(k): np.asarray(v) for k, v in state['pmfs'].items()},
                         base['proxy'], reactive=True)
    audit = audit_gates(p, risk, base['selection'])
    if not audit['views']['production_strict'] or any(identity(audit[k]) != identity(row[k]) for k in audit):
        raise ValueError('Locked original pair gates no longer reproduce')
    proposals = {}
    for name, lam in candidates.items():
        h, metadata = build_h(p, risk, lam)
        proposals[name] = {'ids': row['actor_ids'], 'h': h, **metadata,
                           'factor_tv': factor_tv(p, h, epsilon)}
    return p, risk, proposals


def replay_event_coverage(case, p, risk, proposals, epsilon):
    """Descriptive score coverage of old training events, not new-Q estimates."""
    result = []
    for arm, old_epsilon in case['origin_config']['arms'].items():
        raw = read(Path(case['origin_directory'])/f'case_{case["original_case_index"]:02d}_{arm}.json')
        for sample in raw['sequences']:
            audit_sample(sample, old_epsilon)
            if not sample['event_result']:
                continue
            log = sample['initial_likelihood_audit']; a, b = log['action_indices']
            covered = {}
            for name, chosen in proposals.items():
                p1, p2, h1, h2, q1, q2 = proposal_factors(p, chosen['h'], epsilon)
                if name == 'interaction' and not np.allclose([h1[a], h2[a, b]], log['h_factors'], atol=0, rtol=1e-10):
                    raise ValueError('Reconstructed original H differs from old event')
                covered[name] = {'h_factors': [float(h1[a]), float(h2[a, b])],
                    'category': support_category([h1[a], h2[a, b]]), 'joint_H_mass': float(chosen['h'][a, b]),
                    'joint_Q_mass': float(q1[a]*q2[a, b]),
                    'initial_action_lr_if_drawn': float(p[a, b]/(q1[a]*q2[a, b]))}
            result.append({'source_seed': case['source_seed'], 'time_s': case['time_s'], 'old_arm': arm,
                'seed': sample['seed'], 'action_indices': [a, b], 'proxy_clearance_risk': float(risk[a, b]),
                'old_generation_log_weight': sample['generation_log_weight'], 'candidates': covered})
    return result


def summarize(records, stats):
    """Exact source-then-state target; all zero-event strata remain in the law."""
    masses = case_masses(records); source_rows = defaultdict(list)
    categories = {k: {'events': 0, 'M2_contribution': 0.} for k in (
        'H1_positive_H2_positive', 'H1_zero_H2_positive', 'H1_positive_H2_zero', 'H1_zero_H2_zero')}
    mean = second = variance = weight_mean = weight_variance = raw_rate = top_event = 0.
    events = 0; max_log_error = 0.; tvs = []
    for r, mass in zip(records, masses):
        seqs = r['sequences']; n = len(seqs)
        if n < 2 or len({s['seed'] for s in seqs}) != n:
            raise ValueError('Invalid fixed continuation stratum')
        weights = np.exp([s['generation_log_weight'] for s in seqs])
        y = weights * np.asarray([s['event_result'] for s in seqs], float)
        moment = float(np.mean(y*y)); count = int(np.count_nonzero(y))
        mean += mass*y.mean(); second += mass*moment; raw_rate += mass*count/n
        variance += mass*mass*y.var(ddof=1)/n
        weight_mean += mass*weights.mean(); weight_variance += mass*mass*weights.var(ddof=1)/n
        events += count; top_event = max(top_event, float(mass*np.max(y*y)/n))
        tvs.append(r['factor_tv'])
        source_rows[r['source_seed']].append({'events': count, 'M2': moment, 'mean': float(y.mean()),
            'variance_of_case_mean': float(y.var(ddof=1)/n),
            'effective_dual_events': count if min(r['factor_tv']) >= stats['minimum_effective_factor_tv'] else 0})
        for s, value in zip(seqs, y):
            a = audit_sample(s, r['epsilon'])
            log = s['initial_likelihood_audit']
            max_log_error = max(max_log_error, abs(log['full_log_p']-log['full_log_q']-s['generation_log_weight']))
            if s['event_result']:
                categories[a['category']]['events'] += 1
                categories[a['category']]['M2_contribution'] += float(mass*value*value/n)
    sources = []
    k = len(source_rows)
    for seed, rows in sorted(source_rows.items()):
        m2 = float(np.mean([r['M2'] for r in rows]))
        sources.append({'source_seed': seed, 'events': sum(r['events'] for r in rows),
            'effective_dual_events': sum(r['effective_dual_events'] for r in rows),
            'source_M2': m2, 'M2_contribution': m2/k,
            'source_mean': float(np.mean([r['mean'] for r in rows])),
            'source_variance_component': sum(r['variance_of_case_mean'] for r in rows)/len(rows)**2,
            'share': m2/k/second if second > 0 else None})
    shares = [r['share'] for r in sources] if second > 0 else []
    effective = 1/sum(c*c for c in shares) if shares else None
    event_sources = [r['source_seed'] for r in sources if r['events']]
    effective_event_sources = [r['source_seed'] for r in sources if r['effective_dual_events']]
    evidence_guard = events >= stats['minimum_events'] and len(event_sources) >= stats['minimum_event_sources']
    readiness = (len(effective_event_sources) >= stats['minimum_event_sources'] and shares
                 and max(shares) <= stats['maximum_source_share']
                 and effective >= stats['minimum_effective_second_moment_sources'])
    for cat in categories.values():
        cat['M2_share'] = cat['M2_contribution']/second if second > 0 else None
    return {'cases': len(records), 'source_episodes': k, 'continuations': sum(len(r['sequences']) for r in records),
        'events': events, 'event_sources': event_sources, 'effective_dual_event_sources': effective_event_sources,
        'source_weighted_raw_event_rate': float(raw_rate), 'conditional_IS_mean': float(mean),
        'conditional_M2': float(second), 'fixed_state_stratified_estimator_variance': float(variance),
        'fixed_state_mean_MCSE': float(np.sqrt(variance)),
        'descriptive_relative_half_width': float(NormalDist().inv_cdf((1+stats['confidence'])/2)*np.sqrt(variance)/mean) if mean else None,
        'maximum_source_M2_share': max(shares) if shares else None,
        'effective_second_moment_sources_descriptive': effective,
        'maximum_single_event_M2_share': top_event/second if second > 0 else None,
        'mean_weight': float(weight_mean), 'mean_weight_MCSE': float(np.sqrt(weight_variance)),
        'source_weighted_factor_TV': np.sum(masses[:, None]*np.asarray(tvs), axis=0).tolist(),
        'factor_TV_min': np.min(tvs, axis=0).tolist(), 'factor_TV_max': np.max(tvs, axis=0).tolist(),
        'event_H_categories': categories, 'by_source': sources, 'event_interval_guard': bool(evidence_guard),
        'engineering_readiness_checks_passed': bool(readiness),
        'maximum_full_vs_factor_log_weight_error': float(max_log_error)}


def compare_reports(reports, stats):
    """Matched state law; paired whole-source bootstrap, never pooled draws."""
    baseline = reports['interaction']; result = {}
    rng = np.random.default_rng(stats['bootstrap_seed'])
    k = baseline['source_episodes']
    draws = rng.integers(0, k, size=(stats['bootstrap_repeats'], k))
    for name, report in reports.items():
        if [r['source_seed'] for r in report['by_source']] != [r['source_seed'] for r in baseline['by_source']]:
            raise ValueError('Candidate source targets differ')
        m2s, variances = [], []
        for indices in draws:
            unique = set(int(i) for i in indices)
            eligible = all(sum(group['by_source'][i]['events'] for i in indices) >= stats['minimum_events']
                and len([i for i in unique if group['by_source'][i]['events']]) >= stats['minimum_event_sources']
                for group in (baseline, report))
            if not eligible:
                continue
            b2, r2 = (sum(g['by_source'][i]['source_M2'] for i in indices)/k for g in (baseline, report))
            bv, rv = (sum(g['by_source'][i]['source_variance_component'] for i in indices)/k**2 for g in (baseline, report))
            if b2 > 0: m2s.append(r2/b2)
            if bv > 0: variances.append(rv/bv)
        guard = baseline['event_interval_guard'] and report['event_interval_guard']
        def interval(values):
            censored = stats['bootstrap_repeats']-len(values)
            usable = guard and censored <= .1*stats['bootstrap_repeats']
            q = [(1-stats['confidence'])/2, (1+stats['confidence'])/2]
            return {'interval': np.quantile(values, q).tolist() if usable else None,
                    'censored_replicates': censored,
                    'status': 'supported_source_bootstrap' if usable else 'insufficient_event_sources_or_replicates'}
        result[name] = {'M2_ratio_to_interaction': report['conditional_M2']/baseline['conditional_M2'] if baseline['conditional_M2'] > 0 else None,
            'fixed_state_variance_ratio_to_interaction': report['fixed_state_stratified_estimator_variance']/baseline['fixed_state_stratified_estimator_variance'] if baseline['fixed_state_stratified_estimator_variance'] > 0 else None,
            'M2_ratio_source_interval': interval(m2s), 'variance_ratio_source_interval': interval(variances)}
    return {'bootstrap_unit': 'Whole source episode with all states and draws retained',
            'same_conditional_target': True, 'comparisons': result}


def run(config, lock_path, output):
    cfg, cases, plans, base, physical, assets, current_lock = load_inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in current_lock.items()):
        raise ValueError('Training rule or dependencies changed after freeze')
    destination = Path(output); destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(destination/'opened_protocol.json', {'config': cfg, 'lock': lock, 'rule_lock_sha256': digest(lock_path)})
    save(destination/'case_plan.json', {'selection_uses_new_native_outcomes': False, 'cases': plans})
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    preparation_start = time.process_time()
    prepared, replay = [], []
    for i, case in enumerate(cases):
        p, risk, proposals = prepare(case, model, physical, base, cfg['candidates'], cfg['epsilon'])
        save(destination/f'case_{i:02d}_proposals.json', {'plan': plans[i], 'P': p.tolist(), 'risk': risk.tolist(),
             'proposals': {name: {**v, 'h': v['h'].tolist()} for name, v in proposals.items()}})
        prepared.append(proposals)
        replay.extend(replay_event_coverage(case, p, risk, proposals, cfg['epsilon']))
    save(destination/'old_training_event_coverage.json', {'events': replay,
        'scope': 'Descriptive old-arm action coverage only. No reweighting, pooling or counterfactual event estimate.'})
    preparation_cpu = time.process_time()-preparation_start
    records = {name: [] for name in cfg['candidates']}
    n = cfg['replicates_per_case_candidate']
    for i, (case, plan, proposals) in enumerate(zip(cases, plans, prepared)):
        for name, chosen in proposals.items():
            seqs = [sample_continuation(model, assets, physical, case['state'], chosen, cfg['epsilon'],
                    plan['continuation_seed_start']+j, cfg, base['proxy']['clearance_scale_m']) for j in range(n)]
            save(destination/f'case_{i:02d}_{name}.json', {'plan': plan, 'candidate': name,
                 'interaction_fraction': cfg['candidates'][name], 'epsilon': cfg['epsilon'], 'sequences': seqs})
            records[name].append({'source_seed': plan['source_seed'], 'epsilon': cfg['epsilon'],
                                 'factor_tv': chosen['factor_tv'], 'sequences': seqs})
            print({'case': i+1, 'total': len(cases), 'source': plan['source_seed'], 'candidate': name,
                   'events': sum(s['event_result'] for s in seqs)}, flush=True)
    reports = {name: summarize(rows, cfg['statistics']) for name, rows in records.items()}
    for name, rows in records.items():
        reports[name]['native_CPU_s'] = sum(s.get('cpu_s', 0.) for r in rows for s in r['sequences'])
    result = {'complete': True, 'split': 'train', 'config': cfg, 'rule_lock_sha256': digest(lock_path),
        'lock': lock, 'reports': reports, 'matched_comparison': compare_reports(reports, cfg['statistics']),
        'artifact_hashes': {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
        'wall_s': time.perf_counter()-start, 'proxy_preparation_and_replay_CPU_s': preparation_cpu,
        'validation_sources_opened': False,
        'ppo_trained': False, 'training_exported': False, 'population_acceleration': None,
        'automatically_selected_candidate': None,
        'scope': 'Training-only fixed-state, fixed-pair proposal comparison; no fresh generalization claim'}
    save(destination/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--lock')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze(args.config, args.output)
        print({'proposal_structure_sha256': result['proposal_structure_sha256'], 'validation_sources_opened': False})
    else:
        if not args.lock: parser.error('run requires --lock')
        result = run(args.config, args.lock, args.output)
        print({name: {k: r[k] for k in ('events', 'event_sources', 'conditional_M2', 'maximum_source_M2_share')}
               for name, r in result['reports'].items()})


if __name__ == '__main__':
    main()
