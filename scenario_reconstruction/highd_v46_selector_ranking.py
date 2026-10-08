"""Three outcome-blind rankings on locked training caches, without simulation."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from .highd_v46_assets import digest, read, save, ensure_assets
from .highd_v46_conditional_baseline import make_sequences
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import environment
from .highd_v46_native_calibration import case_masses
from .highd_v46_training_expansion import load_calibration, concentration


METHODS = ('interaction_only', 'absolute_risk_only', 'absolute_risk_times_interaction')


def key(row):
    return (int(row['seed']), float(row['time_s']), tuple(row['actor_ids']))


def score(row, method):
    interaction = float(row['metrics']['positive_interaction_mass'])
    risk = float(row['ranker_features'][2])
    if not np.isfinite([interaction, risk]).all() or interaction < 0 or not 0 <= risk <= 1:
        raise ValueError('Invalid cached risk/interaction score')
    if method not in METHODS:
        raise ValueError('Unknown ranking method')
    return interaction if method == METHODS[0] else risk if method == METHODS[1] else risk*interaction


def select(rows, method, budget, maximum_states_per_source=2):
    if budget < 1 or maximum_states_per_source < 1:
        raise ValueError('Ranking budgets must be positive')
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate cached state-pair')
    eligible = [r for r in rows if r['views']['production_strict']]
    ranked = sorted(eligible, key=lambda r: (-score(r, method), key(r)))
    selected, states, counts = [], set(), Counter()
    for r in ranked:
        state = key(r)[:2]
        if state in states or counts[r['seed']] >= maximum_states_per_source:
            continue
        selected.append(r); states.add(state); counts[r['seed']] += 1
        if len(selected) == budget:
            break
    return selected


def tv_summary(rows, values):
    if not rows:
        return None
    if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
        raise ValueError('Invalid total variation')
    masses = case_masses([{'source_seed': r['seed']} for r in rows])
    return {'minimum': float(min(values)), 'maximum': float(max(values)),
            'source_state_weighted_mean': float(np.dot(masses, values))}


def native_metrics(rows, labels):
    if not rows or any(key(r) not in labels for r in rows):
        return None
    selected = [labels[key(r)] for r in rows]
    cases = [v['case'] for v in selected]; raw = [v['raw'] for v in selected]
    seqs = make_sequences(cases, raw)
    masses = case_masses(cases)
    counts = [sum(s['event_result'] for s in group) for group in raw]
    concentration_report = concentration(seqs)
    return {'continuations': len(seqs), 'cav_events': sum(counts),
            'continuation_collision_rate': sum(counts)/len(seqs),
            'source_state_weighted_collision_rate': float(sum(m*c/len(s) for m,c,s in zip(masses,counts,raw))),
            'event_source_episodes': concentration_report['event_source_episodes'],
            'concentration': concentration_report,
            'q_second_factor_tv': tv_summary(rows, [v['record']['q_second_tv'] for v in selected]),
            'scope': 'Observed fixed-Q continuation yield, not natural risk; selected conditional law differs across rankings'}


def evaluate(rows, labels, reference_sources, reference_contributions):
    known = [r for r in rows if key(r) in labels]
    unknown = [r for r in rows if key(r) not in labels]
    retained_sources = {r['seed'] for r in known if any(s['event_result'] for s in labels[key(r)]['raw'])}
    reference_total = sum(reference_contributions.values())
    return {'strict_positive_states': len({key(r)[:2] for r in rows}),
            'selected_pairs': len(rows), 'source_episodes': len({r['seed'] for r in rows}),
            'mechanisms': dict(Counter(r['mechanism'] for r in rows)),
            'labelled_pairs': len(known), 'unknown_native_pairs': len(unknown),
            'known_collision_source_recall': len(retained_sources)/len(reference_sources) if reference_sources else None,
            'known_collision_source_recall_denominator': len(reference_sources),
            'retained_observed_collision_sources': sorted(retained_sources),
            'recall_scope': 'Retention of observed positive state-pair witnesses only; unknown pairs cannot establish additional collision sources',
            'h_second_factor_tv': tv_summary(rows, [r['actual_H_tv'][1] for r in rows]),
            'native': native_metrics(rows, labels),
            'reference_law_retained_observed_second_moment_fraction':
                sum(reference_contributions.get(key(r), 0.) for r in known)/reference_total if reference_total > 0 else None,
            'reference_fraction_scope': 'Known labelled reference-bank contributions only, not an estimate for unlabelled pairs',
            'selected_keys': [[r['seed'], r['time_s'], r['actor_ids']] for r in rows]}


def run(config, output):
    cfg = read(config)
    if (tuple(cfg['methods']) != METHODS or cfg['maximum_states_per_source'] != 2 or cfg['pair_per_state'] != 1
            or cfg['native_epsilon'] != [.1, .1] or environment() != cfg['environment']):
        raise ValueError('Changed ranking or environment contract')
    tick = time.perf_counter()
    _, frozen = ensure_assets()
    rows, labels, sources, parent = [], {}, set(), None
    for entry in cfg['sources']:
        summary, seeds, cases, raw, records, _ = load_calibration(entry['path'], read(entry['config']), entry['summary_sha256'])
        if sources & seeds:
            raise ValueError('Training source cohorts overlap')
        sources.update(seeds)
        if summary['natural_target_sha256'] != frozen['natural_target_sha256']:
            raise ValueError('Frozen natural target differs')
        target = summary['experiment_target_sha256']
        if parent is not None and parent != target:
            raise ValueError('Physical target differs between caches')
        parent = target
        rows.extend(read(Path(entry['path'])/'pair_audit.json')['rows'])
        for case, seqs, record in zip(cases, raw, records):
            k = key(case['row'])
            if k in labels:
                raise ValueError('Duplicate native-labelled case')
            if not np.allclose(case['row']['actual_H_tv'], record['factor_tv'], rtol=0, atol=1e-12):
                raise ValueError('Recorded factor TV differs')
            labels[k] = {'case': case, 'raw': seqs, 'record': record}
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate audited pair')
    if any(k not in {key(r) for r in rows} for k in labels):
        raise ValueError('Labelled case missing from audited universe')
    reference_sources = {v['case']['source_seed'] for v in labels.values() if any(s['event_result'] for s in v['raw'])}
    all_cases = [v['case'] for v in labels.values()]; all_raw = [v['raw'] for v in labels.values()]
    reference_contributions = {k: float(m*np.mean([s['event_result']*np.exp(2*s['generation_log_weight']) for s in labels[k]['raw']]))
                               for k,m in zip(labels, case_masses(all_cases))}
    panels = {}
    for panel, candidates in [('common_native_labelled_cases', [v['case']['row'] for v in labels.values()]),
                              ('all_cached_strict_pairs', rows)]:
        panels[panel] = {}
        for budget in [cfg['primary_state_budget']]+cfg['diagnostic_state_budgets']:
            comparisons = {}
            for method in METHODS:
                cpu = time.process_time()
                chosen = select(candidates, method, budget, cfg['maximum_states_per_source'])
                cost = time.process_time()-cpu
                comparisons[method] = {**evaluate(chosen, labels, reference_sources, reference_contributions),
                                       'ranking_only_cpu_s': cost, 'requested_state_budget': budget}
            panels[panel][str(budget)] = comparisons
    positive = [r for r in rows if r['views']['production_strict']]
    report = {'scope': cfg['scope'], 'config': cfg, 'config_sha256': digest(config),
              'implementation_sha256': digest(__file__),
              'dependency_sha256': {n: digest(Path(__file__).with_name(n)) for n in
                  ('highd_v46_training_expansion.py', 'highd_v46_conditional_baseline.py', 'highd_v46_native_calibration.py')},
              'natural_target_sha256': frozen['natural_target_sha256'], 'environment': environment(),
              'universe': {'source_episodes': len(sources), 'cached_pairs': len(rows),
                           'strict_pairs': len(positive), 'strict_states': len({key(r)[:2] for r in positive}),
                           'strict_source_episodes': len({r['seed'] for r in positive}),
                           'native_labelled_cases': len(labels), 'observed_collision_source_seeds': sorted(reference_sources)},
              'reference_concentration': concentration(make_sequences(all_cases, all_raw)), 'panels': panels,
              'decision': {'new_collision_sources_demonstrated': False, 'ppo_ready': False,
                           'winner_selected': None, 'reason': 'Fixed labels contain only the existing event sources; incomplete selections have unknown native outcomes'},
              'fresh_validation_performed': False, 'new_continuations': 0, 'new_proxy_matrices': 0,
              'wall_s': time.perf_counter()-tick}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save(output/'ranking_comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v46_selector_ranking_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({'universe': result['universe'], 'decision': result['decision']})


if __name__ == '__main__':
    main()
