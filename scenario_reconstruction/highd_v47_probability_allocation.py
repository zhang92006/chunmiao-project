"""Training-only source-cross-fitted probability allocation audit.

Reads all seven frozen behavior components on the same 31-case training bank.
No new native rollout, validation source, PPO fit or deployment is performed.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import time

import numpy as np

from .highd_v46_assets import digest, read, save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_cross_unit import positive_interaction, proposal_factors
from .highd_v46_d2rl import identity
from .highd_v46_proposal_comparison import load_inputs as training_inputs, build_h
from .highd_v47_allocation import (
    CONTRACT, allocate, bin_indices, check_q, factor_tv, fit_calibrator, second_moment,
)

DEFAULT_CONFIG = 'configs/highd_v47_probability_allocation_v1.json'
CANDIDATES = ('interaction', 'absolute_risk', 'mixed', 'calibrated_linear',
              'allocated_first', 'allocated_dual')


def validate_config(cfg):
    if (cfg['contract'] != CONTRACT or cfg['split'] != 'train'
            or cfg['epsilon'] != [.1, .1] or cfg['candidates'] != list(CANDIDATES)
            or cfg['expected_cases'] != 31 or cfg['expected_sources'] != 24
            or cfg['behavior_components'] != ['old_natural', 'old_first_factor_only',
                'old_second_factor_only', 'old_dual', 'new_interaction', 'new_absolute_risk', 'new_mixed']
            or cfg['replicates_per_case_component'] != 64
            or cfg['fit_split'] != 'leave_one_complete_source_out'
            or cfg['new_native_continuations'] != 0
            or cfg['allowed_source_ranges'] != [[200000, 200063], [230000, 230063]]):
        raise ValueError('Unsupported V47 training-only allocation protocol')
    cal = cfg['calibration']
    if cal != {'risk_edges': [0., .2, .4, .6, .8, 1.], 'interaction_threshold': .05,
               'regularization_mass': .01, 'probability_floor': 1e-6}:
        raise ValueError('Version the protocol before changing calibration choices')
    if cfg['minimum_event_sources_for_calibration_review'] < 5:
        raise ValueError('Do not relax the source-diversity review guard')


def load_inputs(config):
    cfg = read(config)
    validate_config(cfg)
    source = cfg['training_comparison']; root = Path(source['path'])
    if digest(root/'summary.json') != source['summary_sha256']:
        raise ValueError('Checksum-locked training comparison differs')
    summary = read(root/'summary.json')
    old_cfg, cases, plans, _, _, _, old_lock = training_inputs(source['config'])
    if (not summary['complete'] or summary['split'] != 'train'
            or summary['config'] != old_cfg or old_cfg['environment'] != cfg['environment']
            or any(summary['lock'].get(k) != v for k, v in old_lock.items())):
        raise ValueError('Expected unchanged complete training comparison')
    for name, expected in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != expected:
            raise ValueError('Training artifact checksum/path differs: '+name)
    saved_plan = read(root/'case_plan.json')
    if saved_plan['selection_uses_new_native_outcomes'] or saved_plan['cases'] != plans:
        raise ValueError('Locked full case plan changed')
    if len(cases) != cfg['expected_cases'] or len({c['source_seed'] for c in cases}) != cfg['expected_sources']:
        raise ValueError('Keep the entire source/state bank, including negatives')
    records, observations = [], []
    for i, (case, plan) in enumerate(zip(cases, plans)):
        seed = case['source_seed']
        if not any(a <= seed <= b for a, b in cfg['allowed_source_ranges']):
            raise ValueError('Validation/test or undeclared source cannot enter fitting')
        matrices = read(root/f'case_{i:02d}_proposals.json')
        if matrices['plan'] != plan:
            raise ValueError('Matrix/case identity differs')
        p, risk = np.asarray(matrices['P']), np.asarray(matrices['risk'])
        proposals = {}
        for name, lam in old_cfg['candidates'].items():
            h, _ = build_h(p, risk, lam)
            if not np.allclose(h, matrices['proposals'][name]['h'], atol=0, rtol=1e-12):
                raise ValueError('Frozen H no longer reproduces')
            proposals[name] = h
        bins = bin_indices(risk, positive_interaction(p, risk), cfg['calibration'])
        record = {'case_id': i, 'source_seed': seed, 'target_mass': plan['target_mass'],
                  'state_sha256': plan['state_sha256'], 'actor_ids': plan['actor_ids'],
                  'p': p, 'risk': risk, 'bins': bins, 'proposals': proposals}
        records.append(record)
        for component in cfg['behavior_components']:
            if component.startswith('old_'):
                arm = component[4:]; origin = Path(case['origin_directory'])
                raw = read(origin/f'case_{case["original_case_index"]:02d}_{arm}.json')
                eps = case['origin_config']['arms'][arm]
                first_seed = case['origin_config']['continuation_seed_start'] + case['original_case_index']*64
                if raw['conditional_case'] != case['original_plan'] or raw['epsilon'] != eps:
                    raise ValueError('Original behavior case differs')
                h = proposals['interaction']
            else:
                arm = component[4:]
                raw = read(root/f'case_{i:02d}_{arm}.json')
                eps = old_cfg['epsilon']; first_seed = plan['continuation_seed_start']
                if raw['plan'] != plan or raw['epsilon'] != eps or raw['candidate'] != arm:
                    raise ValueError('Comparison behavior case differs')
                h = proposals[arm]
            samples = raw['sequences']
            if [s['seed'] for s in samples] != list(range(first_seed, first_seed+64)):
                raise ValueError('Missing, repeated or changed continuation schedule')
            factors = proposal_factors(p, h, eps)
            for sample in samples:
                audit_sample(sample, eps)
                log = sample['initial_likelihood_audit']; a, b = log['action_indices']
                if not (0 <= a < p.shape[0] and 0 <= b < p.shape[1]) or p[a, b] <= 0:
                    raise ValueError('Sample outside natural support')
                for offset, key in ((0, 'p_factors'), (2, 'h_factors'), (4, 'q_factors')):
                    if not np.allclose(log[key], [factors[offset][a], factors[offset+1][a, b]], atol=1e-15, rtol=1e-10):
                        raise ValueError('Sampled factor does not match locked matrix')
                event = any(case['state']['cav_id'] in pair for pair in sample['collision_pairs'])
                if event != sample['event_result'] or sample['bv_only_collision'] != (bool(sample['collision_pairs']) and not event):
                    raise ValueError('CAV/competing-endpoint labels differ')
                observations.append({'case_id': i, 'source_seed': seed, 'component': component,
                    'seed': sample['seed'], 'action': [a, b], 'event': event,
                    'bin': int(bins[a, b]), 'p': float(p[a, b]),
                    'behavior_q': float(factors[4][a]*factors[5][a, b]),
                    'design_mass': plan['target_mass']/(7*64)})
    if len(observations) != 31*7*64 or not np.isclose(sum(r['design_mass'] for r in observations), 1.):
        raise ValueError('Incomplete source-balanced behavior design')
    lock = {'contract': CONTRACT, 'config_sha256': digest(config), 'config_identity': identity(cfg),
        'implementation_sha256': digest(__file__),
        'allocation_sha256': digest(Path(__file__).with_name('highd_v47_allocation.py')),
        'training_comparison_summary_sha256': source['summary_sha256'],
        'training_input_lock': old_lock, 'observation_table_sha256': identity(observations),
        'conditional_target_sha256': old_lock['conditional_target_sha256'],
        'environment': cfg['environment'], 'validation_sources_opened': False}
    lock['proposal_structure_sha256'] = identity(lock)
    return cfg, records, observations, lock


def freeze(config, output):
    *_, lock = load_inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def candidates_for_case(record, model, epsilon):
    p = record['p']; failure = np.asarray(model['probabilities'])[record['bins']]
    candidates = {name: proposal_factors(p, h, epsilon)[-2:] for name, h in record['proposals'].items()}
    h = p*failure; h /= h.sum()
    candidates['calibrated_linear'] = proposal_factors(p, h, epsilon)[-2:]
    candidates['allocated_first'] = allocate(p, failure, epsilon, first_only=True)
    candidates['allocated_dual'] = allocate(p, failure, epsilon)
    result = {}
    for name, (q1, q2) in candidates.items():
        check_q(p, q1, q2, epsilon)
        result[name] = {'q1': q1, 'q2': q2, 'q': q1[:, None]*q2,
            'factor_tv': factor_tv(p, q1, q2, epsilon),
            'plugin_M2': second_moment(p, failure, q1[:, None]*q2)}
    optimum = result['allocated_dual']['plugin_M2']
    if any(optimum > r['plugin_M2']*(1+1e-10)+1e-15 for r in result.values()):
        raise ArithmeticError('Analytic allocator exceeds a feasible comparator objective')
    return failure, result


def crossfit(records, rows, cfg):
    sources = sorted({r['source_seed'] for r in records})
    models = {s: fit_calibrator(rows, cfg['calibration'], excluded_source=s) for s in sources}
    prepared = {}
    for r in records:
        model = models[r['source_seed']]
        if r['source_seed'] in model['source_seeds']:
            raise ValueError('Source leaked into its fitted probability table')
        failure, proposals = candidates_for_case(r, model, cfg['epsilon'])
        prepared[r['case_id']] = {'failure': failure, 'proposals': proposals}
    return models, prepared


def summarize_offpolicy(records, rows, prepared):
    """Each behavior estimates J via E_g[Y P**2/(g Q)]; average components.

    Case/source masses are external target weights. No behavior mixture is
    mislabelled as a new-Q sample, and paired component rows are not IID counts.
    """
    sources = sorted({r['source_seed'] for r in records})
    result = {}
    for name in CANDIDATES:
        by_source = dict.fromkeys(sources, 0.)
        by_component, paired_blocks = defaultdict(float), defaultdict(float)
        native_mean = 0.
        event_ratios = []
        for row in rows:
            a, b = row['action']; candidate = prepared[row['case_id']]['proposals'][name]
            q = float(candidate['q'][a, b])
            term = row['design_mass']*row['event']*(row['p']/row['behavior_q'])*(row['p']/q)
            by_source[row['source_seed']] += term
            by_component[row['component']] += 7*term
            paired_blocks[(row['source_seed'], row['case_id'], row['seed'])] += term
            native_mean += row['design_mass']*row['event']*row['p']/row['behavior_q']
            if row['event']:
                event_ratios.append(row['p']/q)
        total = float(sum(by_source.values()))
        shares = [x/total for x in by_source.values()] if total else []
        result[name] = {'offpolicy_conditional_M2': total,
            'behavior_IS_conditional_mean': float(native_mean),
            'plugin_conditional_M2': float(sum(r['target_mass']*prepared[r['case_id']]['proposals'][name]['plugin_M2'] for r in records)),
            'source_weighted_factor_TV': np.sum([r['target_mass']*np.asarray(prepared[r['case_id']]['proposals'][name]['factor_tv']) for r in records], axis=0).tolist(),
            'maximum_source_M2_share': max(shares) if shares else None,
            'effective_M2_sources_descriptive': 1/sum(x*x for x in shares) if shares else None,
            'maximum_paired_seed_block_contribution_share': max(paired_blocks.values())/total if total else None,
            'candidate_action_LR_range_at_logged_events': [min(event_ratios), max(event_ratios)] if event_ratios else None,
            'by_behavior_component_M2': dict(by_component),
            'by_source': [{'source_seed': s, 'M2_contribution': m, 'share': m/total if total else None} for s, m in by_source.items()],
            'new_Q_event_count': None, 'source_generalization_interval': None}
    for report in result.values():
        for baseline in ('interaction', 'mixed'):
            denominator = result[baseline]['offpolicy_conditional_M2']
            report['offpolicy_M2_ratio_to_'+baseline] = report['offpolicy_conditional_M2']/denominator if denominator else None
    return result


def calibration_report(records, rows, models, prepared, cfg):
    metrics = {name: {'weighted_Brier': 0., 'weighted_log_loss': 0.} for name in ('raw_clearance', 'source_held_out_global', 'source_held_out_bins')}
    bin_stats = [{'weighted_mass': 0., 'weighted_events': 0., 'weighted_prediction': 0.} for _ in range(10)]
    lookup = {r['case_id']: r for r in records}
    for row in rows:
        a, b = row['action']; key = row['case_id']; source = row['source_seed']
        predictions = {'raw_clearance': lookup[key]['risk'][a, b],
            'source_held_out_global': models[source]['global_probability'],
            'source_held_out_bins': prepared[key]['failure'][a, b]}
        weight = row['design_mass']*row['p']/row['behavior_q']; y = int(row['event'])
        for name, prob in predictions.items():
            safe = np.clip(prob, cfg['calibration']['probability_floor'], 1-cfg['calibration']['probability_floor'])
            metrics[name]['weighted_Brier'] += float(weight*(y-prob)**2)
            metrics[name]['weighted_log_loss'] -= float(weight*(y*np.log(safe)+(1-y)*np.log1p(-safe)))
        group = bin_stats[row['bin']]
        group['weighted_mass'] += weight; group['weighted_events'] += weight*y
        group['weighted_prediction'] += weight*predictions['source_held_out_bins']
    for group in bin_stats:
        mass = group['weighted_mass']
        group['observed_probability'] = group['weighted_events']/mass if mass else None
        group['mean_prediction'] = group['weighted_prediction']/mass if mass else None
    event_sources = sorted({r['source_seed'] for r in rows if r['event']})
    return {'metrics': metrics, 'out_of_fold_bins': bin_stats, 'event_sources': event_sources,
        'minimum_fit_event_sources_across_folds': min(len(m['event_sources']) for m in models.values()),
        'source_diversity_review_guard_passed': len(event_sources) >= cfg['minimum_event_sources_for_calibration_review'],
        'shared_seed_blocks': len({(r['case_id'], r['seed']) for r in rows}),
        'scope': 'Source-cross-fitted training diagnostics; smoothing is not a posterior and no source-bootstrap interval is asserted'}


def run(config, lock_path, output):
    start = time.perf_counter()
    cfg, records, rows, current = load_inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in current.items()):
        raise ValueError('Training inputs, implementation or protocol changed after freeze')
    destination = Path(output); destination.mkdir(parents=True, exist_ok=False)
    save(destination/'opened_protocol.json', {'config': cfg, 'lock': lock, 'rule_lock_sha256': digest(lock_path)})
    models, prepared = crossfit(records, rows, cfg)
    save(destination/'source_folds.json', {str(k): v for k, v in models.items()})
    save(destination/'full_training_calibrator.json', fit_calibrator(rows, cfg['calibration']))
    for r in records:
        item = prepared[r['case_id']]
        save(destination/f'case_{r["case_id"]:02d}_allocation.json', {
            **{k: r[k] for k in ('case_id', 'source_seed', 'target_mass', 'state_sha256', 'actor_ids')},
            'fitting_sources': models[r['source_seed']]['source_seeds'],
            'failure_probability_estimate': item['failure'].tolist(),
            'proposals': {name: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in q.items()}
                          for name, q in item['proposals'].items()}})
    reports = summarize_offpolicy(records, rows, prepared)
    calibration = calibration_report(records, rows, models, prepared, cfg)
    components = {name: {'rows': sum(r['component'] == name for r in rows),
        'events': sum(r['component'] == name and r['event'] for r in rows),
        'event_sources': sorted({r['source_seed'] for r in rows if r['component'] == name and r['event']})}
        for name in cfg['behavior_components']}
    result = {'complete': True, 'split': 'train', 'contract': CONTRACT, 'config': cfg,
        'lock': lock, 'rule_lock_sha256': digest(lock_path), 'cases': len(records),
        'source_episodes': len(models), 'behavior_rows': len(rows), 'behavior_components': components,
        'calibration': calibration, 'reports': reports,
        'artifact_hashes': {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
        'wall_s': time.perf_counter()-start, 'new_native_continuations': 0,
        'validation_sources_opened': False, 'ppo_trained': False, 'training_exported': False,
        'automatically_selected_candidate': None, 'population_acceleration': None,
        'scope': 'Fixed training state law and seven known behavior components; corrected off-policy M2 diagnostics, not new-Q native performance. Models exclude the entire evaluated source. Fold models share other training sources; no naive IID or bootstrap significance claim.'}
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
        lock = freeze(args.config, args.output)
        print({'proposal_structure_sha256': lock['proposal_structure_sha256'], 'validation_sources_opened': False})
    else:
        if not args.lock:
            parser.error('run requires --lock')
        result = run(args.config, args.lock, args.output)
        print({'event_sources': result['calibration']['event_sources'], 'new_native_continuations': 0,
            'offpolicy_M2': {k: v['offpolicy_conditional_M2'] for k, v in result['reports'].items()}})


if __name__ == '__main__':
    main()
