"""Fresh matched-source validation of the predeclared normalized-H family.

All proposals share the frozen strict selector's state/pair and conditional
target. No consumed validation case, native label or PPO model is fitted here.
"""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np

from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, read, save
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import environment, sample_continuation
from .highd_v46_proposal_comparison import (
    DEFAULT_CONFIG as TRAIN_CONFIG, compare_reports, load_inputs, prepare, summarize,
)
from .highd_v46_source_selector_validation import discover, mechanism, select_per_source
from .highd_v46_source_statistics import summarize_confirmation

DEFAULT_CONFIG = 'configs/highd_v46_proposal_validation_v1.json'
CONSUMED_SOURCES = ((100000, 101023), (110000, 110015), (120000, 120031), (140000, 140007),
    (160000, 160015), (170000, 170007), (180000, 180031), (190000, 190031), (200000, 200063),
    (210000, 210031), (220000, 220031), (230000, 230063), (240000, 240255))


def validate_cohort(cfg):
    source = cfg['source']; n = cfg['replicates_per_case_candidate']
    seeds = set(range(source['seed_start'], source['seed_start']+source['episodes']))
    if (source['split'] != 'validation' or source['episodes'] != 256 or source['seed_start'] < 250000
            or n != 64 or any(seeds & set(range(a, b+1)) for a, b in CONSUMED_SOURCES)):
        raise ValueError('Use a fresh episode-disjoint fixed validation cohort')
    if (cfg['candidates'] != {'interaction': 1., 'absolute_risk': 0., 'mixed': .5}
            or cfg['selector'] != 'Frozen strict B, maximum supported proxy risk, time/actor-ID ties'
            or cfg['bank_method'] != 'mixed5_top2' or cfg['horizon_s'] != 4
            or cfg['epsilon'] != [.1, .1]
            or cfg['later_actions'] != 'Unchanged natural NDE; only initial decision is tilted'):
        raise ValueError('Keep the fixed selector, three scores and native budget')
    confirm = cfg['confirmation']
    if (confirm['maximum_source_episodes'] != 16 or confirm['replicates_per_corner'] != 16
            or confirm['minimum_native_interaction'] != .05):
        raise ValueError('Unsupported eligible-source confirmation budget')
    q = set(range(cfg['continuation_seed_start'], cfg['continuation_seed_start']+source['episodes']*n))
    c = set(range(confirm['continuation_seed_start'], confirm['continuation_seed_start']+source['episodes']*confirm['replicates_per_corner']))
    if q & c or min(q | c) < 4200000:
        raise ValueError('Validation/confirmation seeds overlap each other or consumed training draws')


def inputs(config):
    cfg = read(config); validate_cohort(cfg)
    train_cfg, _, _, base, physical, assets, training_rule = load_inputs(TRAIN_CONFIG)
    if cfg['statistics'] != train_cfg['statistics'] or cfg['environment'] != environment():
        raise ValueError('Keep the locked evidence guards and environment')
    train = cfg['training_comparison']; root = Path(train['path'])
    if digest(root/'summary.json') != train['summary_sha256']:
        raise ValueError('Completed training comparison checksum differs')
    summary = read(root/'summary.json')
    if not summary['complete'] or summary['split'] != 'train' or summary['config'] != train_cfg:
        raise ValueError('Expected complete training-only proposal evidence')
    if any(summary['lock'].get(k) != v for k, v in training_rule.items()):
        raise ValueError('Training implementation/target lock differs')
    for name, sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Training artifact differs: '+name)
    dc = read(train_cfg['discovery_config'])
    if ([dc['bank'][k] for k in ('geometry_states', 'uniform_states', 'phase_states')] != [2, 2, 1]
            or dc['pairs']['ranked'] != 2 or dc['pairs']['audit'] != 1):
        raise ValueError('Keep bounded discovery, not exhaustive pair enumeration')
    deps = dict(training_rule['dependencies'])
    deps.update({name: digest(Path(__file__).with_name(name)) for name in (
        'highd_v46_proposal_comparison.py', 'highd_v46_source_selector_validation.py', 'highd_v46_source_statistics.py')})
    lock = {'config_sha256': digest(config), 'config_identity': identity(cfg),
        'implementation_sha256': digest(__file__), 'dependencies': deps,
        'environment': environment(), 'training_summary_sha256': train['summary_sha256'],
        'natural_target_sha256': training_rule['natural_target_sha256'],
        'physical_target_sha256': training_rule['physical_target_sha256'],
        'proposal_family_sha256': training_rule['proposal_structure_sha256'],
        'proposal_structure_sha256': identity({'family': training_rule['proposal_structure_sha256'],
            'selector': cfg['selector'], 'bank_method': cfg['bank_method'],
            'collector_sha256': digest(__file__), 'dependencies': deps}),
        'validation_sources_opened': False}
    return cfg, dc, base, physical, assets, lock


def freeze(config, output):
    *_, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def make_plan(bank, rows, cfg):
    """One common case/source and eligible-source probes before any outcomes."""
    source = cfg['source']; seeds = list(range(source['seed_start'], source['seed_start']+source['episodes']))
    selections = select_per_source(rows, seeds)['B']
    plans = []
    for seed in seeds:
        index = selections[seed]
        if index is None:
            continue
        row = rows[index]; state = bank[row['state_index']]
        if state['seed'] != seed or state['time_s'] != row['time_s']:
            raise ValueError('Selected state/source differs')
        a0, b0 = row['metrics']['baseline_actions']; a, b = row['metrics']['max_interaction_actions']
        plans.append({'case_id': len(plans), 'source_seed': seed, 'time_s': state['time_s'],
            'row_index': index, 'state_index': row['state_index'], 'state_sha256': identity(state),
            'actor_ids': row['actor_ids'], 'confirmation_corners': [(a0, b0), (a, b0), (a0, b), (a, b)],
            'proxy_interaction': row['metrics']['max_positive_interaction'],
            'continuation_seed_start': cfg['continuation_seed_start']+(seed-seeds[0])*cfg['replicates_per_case_candidate']})
    pool = [p['source_seed'] for p in plans]
    rng = np.random.default_rng(cfg['confirmation']['source_selection_seed'])
    probes = sorted(int(s) for s in rng.choice(pool, min(len(pool), cfg['confirmation']['maximum_source_episodes']), replace=False))
    for p in plans: p['confirmation_planned'] = p['source_seed'] in probes
    return plans, seeds, probes


def run(config, lock_path, output):
    cfg, dc, base, physical, assets, current = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in current.items()):
        raise ValueError('Proposal-validation rule changed after freeze')
    destination = Path(output); destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(destination/'opened_protocol.json', {'config': cfg, 'lock': lock, 'rule_lock_sha256': digest(lock_path)})
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    bank, rows, discovery = discover(cfg, dc, base, physical, model, assets)
    plans, seeds, probes = make_plan(bank, rows, cfg)
    if [e['seed'] for e in discovery['episodes']] != seeds:
        raise ValueError('Incomplete fixed source cohort')
    save(destination/'state_bank.json', {'split': 'validation', 'states': bank})
    save(destination/'pair_audit.json', {'rows': rows})
    prepared = []; tick = time.process_time()
    for plan in plans:
        case = {'state': bank[plan['state_index']], 'row': rows[plan['row_index']]}
        p, risk, proposals = prepare(case, model, physical, base, cfg['candidates'], cfg['epsilon'])
        if not all(p[a, b] > 0 for a, b in plan['confirmation_corners']):
            raise ValueError('Locked native confirmation corners are unsupported')
        save(destination/f'case_{plan["case_id"]:03d}_proposals.json', {'plan': plan, 'P': p.tolist(), 'risk': risk.tolist(),
             'proposals': {name: {**v, 'h': v['h'].tolist()} for name, v in proposals.items()}})
        prepared.append(proposals)
    selector_cpu = time.process_time()-tick
    save(destination/'case_plan.json', {'source_seeds': seeds, 'cases': plans, 'confirmation_source_seeds': probes,
        'selection_uses_native_outcomes': False, 'conditional_law': 'Uniform selected source and its one fixed strict-B state'})
    records = {name: [] for name in cfg['candidates']}; confirmations = []
    n = cfg['replicates_per_case_candidate']
    for plan, proposals in zip(plans, prepared):
        state = bank[plan['state_index']]
        for name, chosen in proposals.items():
            seqs = [sample_continuation(model, assets, physical, state, chosen, cfg['epsilon'],
                    plan['continuation_seed_start']+j, cfg, base['proxy']['clearance_scale_m']) for j in range(n)]
            save(destination/f'case_{plan["case_id"]:03d}_{name}.json', {'plan': plan, 'candidate': name,
                 'interaction_fraction': cfg['candidates'][name], 'epsilon': cfg['epsilon'], 'sequences': seqs})
            records[name].append({'source_seed': plan['source_seed'], 'epsilon': cfg['epsilon'],
                                 'factor_tv': chosen['factor_tv'], 'sequences': seqs})
        if plan['confirmation_planned']:
            probe = mechanism(model, assets, physical, state, plan, cfg, base['proxy']['clearance_scale_m'])
            probe['proxy_interaction'] = plan['proxy_interaction']
            confirmations.append(probe); save(destination/f'confirmation_{plan["case_id"]:03d}.json', probe)
        print({'phase': 'proposal_native_validation', 'case': plan['case_id']+1, 'total': len(plans),
               'source': plan['source_seed']}, flush=True)
    reports = {name: summarize(group, cfg['statistics']) for name, group in records.items()} if plans else {}
    for name, report in reports.items():
        report['observed_event_source_yield_all_sources'] = len(report['event_sources'])/len(seeds)
        report['effective_pair_source_coverage_all_sources'] = sum(min(r['factor_tv']) >= cfg['statistics']['minimum_effective_factor_tv'] for r in records[name])/len(seeds)
        report['native_CPU_s'] = sum(s['cpu_s'] for r in records[name] for s in r['sequences'])
    result = {'complete': True, 'split': 'validation', 'config': cfg, 'lock': lock,
        'rule_lock_sha256': digest(lock_path), 'natural_sources': len(seeds), 'selected_sources': len(plans),
        'all_source_candidate_coverage': len(plans)/len(seeds), 'discovery': discovery,
        'bank_states': len(bank), 'bounded_pair_audits': len(rows),
        'strict_positive_pairs': sum(r['views']['production_strict'] for r in rows),
        'strict_positive_states': len({r['state_index'] for r in rows if r['views']['production_strict']}),
        'proxy_CPU_s': sum(r['proxy_cpu_s'] for r in rows),
        'selector_reaudit_CPU_s': selector_cpu, 'reports': reports,
        'matched_comparison': compare_reports(reports, cfg['statistics']) if plans else {'status': 'no_eligible_cases'},
        'native_confirmation': summarize_confirmation(confirmations, cfg['confirmation']['minimum_native_interaction']),
        'proxy_native_calibration': [{k: r[k] for k in ('source_seed', 'proxy_interaction',
             'native_interaction_mean', 'fixed_state_interaction_mcse')} for r in confirmations if r['status'] == 'measured'],
        'conditional_target_sha256': identity([(p['state_sha256'], p['actor_ids'], 1/len(plans)) for p in plans]),
        'artifact_hashes': {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
        'wall_s': time.perf_counter()-start, 'ppo_trained': False, 'training_exported': False,
        'population_acceleration': None, 'automatically_selected_candidate': None,
        'scope': 'Matched conditional states; source generalization only, not population-initialized testing'}
    save(destination/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--lock'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze(args.config, args.output)
        print({'proposal_structure_sha256': result['proposal_structure_sha256'], 'validation_sources_opened': False})
    else:
        if not args.lock: parser.error('run requires --lock')
        result = run(args.config, args.lock, args.output)
        print({'selected_sources': result['selected_sources'], 'events': {k: v['events'] for k, v in result['reports'].items()}})


if __name__ == '__main__':
    main()
