"""Audit and export a training-only union of two locked native calibrations.

All eligible cases, including zero-event cases, enter the uniform-source law.
No old validation bank is relabelled and no policy is trained by this module.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_v46_conditional_baseline import conditional_observation, make_sequences, MANIFEST
from .highd_v46_contract import CONDITIONAL_OBSERVATION
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import dependencies
from .highd_v46_native_calibration import aggregate, case_masses, select_cases, validate_config
from .highd_v46_statistics import episode_values


def load_calibration(path, expected_config, expected_sha):
    path = Path(path)
    if digest(path/'summary.json') != expected_sha:
        raise ValueError('Calibration summary checksum differs')
    summary = read(path/'summary.json')
    cfg = expected_config
    if not summary['complete'] or summary['split'] != 'train' or summary['config'] != cfg:
        raise ValueError('Only complete matching training calibrations are accepted')
    if summary['config_identity'] != identity(cfg):
        raise ValueError('Calibration config identity differs')
    dc = read(cfg['discovery_config']); base = read(dc['base_discovery_config'])
    validate_config(cfg, dc)
    deps = dependencies(cfg, dc, base)
    deps['highd_v46_multistate_q.py'] = digest(Path(__file__).with_name('highd_v46_multistate_q.py'))
    if (summary['dependencies'] != deps or summary['environment'] != cfg['environment']
            or summary['implementation_sha256'] != digest(Path(__file__).with_name('highd_v46_native_calibration.py'))):
        raise ValueError('Calibration implementation or environment differs')
    for name, sha in summary['artifacts'].items():
        if Path(name).name != name or digest(path/name) != sha:
            raise ValueError('Calibration artifact checksum/path differs: '+name)
    required = {'state_bank.json', 'pair_audit.json', 'case_plan.json', 'native_calibration.json'}
    if not required <= summary['artifacts'].keys():
        raise ValueError('Missing artifact lock')
    bank = read(path/'state_bank.json')['states']; rows = read(path/'pair_audit.json')['rows']
    plan = read(path/'case_plan.json')
    expected = select_cases(bank, rows, cfg['cases'])
    plans = plan['cases']
    if plan['selection_uses_native_outcomes'] or len(plans) != len(expected):
        raise ValueError('Case plan differs from outcome-blind selection')
    source = cfg['source']
    source_seeds = set(range(source['seed_start'], source['seed_start']+source['episodes']))
    if [e['seed'] for e in summary['discovery']['episodes']] != sorted(source_seeds):
        raise ValueError('Missing or duplicated natural source episode')
    cases, raw, records, values = [], [], [], {a: [] for a in cfg['arms']}
    for i, (p, chosen) in enumerate(zip(plans, expected)):
        if any(p[k] != value for k, value in chosen.items()):
            raise ValueError('Case plan was changed after selection')
        state = bank[p['state_index']]
        if (state['seed'] not in source_seeds or p['source_seed'] != state['seed']
                or p['state_sha256'] != identity(state) or p['time_s'] != state['time_s']):
            raise ValueError('State/source identity differs')
        arms = {}
        for arm, epsilon in cfg['arms'].items():
            name = f'case_{i:02d}_{arm}.json'
            if name not in summary['artifacts']:
                raise ValueError('Missing continuation artifact lock')
            data = read(path/name); seqs = data['sequences']; n = cfg['replicates_per_case_arm']
            if data['conditional_case'] != p or data['epsilon'] != epsilon:
                raise ValueError('Continuation case or behavior differs')
            if [s['seed'] for s in seqs] != list(range(cfg['continuation_seed_start']+i*n, cfg['continuation_seed_start']+(i+1)*n)):
                raise ValueError('Incomplete or changed fixed continuation schedule')
            for s in seqs:
                audit = s['initial_likelihood_audit']
                if s['event_result'] and s['bv_only_collision']:
                    raise ValueError('Conflicting endpoint labels')
                if not audit.get('abstained'):
                    from d2rl_training.highd_v46_sequence_env import log_weight
                    if (audit['generation_epsilon'] != epsilon
                            or not np.allclose(audit['q_factors'], np.asarray(epsilon)*audit['p_factors']
                                               + (1-np.asarray(epsilon))*audit['h_factors'], rtol=1e-10, atol=0)
                            or not np.isclose(log_weight(audit, epsilon), s['generation_log_weight'], atol=1e-10, rtol=0)):
                        raise ValueError('Behavior likelihood differs')
                    if not np.isclose(audit['full_log_p']-audit['full_log_q'], s['generation_log_weight'], atol=1e-10, rtol=0):
                        raise ValueError('Full/factor likelihood mismatch')
            arms[arm] = seqs
        if p['role'] != 'eligible':
            continue  # Controls retained in source artifacts, outside eligible law.
        row = rows[p['row_index']]
        if not row['views']['production_strict'] or row['seed'] != state['seed'] or row['actor_ids'] != p['selected_pair']:
            raise ValueError('Ineligible or mismatched pair')
        cases.append({'source_seed': state['seed'], 'time_s': state['time_s'], 'state_sha256': identity(state),
                      'row': row, 'observation': conditional_observation(state, row),
                      'origin_summary_sha256': expected_sha, 'original_case_index': i})
        raw.append(arms['dual']); records.append(p)
        for arm in values:
            values[arm].append(episode_values(arms[arm]))
    return summary, source_seeds, cases, raw, records, values


def concentration(sequences):
    """Unscaled empirical behavior-Q second moment; no clipping or deletion."""
    sources = defaultdict(lambda: {'events': 0, 'second_moment': 0., 'h2_zero_events': 0, 'h2_zero_second_moment': 0.})
    for s in sequences:
        row = sources[s['source_seed']]
        if s['event_result']:
            value = float(s['target_mass'] * np.exp(2*s['generation_log_weight']))
            row['events'] += 1; row['second_moment'] += value
            if s['steps'][0]['h_factors'][1] == 0:
                row['h2_zero_events'] += 1; row['h2_zero_second_moment'] += value
    total = sum(r['second_moment'] for r in sources.values())
    for r in sources.values():
        r['fraction'] = r['second_moment']/total if total > 0 else None
    fractions = [r['fraction'] for r in sources.values()] if total > 0 else []
    return {'weighted_event_second_moment': total, 'source_episodes': len(sources),
            'event_source_episodes': sum(r['events'] > 0 for r in sources.values()),
            'top_source_fraction': max(fractions) if fractions else None,
            'effective_second_moment_sources_descriptive': 1/sum(x*x for x in fractions) if fractions else None,
            'by_source': dict(sources)}


def export(config, expansion, expansion_sha, output):
    cfg = read(config)
    original_cfg = read('configs/highd_v46_native_calibration_v1.json')
    original = cfg['original_training']
    bundles = [load_calibration(original['path'], original_cfg, original['summary_sha256']),
               load_calibration(expansion, cfg, expansion_sha)]
    if bundles[0][1] & bundles[1][1]:
        raise ValueError('Training source cohorts overlap')
    _, frozen = ensure_assets()
    for key in ('natural_target_sha256', 'experiment_target_sha256'):
        if bundles[0][0][key] != bundles[1][0][key]:
            raise ValueError('Physical targets differ')
    if bundles[0][0]['natural_target_sha256'] != frozen['natural_target_sha256']:
        raise ValueError('Frozen natural target differs')
    cases = bundles[0][2]+bundles[1][2]; raw = bundles[0][3]+bundles[1][3]
    sequences = make_sequences(cases, raw)
    if len({s['seed'] for s in sequences}) != len(sequences):
        raise ValueError('Training continuation seeds overlap')
    locks = [original['summary_sha256'], expansion_sha]
    structure = identity({'scope': 'training_expansion_v1', 'config': cfg, 'sources': locks,
                          'exporter_sha256': digest(__file__), 'converter_sha256': digest(Path(__file__).with_name('highd_v46_conditional_baseline.py'))})
    law = identity({'source_state_masses': [(c['state_sha256'], float(m)) for c, m in zip(cases, case_masses(cases))]})
    data = {'contract': 'highd_v46_conditional_d2rl_sequence_v1', 'observation_contract': CONDITIONAL_OBSERVATION,
            'action_dimension': 2, 'intervention_mode': 'dual', 'sampling_scope': 'conditional_state_bank',
            'natural_target_sha256': frozen['natural_target_sha256'],
            'parent_physical_target_sha256': bundles[0][0]['experiment_target_sha256'],
            'experiment_target_sha256': identity({'training_expansion': structure}),
            'proposal_structure_sha256': structure, 'conditional_target_sha256': law,
            'initialization_law': cfg['conditional_law'], 'origin_summary_sha256': locks, 'sequences': sequences}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save(output/'sequences.json', data)
    save(output/MANIFEST, {'data_path': 'sequences.json', 'sha256': digest(output/'sequences.json'),
                          **{k: data[k] for k in ('natural_target_sha256', 'experiment_target_sha256',
                              'conditional_target_sha256', 'proposal_structure_sha256', 'intervention_mode')}})
    from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
    env = V46SequenceEnv({'sequence_manifest': str(output/MANIFEST), 'evaluation_only': True})
    measured = env.empirical_objective(lambda obs: [.1, .1]) * np.exp(env.log_scale)
    report = {'scope': 'Training-only union; no fresh validation or policy improvement claim',
              'config_sha256': digest(config), 'implementation_sha256': digest(__file__), 'source_summary_sha256': locks,
              'natural_source_episodes_scanned': sum(len(b[1]) for b in bundles),
              'cases': len(cases), 'sequences': len(sequences), 'events': sum(s['event_result'] for s in sequences),
              'source_mechanisms': dict(Counter(c['row']['mechanism'] for c in cases)),
              'before': concentration(make_sequences(bundles[0][2], bundles[0][3])), 'after': concentration(sequences),
              'four_arm_training_diagnostic': aggregate(bundles[0][4]+bundles[1][4],
                  {a: bundles[0][5][a]+bundles[1][5][a] for a in cfg['arms']}, cfg['statistics']),
              'manifest_sha256': digest(output/MANIFEST), 'ppo_trained': False, 'fresh_validation_performed': False,
              'population_acceleration': None, 'checkpoint_selection_ready': False}
    if not np.isclose(measured, report['after']['weighted_event_second_moment'], rtol=1e-10, atol=0):
        raise ValueError('Replay and native second moment differ')
    save(output/'training_quality.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v46_training_expansion_v1.json')
    parser.add_argument('--expansion', required=True)
    parser.add_argument('--expansion-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = export(args.config, args.expansion, args.expansion_sha256, args.output)
    print({k: report[k] for k in ('cases', 'sequences', 'events', 'checkpoint_selection_ready')})


if __name__ == '__main__':
    main()
