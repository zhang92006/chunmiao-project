"""Prepare, train and natively evaluate a conditional dual-BV PPO baseline.

The state law is explicit and source-stratified. This entry point never exports
natural-initialization collections or claims population testing acceleration.
"""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import numpy as np

from .highd_v46_assets import read, save, digest, ensure_assets
from .highd_v46_contract import CONDITIONAL_OBSERVATION, enrich_observation
from .highd_v46_d2rl import identity, observation, experiment_target
from .highd_v46_native_calibration import discover, select_cases, case_masses, aggregate
from .highd_v46_multistate_q import environment, dependencies, prepare_case, sample_continuation, compare_strata
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_statistics import episode_values


DEFAULT_CONFIG = 'configs/highd_v46_conditional_baseline_v1.json'
MANIFEST = 'conditional_sequence_manifest.json'


def protocol(path):
    cfg = read(path)
    calibration = read(cfg['calibration_config'])
    dc = read(calibration['discovery_config'])
    base = read(dc['base_discovery_config'])
    physical = read(base['physical_config'])
    splits = []
    for name in ('validation', 'test'):
        part = cfg[name]
        seeds = set(range(part['seed_start'], part['seed_start'] + part['episodes']))
        if not seeds or part['replicates'] < 2 or any(seeds & set(range(a, b+1)) for a, b in cfg['forbidden_source_ranges']):
            raise ValueError('Invalid or consumed source cohort: ' + name)
        splits.append(seeds)
    if splits[0] & splits[1]:
        raise ValueError('Validation and test source episodes overlap')
    if environment() != calibration['environment']:
        raise ValueError('Use the pinned Python 3.9 environment and set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1')
    deps = dependencies(calibration, dc, base)
    deps.update({name: digest(Path(__file__).with_name(name)) for name in
                 ('highd_v46_native_calibration.py', 'highd_v46_multistate_q.py', 'highd_v46_contract.py')})
    structure = identity({'contract': 'conditional_one_decision_dual_v1', 'config': cfg,
                          'dependencies': deps, 'implementation': digest(__file__)})
    return cfg, calibration, dc, base, physical, structure


def conditional_observation(state, row):
    """Snapshots lack past acceleration: explicit zero slots in a new contract."""
    cars = [SimpleNamespace(**c) for c in state['vehicles']]
    actors = {c.id: c for c in cars}
    road = SimpleNamespace(vehicles=cars, cav=actors[state['cav_id']], cav_acceleration=0., last_acceleration={})
    ids = row['actor_ids']; rear, front = [actors[i] for i in ids]
    # Positive four-corner interaction always has a zero baseline row/column.
    return enrich_observation(observation(road, ids), row['metrics']['positive_interaction_mass'],
                              1., 0., 0., 4., [1, 1], front.x-front.length-rear.x, front.v-rear.v)


def make_sequences(cases, raw_cases):
    """Convert every dual continuation, with exact fixed source/state masses."""
    sequences = []
    masses = case_masses(cases)
    for i, (case, raw, mass) in enumerate(zip(cases, raw_cases, masses)):
        if not raw:
            raise ValueError('Empty conditional stratum')
        for seq in raw:
            audit = seq['initial_likelihood_audit']
            if audit.get('abstained') or len(audit['p_factors']) != 2:
                raise ValueError('Expected the selected dual-factor behavior law')
            step = {**audit, 'observation': case['observation'], 'factor_mask': [1, 1],
                    'actor_ids': case['row']['actor_ids'], 'time_s': 0., 'source_time_s': case['time_s']}
            sequences.append({'seed': seq['seed'], 'source_seed': case['source_seed'], 'case_id': str(i),
                'target_mass': float(mass/len(raw)), 'event_result': seq['event_result'],
                'bv_only_collision': seq['bv_only_collision'], 'generation_log_weight': seq['generation_log_weight'],
                'steps': [step]})
    if len(raw_cases) != len(cases) or not cases:
        raise ValueError('Missing conditional cases')
    return sequences


def prepare(config, split, output, policy_path=None, fixed_path=None):
    cfg, cal, dc, base, physical, structure = protocol(config)
    test_lock = None
    if split == 'test':
        if not policy_path or not fixed_path: raise ValueError('Freeze learned and selected-fixed policies before opening test sources')
        from .highd_v46_policy import NumpyPolicy
        policies = [NumpyPolicy(p) for p in (policy_path, fixed_path)]
        if any(p.metadata.get('proposal_structure_sha256') != structure for p in policies):raise ValueError('Test policy protocol differs')
        test_lock = {name: {'weights_sha256': p.metadata['weights_sha256'], 'metadata_sha256': digest(Path(path)/'policy.json')}
                     for name, p, path in zip(('ppo', 'fixed'), policies, (policy_path, fixed_path))}
    assets, frozen = ensure_assets()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    source_summary_hash = None
    if split == 'train':
        source = Path(cfg['training_source'])
        if digest(source/'summary.json') != cfg['training_summary_sha256']:
            raise ValueError('Training calibration lock differs; do not silently repin it')
        summary = read(source/'summary.json')
        if summary['split'] != 'train' or not summary['complete'] or summary['natural_target_sha256'] != frozen['natural_target_sha256']:
            raise ValueError('Expected a complete frozen-P training calibration')
        for name, sha in summary['artifacts'].items():
            if digest(source/name) != sha: raise ValueError('Training artifact differs: '+name)
        bank = read(source/'state_bank.json')['states']; rows = read(source/'pair_audit.json')['rows']
        plans = read(source/'case_plan.json')['cases']; discovery = summary['discovery']
        source_summary_hash = digest(source/'summary.json')
        retained = [(i, p) for i, p in enumerate(plans) if p['role'] == 'eligible']
    else:
        local = dict(cal, source={'split': split, **cfg[split]})
        model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
        bank, rows, discovery = discover(local, dc, base, physical, model, assets)
        plans = select_cases(bank, rows, cal['cases'])
        retained = [(i, p) for i, p in enumerate(plans) if p['role'] == 'eligible']
        # Retain all discovery outcomes, including unselected/negative states.
        save(output/'state_bank.json', {'split': split, 'states': bank})
        save(output/'pair_audit.json', {'rows': rows})
    cases = []
    for original_index, plan in retained:
        state, row = bank[plan['state_index']], rows[plan['row_index']]
        if not row['views']['production_strict']: raise ValueError('Ineligible case')
        cases.append({'source_seed': state['seed'], 'time_s': state['time_s'], 'state_sha256': identity(state),
                      'state': state, 'row': row, 'observation': conditional_observation(state, row),
                      'original_case_index': original_index})
    parent = identity(experiment_target(frozen, physical))
    target_family = identity({'parent_physical_target': parent, 'protocol': identity(cfg), 'scope': cfg['scope'], 'event': cfg['event']})
    law = identity({'initialization': cfg['initialization_law'], 'states':
                    [(c['state_sha256'], float(m)) for c, m in zip(cases, case_masses(cases))]})
    bundle = {'split': split, 'config_identity': identity(cfg), 'proposal_structure_sha256': structure,
              'natural_target_sha256': frozen['natural_target_sha256'], 'experiment_target_sha256': target_family,
              'parent_physical_target_sha256': parent, 'conditional_target_sha256': law,
              'source_summary_sha256': source_summary_hash, 'cases': cases, 'discovery': discovery, 'test_policy_lock': test_lock}
    # This bank and its target masses exist before validation continuation labels.
    save(output/'bank.json', bundle)
    report = {'split': split, 'cases': len(cases), 'source_episodes': len({c['source_seed'] for c in cases}),
              'bank_sha256': digest(output/'bank.json'), 'scope': cfg['scope'], 'config_identity': identity(cfg)}
    if split == 'test':
        report['native_outcomes_opened'] = False
    elif cases:
        raw_cases = []
        for i, case in enumerate(cases):
            if split == 'train':
                raw = read(source/f"case_{case['original_case_index']:02d}_dual.json")['sequences']
            else:
                chosen = prepare_case(case['state'], case['row'], model, physical, base)
                part = cfg[split]
                raw = [sample_continuation(model, assets, physical, case['state'], chosen, [.1, .1],
                       part['continuation_seed_start']+i*part['replicates']+j, cal, base['proxy']['clearance_scale_m'])
                       for j in range(part['replicates'])]
            raw_cases.append(raw)
            save(output/f'native_case_{i:02d}.json', {'sequences': raw})
            print({'split': split, 'case': i+1, 'total': len(cases), 'events': sum(s['event_result'] for s in raw)}, flush=True)
        sequences = make_sequences(cases, raw_cases)
        data = {k: bundle[k] for k in ('natural_target_sha256', 'experiment_target_sha256', 'conditional_target_sha256',
                                      'proposal_structure_sha256', 'parent_physical_target_sha256')}
        data.update(contract='highd_v46_conditional_d2rl_sequence_v1', observation_contract=CONDITIONAL_OBSERVATION,
                    action_dimension=2, intervention_mode='dual', sampling_scope='conditional_state_bank',
                    initialization_law=cfg['initialization_law'], sequences=sequences)
        save(output/'sequences.json', data)
        save(output/MANIFEST, {'data_path': 'sequences.json', 'sha256': digest(output/'sequences.json'),
             'conditional_target_sha256': law,
             **{k: data[k] for k in ('natural_target_sha256', 'experiment_target_sha256', 'proposal_structure_sha256', 'intervention_mode')}})
        from d2rl_training.highd_v46_sequence_env import V46SequenceEnv
        V46SequenceEnv({'sequence_manifest': str(output/MANIFEST), 'evaluation_only': True})
        report.update(sequences=len(sequences), cav_events=sum(s['event_result'] for s in sequences),
                      event_source_episodes=len({s['source_seed'] for s in sequences if s['event_result']}),
                      sequence_sha256=digest(output/'sequences.json'), manifest=MANIFEST)
    report['wall_s'] = time.perf_counter()-tick
    save(output/'preparation.json', report)
    return report


def load_bank(path, cfg, structure):
    path = Path(path); report = read(path/'preparation.json')
    if digest(path/'bank.json') != report['bank_sha256']: raise ValueError('Bank checksum differs')
    bank = read(path/'bank.json')
    if bank['config_identity'] != identity(cfg) or bank['proposal_structure_sha256'] != structure:
        raise ValueError('Bank protocol or implementation changed')
    if 'sequence_sha256' in report:
        if digest(path/'sequences.json') != report['sequence_sha256']:raise ValueError('Prepared sequence checksum differs')
        if read(path/'sequences.json')['conditional_target_sha256'] != bank['conditional_target_sha256']:raise ValueError('Sequence and bank target differ')
    return bank


def train(config, data, output, seed=7, iterations=None, interface_only=False):
    cfg, _, _, _, _, structure = protocol(config)
    data = Path(data); output = Path(output)
    training = load_bank(data/'train', cfg, structure)
    if training['split'] != 'train': raise ValueError('Training bank has the wrong split')
    if seed not in cfg['training_seeds']: raise ValueError('Use a predeclared training seed')
    if not interface_only:
        validation = load_bank(data/'validation', cfg, structure)
        if validation['split'] != 'validation': raise ValueError('Validation bank has the wrong split')
    output.mkdir(parents=True, exist_ok=False)
    ppo_config = cfg['ppo_config']
    if interface_only:
        ppo = dict(read(ppo_config), require_validation=False)
        ppo_config = output/'interface_config.json'; save(ppo_config, ppo)
    command = [sys.executable, '-m', 'scenario_reconstruction.highd_v46_train', '--sequence-manifest', str(data/'train'/MANIFEST),
               '--config', str(ppo_config), '--output', str(output/'ppo'), '--seed', str(seed)]
    if iterations is not None: command += ['--iterations', str(iterations)]
    if not interface_only: command += ['--validation-manifest', str(data/'validation'/MANIFEST)]
    subprocess.run(command, check=True)
    if not interface_only:
        from .highd_v46_fixed_baseline import select_fixed
        select_fixed(data/'validation'/MANIFEST, output/'selected_fixed', cfg['fixed_grid'])
    report = {'seed': seed, 'interface_only': interface_only, 'config_identity': identity(cfg),
              'ppo_summary': read(output/'ppo/training_summary.json'),
              'fixed_selection': read(output/'selected_fixed/selection.json') if not interface_only else None,
              'scope': cfg['scope'], 'population_acceleration': None}
    save(output/'baseline_run.json', report)
    return report


def evaluate(config, bank_path, policy_path, fixed_path, output, smoke=False):
    cfg, cal, dc, base, physical, structure = protocol(config)
    bank = load_bank(bank_path, cfg, structure)
    if bank['split'] != 'test' or not bank['cases']: raise ValueError('Need a nonempty held-out test bank')
    from .highd_v46_policy import NumpyPolicy
    learned, fixed = NumpyPolicy(policy_path), NumpyPolicy(fixed_path)
    for name, policy, path in (('ppo', learned, policy_path), ('fixed', fixed, fixed_path)):
        actual = {'weights_sha256': policy.metadata['weights_sha256'], 'metadata_sha256': digest(Path(path)/'policy.json')}
        if bank.get('test_policy_lock', {}).get(name) != actual:raise ValueError('Policy changed after test source lock')
    sources = {c['source_seed'] for c in bank['cases']}
    for policy in (learned, fixed):
        for key, value in [('observation_contract', CONDITIONAL_OBSERVATION), ('proposal_structure_sha256', structure),
                           ('experiment_target_sha256', bank['experiment_target_sha256']), ('sampling_scope', 'conditional_state_bank')]:
            if policy.metadata.get(key) != value: raise ValueError('Policy differs: '+key)
        if sources & set(policy.metadata.get('training_source_seeds', []) + policy.metadata.get('validation_source_seeds', [])):
            raise ValueError('Test source episode overlaps fitting/selection')
    arms = {'natural': lambda obs: [1., 1.], 'first_factor_only': lambda obs: [.1, 1.],
            'second_factor_only': lambda obs: [1., .1], 'fixed_dual': lambda obs: [.1, .1],
            'selected_fixed': fixed, 'ppo': learned}
    n = cfg['smoke_evaluation_replicates'] if smoke else cfg['test']['replicates']
    assets, _ = ensure_assets(); model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    # Lock both policies before opening any new test continuation outcomes.
    save(output/'evaluation_protocol.json', {'config': cfg, 'bank_sha256': digest(Path(bank_path)/'bank.json'),
         'ppo_weights_sha256': learned.metadata['weights_sha256'], 'fixed_weights_sha256': fixed.metadata['weights_sha256'],
         'replicates': n, 'smoke': smoke, 'arm_order': list(arms), 'conditional_target_sha256': bank['conditional_target_sha256']})
    values = {a: [] for a in arms}; records = []; wall = time.perf_counter(); totals = {a: Counter() for a in arms}
    epsilon_records = []
    for i, case in enumerate(bank['cases']):
        chosen = prepare_case(case['state'], case['row'], model, physical, base)
        for arm, policy in arms.items():
            epsilon = policy(case['observation'])
            # Full-sized stride keeps per-case seeds stable for the smoke prefix.
            seeds = range(cfg['test']['continuation_seed_start']+i*cfg['test']['replicates'],
                          cfg['test']['continuation_seed_start']+i*cfg['test']['replicates']+n)
            seq = [sample_continuation(model, assets, physical, case['state'], chosen, epsilon,
                                      seed, cal, base['proxy']['clearance_scale_m']) for seed in seeds]
            save(output/f'case_{i:02d}_{arm}.json', {'epsilon': epsilon, 'sequences': seq})
            values[arm].append(episode_values(seq))
            totals[arm].update(cav_events=sum(s['event_result'] for s in seq), bv_only=sum(s['bv_only_collision'] for s in seq),
                               cpu_s=sum(s['cpu_s'] for s in seq), wall_s=sum(s['wall_s'] for s in seq))
            epsilon_records.append({'case': i, 'source_seed': case['source_seed'], 'arm': arm, 'epsilon': epsilon})
        records.append({'source_seed': case['source_seed']})
        print({'test_case': i+1, 'total': len(bank['cases']), 'cav_events': {a: t['cav_events'] for a, t in totals.items()}}, flush=True)
    primary = aggregate(records, {('dual' if a == 'ppo' else a): v for a, v in values.items()}, cal['statistics'])['arms']
    primary['ppo'] = primary.pop('dual')
    masses = case_masses(records); transformed = {a: [np.sqrt(len(records)*m)*x for m, x in zip(masses, v)] for a, v in values.items()}
    comparisons = {}
    for reference in arms:
        if reference == 'ppo': continue
        aliases = ('natural', 'first_factor_only', 'second_factor_only')
        comparisons[reference] = compare_strata({'dual': transformed['ppo'], **{k: transformed[reference] for k in aliases}},
            {'dual': primary['ppo'], **{k: primary[reference] for k in aliases}}, cal['statistics'])['natural']
    report = {'scope': cfg['scope'], 'conditional_target_sha256': bank['conditional_target_sha256'],
              'smoke': smoke, 'replicates_per_case_arm': n, 'arms': primary, 'ppo_comparisons': comparisons,
              'raw_totals': {a: dict(t) for a, t in totals.items()}, 'policy_actions': epsilon_records,
              'wall_s': time.perf_counter()-wall, 'population_acceleration': None,
              'interpretation': 'Fixed-state conditional evidence. Source generalization and population acceleration are separate questions.'}
    save(output/'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--split', choices=['train', 'validation', 'test'], required=True); p.add_argument('--output', required=True)
    p.add_argument('--policy'); p.add_argument('--fixed')
    p = commands.add_parser('train'); p.add_argument('--data', required=True); p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--iterations', type=int); p.add_argument('--interface-only', action='store_true')
    p = commands.add_parser('evaluate'); p.add_argument('--bank', required=True); p.add_argument('--policy', required=True); p.add_argument('--fixed', required=True)
    p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.config, args.split, args.output, args.policy, args.fixed)
    elif args.command == 'train': result = train(args.config, args.data, args.output, args.seed, args.iterations, args.interface_only)
    else: result = evaluate(args.config, args.bank, args.policy, args.fixed, args.output, args.smoke)
    print({k: v for k, v in result.items() if k not in ('ppo_summary', 'cases', 'policy_actions')})


if __name__ == '__main__':
    main()
