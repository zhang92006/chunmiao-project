"""Training-only coverage audit of a locked state bank and its pair budget.

Exhaustive means all pairs within the existing nearest-eight candidate universe,
not all actors, states or traffic mechanisms. No production component is changed.
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
from .highd_v46_state_discovery import audit_gates, screen_pairs, validate_split


def state_key(state):
    return state['seed'], round(state['time_s'], 6)


def choose_omitted(states, selected_keys, count, horizon, remaining, rng):
    """Uniform state audit; separate RNG never perturbs the natural trajectory."""
    eligible = [i for i, s in enumerate(states)
                if state_key(s) not in selected_keys and s['time_s'] + remaining <= horizon]
    if count < 1:
        raise ValueError('Omitted-state budget must be positive')
    return sorted(int(i) for i in rng.choice(eligible, min(count, len(eligible)), replace=False))


def check_source(summary, cfg, source_digest):
    if summary['split'] != 'train':
        raise ValueError('Coverage audit must not inspect validation or final-test states')
    if source_digest != cfg['source_summary_sha256']:
        raise ValueError('Unexpected training source summary')
    validate_split(summary['config'], 'train')
    if cfg['source_method'] not in summary['config']['bank']['methods']:
        raise ValueError('Unknown source bank method')
    if cfg['minimum_remaining_s'] < summary['config']['proxy']['horizon_s']:
        raise ValueError('Audit must fit the unchanged proxy horizon within the source episode')


def summarize(rows, states, views):
    result = {}
    for group in ('selected_bank', 'omitted_uniform'):
        group_rows = [r for r in rows if r['group'] == group]
        report = {
            'states': sum(s['group'] == group for s in states),
            'evaluated_pairs': len(group_rows),
            'first_failure_counts': dict(Counter(r['first_failed_gate'] for r in group_rows)),
            'views': {},
        }
        for view in views:
            passing = [r for r in group_rows if r['views'][view]]
            n = len(passing)
            original = [r for r in passing if r['was_v1_audited']]
            report['views'][view] = {
                'pairs': n,
                'states': len({r['state_index'] for r in passing}),
                'source_episodes': len({r['seed'] for r in passing}),
                'mechanisms': dict(Counter(r['mechanism'] for r in passing)),
                'pair_ranks': [r['rank'] for r in passing],
                'v1_audited_positive_pairs': len(original),
                'new_positive_pairs': sum(not r['was_v1_audited'] for r in passing),
                'states_without_any_v1_positive_pair': len(
                    {r['state_index'] for r in passing} - {r['state_index'] for r in original}),
                'within_audited_universe_v1_pair_recall': len(original) / n if n and group == 'selected_bank' else None,
                'top_k_positive_state_hits': {
                    str(k): len({r['state_index'] for r in passing if r['rank'] <= k})
                    for k in (1, 2, 3, 5, 10, 28)
                },
            }
        result[group] = report
    return result


def run(source_path, config_path, output):
    source = Path(source_path)
    cfg = read(config_path)
    summary = read(source / 'summary.json')
    check_source(summary, cfg, digest(source / 'summary.json'))
    dc = summary['config']
    for field, name in [('state_bank_sha256', 'state_bank.json'), ('pair_audit_sha256', 'pair_audit.json')]:
        if digest(source / name) != summary[field]:
            raise ValueError('Source artifact checksum differs')
    if digest(Path(__file__).with_name('highd_v46_state_discovery.py')) != summary['implementation_sha256']:
        raise ValueError('Source discovery implementation changed')
    for name, expected in summary['dependencies'].items():
        path = dc['physical_config'] if name == 'physical_config' else Path(__file__).with_name(name)
        if digest(path) != expected:
            raise ValueError('Source dependency changed: ' + name)
    assets, lock = ensure_assets()
    if summary['natural_target_sha256'] != lock['natural_target_sha256']:
        raise ValueError('Natural target changed')
    physical = read(dc['physical_config'])
    horizon = physical['horizon_s']
    original_bank = read(source / 'state_bank.json')['states']
    original_rows = read(source / 'pair_audit.json')['rows']
    old = {state_key(s): s for s in original_bank if cfg['source_method'] in s['methods']}
    previous = {(r['seed'], round(r['time_s'], 6), tuple(r['actor_ids'])): r
                for r in original_rows if cfg['source_method'] in r['methods']}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = PairModelQueries(assets / 'model', assets / 'lateral', **SINGLE_OPTIONS)
    started = time.perf_counter()
    rng = np.random.default_rng(cfg['omitted_state_seed'])
    bank = []
    sources = []

    class Observed(NativeExperiment):
        def choose(self, units, actors, pmfs, mode):
            if mode != 'natural':
                raise ValueError('Coverage audit requires natural rollouts')
            fast = screen_pairs(self, pmfs, dc['screen'])
            item = snapshot(self, units, pmfs, seed, {'discovery_only': True})
            item.update(screen_pairs=fast, screen_score=fast[0]['score'] if fast else 0.)
            episode_states.append(item)
            return None

    for source_episode in summary['source_episodes']:
        seed = source_episode['seed']
        episode_states = []
        road = Observed(model, assets / 'pair', assets / 'lateral', seed, physical)
        while road.time < horizon - 1e-9:
            if not road.step_policy('natural', 1.):
                break
        if len(episode_states) != source_episode['states']:
            raise ValueError('Source episode replay diverged')
        replay = {state_key(s): s for s in episode_states}
        for key, stored in old.items():
            if key[0] == seed and identity(replay[key]) != identity({k: v for k, v in stored.items() if k != 'methods'}):
                raise ValueError('Natural pre-draw snapshot replay diverged')
        chosen = {i: 'selected_bank' for i, s in enumerate(episode_states)
                  if state_key(s) in old and s['time_s'] + cfg['minimum_remaining_s'] <= horizon}
        extra = choose_omitted(episode_states, old, cfg['omitted_states_per_episode'], horizon,
                              cfg['minimum_remaining_s'], rng)
        chosen.update({i: 'omitted_uniform' for i in extra})
        for i, group in sorted(chosen.items()):
            bank.append(dict(episode_states[i], group=group, methods=[group]))
        sources.append({'seed': seed, 'natural_states': len(episode_states),
                        'selected_bank': sum(g == 'selected_bank' for g in chosen.values()),
                        'omitted_uniform': len(extra)})
    scan_time = time.perf_counter() - started
    save(output / 'state_bank.json', {'split': 'train', 'config': dc, 'coverage_config': cfg,
         'natural_target_sha256': lock['natural_target_sha256'], 'states': bank,
         'scope': 'Training diagnostic only, not a natural reset law or PPO dataset'})
    rows = []
    for index, state in enumerate(bank):
        cars = [SimpleNamespace(**c) for c in state['vehicles']]
        road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']),
                               model=model, config=physical)
        units = [{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']]
        pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}
        for rank, candidate in enumerate(state['screen_pairs'], 1):
            ids = candidate['actor_ids']
            p, same = pair_marginal(units, ids)
            risk = response_risk(road, ids, pmfs, dc['proxy'], reactive=True)
            audit = audit_gates(p, risk, dc['selection'])
            prior = previous.get((*state_key(state), tuple(ids)))
            if prior is not None and identity(audit) != identity({k: prior[k] for k in audit}):
                raise ValueError('Repeated proxy audit differs from locked source')
            methods = [state['group']] + ([] if prior is not None else ['missed_by_v1'])
            rows.append({'state_index': index, 'seed': state['seed'], 'time_s': state['time_s'],
                         'group': state['group'], 'methods': methods, 'selection': 'exhaustive',
                         'rank': rank, 'was_v1_audited': prior is not None,
                         'same_natural_unit': same, **candidate, **audit})
        save(output / f'audit_state_{index:03d}.json', {'rows': [r for r in rows if r['state_index'] == index]})
        print({'completed_states': index + 1, 'total_states': len(bank), 'pairs': len(rows)}, flush=True)
    save(output / 'pair_audit.json', {'rows': rows})
    result = {
        'split': 'train', 'scope': cfg['scope'], 'config': cfg,
        'config_sha256': digest(config_path), 'implementation_sha256': digest(__file__),
        'source_summary_sha256': digest(source / 'summary.json'),
        'dependencies': summary['dependencies'], 'natural_target_sha256': lock['natural_target_sha256'],
        'source_episodes': sources, 'state_bank_sha256': digest(output / 'state_bank.json'),
        'pair_audit_sha256': digest(output / 'pair_audit.json'),
        'scan_wall_time_s': scan_time, 'total_wall_time_s': time.perf_counter() - started,
        'groups': summarize(rows, bank, dc['funnel_views']),
        'limitations': ['All-pair coverage is limited to the unchanged nearest-eight universe',
                        'Omitted states are sampled, not an exhaustive temporal audit',
                        'Proxy positives require native confirmation; not population risk or significance',
                        'Training data only; no held-out generalization claim'],
    }
    save(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', default='configs/highd_v46_discovery_coverage_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(args.source, args.config, args.output))


if __name__ == '__main__':
    main()
