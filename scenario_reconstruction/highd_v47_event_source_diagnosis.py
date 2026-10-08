"""Training-only source funnel and native opportunity checks before fitting."""
import argparse
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, read, save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import sample_continuation
from .highd_v46_pair_synergy import pair_marginal, response_risk
from .highd_v46_pair_witness import continue_state
from .highd_v46_proposal_comparison import load_inputs, build_h
from .highd_v46_state_discovery import audit_gates

DEFAULT = 'configs/highd_v47_event_source_diagnosis_v1.json'


def select_rows(rows, seeds, threshold):
    """Selection never reads native outcome labels; zero/absent sources survive."""
    if not 0 <= threshold <= 1 or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid source plan or diagnostic threshold')
    result = []
    for seed in seeds:
        indices = sorted((i for i, r in enumerate(rows) if r['seed'] == seed),
                         key=lambda i: (-rows[i]['ranker_features'][2],
                                        rows[i]['time_s'], tuple(rows[i]['actor_ids'])))
        i = indices[0] if indices else None
        r = rows[i] if i is not None else None
        result.append({'source_seed': seed, 'row_index': i,
                       'maximum_risk': r['ranker_features'][2] if r else None,
                       'probe_selected': bool(r and r['ranker_features'][2] >= threshold)})
    return result


def inputs(config):
    cfg = read(config)
    if (cfg['version'] != 1 or cfg['split'] != 'train'
            or cfg['source_ranges'] != [[200000, 200063], [230000, 230063]]
            or cfg['minimum_proxy_risk'] != .5 or cfg['replicates'] != 64
            or cfg['forced_replicates'] != 16 or cfg['horizon_s'] != 4
            or cfg['candidates'] != {'interaction': 1., 'mixed': .5}
            or cfg['epsilon'] != [.1, .1]
            or cfg['continuation_seed_start'] != 4400000 or cfg['forced_seed_start'] != 4500000):
        raise ValueError('Version the locked training diagnosis before changing its budget')
    train, _, _, base, physical, assets, parent_lock = load_inputs(cfg['training_config'])
    funnel, cases = [], []
    for entry, (lo, hi) in zip(train['training_locks'], cfg['source_ranges']):
        root = Path(entry['path'])
        bank = read(root/'state_bank.json')['states']
        rows = read(root/'pair_audit.json')['rows']
        old = read(root/'case_plan.json')['cases']
        if any(not lo <= r['seed'] <= hi for r in rows):
            raise ValueError('Undeclared source in cached training bank')
        selected = select_rows(rows, list(range(lo, hi+1)), cfg['minimum_proxy_risk'])
        for item in selected:
            index = item['row_index']
            row = rows[index] if index is not None else None
            item.update(origin=entry['path'], previously_queried=any(p['row_index'] == index for p in old) if row else False,
                        strict_eligible=bool(row and row['views']['production_strict']),
                        first_failed_gate=row['first_failed_gate'] if row else None)
            if item['probe_selected']:
                state = bank[row['state_index']]
                cases.append({'state': state, 'row': row, 'plan': dict(item,
                    case_id=len(cases), state_sha256=identity(state), time_s=state['time_s'], actor_ids=row['actor_ids'])})
        funnel.extend(selected)
    lock = {'config_sha256': digest(config), 'implementation_sha256': digest(__file__),
            'parent_lock': parent_lock, 'case_plan_sha256': identity([c['plan'] for c in cases]),
            'funnel_sha256': identity(funnel)}
    return cfg, base, physical, assets, funnel, cases, lock


def prepare(case, model, physical, base):
    state, row = case['state'], case['row']
    cars = [SimpleNamespace(**c) for c in state['vehicles']]
    road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']), model=model, config=physical)
    p, _ = pair_marginal(state['units'], row['actor_ids'])
    risk = response_risk(road, row['actor_ids'], {int(k): np.asarray(v) for k, v in state['pmfs'].items()}, base['proxy'], reactive=True)
    audit = audit_gates(p, risk, base['selection'])
    if any(identity(audit[k]) != identity(row[k]) for k in audit):
        raise ValueError('Cached risk/gate audit no longer reproduces')
    maximum = np.unravel_index(np.argmax(np.where(p > 0, risk, -np.inf)), p.shape)
    modal = np.unravel_index(np.argmax(p), p.shape)
    return p, risk, {'maximum_risk': list(map(int, maximum)), 'natural_mode': list(map(int, modal))}


def freeze(config, output):
    *_, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def run(config, lock_path, output):
    cfg, base, physical, assets, funnel, cases, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen protocol or inputs changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(out/'opened_protocol.json', {'config': cfg, 'lock': lock})
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    prepared = [prepare(case, model, physical, base) for case in cases]
    plans = [dict(c['plan'], forced_actions=pr[2], continuation_seed_start=cfg['continuation_seed_start']+i*64,
                  forced_seed_start=cfg['forced_seed_start']+i*16) for i, (c, pr) in enumerate(zip(cases, prepared))]
    save(out/'case_plan.json', {'selection_reads_new_outcomes': False, 'cases': plans, 'funnel': funnel})
    records = []
    for i, (case, (p, risk, actions), plan) in enumerate(zip(cases, prepared, plans)):
        arms, probes = {}, {}
        for name, lam in cfg['candidates'].items():
            h, _ = build_h(p, risk, lam)
            chosen = {'ids': plan['actor_ids'], 'h': h}
            samples = [sample_continuation(model, assets, physical, case['state'], chosen, cfg['epsilon'],
                plan['continuation_seed_start']+j, cfg, base['proxy']['clearance_scale_m']) for j in range(64)]
            for sample in samples:
                audit_sample(sample, cfg['epsilon'])
            save(out/f'case_{i:03d}_{name}.json', {'plan': plan, 'P': p.tolist(), 'H': h.tolist(), 'sequences': samples})
            weights = np.asarray([float(s['event_result'])*np.exp(s['generation_log_weight']) for s in samples])
            arms[name] = {'events': sum(s['event_result'] for s in samples), 'draws': 64,
                          'conditional_IS_mean': float(weights.mean()), 'M2': float(np.mean(weights**2))}
        for name, action in actions.items():
            samples = [continue_state(model, assets, physical, case['state'], dict(zip(plan['actor_ids'], action)),
                plan['forced_seed_start']+j, 4, base['proxy']['clearance_scale_m']) for j in range(16)]
            save(out/f'case_{i:03d}_forced_{name}.json', {'plan': plan, 'action': action, 'samples': samples})
            probes[name] = {'events': sum(s['cav_first_collision'] for s in samples), 'draws': 16,
                            'proxy_risk': float(risk[tuple(action)]), 'natural_action_probability': float(p[tuple(action)])}
        record = dict(plan, arms=arms, forced_probes=probes)
        records.append(record)
        print({'completed': i+1, 'total': len(cases), 'source': plan['source_seed'], 'arms': arms, 'forced': probes}, flush=True)
    summary = {'complete': True, 'config': cfg, 'lock': lock, 'funnel': funnel, 'records': records,
               'Q_event_sources': {name: [r['source_seed'] for r in records if r['arms'][name]['events']] for name in cfg['candidates']},
               'forced_event_sources': {name: [r['source_seed'] for r in records if r['forced_probes'][name]['events']] for name in ('maximum_risk', 'natural_mode')},
               'total_wall_s': time.perf_counter()-start, 'ppo_trained': False, 'validation_opened': False,
               'artifact_hashes': {p.name: digest(p) for p in sorted(out.glob('*.json'))}}
    save(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run'])
    parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--lock'); parser.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.mode == 'freeze':
        print(freeze(a.config, a.output))
    else:
        r = run(a.config, a.lock, a.output)
        print({'Q_event_sources': r['Q_event_sources'], 'forced_event_sources': r['forced_event_sources']})


if __name__ == '__main__':
    main()
