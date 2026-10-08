"""Opt-in action-conditional training exploration, separate from efficient testing Q."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import time

import numpy as np

from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import read, save, digest
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import sample_continuation
from .highd_v46_proposal_comparison import load_inputs
from .highd_v47_event_source_diagnosis import inputs as diagnosis_inputs, prepare

DEFAULT = 'configs/highd_v47_action_teacher_v1.json'


def exploration_h(p, risk, edges):
    p, risk, edges = np.asarray(p, float), np.asarray(risk, float), np.asarray(edges, float)
    if (p.ndim != 2 or p.shape != risk.shape or not np.isfinite(p).all()
            or not np.isfinite(risk).all() or np.any(p < 0) or not np.isclose(p.sum(), 1)
            or edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1
            or not np.all(np.diff(edges) > 0) or np.any(risk < 0) or np.any(risk > 1)):
        raise ValueError('Invalid supported risk-bin teacher')
    bins = np.searchsorted(edges[1:-1], risk, side='right')
    supported = p > 0
    occupied = np.unique(bins[supported])
    h = np.zeros_like(p)
    for b in occupied:
        cells = supported & (bins == b)
        h[cells] = 1/(len(occupied)*int(cells.sum()))
    return h, bins


def union_cases(original, added):
    """Keep negatives and original cases, with deterministic identity deduplication."""
    seen, result = set(), []
    for c in list(original)+list(added):
        key = (c['state']['seed'], identity(c['state']), tuple(c['row']['actor_ids']))
        if key in seen:
            continue
        seen.add(key); result.append({'state': c['state'], 'row': c['row']})
    return result


def inputs(config):
    cfg = read(config)
    if (cfg['version'] != 1 or cfg['split'] != 'train' or cfg['risk_edges'] != [0, .2, .4, .6, .8, 1]
            or cfg['replicates'] != 64 or cfg['epsilon'] != [.1, .1] or cfg['horizon_s'] != 4
            or cfg['continuation_seed_start'] != 4600000):
        raise ValueError('Version teacher protocol before changing fixed budget')
    dc, base, physical, assets, funnel, added, parent = diagnosis_inputs(cfg['diagnosis_config'])
    _, original, _, _, _, _, _ = load_inputs(dc['training_config'])
    cases = union_cases(original, added)
    counts = Counter(c['state']['seed'] for c in cases)
    plans = [dict(case_id=i, source_seed=c['state']['seed'], time_s=c['state']['time_s'],
                  state_sha256=identity(c['state']), actor_ids=c['row']['actor_ids'],
                  production_strict=c['row']['views']['production_strict'],
                  target_mass=1/(len(counts)*counts[c['state']['seed']]),
                  continuation_seed_start=cfg['continuation_seed_start']+i*64) for i, c in enumerate(cases)]
    lock = {'config_sha256': digest(config), 'implementation_sha256': digest(__file__), 'parent': parent,
            'case_plan_sha256': identity(plans), 'funnel_sha256': identity(funnel)}
    return cfg, base, physical, assets, cases, plans, lock


def freeze(config, output):
    *_, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def run(config, lock_path, output):
    cfg, base, physical, assets, cases, plans, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Teacher lock changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(out/'opened_protocol.json', {'config': cfg, 'lock': lock})
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    prepared = []
    for case, plan in zip(cases, plans):
        p, risk, _ = prepare(case, model, physical, base)
        h, bins = exploration_h(p, risk, cfg['risk_edges'])
        prepared.append((p, risk, h, bins))
        save(out/f'case_{plan["case_id"]:03d}_design.json', {'plan': plan, 'state': case['state'],
            'P': p.tolist(), 'risk': risk.tolist(), 'H': h.tolist(), 'bins': bins.tolist()})
    save(out/'case_plan.json', {'cases': plans, 'outcome_based_selection': False})
    records = []
    for case, plan, (p, risk, h, bins) in zip(cases, plans, prepared):
        samples = [sample_continuation(model, assets, physical, case['state'], {'ids': plan['actor_ids'], 'h': h},
            cfg['epsilon'], plan['continuation_seed_start']+j, cfg, base['proxy']['clearance_scale_m']) for j in range(64)]
        labels = []
        for s in samples:
            audit_sample(s, cfg['epsilon'])
            a, b = s['initial_likelihood_audit']['action_indices']
            labels.append({'seed': s['seed'], 'action_indices': [a, b], 'event': s['event_result'],
                           'risk_bin': int(bins[a, b]), 'proxy_risk': float(risk[a, b]),
                           'P': float(p[a, b]), 'generation_Q': float(np.prod(s['initial_likelihood_audit']['q_factors']))})
        save(out/f'case_{plan["case_id"]:03d}_labels.json', {'plan': plan, 'labels': labels, 'sequences': samples})
        y = np.array([s['event_result'] for s in samples])
        w = np.exp([s['generation_log_weight'] for s in samples])
        record = dict(plan, draws=64, events=int(y.sum()), conditional_IS_mean=float(np.mean(y*w)),
                      M2=float(np.mean(y*w*w)), event_action_P=[r['P'] for r in labels if r['event']])
        records.append(record)
        print({'case': plan['case_id'], 'source': plan['source_seed'], 'events': int(y.sum())}, flush=True)
    contributions = {s: sum(r['target_mass']*r['M2'] for r in records if r['source_seed'] == s)
                     for s in {r['source_seed'] for r in records}}
    total = sum(contributions.values())
    result = {'complete': True, 'config': cfg, 'lock': lock, 'records': records,
              'source_episodes': len(contributions), 'event_sources': sorted({r['source_seed'] for r in records if r['events']}),
              'events': sum(r['events'] for r in records), 'draws': sum(r['draws'] for r in records),
              'source_M2_contributions': contributions, 'maximum_source_M2_share': max(contributions.values())/total if total else None,
              'conditional_IS_mean': sum(r['target_mass']*r['conditional_IS_mean'] for r in records), 'M2': total,
              'total_wall_s': time.perf_counter()-start, 'ppo_ready': False, 'validation_opened': False,
              'artifact_hashes': {p.name: digest(p) for p in sorted(out.glob('*.json'))}}
    save(out/'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'run']); p.add_argument('--config', default=DEFAULT)
    p.add_argument('--lock'); p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.mode == 'freeze':
        print(freeze(a.config, a.output))
    else:
        r = run(a.config, a.lock, a.output)
        print({k: r[k] for k in ('event_sources', 'events', 'draws', 'maximum_source_M2_share', 'total_wall_s')})


if __name__ == '__main__':
    main()
