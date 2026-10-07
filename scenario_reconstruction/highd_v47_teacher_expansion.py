"""Restore-gated, fixed-budget expansion of the collision-producing teacher.

Stages: freeze -> discover (all source/case plans) -> collect -> audit.
There is deliberately no PPO or validation entry point and no outcome extension.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np

from . import highd_v47_action_teacher as teacher
from . import highd_v47_teacher_restore as restore
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_cross_unit import draw_intervention, proposal_factors
from .highd_v46_d2rl import experiment_target, identity
from .highd_v46_multistate_q import environment, sample_continuation
from .highd_v46_native_calibration import discover as natural_discover
from .highd_v46_pair_synergy import pair_marginal

DEFAULT = 'configs/highd_v47_teacher_expansion_v1.json'
_RUNTIME = None


def json_value(value):
    # State dictionaries have integer keys in memory and string keys on disk.
    # Hash the serialized representation from the outset, without touching old locks.
    return json.loads(json.dumps(value, allow_nan=False))


def protocol(config):
    cfg = read(config)
    fixed = dict(version=1, source=dict(split='train', seed_start=1800000000, episodes=4096),
        reserved_validation=[1800010000, 1800011023], source_chunk_size=64, workers=4,
        minimum_proxy_risk=.5, maximum_cases_per_source=3, risk_edges=[0,.2,.4,.6,.8,1],
        epsilon=[.1,.1], horizon_s=4, replicates=64, continuation_seed_start=1900000000,
        state_likelihood_ratio=1., learned_QS=False, minimum_collision_sources=50,
        desired_collision_sources=100, minimum_CAV_events=500,
        discovery_config='configs/highd_v46_discovery_v2.json')
    if any(cfg.get(k) != v for k, v in fixed.items()) or cfg['environment'] != environment():
        raise ValueError('Version changed teacher expansion/source/budget/environment before running')
    dc = read(cfg['discovery_config']); base = read(dc['base_discovery_config'])
    physical = read(base['physical_config']); assets, frozen = ensure_assets()
    # Conservative code lock includes the entire highD implementation surface.
    code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('highd*.py'))}
    configs = {p: digest(p) for p in (cfg['discovery_config'], dc['base_discovery_config'],
                                    base['physical_config'], teacher.DEFAULT)}
    lock = dict(config_sha256=digest(config), implementation_sha256=digest(__file__),
        dependencies=code, configs=configs, environment=environment(),
        natural_target_sha256=frozen['natural_target_sha256'],
        physical_target_sha256=identity(experiment_target(frozen, physical)))
    return cfg, dc, base, physical, assets, lock


def freeze(config, reproduction_audit, output):
    *_, lock = protocol(config)
    report = read(reproduction_audit)
    if (not report.get('full_reproduction_verified') or not report.get('verified')
            or report.get('verifier_sha256') != digest(restore.__file__)):
        raise ValueError('First reproduce the complete original teacher in a separate output')
    actual = restore.verify(report['reproduction'])
    for key in ('events', 'draws', 'event_sources', 'original_summary_sha256',
                'original_lock_sha256', 'reproduction_summary_sha256'):
        if report.get(key) != actual[key]: raise ValueError('Reproduction evidence differs: '+key)
    lock.update(reproduction_audit=str(reproduction_audit), reproduction_audit_sha256=digest(reproduction_audit),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        validation_opened=False, new_sources_opened=False)
    save(output, lock)
    return dict(frozen=True, new_sources_opened=False, source_episodes=4096)


def checked(config, lock_path):
    cfg, dc, base, physical, assets, expected = protocol(config); lock = read(lock_path)
    if (any(lock.get(k) != v for k, v in expected.items())
            or digest(lock['reproduction_audit']) != lock['reproduction_audit_sha256']):
        raise ValueError('Expansion lock changed; no silent repin')
    return cfg, dc, base, physical, assets, lock


def selected_rows(rows, seeds):
    """Calibration's two strict states plus diagnostic maximum, no event labels."""
    selected = []; funnel = []
    by_source = defaultdict(list)
    for i, r in enumerate(rows):
        if r['seed'] not in seeds: raise ValueError('Source outside declared chunk')
        by_source[r['seed']].append(i)
    for seed in seeds:
        candidates = by_source[seed]
        eligible = [i for i in candidates if rows[i]['views']['production_strict']]
        tie = lambda i: (rows[i]['time_s'], tuple(rows[i]['actor_ids']))
        chosen = []
        if eligible:
            first = min(eligible, key=lambda i: (-rows[i]['metrics']['positive_interaction_mass'], tie(i)))
            chosen.append(first)
            remaining = [i for i in eligible if rows[i]['state_index'] != rows[first]['state_index']]
            if remaining:
                chosen.append(min(remaining, key=lambda i: (
                    rows[i]['mechanism'] == rows[first]['mechanism'],
                    -rows[i]['metrics']['positive_interaction_mass'], tie(i))))
        maximum = min(candidates, key=lambda i: (-rows[i]['ranker_features'][2], tie(i))) if candidates else None
        if maximum is not None and rows[maximum]['ranker_features'][2] >= .5:
            chosen.append(maximum)
        unique = []; seen = set()
        for i in chosen:
            key = (rows[i]['state_index'], tuple(rows[i]['actor_ids']))
            if key not in seen: unique.append(i); seen.add(key)
        selected.extend(unique)
        funnel.append(dict(source_seed=seed, evaluated_pairs=len(candidates), strict_pairs=len(eligible),
            maximum_risk=rows[maximum]['ranker_features'][2] if maximum is not None else None,
            selected_rows=unique))
    return selected, funnel


def init_worker(assets, physical, base, dc, cfg):
    global _RUNTIME
    assets = Path(assets)
    _RUNTIME = (PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS),
                assets, physical, base, dc, cfg)


def discovery_worker(task):
    chunk, root = task; root = Path(root)
    model, assets, physical, base, dc, cfg = _RUNTIME
    start = cfg['source']['seed_start']+chunk*cfg['source_chunk_size']
    source = dict(split='train', seed_start=start, episodes=cfg['source_chunk_size'])
    bank, rows, report = natural_discover({'source': source}, dc, base, physical, model, assets)
    bank = json_value(bank)
    seeds = list(range(start, start+cfg['source_chunk_size']))
    indices, funnel = selected_rows(rows, seeds)
    if [e['seed'] for e in report['episodes']] != seeds:
        raise ValueError('Incomplete discovery chunk')
    designs = []; design_hashes = {}
    for index in indices:
        row = rows[index]; state = bank[row['state_index']]
        if state['seed'] != row['seed']: raise ValueError('Selected source/state mismatch')
        p, risk, _ = teacher.prepare(dict(state=state, row=row), model, physical, base)
        h, _ = teacher.exploration_h(p, risk, cfg['risk_edges'])
        design = dict(state=state, row=row, P=p.tolist(), risk=risk.tolist(), H=h.tolist(), row_index=index)
        design_name = f'design_{chunk:03d}_{len(designs):03d}.json'
        save(root/design_name,design); design_hashes[design_name] = digest(root/design_name)
        designs.append(design)
    name = f'chunk_{chunk:03d}.json'
    save(root/name, dict(source=source, bank=bank, rows=rows, report=report, funnel=funnel, designs=designs))
    return name, digest(root/name), design_hashes


def collect_worker(task):
    case, bank, output = task
    model, assets, physical, base, dc, cfg = _RUNTIME
    design = read(Path(bank)/case['design_file'])
    if identity(design['state']) != case['state_sha256']:
        raise ValueError('Serialized state identity differs')
    samples = [sample_continuation(model, assets, physical, design['state'],
        {'ids': case['actor_ids'], 'h': np.asarray(design['H'])}, cfg['epsilon'],
        case['continuation_seed_start']+j, cfg, base['proxy']['clearance_scale_m'])
        for j in range(cfg['replicates'])]
    for sample in samples: audit_sample(sample, cfg['epsilon'])
    name = f'case_{case["case_id"]:05d}.json'
    save(Path(output)/name, dict(plan=case, sequences=samples))
    return name, digest(Path(output)/name), statistics(case, samples)


def pool(args):
    cfg, dc, base, physical, assets = args[:5]
    return ProcessPoolExecutor(cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
        initializer=init_worker, initargs=(str(assets), physical, base, dc, cfg))


def build_plan(chunks, cfg):
    cases = []; funnel = []
    for name, chunk in chunks:
        funnel.extend(chunk['funnel'])
        for i, design in enumerate(chunk['designs']):
            state = design['state']; row = design['row']; case_id = len(cases)
            cases.append(dict(case_id=case_id, source_seed=state['seed'], time_s=state['time_s'],
                actor_ids=row['actor_ids'], discovery_file=name, design_index=i,
                design_file=f'{Path(name).stem.replace("chunk_", "design_")}_{i:03d}.json',
                state_sha256=identity(state),
                continuation_seed_start=cfg['continuation_seed_start']+cfg['replicates']*case_id))
    start = cfg['source']['seed_start']
    if [r['source_seed'] for r in funnel] != list(range(start,start+cfg['source']['episodes'])):
        raise ValueError('Full ordered source cohort required before outcomes')
    counts = Counter(c['source_seed'] for c in cases)
    if counts and max(counts.values()) > cfg['maximum_cases_per_source']:
        raise ValueError('Case budget exceeded')
    if len({(c['source_seed'],c['state_sha256'],tuple(c['actor_ids'])) for c in cases}) != len(cases):
        raise ValueError('Duplicate physical case')
    for c in cases:
        c.update(alpha=1/(len(counts)*counts[c['source_seed']]), state_likelihood_ratio=1.,
                 stratum_replay_correction=len(cases)/(len(counts)*counts[c['source_seed']]))
    return dict(cases=cases, sources=funnel, state_law=cfg['state_law'],
        selection_uses_native_outcomes=False, continuation_draws=cfg['replicates']*len(cases),
        source_episodes=cfg['source']['episodes'], selected_sources=len(counts))


def discover(config, lock_path, output):
    args = checked(config, lock_path); cfg = args[0]; out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', dict(lock=args[-1], config=cfg))
    count = cfg['source']['episodes']//cfg['source_chunk_size']; hashes = {}
    with pool(args) as workers:
        jobs = [workers.submit(discovery_worker, (i, str(out))) for i in range(count)]
        for future in as_completed(jobs):
            name, sha, design_hashes = future.result(); hashes[name] = sha; hashes.update(design_hashes)
            print(dict(phase='teacher_source_discovery',
                completed_chunks=sum(n.startswith('chunk_') for n in hashes), total_chunks=count), flush=True)
    plan = build_plan(((name,read(out/name)) for name in sorted(hashes) if name.startswith('chunk_')), cfg)
    cases = plan['cases']
    save(out/'case_plan.json', plan)
    hashes.update({'case_plan.json': digest(out/'case_plan.json'), 'opened_protocol.json': digest(out/'opened_protocol.json')})
    save(out/'summary.json', dict(complete=True, lock_sha256=identity(args[-1]), artifact_hashes=hashes,
        source_episodes=cfg['source']['episodes'], selected_sources=plan['selected_sources'], cases=len(cases), continuation_draws=64*len(cases)))
    return dict(complete=True, source_episodes=cfg['source']['episodes'], selected_sources=plan['selected_sources'], cases=len(cases),
                continuation_draws=64*len(cases), native_teacher_outcomes_opened=False)


def check_artifacts(root, lock):
    root = Path(root); s = read(root/'summary.json')
    if not s['complete'] or s['lock_sha256'] != identity(lock): raise ValueError('Incomplete or wrong lock')
    for name, sha in s['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha: raise ValueError('Artifact changed: '+name)
    return s


def statistics(case, samples):
    y = np.asarray([s['event_result'] for s in samples], float)
    w = np.exp([s['generation_log_weight'] for s in samples]); values = y*w
    return dict(case_id=case['case_id'], source_seed=case['source_seed'], alpha=case['alpha'],
        draws=len(samples), events=int(y.sum()), bv_only_events=sum(s['bv_only_collision'] for s in samples),
        IS_mean=float(values.mean()), M2=float(np.mean(values**2)),
        within_variance=float(values.var(ddof=1)),
        largest_event_M2_contribution=float(case['alpha']*np.max(values**2)/len(samples)))


def aggregate(records, cfg):
    contributions = defaultdict(float); event_counts = Counter()
    for r in records:
        contributions[r['source_seed']] += r['alpha']*r['M2']
        event_counts[r['source_seed']] += r['events']
    total = sum(contributions.values()); events = sum(event_counts.values())
    sources = sorted(s for s, n in event_counts.items() if n)
    shares = [v/total for v in contributions.values()] if total else []
    draws = sum(r['draws'] for r in records)
    return dict(events=events, draws=draws, event_sources=sources, collision_sources=len(sources),
        event_counts_by_source=dict(event_counts), raw_CAV_fraction=events/draws if draws else None,
        event_source_yield_all_sources=len(sources)/cfg['source']['episodes'],
        conditional_IS_mean=sum(r['alpha']*r['IS_mean'] for r in records), M2=total,
        stratified_estimator_variance=sum(r['alpha']**2*r['within_variance']/r['draws'] for r in records),
        maximum_source_M2_share=max(shares) if shares else None,
        effective_M2_sources=1/sum(v*v for v in shares) if shares else None,
        maximum_event_M2_share=max((r['largest_event_M2_contribution'] for r in records), default=0)/total if total else None,
        collision_count_gate_passed=len(sources)>=cfg['minimum_collision_sources'] and events>=cfg['minimum_CAV_events'],
        desired_100_collision_sources_reached=len(sources)>=cfg['desired_collision_sources'],
        ppo_ready=False, validation_opened=False,
        remaining_gate='Shared-data independent Single/Dual target likelihood, useful reward/gradient diversity and replay audit; counts alone do not authorize PPO')


def collect(config, lock_path, bank, output):
    args = checked(config, lock_path); cfg = args[0]
    bank_summary = check_artifacts(bank, args[-1]); plan = read(Path(bank)/'case_plan.json')
    expected = build_plan(((name,read(Path(bank)/name)) for name in sorted(bank_summary['artifact_hashes'])
                           if name.startswith('chunk_')),cfg)
    if plan != expected: raise ValueError('Bank plan differs from complete outcome-blind selection')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', dict(lock=args[-1], bank=str(bank), bank_summary_sha256=digest(Path(bank)/'summary.json'),
        case_plan_sha256=digest(Path(bank)/'case_plan.json'), config=cfg))
    records = []; hashes = {}
    with pool(args) as workers:
        jobs = [workers.submit(collect_worker, (c, str(bank), str(out))) for c in plan['cases']]
        for future in as_completed(jobs):
            name, sha, record = future.result(); hashes[name] = sha; records.append(record)
            if len(records)%16 == 0 or len(records)==len(jobs):
                print(dict(phase='teacher_native_collection', completed_cases=len(records), total_cases=len(jobs),
                    CAV_events=sum(r['events'] for r in records),
                    collision_sources=len({r['source_seed'] for r in records if r['events']})), flush=True)
    records.sort(key=lambda r:r['case_id'])
    hashes['opened_protocol.json'] = digest(out/'opened_protocol.json')
    result = dict(complete=True, lock_sha256=identity(args[-1]), artifact_hashes=hashes,
                  records=records, bank_summary_sha256=digest(Path(bank)/'summary.json'), **aggregate(records,cfg))
    save(out/'summary.json', result)
    return {k:result[k] for k in ('complete','events','draws','collision_sources','collision_count_gate_passed','ppo_ready')}


def audit(config, lock_path, bank, data, output, replay=False):
    args = checked(config, lock_path); cfg, dc, base, physical, assets, lock = args
    b = check_artifacts(bank, lock); s = check_artifacts(data, lock)
    if s['bank_summary_sha256'] != digest(Path(bank)/'summary.json'): raise ValueError('Wrong bank')
    plan = read(Path(bank)/'case_plan.json'); cases = plan['cases']
    expected_sources = list(range(cfg['source']['seed_start'], cfg['source']['seed_start']+cfg['source']['episodes']))
    if [r['source_seed'] for r in plan['sources']] != expected_sources:
        raise ValueError('Incomplete source cohort')
    if set(s['artifact_hashes']) != {'opened_protocol.json'} | {f'case_{i:05d}.json' for i in range(len(cases))}:
        raise ValueError('Missing or extra case outcomes')
    counts = Counter(c['source_seed'] for c in cases); records = []; replays = 0; max_error = 0.
    model = PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS) if replay else None
    def checked_chunks():
        # Stream potentially multi-GB discovery banks; never retain all chunks.
        for name in sorted(n for n in b['artifact_hashes'] if n.startswith('chunk_')):
            chunk = read(Path(bank)/name)
            seeds = [e['seed'] for e in chunk['report']['episodes']]
            indices, funnel = selected_rows(chunk['rows'], seeds)
            if funnel != chunk['funnel'] or indices != [d['row_index'] for d in chunk['designs']]:
                raise ValueError('Outcome-blind selection no longer reproduces')
            for i,d in enumerate(chunk['designs']):
                filename = f'{Path(name).stem.replace("chunk_", "design_")}_{i:03d}.json'
                if filename not in b['artifact_hashes'] or read(Path(bank)/filename) != d:
                    raise ValueError('Case design detached from discovery')
                row = chunk['rows'][d['row_index']]
                if d['row'] != row or d['state'] != chunk['bank'][row['state_index']]:
                    raise ValueError('Selected case does not match the natural state bank')
            yield name,chunk
    if plan != build_plan(checked_chunks(),cfg): raise ValueError('Selected cases or masses changed')
    for i,c in enumerate(cases):
        d = read(Path(bank)/c['design_file']); state = d['state']
        alpha = 1/(len(counts)*counts[c['source_seed']])
        if (c['case_id'] != i or c['source_seed'] != state['seed'] or c['actor_ids'] != d['row']['actor_ids']
                or c['state_sha256'] != identity(state) or c['alpha'] != alpha
                or c['stratum_replay_correction'] != len(cases)/(len(counts)*counts[c['source_seed']])
                or c['state_likelihood_ratio'] != 1.
                or c['continuation_seed_start'] != cfg['continuation_seed_start']+i*64):
            raise ValueError('Case/state/mass/schedule mismatch')
        p,_ = pair_marginal(state['units'],c['actor_ids']); h,_ = teacher.exploration_h(p,d['risk'],cfg['risk_edges'])
        np.testing.assert_allclose(p,d['P'],rtol=1e-12,atol=0)
        np.testing.assert_allclose(h,d['H'],rtol=1e-12,atol=0)
        factors = proposal_factors(p,h,cfg['epsilon'])
        raw = read(Path(data)/f'case_{i:05d}.json'); samples = raw['sequences']
        if raw['plan'] != c or [r['seed'] for r in samples] != list(range(c['continuation_seed_start'],c['continuation_seed_start']+64)):
            raise ValueError('Incomplete real-Q schedule')
        for sample in samples:
            audit_sample(sample,cfg['epsilon']); log = sample['initial_likelihood_audit']; a,z = log['action_indices']
            # Reconstruct the sampled actions and the FULL natural conditional
            # partner law for every draw, not only the logged factor product.
            _, rebuilt = draw_intervention(state['units'],c['actor_ids'],h,cfg['epsilon'],
                                            np.random.default_rng(sample['seed']))
            if restore.physical_record(rebuilt) != restore.physical_record(log):
                raise ValueError('Initial action/conditional-partner likelihood replay differs')
            for offset,key in ((0,'p_factors'),(2,'h_factors'),(4,'q_factors')):
                np.testing.assert_allclose(log[key],[factors[offset][a],factors[offset+1][a,z]],rtol=1e-11,atol=1e-15)
            if sample['event_result'] != any(state['cav_id'] in pair for pair in sample['collision_pairs']):
                raise ValueError('CAV label differs from native stopped collision')
            if sample['bv_only_collision'] != (bool(sample['collision_pairs']) and not sample['event_result']):
                raise ValueError('Competing endpoint differs')
            max_error = max(max_error,abs(log['full_log_p']-log['full_log_q']-sample['generation_log_weight']))
        if replay:
            teacher.prepare(dict(state=state,row=d['row']),model,physical,base)
            # Predetermined first continuation of EVERY case, not positive selection.
            actual = sample_continuation(model,assets,physical,state,{'ids':c['actor_ids'],'h':h},
                cfg['epsilon'],c['continuation_seed_start'],cfg,base['proxy']['clearance_scale_m'])
            if restore.physical_record(actual) != restore.physical_record(samples[0]):
                raise ValueError('Predetermined native replay differs')
            replays += 1
        records.append(statistics(c,samples))
        if (i+1)%16==0 or i+1==len(cases):
            print(dict(phase='teacher_audit',completed_cases=i+1,total_cases=len(cases),native_replays=replays),flush=True)
    if records != s['records'] or any(s[k] != v for k,v in aggregate(records,cfg).items() if k!='event_counts_by_source'):
        raise ValueError('Collection statistics differ')
    if {str(k):v for k,v in aggregate(records,cfg)['event_counts_by_source'].items()} != s['event_counts_by_source']:
        raise ValueError('Source counts differ')
    result = dict(verified=True, summary_sha256=digest(Path(data)/'summary.json'), native_replays=replays,
        events=s['events'], collision_sources=s['collision_sources'],
        collision_count_gate_passed=s['collision_count_gate_passed'], maximum_full_log_error=max_error,
        ppo_ready=False, validation_opened=False, auditor_sha256=digest(__file__))
    save(output,result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze','discover','collect','audit'])
    p.add_argument('--config',default=DEFAULT); p.add_argument('--lock'); p.add_argument('--bank')
    p.add_argument('--data'); p.add_argument('--reproduction-audit'); p.add_argument('--replay',action='store_true')
    p.add_argument('--output',required=True); a=p.parse_args()
    if a.command=='freeze': result=freeze(a.config,a.reproduction_audit,a.output)
    elif a.command=='discover': result=discover(a.config,a.lock,a.output)
    elif a.command=='collect': result=collect(a.config,a.lock,a.bank,a.output)
    else: result=audit(a.config,a.lock,a.bank,a.data,a.output,a.replay)
    print(result,flush=True)


if __name__=='__main__': main()
