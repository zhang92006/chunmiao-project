"""Bounded supported-action probes, not another blind PPO collection.

All cases and negatives survive. Deterministic forced actions cannot provide
ordinary IS risk estimates and are never exported as PPO sequences.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from .highd_v46_assets import ROOT, read, save, digest
from .highd_v46_d2rl import challenge, identity
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_cross_unit import natural_log_probability
from .highd_v47_fixed_alpha_native import single_marginal
from . import highd_v47_formal_native as native
from . import highd_v47_formal_recovery as recovery

DEFAULT = 'configs/highd_v47_formal_reachability_v1.json'
ARMS = ('initial_peak', 'reselected_peak', 'persistent_peak')
_WORK = None


def peak(p, risk):
    """Never use an unsupported cell, including a larger unsupported score."""
    p, risk = np.asarray(p, float), np.asarray(risk, float)
    if (p.shape != risk.shape or p.ndim not in (1, 2)
            or not np.isfinite(p).all() or not np.isfinite(risk).all()
            or np.any(p < 0) or not np.isclose(p.sum(), 1.)):
        raise ValueError('Invalid supported action matrix')
    candidates = np.flatnonzero(p.ravel() > 0)
    scores = risk.ravel()[candidates]
    candidates = candidates[scores == scores.max()]
    masses = p.ravel()[candidates]
    index = int(candidates[np.argmax(masses)])
    return tuple(int(i) for i in np.unravel_index(index, p.shape))


def controlled_ids(arm, initial_ids, current_ids, decision):
    if arm not in ARMS:
        raise ValueError('Unknown diagnostic arm')
    if arm == 'initial_peak':
        return list(initial_ids) if decision == 0 else []
    return list(initial_ids if arm == 'persistent_peak' else current_ids)


def actor_role(road, actor_id):
    actor = next(c for c in road.vehicles if c.id == actor_id)
    cav = road.cav
    if actor.lane != cav.lane:
        return 'other_lane'
    if actor.x <= cav.x:
        return 'rear_same_lane'
    front = min((c for c in road.vehicles if c.id != cav.id
                 and c.lane == cav.lane and c.x > cav.x), key=lambda c: c.x)
    return 'direct_leader' if actor.id == front.id else 'other_front_same_lane'


def marginal(units, ids):
    if len(ids) == 1:
        return single_marginal(units, ids[0])
    if len(ids) == 2:
        return pair_marginal(units, ids)[0]
    raise ValueError('Expected one or two distinct BV actors')


def probe(model, assets, physical, case, mode, arm, seed):
    if mode not in ('single', 'dual') or arm not in ARMS or case['horizon_s'] != 4:
        raise ValueError('Wrong fixed diagnostic task')
    road = native.restore_prepared_state(model, assets, physical, case['state'], seed)
    road.last_acceleration = {int(k): v for k, v in case['state']['last_acceleration'].items()}
    road.cav_acceleration = case['state']['cav_acceleration']
    initial_time = road.time
    records, initial_ids = [], None
    while not road.crashes and road.time < initial_time + case['horizon_s'] - 1e-9:
        if not records:
            st = case['state']
            pmfs = {int(k): np.asarray(v) for k, v in st['pmfs'].items()}
            units = [dict(ids=u['ids'], p=np.asarray(u['p'])) for u in st['units']]
            actors = {c.id: c for c in road.vehicles}
        else:
            pmfs = road.action_pmfs()
            units, actors = road.prepare(pmfs)
        selected = native.common_choice(road, units, actors, pmfs, mode)
        target_ids = [] if selected is None else selected['ids']
        if initial_ids is None:
            if not target_ids:
                raise ValueError('Locked original-critical state no longer reproduces')
            initial_ids = list(target_ids)
        ids = controlled_ids(arm, initial_ids, target_ids, len(records))
        forced, scores = {}, None
        if ids:
            p = marginal(units, ids)
            r = challenge(road, ids, pmfs, physical['criticality'])
            action = peak(p, r)
            forced = dict(zip(ids, action))
            scores = dict(action_indices=list(action), natural_mass=float(p[action]),
                          proxy_risk=float(r[action]))
        roles = [actor_role(road, i) for i in ids]
        actions = native.conditional_actions(units, forced, road.rng)
        log_p = natural_log_probability(units, actions)
        # Conditioning is a diagnostic intervention, NOT a full-support Q.
        log_conditional = log_p - np.log(scores['natural_mass']) if scores else log_p
        record = dict(time_s=road.time, original_target_actor_ids=target_ids,
                      forced_actor_ids=ids, forced_actor_roles=roles, forced=scores,
                      actions={str(k): int(v) for k, v in actions.items()},
                      natural_log_probability=log_p,
                      conditional_partner_log_probability=float(log_conditional))
        native.advance_prepared(road, pmfs, actions)
        record.update(cav_acceleration=road.cav_acceleration,
                      minimum_CAV_gap_m=float(road.minimum_gap) if np.isfinite(road.minimum_gap) else None)
        records.append(record)
    event = any(road.cav.id in pair for pair in road.crashes)
    return dict(case_id=case['case_id'], source_seed=case['source_seed'], seed=seed,
                mode=mode, arm=arm, initial_actor_ids=initial_ids,
                event_result=event, cav_id=road.cav.id, collision_pairs=road.crashes,
                event_type='cav_first_collision' if event else 'bv_only_competing_collision'
                if road.crashes else 'horizon_no_collision',
                minimum_CAV_gap_m=float(road.minimum_gap) if np.isfinite(road.minimum_gap) else None,
                elapsed_s=road.time-initial_time, decisions=records,
                natural_path_log_probability=float(sum(t['natural_log_probability'] for t in records)),
                sampling_role='forced_supported_reachability_only_not_IS_or_PPO')


def task_plan(cfg, bank):
    return [dict(case_index=i, case_id=c['case_id'], source_seed=c['source_seed'],
                 mode=mode, arm=arm, seed=cfg['seed_start']+i)
            for i, c in enumerate(bank['cases']) for mode in cfg['modes'] for arm in cfg['arms']]


def implementation(config):
    cfg = read(config)
    if (cfg != read(ROOT/DEFAULT) or cfg['version'] != 1
            or cfg['modes'] != ['single', 'dual'] or cfg['arms'] != list(ARMS)
            or cfg['cases'] != 'every_locked_training_case_in_bank_order'
            or cfg['replicates_per_case'] != 1 or cfg['horizon_s'] != 4
            or cfg['seed_start'] != 1600000000 or cfg['workers'] != 4
            or cfg['validation_opened'] or cfg['training_export']
            or cfg['automatic_retry_or_extension']):
        raise ValueError('Version the diagnostic design explicitly')
    files = [Path(config), Path(__file__)]
    return cfg, {recovery.relative(p): digest(p) for p in files}


def evidence(recovery_frozen, data):
    checked = recovery.checked(recovery.DEFAULT, recovery_frozen)
    cfg, pc, physical, assets, parent_lock, lock, bank, schedule = checked
    summary, envs = recovery.load_data(data, parent_lock, lock, bank)
    audit = read(Path(data)/'audit.json')
    if (not audit['verified'] or audit['recovery_lock_sha256'] != identity(lock)
            or audit['collection_sha256'] != digest(Path(data)/'summary.json')
            or audit['native_replays'] != 2*len(set(schedule['case_indices']))
            or audit['training_readiness'] != summary['training_readiness']):
        raise ValueError('Complete recovery native audit required')
    if any(v['CAV_events'] for v in summary['training_readiness']['modes'].values()):
        raise ValueError('This diagnostic is for the complete zero-event recovery')
    if any(c['horizon_s'] != 4 for c in bank['cases']):
        raise ValueError('Changed bank horizon')
    modes = {}
    for mode, env in envs.items():
        roles = Counter()
        for s in env.sequences:
            fleet = {c['id']: c for c in s['initial_fleet']}
            cav = fleet[s['cav_id']]
            if s['steps']:
                for i in s['steps'][0]['actor_ids']:
                    c = fleet[i]
                    roles['same_lane' if c['lane'] == cav['lane'] else 'other_lane'] += 1
        modes[mode] = dict(sequences=len(env.sequences), CAV_events=len(env.positive),
                          BV_only=sum(s['event_type'] == 'bv_only_competing_collision' for s in env.sequences),
                          initial_controlled_actor_lanes=dict(roles))
    return physical, assets, bank, dict(modes=modes, formal_PPO_started=False,
        state_likelihood_ratio=1., native_replays=audit['native_replays'], validation_opened=False)


def freeze(config, recovery_frozen, data, output):
    cfg, components = implementation(config)
    _, _, bank, status = evidence(recovery_frozen, data)
    plan = task_plan(cfg, bank)
    if len({t['seed'] for t in plan}) != len(bank['cases']):
        raise ValueError('Case seeds must be disjoint; arms share only CRN')
    paths = [Path(recovery_frozen)/name for name in ('lock.json', 'shared_schedule.json')]
    paths += [Path(data)/name for name in ('summary.json', 'audit.json', 'shared_schedule.json')]
    paths += [Path(data)/mode/name for mode in cfg['modes'] for name in ('sequences.json', 'sequence_manifest.json')]
    lock = dict(config=cfg, components=components, inputs={recovery.relative(p): digest(p) for p in paths},
                recovery_frozen=recovery.relative(recovery_frozen), data=recovery.relative(data),
                state_law_sha256=identity(bank['state_law']), plan_sha256=identity(plan),
                code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'completed_zero_signal.json', status)
    save(out/'plan.json', plan)
    save(out/'lock.json', lock)
    return dict(frozen=True, probes=len(plan), cases=len(bank['cases']),
                sources=len({c['source_seed'] for c in bank['cases']}), new_outcomes_opened=False)


def checked(config, frozen):
    cfg, components = implementation(config)
    lock = read(Path(frozen)/'lock.json')
    if lock['config'] != cfg or lock['components'] != components:
        raise ValueError('Diagnostic implementation changed after freeze')
    for name, sha in lock['inputs'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Protected evidence changed: '+name)
    physical, assets, bank, _ = evidence(ROOT/lock['recovery_frozen'], ROOT/lock['data'])
    plan = read(Path(frozen)/'plan.json')
    if (identity(bank['state_law']) != lock['state_law_sha256'] or plan != task_plan(cfg, bank)
            or identity(plan) != lock['plan_sha256']):
        raise ValueError('Changed state law or complete schedule')
    return cfg, physical, assets, bank, plan, lock


def initialize(assets, physical, cases):
    global _WORK
    from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
    assets = Path(assets)
    _WORK = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS), assets, physical, cases


def worker(task):
    model, assets, physical, cases = _WORK
    return probe(model, assets, physical, cases[task['case_index']], task['mode'], task['arm'], task['seed'])


def summarize(rows):
    groups = {}
    for mode in ('single', 'dual'):
        for arm in ARMS:
            part = [r for r in rows if r['mode'] == mode and r['arm'] == arm]
            positive = [r for r in part if r['event_result']]
            groups[mode+'/'+arm] = dict(probes=len(part), CAV_events=len(positive),
                event_sources=sorted({r['source_seed'] for r in positive}),
                BV_only=sum(r['event_type'] == 'bv_only_competing_collision' for r in part),
                witnesses=[dict(case_id=r['case_id'], source_seed=r['source_seed'], seed=r['seed'],
                                natural_path_log_probability=r['natural_path_log_probability']) for r in positive])
    return dict(groups=groups, PPO_started=False, training_export=False, validation_opened=False,
                conclusion='Supported CAV paths found; inspect actor/action continuity before a separately versioned probabilistic teacher'
                if any(r['event_result'] for r in rows) else
                'No witness in this finite diagnostic; no safety claim, automatic retry, collection or PPO',
                scope='Forced path reachability only; counts are not IS risk or training readiness')


def run(config, frozen, output):
    cfg, physical, assets, bank, plan, lock = checked(config, frozen)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', lock)
    save(out/'plan.json', plan)
    rows, hashes = [], {}
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(str(assets), physical, bank['cases'])) as pool:
        for i, row in enumerate(pool.map(worker, plan, chunksize=1)):
            name = 'probe_%05d.json' % i
            save(out/name, row)
            hashes[name] = digest(out/name)
            rows.append(row)
            if (i+1) % 32 == 0 or i+1 == len(plan):
                print(dict(phase='supported_reachability', completed=i+1, total=len(plan),
                           CAV_witnesses=sum(r['event_result'] for r in rows)), flush=True)
    summary = dict(complete=True, lock_sha256=identity(lock), artifacts=hashes,
                   wall_time_s=time.monotonic()-start, **summarize(rows))
    save(out/'summary.json', summary)
    return {k: v for k, v in summary.items() if k != 'artifacts'}


def audit(config, frozen, data, output, replay=False):
    cfg, physical, assets, bank, plan, lock = checked(config, frozen)
    root = Path(data)
    summary = read(root/'summary.json')
    if (not summary['complete'] or summary['lock_sha256'] != identity(lock)
            or read(root/'opened_protocol.json') != lock or read(root/'plan.json') != plan
            or set(summary['artifacts']) != {'probe_%05d.json' % i for i in range(len(plan))}):
        raise ValueError('Incomplete or mismatched diagnostic')
    rows = []
    if replay:
        initialize(assets, physical, bank['cases'])
    for i, task in enumerate(plan):
        name = 'probe_%05d.json' % i
        if digest(root/name) != summary['artifacts'][name]:
            raise ValueError('Changed probe artifact')
        row = read(root/name)
        if any(row[k] != task[k] for k in ('case_id', 'source_seed', 'mode', 'arm', 'seed')):
            raise ValueError('Probe schedule differs')
        if row['event_result'] != any(row['cav_id'] in p for p in row['collision_pairs']):
            raise ValueError('CAV label differs')
        if replay and worker(task) != row:
            raise ValueError('Native supported-action replay differs')
        rows.append(row)
        if replay and ((i+1) % 32 == 0 or i+1 == len(plan)):
            print(dict(phase='reachability_replay', completed=i+1, total=len(plan)), flush=True)
    if any(summary[k] != v for k, v in summarize(rows).items()):
        raise ValueError('Diagnostic summary differs')
    result = dict(verified=True, summary_sha256=digest(root/'summary.json'),
                  native_replays=len(rows) if replay else 0, training_export=False)
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('freeze')
    p.add_argument('--recovery-frozen', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    for command in ('run', 'audit'):
        p = commands.add_parser(command)
        p.add_argument('--frozen', required=True)
        p.add_argument('--output', required=True)
        if command == 'audit':
            p.add_argument('--data', required=True)
            p.add_argument('--replay', action='store_true')
    a = parser.parse_args()
    print(dict(phase='verifying_completed_training_inputs', command=a.command,
               new_outcomes_opened=False), flush=True)
    if a.command == 'freeze':
        result = freeze(a.config, a.recovery_frozen, a.data, a.output)
    elif a.command == 'run':
        result = run(a.config, a.frozen, a.output)
    else:
        result = audit(a.config, a.frozen, a.data, a.output, a.replay)
    # Keep witnesses in summary.json rather than flooding the terminal.
    result = dict(result)
    if 'groups' in result:
        result['groups'] = {k: {name: val for name, val in v.items() if name != 'witnesses'}
                            for k, v in result['groups'].items()}
    print(result, flush=True)


if __name__ == '__main__':
    main()
