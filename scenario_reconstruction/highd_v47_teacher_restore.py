"""Verify the original collision-producing teacher, without changing its lock."""
import argparse
from pathlib import Path

from . import highd_v47_action_teacher as teacher
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, read, save
from .highd_v46_collision_coverage import audit_sample

OLD = Path('outputs/v47_action_teacher_v1_train')
LOCK = Path('outputs/v47_action_teacher_v1_lock.json')
SUMMARY_SHA = 'b59a511ebf1751f255a7c4039b29e5f315a008a69e4f34e6671a385adc4505d8'
LOCK_SHA = '6a75888149d4ce31afb71b816d194cd9f615fd1a288baa3eca6bcfed7762a235'


def physical_record(value):
    """Only execution timing is excluded; every other field must reproduce."""
    if isinstance(value, dict):
        return {str(k): physical_record(v) for k, v in value.items()
                if k not in ('cpu_s', 'wall_s', 'total_wall_s')}
    if isinstance(value, list):
        return [physical_record(v) for v in value]
    if isinstance(value, tuple):
        return [physical_record(v) for v in value]
    return value


def artifacts(root):
    summary = read(root/'summary.json')
    expected = {'opened_protocol.json', 'case_plan.json'} | {
        f'case_{i:03d}_{kind}.json' for i in range(33) for kind in ('design', 'labels')}
    if not summary['complete'] or set(summary['artifact_hashes']) != expected:
        raise ValueError('Incomplete original 33-case teacher collection')
    for name, sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Teacher artifact changed: '+name)
    return summary


def verify(reproduction=None, replay=False):
    if digest(OLD/'summary.json') != SUMMARY_SHA or digest(LOCK) != LOCK_SHA:
        raise ValueError('Original teacher evidence changed; do not repin')
    cfg, base, physical, assets, cases, plans, expected = teacher.inputs(teacher.DEFAULT)
    lock = read(LOCK)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Original teacher runtime no longer matches its frozen lock')
    original = artifacts(OLD)
    if original['lock'] != lock or read(OLD/'case_plan.json')['cases'] != plans:
        raise ValueError('Original teacher plan/lock mismatch')
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS) if replay else None
    events = 0; sources = set(); replays = 0
    for case, plan in zip(cases, plans):
        raw = read(OLD/f'case_{plan["case_id"]:03d}_labels.json')
        if raw['plan'] != plan or len(raw['sequences']) != 64:
            raise ValueError('Teacher case/schedule differs')
        for j, sample in enumerate(raw['sequences']):
            audit_sample(sample, cfg['epsilon'])
            if sample['seed'] != plan['continuation_seed_start']+j:
                raise ValueError('Teacher continuation seed differs')
            events += int(sample['event_result'])
            if sample['event_result']: sources.add(plan['source_seed'])
        if replay:
            p, risk, _ = teacher.prepare(case, model, physical, base)
            h, _ = teacher.exploration_h(p, risk, cfg['risk_edges'])
            # Diagnostic replay of one scheduled root plus one historical positive
            # per case (when present). Not new data, not a yield estimate.
            indices = {0}
            indices.update(next(([j] for j, s in enumerate(raw['sequences']) if s['event_result']), []))
            for j in sorted(indices):
                actual = teacher.sample_continuation(model, assets, physical, case['state'],
                    {'ids': plan['actor_ids'], 'h': h}, cfg['epsilon'],
                    plan['continuation_seed_start']+j, cfg, base['proxy']['clearance_scale_m'])
                if physical_record(actual) != physical_record(raw['sequences'][j]):
                    raise ValueError('Original native teacher replay differs')
                replays += 1
    if (events, len(sources), original['draws']) != (67, 7, 2112):
        raise ValueError('Historical collision positive control does not reproduce')
    if reproduction is not None:
        root = Path(reproduction)
        if root.resolve() == OLD.resolve():
            raise ValueError('Reproduction must be a separate complete run')
        repeated = artifacts(root)
        for name in original['artifact_hashes']:
            if physical_record(read(root/name)) != physical_record(read(OLD/name)):
                raise ValueError('Full teacher reproduction differs: '+name)
        if physical_record(repeated) != physical_record(dict(original,
                artifact_hashes=repeated['artifact_hashes'])):
            raise ValueError('Reproduction summary differs beyond timing/artifact hashes')
    return dict(verified=True, original_summary_sha256=SUMMARY_SHA, original_lock_sha256=LOCK_SHA,
        events=events, draws=2112, event_sources=sorted(sources), source_episodes=25,
        native_replays=replays, full_reproduction_verified=reproduction is not None,
        reproduction=str(reproduction) if reproduction is not None else None,
        reproduction_summary_sha256=digest(Path(reproduction)/'summary.json') if reproduction is not None else None,
        verifier_sha256=digest(__file__), independent_new_data=False, PPO_started=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reproduction'); p.add_argument('--replay', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args(); result = verify(a.reproduction, a.replay)
    save(a.output, result); print(result, flush=True)


if __name__ == '__main__':
    main()
