"""Frozen-state Q_S allocation with explicit state/action likelihoods."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import time

import numpy as np

from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, read, save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import sample_continuation
from .highd_v47_action_teacher import inputs as teacher_inputs, exploration_h
from .highd_v47_allocation import waterfill

DEFAULT = 'configs/highd_v47_state_allocation_v1.json'


def state_allocation(alpha, m2, floor=.2, regularization=.01):
    alpha, m2 = np.asarray(alpha, float), np.asarray(m2, float)
    if (alpha.ndim != 1 or alpha.shape != m2.shape or not len(alpha)
            or not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(m2))
            or np.any(alpha <= 0) or np.any(m2 < 0) or not np.isclose(alpha.sum(), 1)
            or not 0 < floor <= 1 or not np.isfinite(regularization) or regularization < 0):
        raise ValueError('Invalid state allocation')
    regularized = m2 + regularization*float(alpha @ m2)
    return waterfill(alpha, regularized, floor), regularized


def schedule(alpha, q, n, seed):
    alpha, q = np.asarray(alpha), np.asarray(q)
    if n < 2 or n % len(alpha) or q.shape != alpha.shape or np.any(q <= 0) or not np.isclose(q.sum(), 1):
        raise ValueError('Invalid matched state budget')
    # Same uniforms across IID arms; never consume the native action RNG.
    u = np.random.default_rng(seed).random(n)
    draw = lambda p: np.minimum(np.searchsorted(np.cumsum(p), u, side='right'), len(p)-1).tolist()
    return {'target_iid': draw(alpha), 'allocated_iid': draw(q),
            'equal_state_stratified': np.repeat(np.arange(len(alpha)), n//len(alpha)).tolist()}


def summarize(samples, plans, arm):
    n = len(samples)
    values = np.asarray([s['event_result']*np.exp(s['total_log_weight']) for s in samples])
    source_m2 = {}
    for s, v in zip(samples, values):
        source = str(plans[s['state_id']]['source_seed'])
        source_m2[source] = source_m2.get(source, 0.) + float(v*v/n)
    if arm == 'equal_state_stratified':
        # Total weights use alpha/(n_s/N); pooled average still equals stratified IS.
        variance = 0.
        for i, p in enumerate(plans):
            raw = np.asarray([s['event_result']*np.exp(s['generation_log_weight']) for s in samples if s['state_id'] == i])
            if len(raw) < 2:
                raise ValueError('Missing fixed stratum replicates')
            variance += p['target_mass']**2 * float(raw.var(ddof=1))/len(raw)
    else:
        variance = float(values.var(ddof=1)/n)
    m2 = float(np.mean(values*values))
    return {'draws': n, 'events': sum(s['event_result'] for s in samples),
            'event_sources': sorted({plans[s['state_id']]['source_seed'] for s in samples if s['event_result']}),
            'mean': float(values.mean()), 'pooled_second_moment': m2,
            'estimator_variance': variance, 'MCSE': float(np.sqrt(variance)),
            'mean_total_likelihood': float(np.mean([np.exp(s['total_log_weight']) for s in samples])),
            'state_draw_counts': dict(Counter(s['state_id'] for s in samples)),
            'source_second_moment_contributions': source_m2,
            'maximum_source_second_moment_share': max(source_m2.values())/m2 if m2 > 0 else None,
            'uncertainty_scope': 'Conditional on frozen states and fitted Q_S; not source-population generalization'}


def inputs(config):
    cfg = read(config)
    if (cfg['version'] != 1 or cfg['state_floor'] != .2 or cfg['M2_regularization'] != .01
            or cfg['draws_per_arm'] != 2112 or cfg['arms'] != ['target_iid', 'allocated_iid', 'equal_state_stratified']
            or cfg['epsilon'] != [.1, .1] or cfg['horizon_s'] != 4
            or cfg['state_rng_seed'] != 470017 or cfg['continuation_seed_start'] != 4800000):
        raise ValueError('Version state protocol before changing allocation or budget')
    tc, base, physical, assets, cases, plans, teacher_lock = teacher_inputs(cfg['teacher_config'])
    root = Path(cfg['teacher_path'])
    if digest(root/'summary.json') != cfg['teacher_summary_sha256']:
        raise ValueError('Teacher pilot summary changed')
    pilot = read(root/'summary.json')
    if not pilot['complete'] or pilot['config'] != tc or any(pilot['lock'].get(k) != v for k, v in teacher_lock.items()):
        raise ValueError('Teacher pilot implementation/inputs differ')
    for name, sha in pilot['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Teacher pilot artifact changed')
    if len(plans) != 33 or len({p['source_seed'] for p in plans}) != 25:
        raise ValueError('Do not change target bank')
    alpha = np.array([p['target_mass'] for p in plans]); m2 = []; designs = []; means = []
    for i, plan in enumerate(plans):
        d = read(root/f'case_{i:03d}_design.json'); raw = read(root/f'case_{i:03d}_labels.json')
        if d['plan'] != plan or raw['plan'] != plan or identity(d['state']) != plan['state_sha256']:
            raise ValueError('State identity differs')
        p, risk = np.array(d['P']), np.array(d['risk'])
        h, _ = exploration_h(p, risk, tc['risk_edges'])
        if not np.array_equal(h, d['H']):
            raise ValueError('Action teacher differs')
        if len(raw['sequences']) != 64 or [x['seed'] for x in raw['sequences']] != list(range(plan['continuation_seed_start'], plan['continuation_seed_start']+64)):
            raise ValueError('Incomplete teacher schedule')
        for x in raw['sequences']:
            audit_sample(x, cfg['epsilon'])
        z = np.array([x['event_result']*np.exp(x['generation_log_weight']) for x in raw['sequences']])
        m2.append(float(np.mean(z*z))); means.append(float(z.mean())); designs.append(d)
    q, regularized = state_allocation(alpha, m2, cfg['state_floor'], cfg['M2_regularization'])
    schedules = schedule(alpha, q, cfg['draws_per_arm'], cfg['state_rng_seed'])
    allocation = {'target_masses': alpha.tolist(), 'state_Q': q.tolist(), 'pilot_M2': m2,
                  'pilot_means': means, 'regularized_M2': regularized.tolist(),
                  'source_Q': {str(s): float(sum(q[i] for i, p in enumerate(plans) if p['source_seed'] == s)) for s in sorted({p['source_seed'] for p in plans})},
                  'predicted_iid_M2_ratio': float(np.sum(alpha*alpha*np.array(m2)/q)/(alpha@np.array(m2))),
                  'schedules': schedules}
    lock = {'config_sha256': digest(config), 'implementation_sha256': digest(__file__),
            'allocator_sha256': digest(Path(__file__).with_name('highd_v47_allocation.py')),
            'teacher_lock': teacher_lock, 'pilot_summary_sha256': cfg['teacher_summary_sha256'],
            'target_sha256': identity([(p['state_sha256'], p['actor_ids'], p['target_mass']) for p in plans]),
            'allocation_sha256': identity(allocation)}
    return cfg, base, physical, assets, plans, designs, allocation, lock


def freeze(config, output):
    *_, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def run(config, lock_path, output):
    cfg, base, physical, assets, plans, designs, allocation, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen state proposal or input changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', {'config': cfg, 'lock': lock})
    save(out/'state_allocation.json', dict(allocation, plans=plans))
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    results = {}; started = time.perf_counter()
    alpha = np.array(allocation['target_masses']); q = np.array(allocation['state_Q'])
    for arm in cfg['arms']:
        samples = []; tick = time.perf_counter()
        qs = alpha if arm == 'target_iid' else q if arm == 'allocated_iid' else np.full(len(plans), 1/len(plans))
        for draw_id, i in enumerate(allocation['schedules'][arm]):
            plan, d = plans[i], designs[i]
            x = sample_continuation(model, assets, physical, d['state'], {'ids': plan['actor_ids'], 'h': np.array(d['H'])},
                cfg['epsilon'], cfg['continuation_seed_start']+draw_id, cfg, base['proxy']['clearance_scale_m'])
            audit_sample(x, cfg['epsilon'])
            state_log = float(np.log(alpha[i]/qs[i]))
            x.update(draw_id=draw_id, state_id=i, target_state_mass=float(alpha[i]), generation_state_mass=float(qs[i]),
                     state_log_weight=state_log, total_log_weight=state_log+x['generation_log_weight'])
            samples.append(x)
            if (draw_id+1) % 256 == 0:
                print({'arm': arm, 'draws': draw_id+1, 'total': cfg['draws_per_arm']}, flush=True)
        save(out/f'{arm}_sequences.json', {'arm': arm, 'sequences': samples})
        results[arm] = dict(summarize(samples, plans, arm), wall_s=time.perf_counter()-tick)
        print({'arm': arm, **{k: results[arm][k] for k in ('events', 'mean', 'estimator_variance', 'maximum_source_second_moment_share')}}, flush=True)
    ratios = {arm: results['allocated_iid']['estimator_variance']/results[arm]['estimator_variance']
              if results[arm]['estimator_variance'] > 0 else None for arm in ('target_iid', 'equal_state_stratified')}
    result = {'complete': True, 'config': cfg, 'lock': lock, 'arms': results,
              'allocated_variance_ratio': ratios, 'total_wall_s': time.perf_counter()-started,
              'source_generalization_validated': False, 'population_acceleration': None, 'ppo_ready': False,
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
        print(r['allocated_variance_ratio'])


if __name__ == '__main__':
    main()
