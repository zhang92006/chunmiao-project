"""Matched frozen V46 rollouts with passive 15 Hz observation, no NDE edits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

from .highd_v46_assets import ROOT, ensure_assets, read, save, digest
from .highd_paper_dynamics import Road
from .highd_paper_discrete import DiscreteRoad
from .highd_paper_discrete_pairs import PairModelQueries, DiscretePairs, SINGLE_OPTIONS
from . import highd_paper_discrete as executor

FIELDS = ('id', 'x', 'v', 'lane', 'y', 'target', 'length', 'width')
_MODEL = _ASSETS = _CONFIG = None


def state(vehicles):
    return np.asarray([[getattr(car, key) for key in FIELDS] for car in vehicles], dtype=float)


class RecordedRng:
    def __init__(self, rng):
        self.rng = rng
        self.uniforms = []

    def random(self):
        value = float(self.rng.random())
        self.uniforms.append(value)
        return value


def worker_init(assets, config):
    global _MODEL, _ASSETS, _CONFIG, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _ASSETS = Path(assets)
    _CONFIG = config
    _MODEL = PairModelQueries(_ASSETS/'model', _ASSETS/'lateral', **SINGLE_OPTIONS)


def run_arm(seed, snapshot, arm, output):
    model = _MODEL
    sampler = DiscretePairs(_ASSETS/'pair', model, _ASSETS/'lateral',
                            mode=_CONFIG['arms'][arm], **_CONFIG['pair_options'])
    road = DiscreteRoad(model, seed, snapshot, sampler)
    road.decisions = []
    road.rng = RecordedRng(road.rng)
    states = [state(road.vehicles)]
    original_collision = executor.collisions

    def observe(vehicles):
        # The frozen executor calls this after its final lateral update and
        # advances its clock. The observer neither draws nor edits anything.
        states.append(state(vehicles))
        return original_collision(vehicles)

    start = time.perf_counter()
    executor.collisions = observe
    try:
        while road.time < _CONFIG['horizon_s']-1e-9:
            if not road.step():
                break
    finally:
        executor.collisions = original_collision
    elapsed = time.perf_counter()-start
    actions = np.asarray([[d['accelerations'][int(car_id)] for car_id in states[0][:, 0]]
                          for d in road.decisions])
    codes = np.asarray([[d['actions'][int(car_id)] for car_id in states[0][:, 0]]
                        for d in road.decisions], dtype=np.int16)
    trajectory = np.asarray(states)
    uniforms = np.asarray(road.rng.uniforms).reshape(len(road.decisions), len(road.vehicles))
    path = Path(output)/f'{arm}_seed{seed}.npz'
    np.savez_compressed(path, states=trajectory, actions=actions, action_codes=codes, uniforms=uniforms)
    report = {
        'seed': seed, 'arm': arm, 'elapsed_s': road.time,
        'complete': not road.crashes and road.time >= _CONFIG['horizon_s']-1e-9,
        'collision_pairs': road.crashes, 'collision_context': road.crash_context,
        'cav_collision_count': None, 'bv_collision_episode': bool(road.crashes),
        'diagnostics': dict(road.diagnostics), 'runtime_s': elapsed,
        'trace_sha256': digest(path), 'fields': list(FIELDS),
        'pair_decisions': sum(len(r['pairs']) for r in sampler.records),
        'correlated_pair_decisions': sum(u.get('independent_mass', 1.) < 1.
                                        for r in sampler.records for u in r['units']),
        'maximum_simultaneous_pairs': max(len(r['pairs']) for r in sampler.records),
        'pair_additions': sum(len(r['added']) for r in sampler.records),
        'pair_removals': sum(len(r['removed']) for r in sampler.records),
        'maximum_marginal_error': sampler.maximum_marginal_error,
        'log_importance_ratio': 0., 'actor_count': len(road.vehicles),
    }
    save(Path(output)/f'{arm}_seed{seed}.json', report)
    return report, trajectory, uniforms


def run_seed(seed, output):
    initial = Road(_MODEL, seed)
    snapshot = {'seed': seed, 'model_sha256': _MODEL.summary['model_sha256'],
                'vehicles': [vars(car).copy() for car in initial.vehicles],
                'diagnostics': initial.initialization,
                'rng_state_after_initialization': initial.rng.bit_generator.state}
    save(Path(output)/f'initial_seed{seed}.json', snapshot)
    reports = {}; traces = {}; uniforms = {}
    # Counterbalance process-local cache/timing order across the fixed seeds.
    for arm in (('single', 'dual') if seed % 2 == 0 else ('dual', 'single')):
        reports[arm], traces[arm], uniforms[arm] = run_arm(seed, snapshot, arm, output)
    np.testing.assert_array_equal(traces['single'][0], traces['dual'][0])
    n = min(len(uniforms['single']), len(uniforms['dual']))
    np.testing.assert_array_equal(uniforms['single'][:n], uniforms['dual'][:n])
    reports['matched'] = {'initial_states_identical': True, 'rng_common_prefix_identical': True,
                          'common_decisions': n, 'initial_sha256': digest(Path(output)/f'initial_seed{seed}.json')}
    save(Path(output)/f'pair_seed{seed}.json', reports)
    return {arm: {key: reports[arm][key] for key in ('seed', 'elapsed_s', 'complete', 'runtime_s')}
            for arm in ('single', 'dual')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/highd_v46_matched_fidelity_v1.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assets, lock = ensure_assets(); cfg = read(args.config)
    if cfg['natural_target_sha256'] != lock['natural_target_sha256'] or cfg['single_options'] != SINGLE_OPTIONS:
        raise ValueError('Frozen target mismatch')
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    save(root/'protocol.json', {'config': cfg, 'config_sha256': digest(args.config),
                               'collector_sha256': digest(__file__), 'frozen_target': lock['natural_target_sha256']})
    seeds = range(cfg['seed_start'], cfg['seed_start']+cfg['episodes'])
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(str(assets), cfg)) as pool:
        futures = {pool.submit(run_seed, seed, str(root)): seed for seed in seeds}
        for index, future in enumerate(as_completed(futures), 1):
            print({'completed_pairs': index, 'result': future.result()}, flush=True)
    save(root/'completion.json', {'matched_seed_pairs': cfg['episodes'], 'all_seeds_retained': True})


if __name__ == '__main__':
    main()
