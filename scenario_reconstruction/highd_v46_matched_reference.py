"""Fixed-bin highD references for the frozen V46 matched experiment (no fit)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .highd_v46_assets import ROOT, ensure_assets, read, save, digest
from .highd_paper_data import clock, histogram
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_paper_pair_calibration import samples


def bin_indices(values, spec):
    first, last, step = spec
    edges = first-step/2+np.arange(round((last-first)/step)+2)*step
    return np.searchsorted(np.round(edges, 10), np.round(values, 10), side='right')


def bin_count(spec):
    return round((spec[1]-spec[0])/spec[2])+3


def hist2(first, second, first_spec, second_spec):
    n, m = bin_count(first_spec), bin_count(second_spec)
    indices = bin_indices(first, first_spec)*m+bin_indices(second, second_spec)
    return np.bincount(indices, minlength=n*m).reshape(n, m)


def action_counts(front, rear, bins):
    return {'joint_accel': hist2(front, rear, bins['accel'], bins['accel']),
            'relative_accel': histogram(front-rear, *bins['relative_accel']),
            'accel_moments': np.array([len(front), front.sum(), rear.sum(),
                (front*front).sum(), (rear*rear).sum(), (front*rear).sum()], dtype=float)}


def record_reference(source, rec, model, config):
    root = Path(source)/'data'; bins = config['bins']
    hashes = {kind: digest(root/f'{rec}_{kind}.csv') for kind in ('tracks', 'tracksMeta', 'recordingMeta')}
    if hashes != model.summary['source_sha256'][rec]:
        raise ValueError('Frozen train source changed: '+rec)
    meta = pd.read_csv(root/f'{rec}_tracksMeta.csv')
    if int(pd.read_csv(root/f'{rec}_recordingMeta.csv').iloc[0].frameRate) != 25:
        raise ValueError('Expected 25 Hz highD source')
    cols = ['frame', 'id', 'laneId', 'xVelocity', 'yVelocity', 'precedingId', 'precedingXVelocity', 'dhw']
    rows = pd.read_csv(root/f'{rec}_tracks.csv', usecols=cols).sort_values(['id', 'frame']).reset_index(drop=True)
    supported = meta.loc[meta['class'].isin(model.config['vehicle_classes']) & (meta.numFrames >= 75), 'id']
    speed = rows.xVelocity.abs().to_numpy(); gap = rows.dhw.to_numpy()
    rr = rows.precedingXVelocity.abs().to_numpy()-speed
    frame = rows.frame.to_numpy(); actor = rows.id.to_numpy(); leader = rows.precedingId.to_numpy()
    lane = rows.laneId.to_numpy()
    domain = rows.id.isin(supported).to_numpy() & (speed >= 20) & (speed <= 40)
    timed = clock(frame, 25, 10)
    following = domain & (leader > 0) & (gap > 0) & (gap <= 115) & np.isfinite(rr)
    counts = {'speed': histogram(speed[domain & timed], *bins['speed']),
              'gap': histogram(gap[following & timed], *bins['gap']),
              'rr': histogram(rr[following & timed], *bins['rr']),
              'joint_state': hist2(gap[following & timed], rr[following & timed], bins['gap'], bins['rr'])}

    # Stable pair/lane identities, complete source endpoints, same speed domain
    # for both actors. The windows are disjoint within each relationship spell.
    lookup = pd.MultiIndex.from_frame(rows[['frame', 'id']])
    lead_index = lookup.get_indexer(pd.MultiIndex.from_arrays([frame, leader]))
    safe = np.maximum(lead_index, 0)
    direction = rows.id.map(dict(zip(meta.id, meta.drivingDirection))).to_numpy()
    quiet = rows.yVelocity.abs().to_numpy() < .2
    valid = (following & (lead_index >= 0) & domain[safe] & (lane == lane[safe]) &
             (direction == direction[safe]) & quiet & quiet[safe])
    changed = np.ones(len(rows), dtype=bool)
    changed[1:] = ((actor[1:] != actor[:-1]) | (frame[1:] != frame[:-1]+1) |
                   (leader[1:] != leader[:-1]) | (lane[1:] != lane[:-1]) |
                   (lane[safe][1:] != lane[safe][:-1]) | ~valid[1:] | ~valid[:-1])
    starts = np.flatnonzero(changed); ends = np.r_[starts[1:], len(rows)]
    persistent = np.zeros(len(rows), dtype=bool); windows = strict60 = 0; max_duration = 0.
    width = round(config['persistent_alternative_window_s']*25)
    for start, stop in zip(starts, ends):
        if not valid[start]:
            continue
        duration = (stop-start-1)/25
        max_duration = max(max_duration, duration)
        strict60 += duration >= config['persistent_duration_s']
        for left in range(start, stop-width, width):
            persistent[left:left+width] = True
            windows += 1
    keep = persistent & timed
    counts.update({'persistent_gap': histogram(gap[keep], *bins['gap']),
                   'persistent_rr': histogram(rr[keep], *bins['rr']),
                   'persistent_joint_state': hist2(gap[keep], rr[keep], bins['gap'], bins['rr'])})
    del rows, lookup
    data, action_info = samples(model, source, rec, return_data=True, decision_hz=1, one_second_actions=True)
    # Include observed zero-probability labels; do not condition this fidelity
    # reference on a fitted model having assigned them positive mass.
    front = model.a[data['b']]; rear = model.a[data['a']]
    counts.update(action_counts(front, rear, bins))
    summary = {'source_sha256': hashes, 'speed_samples': int(counts['speed'].sum()),
               'following_samples': int(counts['gap'].sum()), 'action_selector': action_info,
               'persistent_5s_windows': windows, 'persistent_60s_spells': int(strict60),
               'max_supported_persistent_s': max_duration, 'max_source_track_s': float(meta.numFrames.max()/25)}
    return counts, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--config', default=str(ROOT/'configs/highd_v46_matched_fidelity_v1.json'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); assets, lock = ensure_assets(); cfg = read(args.config)
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    if cfg['reference_recordings'] != model.summary['train_recordings']:
        raise ValueError('Reference recordings differ from frozen train exposure')
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    pooled = {}; per_record = {}
    for rec in cfg['reference_recordings']:
        own, info = record_reference(args.source_root, rec, model, cfg)
        for key, value in own.items():
            pooled[key] = pooled.get(key, np.zeros_like(value))+value
        per_record[rec] = info
        np.savez_compressed(root/f'reference_{rec}.npz', **own)
        print({'recording': rec, 'speed_samples': info['speed_samples'],
               'action_pairs': info['action_selector']['eligible_pairs'],
               'persistent_5s_windows': info['persistent_5s_windows']}, flush=True)
    with np.load(assets/'reference/reference.npz') as original:
        for key in ('speed', 'gap', 'rr'):
            np.testing.assert_array_equal(pooled[key], original['reference_'+key+'_counts'])
    np.savez_compressed(root/'reference.npz', **pooled)
    save(root/'reference_summary.json', {'config_sha256': digest(args.config), 'exporter_sha256': digest(__file__),
        'natural_target_sha256': lock['natural_target_sha256'], 'reference_sha256': digest(root/'reference.npz'),
        'original_three_marginal_references_exactly_reproduced': True, 'recordings': per_record,
        'action_order': ['front', 'rear'], 'action_time_unit_s': 1.,
        'scope': 'Existing train exposure, fixed reference. No refitting, selection or independent highD holdout claim.'})


if __name__ == '__main__':
    main()
