"""Measured longitudinal motion over the current one-second LC proxy window.

Future samples are used ONLY as audit outcomes, never as policy input or a safety
mask. The proxy starts at the existing LC label, not true maneuver initiation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .highd_lane_change_baseline import _decision_rows, _direction_map
from .highd_ndd_baseline import _load_split_manifest


def window_metrics(window, start_frame, source_hz, duration_s, direction):
    count = int(round(source_hz * duration_s))
    if source_hz <= 0 or count < 1 or not np.isclose(count / source_hz, duration_s):
        raise ValueError('Window must span a positive integer number of source steps')
    if direction not in (1, 2):
        raise ValueError('Unknown driving direction')
    frames = window['frame'].to_numpy(dtype=int)
    if not np.array_equal(frames, np.arange(start_frame, start_frame + count + 1)):
        raise ValueError('incomplete_or_duplicate_frames')
    velocity = window['xVelocity'].to_numpy(dtype=float)
    acceleration = window['xAcceleration'].to_numpy(dtype=float)
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
        raise ValueError('nonfinite_motion')
    sign = -1.0 if direction == 1 else 1.0
    speed, accel = sign * velocity, sign * acceleration
    if np.any(speed < 0):
        raise ValueError('direction_velocity_mismatch')
    delta = float(speed[-1] - speed[0])
    return {
        'duration_s': duration_s, 'source_samples': len(frames),
        'initial_speed_mps': float(speed[0]), 'final_speed_mps': float(speed[-1]),
        'speed_change_mps': delta, 'mean_acceleration_from_speed_mps2': delta / duration_s,
        'mean_recorded_acceleration_mps2': float(np.trapz(accel, dx=1/source_hz) / duration_s),
        'minimum_recorded_acceleration_mps2': float(accel.min()),
        'maximum_recorded_acceleration_mps2': float(accel.max()),
        'zero_acceleration_endpoint_speed_error_mps': abs(delta),
        'zero_acceleration_speed_path_rmse_mps': float(np.sqrt(np.mean((speed-speed[0])**2))),
    }


def describe_events(events):
    result = {'event_count': len(events),
              'vehicle_count': len({(r['recording'], r['vehicle_id']) for r in events}),
              'recording_count': len({r['recording'] for r in events})}
    if not events:
        return result
    a = np.array([r['mean_acceleration_from_speed_mps2'] for r in events])
    result['near_zero_mean_abs_le_0p1_count'] = int(np.sum(abs(a) <= .1 + 1e-12))
    result['mean_deceleration_counts'] = {
        str(threshold): int(np.sum(a <= -threshold + 1e-12)) for threshold in (.2, .5, 1, 2)}
    result['mean_acceleration_ge_0p5_count'] = int(np.sum(a >= .5 - 1e-12))
    for key in ('mean_acceleration_from_speed_mps2', 'zero_acceleration_endpoint_speed_error_mps',
                'zero_acceleration_speed_path_rmse_mps'):
        values = np.array([r[key] for r in events])
        result[key] = {'mean': float(values.mean()), 'median': float(np.median(values)),
                       'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95))}
    return result


def audit_recording(source_root, recording, context, duration_s):
    rows = _decision_rows(Path(source_root), recording,
        source_hz=context['source_frequency_hz'], target_hz=context['target_frequency_hz'],
        decision_lead_s=context['decision_lead_s'], execution_tail_s=context['execution_tail_s'])
    speed = rows.xVelocity.abs()
    leader = rows.precedingId > 0
    gap = np.where(leader, rows.dhw, context['maximum_gap_m'])
    rr = np.where(leader, rows.precedingXVelocity.abs()-speed, 0)
    eligible = (speed.between(*context['speed_range_mps']) & (gap >= 0)
        & (gap <= context['maximum_gap_m']) & (rr >= context['relative_speed_range_mps'][0])
        & (rr <= context['relative_speed_range_mps'][1]) & (rows.action_index != 1))
    if context.get('require_current_leader_for_lateral', False):
        eligible &= leader
    rows = rows.loc[eligible]
    tracks = pd.read_csv(Path(source_root)/'data'/f'{recording}_tracks.csv',
        usecols=['frame','id','x','width','xVelocity','xAcceleration'])
    lookup = tracks.set_index(['frame','id'])
    groups = {int(i): g.sort_values('frame') for i, g in tracks.groupby('id', sort=False)}
    directions = _direction_map(Path(source_root), recording)
    counts = Counter(eligible_lc_labels=len(rows), audited=0)
    events, excluded = [], []
    for _, row in rows.iterrows():
        vehicle_id, start = int(row.id), int(row.frame)
        group = groups[vehicle_id]
        window = group.loc[group.frame.between(start, start + round(duration_s*context['source_frequency_hz']))]
        try:
            metrics = window_metrics(window, start, context['source_frequency_hz'], duration_s, directions[vehicle_id])
        except ValueError as exc:
            counts[str(exc)] += 1
            excluded.append({'vehicle_id': vehicle_id, 'frame': start, 'reason': str(exc)})
            continue
        side = 'left' if row.action_index == 0 else 'right'
        neighbor_id = int(row[side+'PrecedingId'])
        target_gap = closing = ttc = None
        status = 'no_reported_target_leader'
        if neighbor_id:
            if (start, neighbor_id) not in lookup.index:
                status = 'missing_target_leader'
            else:
                neighbor = lookup.loc[(start, neighbor_id)]
                sign = -1 if directions[vehicle_id] == 1 else 1
                target_gap = float(sign*(neighbor.x+neighbor.width/2-row.x-row.width/2)
                                   - (neighbor.width+row.width)/2)
                closing = float(abs(row.xVelocity)-abs(neighbor.xVelocity))
                status = 'visible' if target_gap <= context['maximum_gap_m'] else 'beyond_observation_range'
                if status == 'visible' and target_gap > 0 and closing > 0:
                    ttc = target_gap / closing
        events.append({'recording': recording, 'vehicle_id': vehicle_id, 'decision_frame': start,
            'side': side, 'has_current_leader': bool(row.precedingId > 0),
            'target_leader_id': neighbor_id, 'target_leader_status': status,
            'initial_target_gap_m': target_gap, 'initial_target_closing_speed_mps': closing,
            'initial_target_ttc_s': ttc, **metrics})
        counts['audited'] += 1
    return {'counts': dict(counts), 'events': events, 'excluded': excluded}


def audit(source_root, manifest_path, context_path, output, recordings=None, duration_s=1.0):
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Duration must be finite and positive')
    splits = _load_split_manifest(Path(manifest_path))
    flat = [r for values in splits.values() for r in values]
    if len(flat) != len(set(flat)):
        raise ValueError('Recording splits overlap')
    allowed = splits['train'] + splits['calibration']
    chosen = allowed if recordings is None else recordings
    if not chosen or len(set(chosen)) != len(chosen) or any(r not in allowed for r in chosen):
        raise ValueError('Only unique train/calibration recordings are allowed')
    context = json.loads(Path(context_path).read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, events = {}, []
    for recording in chosen:
        result = audit_recording(source_root, recording, context, duration_s)
        results[recording] = result
        events.extend(result['events'])
        print(json.dumps({'recording': recording, **result['counts']}), flush=True)
    cohorts = {'all': events,
        'without_current_leader': [r for r in events if not r['has_current_leader']],
        'target_ttc_lt3': [r for r in events if r['initial_target_ttc_s'] is not None and r['initial_target_ttc_s'] < 3]}
    for split in ('train','calibration'):
        cohorts[split] = [r for r in events if r['recording'] in splits[split]]
    summary = {'schema_version': 1, 'recordings_read': chosen,
        'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        'context_config_sha256': hashlib.sha256(Path(context_path).read_bytes()).hexdigest(),
        'duration_s': duration_s, 'source_frequency_hz': context['source_frequency_hz'],
        'decision_lead_frames': round(context['source_frequency_hz']*context['decision_lead_s']),
        'complete_train_calibration': set(chosen)==set(allowed),
        'cohorts': {k: describe_events(v) for k,v in cohorts.items()}, 'by_recording': results,
        'scope': 'Descriptive longitudinal motion around an LC-label proxy, NOT true maneuver '
                 'initiation. Uses future only as audit outcome. No independence, causal safety, '
                 'or target crash-probability claim. No policy or threshold fitted here.'}
    (output/'lane_change_motion_audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_root','manifest','context_config','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--recordings', nargs='+')
    args = parser.parse_args()
    result = audit(args.source_root, args.manifest, args.context_config, args.output, args.recordings)
    print(json.dumps(result['cohorts']))


if __name__ == '__main__':
    main()
