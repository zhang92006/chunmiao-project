"""Train-only gap-preserving priors, evaluated on calibration only.

This does not replace the runtime empty-cell fallback or alter lateral actions.
Partial recording scans are diagnostics and cannot export a selected model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .highd_ndd_baseline import (
    _add_counts, _empty_counts, _load_split_manifest, _observations, _recording_rows,
)
from .highd_longitudinal_calibration import score_counts


def gap_preserving_probabilities(counts, concentration, parent_concentration=100.0):
    """Global -> relative speed -> (gap, relative speed) -> full CF state."""
    cf = np.asarray(counts, dtype=float)
    if cf.ndim != 4 or not np.isfinite(cf).all() or np.any(cf < 0) or cf.sum() <= 0:
        raise ValueError('Expected nonempty finite CF[gap,rr,speed,action] counts')
    if any(not np.isfinite(v) or v <= 0 for v in (concentration, parent_concentration)):
        raise ValueError('Prior concentrations must be finite and positive')

    def posterior(values, prior, strength):
        return (values + strength * prior) / (values.sum(axis=-1, keepdims=True) + strength)

    global_prior = cf.sum(axis=(0, 1, 2)) + 0.5
    global_prior /= global_prior.sum()
    rr_prior = posterior(cf.sum(axis=(0, 2)), global_prior, parent_concentration)
    gap_rr_prior = posterior(cf.sum(axis=2), rr_prior[None, :, :], parent_concentration)
    return posterior(cf, gap_rr_prior[:, :, None, :], concentration)


def support_report(observed, train):
    n = train.sum(axis=-1)
    labels = {'empty': n == 0, 'n1_2': (n >= 1) & (n <= 2),
              'n3_9': (n >= 3) & (n <= 9), 'n10_99': (n >= 10) & (n <= 99),
              'n100_plus': n >= 100}
    return {name: {'training_cells': int(mask.sum()),
                   'observed_frames': int(observed[mask].sum())}
            for name, mask in labels.items()}


def select_candidate(candidates, complete, tail_frames, tail_recordings, config):
    """Baseline remains valid; tail count gates prevent claiming unsupported acceptance."""
    if not complete:
        return None, 'partial_calibration_diagnostic_only'
    if tail_frames < config['minimum_covered_tail_frames'] or tail_recordings < config['minimum_tail_recordings']:
        return None, 'insufficient_tail_evidence'
    baseline = candidates[0]
    eligible = []
    for candidate in candidates:
        candidate['passes_non_degradation_gate'] = all(
            candidate[group][metric] is not None and baseline[group][metric] is not None
            and candidate[group][metric] <= baseline[group][metric] + 1e-12
            for group in ('overall', 'tail') for metric in ('covered_nll', 'covered_brier'))
        if candidate['passes_non_degradation_gate']:
            eligible.append(candidate)
    if not eligible:
        return None, 'no_scorable_candidate'
    return min(eligible, key=lambda c: c['overall']['covered_nll']), 'calibration_screen_only'


def calibrate(source_root, manifest_path, model_path, config, output, recordings=None):
    splits = _load_split_manifest(Path(manifest_path))
    flat = [r for values in splits.values() for r in values]
    if len(flat) != len(set(flat)):
        raise ValueError('Recording splits overlap')
    if sorted(splits['train']) != sorted(config['train_recordings']):
        raise ValueError('Train split differs from pinned artifact')
    chosen = list(splits['calibration'] if recordings is None else recordings)
    if not chosen or len(set(chosen)) != len(chosen) or any(r not in splits['calibration'] for r in chosen):
        raise ValueError('Only unique calibration recordings may be read')
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if model_hash != config['source_model_sha256']:
        raise ValueError('Source model differs from pinned baseline')
    with np.load(model_path, allow_pickle=False) as artifact:
        arrays = {k: artifact[k] for k in artifact.files}
    train_cf, train_ff = arrays['car_following_counts'], arrays['free_flow_counts']
    axes = {key: arrays[key + '_axis'] for key in ('speed', 'gap', 'range_rate', 'acceleration')}
    baseline = (arrays['car_following_probability'], arrays['free_flow_probability'])
    cf, ff = _empty_counts(axes)
    tail_cf, tail_ff = _empty_counts(axes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    per_recording = {}
    for recording in chosen:
        rc, rf = _empty_counts(axes)
        tc, tf = _empty_counts(axes)
        for chunk in _recording_rows(Path(source_root), recording,
                chunk_size=config['chunk_size'], frame_stride=None,
                source_frequency_hz=config['source_frequency_hz'],
                target_frequency_hz=config['target_frequency_hz']):
            obs = _observations(chunk, axes)
            _add_counts(rc, rf, obs, axes)
            tail = dict(obs)
            closing = -obs['range_rate']
            # Use measured continuous values, not TTC computed from rounded state bins.
            tail['car_following'] = obs['car_following'] & (closing > 0) & (obs['gap'] > 0) & (
                obs['gap'] < config['tail_ttc_s'] * closing)
            tail['free_flow'] = np.zeros_like(obs['free_flow'])
            _add_counts(tc, tf, tail, axes)
        cf += rc
        ff += rf
        tail_cf += tc
        per_recording[recording] = {
            'eligible_frames': int(rc.sum() + rf.sum()), 'tail_frames': int(tc.sum()),
            'covered_tail_frames': int(tc[train_cf.sum(axis=-1) > 0].sum())}
        print(json.dumps({'recording_completed': recording, **per_recording[recording]}), flush=True)
    candidates = []
    for strength in [None, *config['prior_concentrations']]:
        # Score float32 artifacts as they will actually be loaded by the adapter.
        pdf = baseline if strength is None else (
            gap_preserving_probabilities(train_cf, strength, config['parent_concentration']).astype(np.float32), baseline[1])
        candidates.append({'name': 'frozen_v2' if strength is None else 'gap_preserving',
            'concentration': strength,
            'overall': score_counts(cf, ff, train_cf, train_ff, pdf),
            'tail': score_counts(tail_cf, tail_ff, train_cf, train_ff, pdf)})
    complete = set(chosen) == set(splits['calibration'])
    selected, status = select_candidate(candidates, complete,
        candidates[0]['tail']['runtime_covered_observations'],
        sum(r['covered_tail_frames'] > 0 for r in per_recording.values()), config)
    result = {'schema_version': 1, 'status': status, 'config': config,
        'source_model_sha256': model_hash,
        'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        'recordings_read': chosen, 'complete_calibration': complete,
        'per_recording': per_recording, 'candidates': candidates, 'selected': selected,
        'calibration_cf_support': support_report(cf, train_cf),
        'calibration_tail_cf_support': support_report(tail_cf, train_cf),
        'runtime_empty_state_fallback_changed': False,
        'scope': 'Calibration-only instantaneous longitudinal scoring. No validation/test reads; '
                 'tail frames are correlated, not independent events. Sparse-cell hard fallback '
                 'boundary remains unresolved. This is not deployment or safety acceptance.'}
    if selected is not None and selected['concentration'] is not None:
        arrays['car_following_probability'] = gap_preserving_probabilities(
            train_cf, selected['concentration'], config['parent_concentration']).astype(np.float32)
        path = output / 'highd_longitudinal_gap_candidate.npz'
        np.savez_compressed(path, **arrays)
        result['candidate_artifact'] = path.name
        result['candidate_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'tail_calibration_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_root', 'manifest', 'model', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--recordings', nargs='+', help='Partial calibration diagnostic; no model selection/export')
    args = parser.parse_args()
    result = calibrate(args.source_root, args.manifest, args.model,
        json.loads(Path(args.config).read_text(encoding='utf-8')), args.output, args.recordings)
    print(json.dumps({'status': result['status'], 'selected': result['selected'],
                      'candidate_artifact': result.get('candidate_artifact')}))


if __name__ == '__main__':
    main()
