"""Calibration-only support-aware blend of highD and original longitudinal PDFs.

The blend is conditional-longitudinal only. It does not alter lane-change
probabilities, runtime sampling, or the formal D2RL target distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .highd_longitudinal_tail_calibration import gap_preserving_probabilities
from .highd_ndd_baseline import (
    _add_counts, _empty_counts, _load_split_manifest, _observations, _recording_rows,
)


def support_weight(state_count, tau):
    count = np.asarray(state_count, dtype=float)
    if not np.isfinite(count).all() or np.any(count < 0):
        raise ValueError('State counts must be finite and nonnegative')
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('Tau must be finite and positive')
    return count / (count + float(tau))


def blend_probabilities(highd, reference, state_count, tau):
    highd, reference = np.asarray(highd, dtype=float), np.asarray(reference, dtype=float)
    if highd.shape != reference.shape or highd.ndim < 2 or highd.shape[:-1] != np.shape(state_count):
        raise ValueError('Probability and state-count shapes disagree')
    for pdf in (highd, reference):
        if not np.isfinite(pdf).all() or np.any(pdf < 0) or not np.allclose(pdf.sum(axis=-1), 1):
            raise ValueError('Expected normalized finite nonnegative PDFs')
    weight = support_weight(state_count, tau)[..., None]
    return weight * highd + (1-weight) * reference


def score_all(observed, probability):
    observed, probability = np.asarray(observed, dtype=float), np.asarray(probability, dtype=float)
    if observed.shape != probability.shape or observed.ndim < 2:
        raise ValueError('Observed counts and probabilities must have the same action shape')
    if (not np.isfinite(observed).all() or np.any(observed < 0)
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=-1), 1)):
        raise ValueError('Invalid counts or probabilities')
    total = float(observed.sum())
    positive = probability > 0
    supported = float(observed[positive].sum())
    zero = int(observed[~positive].sum())
    nll = -float(np.sum(observed[positive] * np.log(probability[positive]))) / supported if supported else None
    brier = float(np.sum(observed.sum(axis=-1) * np.sum(probability**2, axis=-1)
                         - 2*np.sum(observed*probability, axis=-1) + observed.sum(axis=-1)))
    return {'observations': int(total), 'positive_support_observations': int(supported),
            'zero_probability_observations': zero,
            'positive_support_fraction': supported/total if total else None,
            'nll_on_positive_support': nll, 'brier_all_observations': brier/total if total else None}


def original_reference_grid(axes):
    """Evaluate the unmodified original longitudinal policy on measured net gaps."""
    from controller.nddcontroller import NDDController
    import conf.conf as conf

    if not np.allclose(axes['acceleration'], np.asarray(conf.acc_list), atol=1e-12, rtol=0):
        raise ValueError('Original and highD acceleration axes differ')
    shape = tuple(len(axes[k]) for k in ('gap','range_rate','speed','acceleration'))
    result = np.empty(shape, dtype=float)
    for gi,gap in enumerate(axes['gap']):
        for ri,rr in enumerate(axes['range_rate']):
            for si,speed in enumerate(axes['speed']):
                obs = {'Ego': {'velocity': float(speed), 'position': [0.0, 0.0]},
                       'Lead': {'velocity': float(speed+rr), 'distance': float(gap),
                                'position': [float(gap+conf.LENGTH), 0.0]}}
                _, pdf = NDDController.Longitudinal_NDD(obs)
                pdf = np.asarray(pdf, dtype=float).copy()
                if pdf.shape != (len(axes['acceleration']),) or not np.isfinite(pdf).all() or pdf.sum() <= 0:
                    raise ValueError('Invalid original longitudinal distribution')
                result[gi,ri,si] = pdf/pdf.sum()
    return result


def choose(candidates, complete):
    if not complete:
        return None, 'partial_calibration_diagnostic_only'
    baseline = candidates[0]
    minimum_zeros = {
        group: min(item[group]['zero_probability_observations'] for item in candidates)
        for group in ('overall','tail')
    }
    eligible = []
    for item in candidates:
        item['passes_support_and_brier_gate'] = all(
            item[group]['zero_probability_observations'] == minimum_zeros[group]
            and item[group]['brier_all_observations'] <= baseline[group]['brier_all_observations'] + 1e-12
            for group in ('overall','tail'))
        if item['passes_support_and_brier_gate']:
            eligible.append(item)
    return (min(eligible, key=lambda x:x['overall']['brier_all_observations']),
            'calibration_screen_only') if eligible else (None, 'no_candidate_passed')


def calibrate(source_root, manifest_path, model_path, config, output, recordings=None,
              reference_builder=original_reference_grid):
    splits = _load_split_manifest(Path(manifest_path))
    flat = [r for values in splits.values() for r in values]
    if len(flat) != len(set(flat)):
        raise ValueError('Recording splits overlap')
    chosen = list(splits['calibration'] if recordings is None else recordings)
    if not chosen or len(set(chosen)) != len(chosen) or any(r not in splits['calibration'] for r in chosen):
        raise ValueError('Only unique calibration recordings may be read')
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if model_hash != config['source_model_sha256']:
        raise ValueError('Source model differs from pinned candidate')
    with np.load(model_path, allow_pickle=False) as artifact:
        arrays = {k: artifact[k] for k in artifact.files}
    axes = {k: arrays[k+'_axis'] for k in ('speed','gap','range_rate','acceleration')}
    train_cf, train_ff = arrays['car_following_counts'], arrays['free_flow_counts']
    highd = arrays['car_following_probability'].astype(float)
    reference = reference_builder(axes)
    observed, observed_ff = _empty_counts(axes)
    tail, tail_ff = _empty_counts(axes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    by_recording = {}
    for recording in chosen:
        rc, rf = _empty_counts(axes)
        tc, tf = _empty_counts(axes)
        for chunk in _recording_rows(Path(source_root), recording, chunk_size=config['chunk_size'],
                frame_stride=None, source_frequency_hz=config['source_frequency_hz'],
                target_frequency_hz=config['target_frequency_hz']):
            obs = _observations(chunk, axes)
            _add_counts(rc, rf, obs, axes)
            subset = dict(obs)
            closing = -obs['range_rate']
            subset['car_following'] = obs['car_following'] & (closing > 0) & (obs['gap'] > 0) & (
                obs['gap'] < config['tail_ttc_s']*closing)
            subset['free_flow'] = np.zeros_like(obs['free_flow'])
            _add_counts(tc, tf, subset, axes)
        observed += rc
        observed_ff += rf
        tail += tc
        tail_ff += tf
        by_recording[recording] = {'car_following_frames': int(rc.sum()), 'tail_frames': int(tc.sum())}
        print(json.dumps({'recording_completed': recording, **by_recording[recording]}), flush=True)
    state_count = train_cf.sum(axis=-1)
    baseline = np.where((state_count > 0)[...,None], highd, reference)
    candidates = [{'name':'hard_fallback_baseline','tau':None,
                   'overall':score_all(observed,baseline),'tail':score_all(tail,baseline),
                   'mean_highd_weight_overall':float(np.average(state_count>0,weights=observed.sum(axis=-1))),
                   'mean_highd_weight_tail':float(np.average(state_count>0,weights=tail.sum(axis=-1)))}]
    candidates.append({'name':'highd_hierarchical_parent_all_states','tau':0.0,
        'overall':score_all(observed,highd),'tail':score_all(tail,highd),
        'mean_highd_weight_overall':1.0,'mean_highd_weight_tail':1.0})
    for tau in config['tau_candidates']:
        pdf = blend_probabilities(highd, reference, state_count, tau)
        weight = support_weight(state_count,tau)
        candidates.append({'name':'support_aware_blend','tau':tau,
            'overall':score_all(observed,pdf),'tail':score_all(tail,pdf),
            'mean_highd_weight_overall':float(np.average(weight,weights=observed.sum(axis=-1))),
            'mean_highd_weight_tail':float(np.average(weight,weights=tail.sum(axis=-1)))})
    complete = set(chosen)==set(splits['calibration'])
    selected,status = choose(candidates,complete)
    result = {'schema_version':1,'status':status,'config':config,'source_model_sha256':model_hash,
        'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        'recordings_read':chosen,'complete_calibration':complete,'by_recording':by_recording,
        'reference':'original NDD longitudinal PDF evaluated with measured net-gap geometry',
        'blend_scope':'conditional longitudinal PDF only; lateral probabilities unchanged',
        'selection_semantics':'First minimize zero-probability observations in both overall and tail groups; '
                'then require all-observation Brier no worse than the hard-fallback baseline and minimize '
                'overall Brier. Positive-support NLL is diagnostic only because candidates with different '
                'support are not comparable on that conditional subset.',
        'candidates':candidates,'selected':selected,
        'scope':'Calibration-only screening. NLL excludes explicitly reported zero-support observations; '
                'Brier includes every observation. Frames are temporally correlated. No validation/test reads, '
                'runtime enablement, safety acceptance, or pure-highD claim.'}
    (output/'continuous_fallback_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source_root','manifest','model','config','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--recordings',nargs='+')
    args=parser.parse_args()
    result=calibrate(args.source_root,args.manifest,args.model,
        json.loads(Path(args.config).read_text(encoding='utf-8')),args.output,args.recordings)
    print(json.dumps({'status':result['status'],'selected':result['selected']}))


if __name__=='__main__':
    main()
