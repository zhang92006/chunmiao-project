"""Selected-case horizon regression, NOT an extended highD reference evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .highd_ndd_shadow import HighDShadowNDD
from .highd_naturalistic_rollout import run_naturalistic


def extended_template(template, duration):
    if not np.isfinite(duration) or duration <= template['duration']:
        raise ValueError('Diagnostic horizon must exceed the original duration')
    result = copy.deepcopy(template)
    result['duration'] = float(duration)
    result.setdefault('bridge_metadata', {})['horizon_diagnostic'] = {
        'original_duration_s': template['duration'],
        'extended_duration_s': float(duration),
        'scope': 'Same initial traffic, no new support vehicles. Beyond original horizon '
                 'is simulation-only extrapolation, not measured reference validation.',
    }
    return result


def check_prefix(original, extended):
    """Exact replay of existing logged history is required before interpreting extension."""
    result = {}
    for kind in ('decisions', 'snapshots'):
        old = original[kind]
        # Include all actors, so unexpected extra records also fail the check.
        cutoff = max((r['time'] for r in old), default=-float('inf'))
        new = [r for r in extended[kind] if r['time'] <= cutoff]
        result[kind + '_identical'] = old == new
    result['passed'] = all(result.values())
    return result


def diagnose(manifest_path, source_root, output, indices, duration, model):
    if not indices or len(set(indices)) != len(indices) or any(i < 0 for i in indices):
        raise ValueError('Episode indices must be nonnegative, unique and nonempty')
    source_root, output = Path(source_root), Path(output)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source_summary = json.loads((source_root / 'naturalistic_summary.json').read_text())
    if source_summary['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Source manifest changed')
    templates = {}
    for rec in manifest['records']:
        path = Path(rec['template_path'])
        value = json.loads(path.read_text(encoding='utf-8'))
        templates[value['template_id']] = (path, value)
    cases = []
    for index in indices:
        source_path = source_root / f'episode_{index:04d}' / 'naturalistic_episode.json'
        ep = json.loads(source_path.read_text(encoding='utf-8'))
        audit = json.loads((source_path.parent / 'naturalistic_audit.json').read_text())
        if not audit['probability_audit_passed'] or not audit['model_reconstruction_checked']:
            raise ValueError('Source audit failed')
        for key in ('longitudinal_model_sha256', 'context_model_sha256', 'context_config_sha256'):
            if ep['metadata'][key] != model.metadata[key]:
                raise ValueError('Source model changed: ' + key)
        path, template = templates[ep['metadata']['template_id']]
        if hashlib.sha256(path.read_bytes()).hexdigest() != ep['metadata']['template_sha256']:
            raise ValueError('Source template changed')
        cases.append((index, source_path, ep, extended_template(template, duration)))
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, source_path, original, template in cases:
        template_path = output / f'template_{index:04d}.json'
        template_path.write_text(json.dumps(template, indent=2), encoding='utf-8')
        run_output = output / f'episode_{index:04d}'
        summary = run_naturalistic(template_path, model, run_output,
                                  seed=original['metadata']['seed'],
                                  guard_config=original['metadata'].get('lane_change_guard_config'))
        extended = json.loads((run_output / 'naturalistic_episode.json').read_text())
        prefix = check_prefix(original, extended)
        core = set(original['metadata']['evaluation_actor_ids'])
        rows = [r for r in extended['snapshots'] if r['vehicle_id'] in core]
        low = [r for r in rows if r['ttc_s'] is not None and 0 <= r['ttc_s'] < 1]
        affected = {r['vehicle_id'] for r in low}
        result = {
            'source_episode_index': index, 'source_episode_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'template_id': original['metadata']['template_id'], 'seed': original['metadata']['seed'],
            'original_duration_s': template['bridge_metadata']['horizon_diagnostic']['original_duration_s'],
            'requested_duration_s': duration, 'last_snapshot_time_s': max(r['time'] for r in rows),
            'termination': extended['termination'], 'prefix_check': prefix,
            'probability_audit_passed': summary['probability_audit_passed'],
            'minimum_positive_core_ttc_s': min((r['ttc_s'] for r in low), default=None),
            'low_ttc_timeline': low,
            'affected_actor_decisions': [r for r in extended['decisions'] if r['vehicle_id'] in affected],
        }
        results.append(result)
        report = {'schema_version': 1, 'model': model.metadata,
                  'manifest_sha256': source_summary['manifest_sha256'],
                  'requested_episode_indices': indices, 'requested_duration_s': duration,
                  'scope': 'Selected failure regression; no population risk '
                  'or extended real-trajectory fidelity claim. Support traffic fixed at original initialization.',
                  'results': results}
        (output / 'horizon_diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k not in ('low_ttc_timeline', 'affected_actor_decisions')}), flush=True)
        if not prefix['passed'] or not summary['probability_audit_passed']:
            raise RuntimeError('Prefix or probability audit failed; do not interpret extension')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--longitudinal_model', required=True)
    parser.add_argument('--context_model', required=True)
    parser.add_argument('--context_config', required=True)
    args = parser.parse_args()
    model = HighDShadowNDD(args.longitudinal_model, args.context_model, args.context_config)
    diagnose(args.manifest, args.source_root, args.output, args.episodes, args.duration, model)


if __name__ == '__main__':
    main()
