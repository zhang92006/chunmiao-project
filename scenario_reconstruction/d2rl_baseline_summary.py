"""Summarize complete published-agent runs without dropping failed workers."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np


def summarize(root, expected=None):
    root=Path(root)
    plan_path=root/'batch_plan.json'
    if plan_path.exists():
        planned=json.loads(plan_path.read_text())['total_episodes']
        if expected is not None and expected!=planned:
            raise ValueError('Requested count differs from recorded batch plan')
        expected=planned
    weights=[];reports=[]
    for path in sorted(root.glob('weight*.npy')):
        worker_id=int(path.stem[len('weight'):])
        report=json.loads((root/f'worker{worker_id}.json').read_text(encoding='utf-8'))
        w=np.load(path,allow_pickle=False)
        if report['errors'] or report['completed']!=report['requested'] or len(w)!=report['completed']:
            raise ValueError('Incomplete/failed worker: '+str(worker_id))
        if w.ndim!=1 or not np.isfinite(w).all() or np.any(w<0):
            raise ValueError('Invalid weights')
        weights.append(w);reports.append(report)
    if not weights:
        raise ValueError('No completed workers')
    w=np.concatenate(weights)
    if expected is not None and len(w)!=expected:
        raise ValueError(f'Expected {expected} tests, received {len(w)}')
    for key in ('model_sha256','upstream_commit','mode','av_model','lanes','seed_base','probability_backend'):
        if any(r[key]!=reports[0][key] for r in reports):
            raise ValueError('Workers used different '+key)
    mean=float(w.mean());variance=float(w.var(ddof=1)) if len(w)>1 else None
    half=NormalDist().inv_cdf(.95)*np.sqrt(variance/len(w)) if variance is not None else None
    crashes=sum(r['collision_episodes'] for r in reports)
    return {'episodes':len(w),'workers':len(reports),'crash_episodes':crashes,
            'crash_episode_share':crashes/len(w),'positive_weight_crash_episodes':int(np.count_nonzero(w)),
            'crash_rate_estimate':mean,
            'relative_half_width_90':float(half/mean) if mean>0 and half is not None else None,
            'normal_ci90':[float(mean-half),float(mean+half)] if half is not None and mean>0 else None,
            'variance_of_weighted_indicator':variance,'maximum_collision_weight':float(w.max()),
            'worker_wall_seconds':[r['wall_seconds'] for r in reports],
            'max_worker_rss_MiB':max(r['max_rss_bytes'] for r in reports)/2**20,
            'max_worker_private_MiB':max(r['max_private_bytes'] for r in reports)/2**20,
            'policy_calls':sum(r['policy_calls'] for r in reports),
            'model_sha256':reports[0]['model_sha256'],
            'scope':'Importance-weighted CAV collision probability per 400m test; raw collision share is not the natural crash probability. No event implies RHW unavailable, not zero.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--expected',type=int)
    a=parser.parse_args()
    print(json.dumps(summarize(a.experiment,a.expected),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
