"""Post-hoc user-margin paired NI and TOST on existing frozen NDE rollouts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .highd_v46_assets import ROOT, read, save, digest
from .highd_v46_matched_report import metric
from .highd_v46_matched_score import state_counts


def paired_test(delta, margin=.01, alpha=.05):
    delta = np.asarray(delta, float)
    if len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError('Need finite independent-seed paired differences')
    n = len(delta); mean = float(delta.mean()); se = float(delta.std(ddof=1)/np.sqrt(n))
    if se == 0:
        p_upper = float(mean >= margin); p_lower = float(mean <= -margin)
    else:
        p_upper = float(t.cdf((mean-margin)/se, n-1))
        p_lower = float(t.sf((mean+margin)/se, n-1))
    width90 = float(t.ppf(1-alpha, n-1)*se)
    width95 = float(t.ppf(1-alpha/2, n-1)*se)
    return {'n': n, 'mean_delta': mean, 'standard_error': se, 'margin': margin,
        'one_sided_95_upper': mean+width90, 'ci90': [mean-width90, mean+width90],
        'ci95': [mean-width95, mean+width95],
        'bonferroni_two_endpoint_simultaneous_95_upper': mean+width95,
        'ni_p': p_upper, 'noninferior_at_alpha_0_05': p_upper < alpha,
        'tost_lower_p': p_lower, 'tost_upper_p': p_upper, 'tost_p': max(p_lower, p_upper),
        'equivalent_at_alpha_0_05': max(p_lower, p_upper) < alpha}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/highd_v46_ni_temporal_v1.json'))
    p.add_argument('--rollout', required=True); p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(); cfg = read(args.config); settings = cfg['noninferiority']
    path = ROOT/'results/highd_v46_matched100_episode_metrics.csv'
    frame = pd.read_csv(path); wide = frame.pivot(index='seed', columns='arm')
    if list(wide.index) != list(range(40000,40100)):
        raise ValueError('Paired seed cohort differs from the frozen comparison')
    primary = {}; bootstrap_rng = np.random.default_rng(settings['bootstrap_seed'])
    indices = bootstrap_rng.integers(100, size=(settings['bootstrap_replicates'],100))
    for key in settings['co_primary']:
        delta = (wide[key]['dual']-wide[key]['single']).to_numpy()
        result = paired_test(delta, settings['margin'], settings['alpha'])
        boot = delta[indices].mean(axis=1)
        result['bootstrap_one_sided_95_upper'] = float(np.quantile(boot,.95))
        result['bootstrap_ci90'] = np.quantile(boot,[.05,.95]).tolist()
        result['bootstrap_ci95'] = np.quantile(boot,[.025,.975]).tolist()
        result['margin_sensitivity'] = [paired_test(delta,m) for m in settings['secondary_margins']]
        primary[key] = result
    # Prevent a pooled-exposure claim from silently extending to the endpoint.
    base_cfg = read(ROOT/'configs/highd_v46_matched_fidelity_v1.json')
    with np.load(Path(args.reference)/'reference.npz') as source:
        refs = {k: source[k] for k in ('speed','gap')}
    differences = {k: [] for k in settings['co_primary']}; complete_seeds = []
    root = Path(args.rollout)
    for seed in range(40000,40100):
        pair = read(root/f'pair_seed{seed}.json')
        if not all(pair[a]['complete'] for a in ('single','dual')):
            continue
        complete_seeds.append(seed); scores = {}
        for arm in ('single','dual'):
            trace = root/f'{arm}_seed{seed}.npz'
            if digest(trace) != pair[arm]['trace_sha256']:
                raise ValueError('Modified trace')
            with np.load(trace) as source:
                counts = state_counts(source['states'][-1:], base_cfg['bins'])
            scores[arm] = {k: metric(counts[k[:-2]].reshape(1,-1),refs[k[:-2]],'hellinger')[0]
                           for k in settings['co_primary']}
        for key in differences:
            differences[key].append(scores['dual'][key]-scores['single'][key])
    secondary = {k: paired_test(v,settings['margin']) for k,v in differences.items()}
    report = {'protocol': cfg, 'protocol_sha256': digest(args.config), 'implementation_sha256': digest(__file__),
        'episode_metrics_sha256': digest(path), 'primary_common_exposure': primary,
        'both_primary_noninferior': all(r['noninferior_at_alpha_0_05'] for r in primary.values()),
        'both_primary_equivalent': all(r['equivalent_at_alpha_0_05'] for r in primary.values()),
        'secondary_60s_common_survivors': secondary, 'secondary_complete_seed_count':len(complete_seeds),
        'scope': 'Fixed empirical train reference; user margin selected after original results; exploratory post-hoc NI/equivalence of mean episode distances, not all-time fidelity, safety or D2RL efficiency.'}
    save(args.output,report)
    print({'primary':primary,'secondary_60s':secondary},flush=True)


if __name__ == '__main__':main()
