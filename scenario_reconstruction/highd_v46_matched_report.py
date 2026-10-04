"""Paired episode bootstrap and three-layer frozen-NDE fidelity report."""
import argparse
import csv
import json
from pathlib import Path
import platform
import numpy as np
from threadpoolctl import threadpool_limits

from .highd_v46_assets import ROOT, read, save, digest
from .highd_v46_matched_score import load_pair


CORE = [('speed_h', 'speed', 'hellinger'), ('speed_mae', 'speed', 'mae'),
        ('gap_h', 'gap', 'hellinger'), ('gap_mae', 'gap', 'mae'),
        ('rr_h', 'rr', 'hellinger'), ('joint_state_h', 'joint_state', 'hellinger')]
INTERACTION = [('joint_accel_h', 'joint_accel', 'hellinger'),
               ('relative_accel_h', 'relative_accel', 'hellinger'),
               ('accel_rho_error', 'accel_moments', 'rho_error')]
PERSISTENT = [('persistent_gap_h', 'persistent_gap', 'hellinger'),
              ('persistent_gap_mae', 'persistent_gap', 'mae'),
              ('persistent_rr_h', 'persistent_rr', 'hellinger'),
              ('persistent_rr_mae', 'persistent_rr', 'mae'),
              ('persistent_joint_state_h', 'persistent_joint_state', 'hellinger')]


def rho(moments):
    m = np.atleast_2d(np.asarray(moments, dtype=float))
    n, sx, sy, sxx, syy, sxy = m.T
    numerator = n*sxy-sx*sy
    denominator = np.sqrt(np.maximum(n*sxx-sx*sx, 0)*np.maximum(n*syy-sy*sy, 0))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def metric(counts, reference, kind):
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    if kind == 'rho_error':
        return np.abs(rho(counts)-rho(reference)[0])
    total = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, total, out=np.full_like(counts, np.nan), where=total > 0)
    q = np.asarray(reference, dtype=float).ravel(); q = q/q.sum()
    if kind == 'hellinger':
        return np.sqrt(((np.sqrt(p)-np.sqrt(q))**2).sum(axis=1)/2)
    if kind == 'mae':
        return np.abs(p-q).mean(axis=1)
    raise ValueError(kind)


def bootstrap_weights(n, repetitions, seed):
    indices = np.random.default_rng(seed).integers(n, size=(repetitions, n))
    weights = np.zeros((repetitions, n), dtype=float)
    np.add.at(weights, (np.arange(repetitions)[:, None], indices), 1)
    return weights


def interval(values):
    values = np.asarray(values); values = values[np.isfinite(values)]
    return np.quantile(values, [.025, .975]).tolist() if len(values) else None


def compare(single, dual, reference, kind, weights, pooled_bootstrap=True):
    single = np.asarray(single, dtype=float).reshape(len(single), -1)
    dual = np.asarray(dual, dtype=float).reshape(len(dual), -1)
    ref = np.asarray(reference).ravel()
    a = metric(single.sum(axis=0), ref, kind)[0]; b = metric(dual.sum(axis=0), ref, kind)[0]
    delta = metric(dual, ref, kind)-metric(single, ref, kind)
    valid = np.isfinite(delta)
    denominator = weights[:, valid].sum(axis=1)
    sampled_mean = np.divide(weights[:, valid] @ delta[valid], denominator,
                             out=np.full(len(weights), np.nan), where=denominator > 0)
    result = {'single': float(a), 'dual': float(b), 'pooled_delta_dual_minus_single': float(b-a),
              'relative_change_percent': float(100*(b/a-1)) if a > 0 else None,
              'mean_seed_delta': float(delta[valid].mean()) if valid.any() else None,
              'mean_seed_delta_ci95': interval(sampled_mean), 'paired_episodes': int(valid.sum()),
              'single_samples': int(single[:, 0].sum()) if kind == 'rho_error' else int(single.sum()),
              'dual_samples': int(dual[:, 0].sum()) if kind == 'rho_error' else int(dual.sum())}
    if pooled_bootstrap:
        boot_delta = []
        for left in range(0, len(weights), 100):
            w = weights[left:left+100]
            boot_delta.extend(metric(w @ dual, ref, kind)-metric(w @ single, ref, kind))
        result['pooled_delta_ci95'] = interval(boot_delta)
    return result


def tables(pairs, reference, config, collection='primary', definitions=CORE, bootstrap=True):
    if not pairs:
        return {}
    weights = bootstrap_weights(len(pairs), config['bootstrap']['replicates'], config['bootstrap']['seed'])
    return {name: compare([p['single'][collection][key] for p in pairs],
                          [p['dual'][collection][key] for p in pairs], reference[key], kind, weights, bootstrap)
            for name, key, kind in definitions}


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_report(root, primary, timeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (metric_name, label) in zip(axes.flat, [('speed_h', 'Speed H'), ('gap_h', 'Gap H'),
                                                 ('rr_h', 'Relative-speed H'), ('joint_state_h', 'Joint (gap, relative speed) H')]):
        for arm, color in [('single', '#3266ad'), ('dual', '#d36926')]:
            ax.plot([r['time_s'] for r in timeline], [r['metrics'][metric_name][arm] for r in timeline],
                    marker='o', label=arm, color=color)
        ax.set(xlabel='Time (s)', ylabel=label); ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle(f"Frozen V46: fixed cohort completing 60 s in both arms (n={timeline[0]['common_reached_pairs']})")
    fig.savefig(root/'rollout_drift.png', dpi=170); fig.savefig(root/'rollout_drift.pdf'); plt.close(fig)
    names = ['speed_h', 'gap_h', 'rr_h', 'joint_state_h', 'joint_accel_h', 'relative_accel_h']
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True); x = np.arange(len(names))
    ax.bar(x-.18, [primary[n]['single'] for n in names], .36, label='single', color='#3266ad')
    ax.bar(x+.18, [primary[n]['dual'] for n in names], .36, label='dual', color='#d36926')
    ax.set_xticks(x, ['Speed', 'Gap', 'Relative\nspeed', 'Joint\nstate', 'Joint\nacceleration', 'Relative\nacceleration'])
    ax.set(ylabel='Hellinger distance (lower is better)', title='All 100 seed pairs: common pre-terminal exposure')
    ax.legend(); ax.grid(axis='y', alpha=.25)
    fig.savefig(root/'pooled_fidelity.png', dpi=170); fig.savefig(root/'pooled_fidelity.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/highd_v46_matched_fidelity_v1.json'))
    parser.add_argument('--rollout', required=True); parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); cfg = read(args.config); run = Path(args.rollout); root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    protocol = read(run/'protocol.json'); ref_summary = read(Path(args.reference)/'reference_summary.json')
    if protocol['config_sha256'] != digest(args.config) or ref_summary['config_sha256'] != digest(args.config):
        raise ValueError('Protocol changed after collection/reference extraction')
    if read(run/'completion.json')['matched_seed_pairs'] != cfg['episodes']:
        raise ValueError('Incomplete declared cohort')
    ref_path = Path(args.reference)/'reference.npz'
    if digest(ref_path) != ref_summary['reference_sha256']:
        raise ValueError('Reference changed')
    with np.load(ref_path) as source:
        reference = {key: source[key] for key in source.files}
    pairs = []
    for index, seed in enumerate(range(cfg['seed_start'], cfg['seed_start']+cfg['episodes']), 1):
        pair = load_pair(run, seed, cfg); pairs.append(pair)
        if index % 10 == 0:
            print({'scored_seed_pairs': index}, flush=True)
    definitions = CORE+INTERACTION+PERSISTENT
    primary = tables(pairs, reference, cfg, definitions=definitions)
    available = tables(pairs, reference, cfg, collection='available')
    complete = [p for p in pairs if all(p[a]['report']['complete'] for a in ('single', 'dual'))]
    complete_table = tables(complete, reference, cfg)
    timelines = {}
    for cohort_name, cohort in [('common_at_time', pairs), ('fixed_60s_complete_cohort', complete)]:
        series = []
        for point in cfg['diagnostic_times_s']:
            local = [p for p in cohort if all(str(point) in p[a]['by_time'] for a in ('single', 'dual'))]
            temporal = [{a: {'snapshot': p[a]['by_time'][str(point)]} for a in ('single', 'dual')} for p in local]
            series.append({'time_s': point, 'common_reached_pairs': len(local),
                'single_reached': sum(str(point) in p['single']['by_time'] for p in pairs),
                'dual_reached': sum(str(point) in p['dual']['by_time'] for p in pairs),
                'metrics': tables(temporal, reference, cfg, collection='snapshot', bootstrap=False)})
        timelines[cohort_name] = series
    rho_data = {'highd': float(rho(reference['accel_moments'])[0])}
    for arm in ('single', 'dual'):
        rho_data[arm] = float(rho(sum(p[arm]['primary']['accel_moments'] for p in pairs))[0])
    robustness = {}
    for arm in ('single', 'dual'):
        rows = [p[arm]['report'] for p in pairs]; safety = [p[arm]['safety'] for p in pairs]
        finite_ttc = [r['minimum_precontact_ttc_s'] for r in safety if r['minimum_precontact_ttc_s'] is not None]
        robustness[arm] = {'episodes': len(rows), 'completed_60s': sum(r['complete'] for r in rows),
            'bv_collision_episodes': sum(r['bv_collision_episode'] for r in rows),
            'cav_collision_episodes': None,
            'near_collision_episodes': sum(r['near_collision_episode'] for r in safety),
            'minimum_precontact_ttc_s': min(finite_ttc) if finite_ttc else None,
            'median_episode_minimum_precontact_ttc_s': float(np.median(finite_ttc)) if finite_ttc else None,
            'initial_overlap_pairs': sum(r['initial_overlap_pairs'] for r in safety),
            'overlap_episodes': sum(r['observed_overlap_pair_ticks'] > 0 for r in safety),
            'max_first_contact_longitudinal_penetration_m': max(r['maximum_contact_longitudinal_overlap_m'] for r in safety),
            'max_first_contact_lateral_penetration_m': max(r['maximum_contact_lateral_overlap_m'] for r in safety),
            'observed_vehicle_seconds': sum(r['actor_count']*r['elapsed_s'] for r in rows),
            'runtime_s_total': sum(r['runtime_s'] for r in rows),
            'median_runtime_per_simulated_vehicle_second': float(np.median([r['runtime_s']/(r['elapsed_s']*r['actor_count']) for r in rows])),
            'lane_change_onsets': sum(r['diagnostics'].get('lane_change_onsets', 0) for r in rows),
            'pair_decisions': sum(r['pair_decisions'] for r in rows),
            'correlated_pair_decisions': sum(r['correlated_pair_decisions'] for r in rows),
            'maximum_simultaneous_pairs': max(r['maximum_simultaneous_pairs'] for r in rows),
            'maximum_marginal_error': max(r['maximum_marginal_error'] for r in rows)}
    weights = bootstrap_weights(len(pairs), cfg['bootstrap']['replicates'], cfg['bootstrap']['seed'])
    collision_delta = np.array([int(p['dual']['report']['bv_collision_episode'])-int(p['single']['report']['bv_collision_episode']) for p in pairs])
    robustness['paired_collision_probability_delta'] = {'estimate': float(collision_delta.mean()),
        'ci95': interval(weights @ collision_delta/len(pairs)),
        'single_only_collision_seeds': [p['single']['report']['seed'] for p in pairs if p['single']['report']['bv_collision_episode'] and not p['dual']['report']['bv_collision_episode']],
        'dual_only_collision_seeds': [p['dual']['report']['seed'] for p in pairs if p['dual']['report']['bv_collision_episode'] and not p['single']['report']['bv_collision_episode']]}
    near_delta = np.array([int(p['dual']['safety']['near_collision_episode'])-int(p['single']['safety']['near_collision_episode']) for p in pairs])
    robustness['paired_near_collision_probability_delta'] = {
        'estimate': float(near_delta.mean()), 'ci95': interval(weights @ near_delta/len(pairs))}
    strict_source = sum(r['persistent_60s_spells'] for r in ref_summary['recordings'].values())
    persistence = {'strict_60s_highd_spells': strict_source,
        'strict_60s_single_pairs': sum(p['single']['persistent_60s_pairs'] for p in pairs),
        'strict_60s_dual_pairs': sum(p['dual']['persistent_60s_pairs'] for p in pairs),
        'strict_60s_common_pairs': sum(p['matched']['common_persistent_60s_pairs'] for p in pairs),
        'strict_60s_fidelity': None if strict_source == 0 else 'not_scored',
        'strict_60s_reason': 'No 60 s observed highD tracks in the fixed source; no synthetic ground truth substitution.',
        'alternative_window_s': cfg['persistent_alternative_window_s'],
        'highd_5s_windows': sum(r['persistent_5s_windows'] for r in ref_summary['recordings'].values()),
        'common_simulation_5s_windows': sum(p['matched']['common_persistent_5s_windows'] for p in pairs),
        'quiet_rule': 'Both highD actors have abs(yVelocity)<0.2 m/s throughout; both simulated actors have no active lane change throughout. Same pair/lane identity and supported gap/speeds at every physical/source frame, including endpoint.',
        'scope': 'Outcome-conditioned common-pair subset; histograms and probability MAE, not ground-truth trajectory MAE.'}
    summary = {'protocol': cfg, 'protocol_sha256': digest(args.config), 'collector_sha256': protocol['collector_sha256'],
        'scorer_sha256': digest(Path(__file__).with_name('highd_v46_matched_score.py')), 'reporter_sha256': digest(__file__),
        'reference_sha256': ref_summary['reference_sha256'], 'reference_exporter_sha256': ref_summary['exporter_sha256'],
        'reference_recording_summary': ref_summary['recordings'],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'primary_common_exposure': primary, 'all_available_exposure': available, 'common_60s_complete': complete_table,
        'common_60s_complete_pairs': len(complete), 'total_common_simulation_seconds': sum(p['matched']['common_duration_s'] for p in pairs),
        'time_resolved': timelines, 'acceleration_correlation': rho_data, 'persistent': persistence, 'robustness': robustness,
        'limitations': ['Single is the frozen highD method-aligned independent adaptation, not exact paper reproduction.',
            'Both arms retain V34 explicit LC feasibility restrictions, same empirical initialization and direct dynamics.',
            'Reference is previously used train data; bootstrap covers Monte Carlo seeds conditional on this reference, not highD uncertainty.',
            'Primary exposure is paired to the earlier collision. Late-time/fixed-completion/persistent cohorts are conditioned on survival.',
            'No 60 s source persistent pairs or matched future real trajectories; no trajectory MAE is invented.',
            'Collision contacts have positive rectangle overlap; these are reported contacts, not an independent claim of a nonphysical artifact.',
            'No inference about D2RL efficiency or numerical comparisons against published paper scores.']}
    save(root/'summary.json', summary)
    csv_rows = []
    for name, result in primary.items():
        row = {'metric': name, **result}
        row = {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}; csv_rows.append(row)
    write_csv(root/'metric_table.csv', csv_rows)
    episode_rows = []
    for p in pairs:
        for arm in ('single', 'dual'):
            r = p[arm]['report']; row = {'seed': r['seed'], 'arm': arm, 'elapsed_s': r['elapsed_s'],
                'completed': r['complete'], 'collision': r['bv_collision_episode'],
                'common_duration_s': p['matched']['common_duration_s'], **p[arm]['safety']}
            for name, key, kind in definitions:
                row[name] = metric(p[arm]['primary'][key].reshape(1, -1), reference[key].ravel(), kind)[0]
            episode_rows.append(row)
    write_csv(root/'episode_metrics.csv', episode_rows)
    for name, series in timelines.items():
        rows = []
        for t in series:
            for m, r in t['metrics'].items():
                rows.append({'time_s': t['time_s'], 'common_reached_pairs': t['common_reached_pairs'], 'metric': m,
                             'single': r['single'], 'dual': r['dual'], 'delta': r['pooled_delta_dual_minus_single'],
                             'mean_seed_delta_ci95': json.dumps(r['mean_seed_delta_ci95'])})
        write_csv(root/f'time_{name}.csv', rows)
    plot_report(root, primary, timelines['fixed_60s_complete_cohort'])
    print(json.dumps({'primary': primary, 'robustness': robustness, 'persistent': persistence}, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
