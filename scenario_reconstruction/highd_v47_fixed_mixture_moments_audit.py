"""Independent read-only audit of fixed-component continuation measurement."""
import argparse
import json
from pathlib import Path

import numpy as np

from .highd_v46_assets import digest, read
from .highd_v47_continuation_moments_audit import reconstruct_path
from .highd_v47_fixed_mixture_moments import inputs, ARMS


def reconstruct(strata, rows, arm, error):
    """Recompute without calling the collector's estimator/statistics functions."""
    j = v = lo = hi = removed = inferred = 0.; rb = np.zeros(2)
    def moments(z, bound):
        z = np.asarray(z, float); n = len(z)
        assert n >= 2 and np.isfinite(z).all() and np.all((z >= 0) & (z <= bound+1e-12))
        mean = float(z.sum()/n); s2 = float(((z-mean)**2).sum()/(n-1))
        radius = np.sqrt(2*s2*np.log(4/error)/n)+7*bound*np.log(4/error)/(3*(n-1))
        return mean, s2, np.array([max(0., mean-radius), min(bound, mean+radius)])
    for h in strata:
        g = [r for r in rows if r['stratum_id'] == h['stratum_id']]; n = h['draws']
        assert len(g) == n and {r['within_stratum_draw'] for r in g} == set(range(n))
        if arm == 'fixed_defensive':
            counts = [sum(r['fixed_tilted_component'] == t for r in g) for t in (False, True)]
            assert counts == [n//2, n//2] and n % 2 == 0
            e = [moments([r[arm]['weighted_event'] for r in g if r['fixed_tilted_component'] == t], 2.)
                 for t in (False, True)]
            mean = (e[0][0]+e[1][0])/2
            var = (e[0][1]+e[1][1])/(2*n)
            bounds = np.minimum(1., (e[0][2]+e[1][2])/2)
            term = ((e[0][0]-e[1][0])**2-(e[0][1]+e[1][1])/(n/2))/(4*n)
            delta = [e[0][2][0]-e[1][2][1], e[0][2][1]-e[1][2][0]]
            squared = [0. if delta[0] <= 0 <= delta[1] else min(x*x for x in delta), max(x*x for x in delta)]
            removed += h['coefficient_mass']**2*term
            inferred += h['coefficient_mass']**2*(var+term)
            rb += h['coefficient_mass']**2*np.array(squared)/(4*n)
        else:
            values = [float(r[arm]['event_result']) if arm == 'natural' else r[arm]['weighted_event'] for r in g]
            mean, sample_var, bounds = moments(values, 1. if arm == 'natural' else 2.)
            var = sample_var/n; bounds = np.minimum(1., bounds)
        c = h['coefficient_mass']; j += c*mean; v += c*c*var; lo += c*bounds[0]; hi += c*bounds[1]
    return dict(J=float(j), variance=float(v), bounds=[lo, hi], removed=float(removed),
                inferred=float(inferred), removed_bounds=rb.tolist())


def audit(config, directory, replay=False):
    cfg, physical, assets, plans, ads, design, lock = inputs(config)
    root = Path(directory); summary = read(root/'summary.json')
    assert summary['complete'] and summary['config'] == cfg
    assert all(summary['lock'].get(k) == v for k, v in lock.items())
    assert read(root/'measurement_plan.json') == design
    for name, sha in summary['artifact_hashes'].items():
        assert Path(name).name == name and digest(root/name) == sha
    family = 8*sum(len(s['strata']) for s in design['states']); error = (1-cfg['confidence'])/family
    estimates = {(e['batch'], e['state_id'], e['arm']): e for e in summary['state_batch_estimates']}
    measurements = {}; seen = set(); paths = 0; max_error = 0.; weights = []; replays = 0
    paired = {(x['batch'], x['state_id']): x for x in summary['paired_differences']}
    if replay:
        from . import highd_v47_fixed_mixture_moments as native
        native.parent._init_worker(str(assets))
    for schedule in design['schedules']:
        b, i = schedule['batch'], schedule['state_id']; hs = design['states'][i]['strata']
        by_stratum = {h['stratum_id']: h for h in hs}
        stored = read(root/f'batch_{b}_state_{i:03d}.json')
        assert len(stored['rows']) == len(schedule['rows']) == design['states'][i]['draws']
        for plan, row in zip(schedule['rows'], stored['rows']):
            assert all(row[k] == v for k, v in plan.items())
            assert row['native_seed'] not in seen; seen.add(row['native_seed'])
            assert any(row['action_indices'] == list(np.unravel_index(cell, np.array(ads[i]['P']).shape))
                       for cell in by_stratum[row['stratum_id']]['cells'])
            for arm in ARMS:
                max_error = max(max_error, reconstruct_path(row[arm]))
            for arm, key in [('iid_defensive', 'iid_tilted_component'), ('fixed_defensive', 'fixed_tilted_component')]:
                np.testing.assert_allclose(row[arm]['initial_cancelled_log_conditional'],
                                           row['natural']['initial_cancelled_log_conditional'], atol=1e-12)
                if not row[key] or not row['natural']['steps']:
                    assert row[arm] == row['natural']
            if row['iid_tilted_component'] == row['fixed_tilted_component']:
                assert row['iid_defensive'] == row['fixed_defensive']
            separate = bool((row['iid_tilted_component'] or row['fixed_tilted_component']) and row['natural']['steps'])
            assert row['simulated_tilted_path'] == separate
            paths += 1+int(separate); weights.append(row['fixed_defensive']['future_weight'])
        for arm in ARMS:
            result = reconstruct(hs, stored['rows'], arm, error); measurements[(b, i, arm)] = result
            e = estimates[(b, i, arm)]
            np.testing.assert_allclose([result['J'], result['variance']], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose(result['bounds'], e['bank_simultaneous_interval'], rtol=1e-12, atol=1e-30)
            if arm == 'fixed_defensive':
                np.testing.assert_allclose([result['removed'], result['inferred']],
                    [e['removed_between_component_variance_unbiased'], e['iid_variance_inferred_same_draws_unbiased']], rtol=1e-10, atol=1e-25)
                np.testing.assert_allclose(result['removed_bounds'], e['removed_between_component_variance_interval'], rtol=1e-12)
        diff = dv = 0.
        for h in hs:
            groups = [[r for r in stored['rows'] if r['stratum_id'] == h['stratum_id'] and r['fixed_tilted_component'] == t]
                      for t in (False, True)]
            d = [np.array([r['fixed_defensive']['weighted_event']-r['iid_defensive']['weighted_event'] for r in g]) for g in groups]
            diff += h['coefficient_mass']*(d[0].mean()+d[1].mean())/2
            dv += h['coefficient_mass']**2*sum(x.var(ddof=1)/len(x) for x in d)/4
        np.testing.assert_allclose([diff, dv], [paired[(b, i)]['fixed_minus_iid_J'], paired[(b, i)]['difference_variance']], rtol=1e-12, atol=1e-30)
        if replay:
            index = (17*i+31*b) % len(stored['rows'])
            task = (dict(batch=b, state_id=i, rows=[schedule['rows'][index]]), ads[i]['state'], plans[i], str(assets), physical, cfg)
            regenerated = native._collect(task)['rows'][0]
            assert json.loads(json.dumps(regenerated)) == stored['rows'][index]; replays += 1
    alpha = np.array(design['target_masses']); info = {}
    for arm in ARMS:
        js = []; vs = []
        for i in range(len(plans)):
            a, b = measurements[(0, i, arm)], measurements[(1, i, arm)]
            j = (a['J']+b['J'])/2; v = (a['variance']+b['variance'])/4
            bounds = (np.array(a['bounds'])+b['bounds'])/2
            e = summary['arms'][arm]['states'][i]
            np.testing.assert_allclose([j, v], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose(bounds, e['bank_simultaneous_interval'], rtol=1e-12, atol=1e-30)
            assert e['precision_supported'] == bool(bounds[0]>0 and (bounds[1]-bounds[0])/2 <= .5*j)
            js.append(j); vs.append(v)
        total = float(alpha@js); total_v = float((alpha*alpha)@vs)
        np.testing.assert_allclose([total, total_v], [summary['arms'][arm]['target_M2'], summary['arms'][arm]['target_measurement_variance']], rtol=1e-12)
        info[arm] = dict(target_M2=total, variance=total_v,
                         precise_states=sum(e['precision_supported'] for e in summary['arms'][arm]['states']))
    removed = float(sum(.25*alpha[i]**2*measurements[(b, i, 'fixed_defensive')]['removed']
                        for b in range(2) for i in range(len(alpha))))
    rb = [sum(.25*alpha[i]**2*measurements[(b, i, 'fixed_defensive')]['removed_bounds'][j]
              for b in range(2) for i in range(len(alpha))) for j in (0, 1)]
    contrast = summary['same_draw_structural_comparison']
    np.testing.assert_allclose(removed, contrast['removed_between_component_variance_unbiased'], rtol=1e-10)
    np.testing.assert_allclose(rb, contrast['removed_between_component_variance_interval'], rtol=1e-12)
    np.testing.assert_allclose(info['fixed_defensive']['variance']+removed, contrast['iid_variance_inferred_unbiased'], rtol=1e-12)
    assert len(seen)*3 == summary['logical_draws'] == design['logical_draws']
    assert paths == summary['native_paths_simulated'] <= design['planned_native_paths_upper_bound']
    return dict(verified=True, summary_sha256=digest(root/'summary.json'),
                artifact_hashes_verified=len(summary['artifact_hashes']), native_paths_simulated=paths,
                logical_draws=summary['logical_draws'], max_full_factor_log_error=max_error,
                fixed_future_weight_range=[min(weights), max(weights)], native_root_replays_matched=replays,
                arms=info, same_draw_structural_comparison=contrast,
                scope='Read-only reconstruction; replayed CRN paths are not extra data.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v47_fixed_mixture_moments_v1.json')
    parser.add_argument('--run', default='outputs/v47_fixed_mixture_moments_v1')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args(); print(audit(args.config, args.run, args.replay))
