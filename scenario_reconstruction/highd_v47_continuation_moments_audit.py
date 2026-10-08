"""Independent read-only reconstruction of the bounded continuation estimator."""
import argparse
import json
from pathlib import Path

import numpy as np

from .highd_v46_assets import digest, read
from .highd_v47_continuation_moments import inputs


def reconstruct_path(path):
    lp = lt = lr = 0.
    for step in path['steps']:
        if step['actor_ids']:
            pf, tf = np.array(step['p_factors']), np.array(step['t_factors'])
            assert len(pf) == len(tf) == 2 and np.all(pf > 0) and np.all(tf >= .1*pf-1e-14)
            assert np.all(pf <= 1) and np.all(tf <= 1)
            ratio = float(np.log(tf).sum()-np.log(pf).sum())
            np.testing.assert_allclose(step['full_log_p'], np.log(pf).sum()+step['cancelled_log_conditional'], atol=1e-12)
            np.testing.assert_allclose(step['full_log_t'], np.log(tf).sum()+step['cancelled_log_conditional'], atol=1e-12)
        else:
            ratio = 0.
        np.testing.assert_allclose(ratio, step['log_t_over_p'], atol=1e-12)
        np.testing.assert_allclose(ratio, step['full_log_t']-step['full_log_p'], atol=1e-12)
        lp += step['full_log_p']; lt += step['full_log_t']; lr += ratio
    # Initial partners' conditional density is common and separately recorded.
    initial = path['initial_cancelled_log_conditional']
    complete_p, complete_t = initial+lp, initial+lt
    np.testing.assert_allclose([lp, lt, lr], [path['full_log_p'], path['full_log_t'], path['log_t_over_p']], atol=1e-12)
    assert not (path['event_result'] and path['bv_only_collision'])
    if path['collision_pairs']:
        assert path['event_result'] or path['bv_only_collision']
    error = abs((complete_t-complete_p)-lr)
    # Independent density-sum calculation; no use of the production path_weight.
    log_g_over_p = np.logaddexp(0., complete_t-complete_p)-np.log(2.)
    weight = float(np.exp(-log_g_over_p))
    assert 0 < weight <= 2.+1e-12
    np.testing.assert_allclose(weight, path['future_weight'], rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(path['weighted_event'], float(path['event_result'])*weight, rtol=1e-11, atol=1e-14)
    return error


def reconstruct_estimate(strata, rows, arm, error):
    j = v = lower = upper = 0.
    for h in strata:
        group = [r for r in rows if r['stratum_id'] == h['stratum_id']]
        assert len(group) == h['draws']
        z = np.array([r['natural']['event_result'] if arm == 'natural'
                      else r['defensive']['weighted_event'] for r in group], float)
        n = len(z); mean = float(z.sum()/n); s2 = float(np.sum((z-mean)**2)/(n-1))
        bound = 1. if arm == 'natural' else 2.
        radius = np.sqrt(2*s2*np.log(4/error)/n)+7*bound*np.log(4/error)/(3*(n-1))
        c = h['coefficient_mass']; j += c*mean; v += c*c*s2/n
        lower += c*min(1., max(0., mean-radius)); upper += c*min(1., mean+radius)
    return j, v, np.array([lower, upper])


def same_draw_variance(strata, rows, error):
    """Post-hoc explanatory contrast using the actual defensive G draws.

    Var_G(Z)-Var_P(Y)=E_G[Z^2-Z], since E_G Z=E_P Y=f.
    The natural variance uses an unbiased distinct-root U-statistic for f^2.
    Do not clip a finite negative variance estimate or substitute this analysis
    for the preregistered two-arm native results. Difference bounds are a
    separately defined 95% empirical-Bernstein family, not generalization.
    """
    vn = vg = diff = lower = upper = 0.
    for h in strata:
        group = [r for r in rows if r['stratum_id'] == h['stratum_id']]
        z = np.array([r['defensive']['weighted_event'] for r in group]); n = len(z)
        assert n == h['draws'] and n >= 2 and np.all((z >= 0) & (z <= 2+1e-12))
        f = z.mean(); second = np.mean(z*z)
        cross = ((z.sum())**2-n*second)/(n*(n-1))
        coefficient = h['coefficient_mass']**2/n
        vn += coefficient*(f-cross)
        vg += coefficient*z.var(ddof=1)
        delta = z*z-z; mean = float(delta.mean())
        diff += coefficient*mean
        # delta lies in [-1/4,2]. No probability clipping for this contrast.
        radius = np.sqrt(2*delta.var(ddof=1)*np.log(4/error)/n)+7*2.25*np.log(4/error)/(3*(n-1))
        lower += coefficient*max(-.25, mean-radius)
        upper += coefficient*min(2., mean+radius)
    np.testing.assert_allclose(vg-vn, diff, rtol=1e-10, atol=1e-15)
    return dict(natural_variance_same_draws_unbiased=float(vn), defensive_variance_same_draws=float(vg),
                variance_difference=float(diff), variance_difference_interval=[float(lower), float(upper)],
                variance_ratio_point_only=float(vg/vn) if vn>0 else None,
                certified_improvement=bool(upper < -64*np.finfo(float).eps*max(abs(lower), abs(upper))))


def audit(config, directory, replay=False):
    cfg, physical, assets, plans, designs, design, expected = inputs(config)
    root = Path(directory); summary = read(root/'summary.json')
    assert summary['complete'] and summary['config'] == cfg
    assert all(summary['lock'].get(k) == v for k, v in expected.items())
    assert read(root/'measurement_plan.json') == design
    for name, sha in summary['artifact_hashes'].items():
        assert Path(name).name == name and digest(root/name) == sha
    family = cfg['batches']*len(cfg['arms'])*sum(len(s['strata']) for s in design['states'])
    error = (1-cfg['confidence'])/family
    by_key = {(r['batch'], r['state_id'], r['arm']): r for r in summary['state_batch_estimates']}
    measurements = {}; seen = set(); simulated = 0; max_error = 0.; weights = []; replays = 0; contrasts = []
    if replay:
        from . import highd_v47_continuation_moments as native
        native._init_worker(str(assets))
    for schedule in design['schedules']:
        b, i = schedule['batch'], schedule['state_id']
        stored = read(root/f'batch_{b}_state_{i:03d}.json')
        assert len(stored['rows']) == len(schedule['rows']) == 256
        for planned, row in zip(schedule['rows'], stored['rows']):
            assert all(row[k] == v for k, v in planned.items())
            assert row['native_seed'] not in seen; seen.add(row['native_seed'])
            for arm in ('natural', 'defensive'):
                max_error = max(max_error, reconstruct_path(row[arm]))
            # CRN starts with the exact same initial natural partner draw.
            np.testing.assert_allclose(row['natural']['initial_cancelled_log_conditional'],
                                       row['defensive']['initial_cancelled_log_conditional'], atol=1e-12)
            shared = not row['tilted_component'] or not row['natural']['steps']
            assert row['shared_physical_path'] == shared
            if shared:
                assert row['natural'] == row['defensive']
            simulated += 1+int(not shared); weights.append(row['defensive']['future_weight'])
        for arm in cfg['arms']:
            measured = reconstruct_estimate(design['states'][i]['strata'], stored['rows'], arm, error)
            e = by_key[(b, i, arm)]
            np.testing.assert_allclose(measured[:2], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose(measured[2], e['bank_simultaneous_interval'], rtol=1e-12, atol=1e-30)
            measurements[(b, i, arm)] = measured
        contrast = same_draw_variance(design['states'][i]['strata'], stored['rows'],
                                     (1-cfg['confidence'])/(family/len(cfg['arms'])))
        contrasts.append(dict(contrast, batch=b, state_id=i))
        if replay:
            # One predetermined row per batch/state, not a selected positive.
            index = (17*i+31*b) % len(stored['rows'])
            task = (dict(batch=b, state_id=i, rows=[schedule['rows'][index]]), designs[i]['state'],
                    plans[i], str(assets), physical, cfg)
            regenerated = native._collect(task)['rows'][0]
            assert json.loads(json.dumps(regenerated)) == stored['rows'][index]
            replays += 1
    alpha = np.array(design['target_masses'])
    for arm in cfg['arms']:
        js = []; vs = []
        for i in range(len(plans)):
            a, b = measurements[(0, i, arm)], measurements[(1, i, arm)]
            j, v, bounds = (a[0]+b[0])/2, (a[1]+b[1])/4, (a[2]+b[2])/2
            e = summary['arms'][arm]['states'][i]
            np.testing.assert_allclose([j, v], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose(bounds, e['bank_simultaneous_interval'], rtol=1e-12, atol=1e-30)
            js.append(j); vs.append(v)
        np.testing.assert_allclose([alpha@js, np.sqrt((alpha*alpha)@vs)],
                [summary['arms'][arm]['target_M2'], summary['arms'][arm]['target_M2_empirical_MCSE']], rtol=1e-12)
    assert len(seen)*2 == summary['logical_draws'] == design['logical_draws']
    assert simulated == summary['native_paths_simulated'] <= design['planned_native_paths_upper_bound']
    total = {key: float(sum(.25*alpha[c['state_id']]**2*c[key] for c in contrasts))
             for key in ('natural_variance_same_draws_unbiased', 'defensive_variance_same_draws', 'variance_difference')}
    total['variance_difference_interval'] = [float(sum(.25*alpha[c['state_id']]**2*c['variance_difference_interval'][j]
                                                      for c in contrasts)) for j in (0, 1)]
    total['certified_improvement'] = total['variance_difference_interval'][1] < 0
    total['variance_ratio_point_only'] = total['defensive_variance_same_draws']/total['natural_variance_same_draws_unbiased'] if total['natural_variance_same_draws_unbiased']>0 else None
    return dict(verified=True, summary_sha256=digest(root/'summary.json'),
                artifact_hashes_verified=len(summary['artifact_hashes']), native_paths_simulated=simulated,
                max_full_factor_log_error=max_error, future_weight_min=min(weights), future_weight_max=max(weights),
                future_weight_mean=float(np.mean(weights)),
                native_root_replays_matched=replays,
                same_draw_variance_contrast=total,
                scope='Arithmetic audit of retained training paths; not independent evaluation data.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v47_continuation_moments_v1.json')
    parser.add_argument('--run', default='outputs/v47_continuation_moments_v1')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args(); print(audit(args.config, args.run, args.replay))
