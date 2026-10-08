"""Read-only, independent arithmetic reconstruction of the v2 moment run."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .highd_v46_assets import digest, read
from .highd_v47_state_moments_v2 import inputs


def precision_record(state):
    """Inherited half-width <= 50% J rule, using the bank-wide interval.

    This is a fail-closed development consumer check, not a new outcome-tuned
    acceptance criterion for the already frozen native experiment.
    """
    lo, hi = state['bank_simultaneous_interval']
    j = state['J']
    if not (np.isfinite([lo, hi, j]).all() and 0 <= lo <= hi and j >= 0):
        raise ValueError('Invalid state moment or uncertainty interval')
    return {'state_id': state['state_id'],
            'precision_supported': bool(lo > 0 and (hi-lo)/2 <= .5*j),
            'zero_observed_moment_is_not_zero_risk': j == 0,
            'relative_interval_half_width': (hi-lo)/(2*j) if j > 0 else None,
            'relative_empirical_MCSE': state['relative_empirical_MCSE'],
            'relative_batch_gap': state['relative_batch_gap']}


def checked_moments(summary, arm):
    """Reject uncertified point moments instead of feeding zeros to Q_S.

    Do not filter out failed states: that would silently change the target.
    Retain the complete bank and its uncertainty for separate planning work.
    """
    if not summary.get('complete'):
        raise ValueError('Incomplete moment measurement')
    states = summary['arms'][arm]['states']
    if not states or not all(precision_record(s)['precision_supported'] for s in states):
        raise ValueError('No precise full-bank point fit: retain target allocation and all states')
    return np.array([s['J'] for s in states])


def cp_interval(k, n, error):
    return np.array([0. if k == 0 else beta.ppf(error/2, k, n-k+1),
                     1. if k == n else beta.ppf(1-error/2, k+1, n-k)])


def audit(config, directory):
    cfg, _, _, _, plans, designs, design, expected = inputs(config)
    root = Path(directory); result = read(root/'summary.json')
    assert result['complete'] and result['config'] == cfg
    assert all(result['lock'].get(k) == v for k, v in expected.items())
    for name, sha in result['artifact_hashes'].items():
        assert Path(name).name == name and digest(root/name) == sha
    assert read(root/'measurement_plan.json') == design
    cells = sum(len(s['strata']) for s in design['states'])
    family = cfg['batches']*len(cfg['arms'])*cells
    by_key = {(e['batch'], e['state_id'], e['arm']): e for e in result['state_estimates']}
    paired = {(e['batch'], e['state_id']): e for e in result['paired_variance']}
    measurements = {}; seen = set(); logical = 0; correction_error = 0.
    for schedule in design['schedules']:
        batch, i = schedule['batch'], schedule['state_id']; state = design['states'][i]
        stored = read(root/f'batch_{batch}_state_{i:03d}.json')
        assert (stored['batch'], stored['state_id']) == (batch, i)
        assert len(stored['probes']) == len(schedule['probes'])
        p = np.array(designs[i]['P']); qa = np.array(state['target_QA'])
        c = np.zeros_like(p); support = p > 0; c[support] = p[support]*(p[support]/qa[support])
        assert sorted(cell for h in state['strata'] for cell in h['cells']) == np.flatnonzero(support.ravel()).tolist()
        for r, planned in zip(stored['probes'], schedule['probes']):
            assert all(r[k] == v for k, v in planned.items())
            assert r['native_seed'] not in seen; seen.add(r['native_seed'])
            assert r['event_result'] == r['native']['cav_first_collision']
            assert not (r['native']['cav_first_collision'] and r['native']['bv_only_collision'])
            a = tuple(r['action_indices'])
            assert r['P_action'] == p[a] and r['target_QA_action'] == qa[a]
            error = abs(c[a]/r['measurement_G_conditional']-r['constant_moment_weight'])
            correction_error = max(correction_error, float(error))
            assert error <= 1e-12*max(1., r['constant_moment_weight'])
            logical += len(r['arms'])
        scores = np.array([s['objective_coefficient'] for s in state['planning']])
        ne, nu = (np.array(state['counts'][arm]) for arm in cfg['arms'])
        assert ne.sum() == nu.sum() == cfg['draws_per_state_per_arm']
        assert np.all(nu >= np.array(state['allocation_floors'])) and np.all(nu >= .5*ne)
        assert np.sum(scores/nu) <= np.sum(scores/ne)+1e-12
        # Verify discrete optimality through the best feasible one-draw exchange.
        for donor in range(len(nu)):
            if nu[donor] > state['allocation_floors'][donor]:
                loss = scores[donor]/(nu[donor]*(nu[donor]-1))
                assert loss >= np.max(scores/(nu*(nu+1)))-1e-12
        for arm in cfg['arms']:
            rows = [r for r in stored['probes'] if arm in r['arms']]
            assert len(rows) == cfg['draws_per_state_per_arm']
            j = variance = 0.; local = np.zeros(2); global_bounds = np.zeros(2)
            for h, n in zip(state['strata'], state['counts'][arm]):
                values = [r for r in rows if r['stratum_id'] == h['stratum_id']]
                assert [r['within_stratum_draw'] for r in values] == list(range(n))
                k = sum(r['event_result'] for r in values); rate = k/n
                mass = float(c.ravel()[h['cells']].sum())
                np.testing.assert_allclose(mass, h['coefficient_mass'], rtol=1e-13)
                np.testing.assert_allclose(c.ravel()[h['cells']]/mass, h['measurement_probabilities'], rtol=1e-13)
                j += mass*rate; variance += mass*mass*rate*(1-rate)/(n-1)
                local += mass*cp_interval(k, n, (1-cfg['confidence'])/len(state['strata']))
                global_bounds += mass*cp_interval(k, n, (1-cfg['confidence'])/family)
            e = by_key[(batch, i, arm)]
            np.testing.assert_allclose([j, variance], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-25)
            np.testing.assert_allclose(local, e['state_interval'], rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(global_bounds, e['bank_simultaneous_interval'], rtol=1e-8, atol=1e-12)
            measurements[(batch, i, arm)] = (j, variance, global_bounds)
        ve = vu = covariance = 0.; variance_bounds = np.zeros(2)
        for index, h in enumerate(state['strata']):
            rows = [r for r in stored['probes'] if r['stratum_id'] == h['stratum_id']]
            count = len(rows); k = sum(r['event_result'] for r in rows); rate = k/count
            assert count == max(ne[index], nu[index])
            s2 = rate*(1-rate)*count/(count-1); c2 = h['coefficient_mass']**2
            ve += c2*s2/ne[index]; vu += c2*s2/nu[index]
            covariance += c2*s2/count
            lo, hi = cp_interval(k, count, (1-cfg['confidence'])/(cfg['batches']*cells))
            vlow = min(lo*(1-lo), hi*(1-hi))
            mode = np.clip(.5, lo, hi); vhigh = mode*(1-mode)
            d = c2*(1/nu[index]-1/ne[index])
            variance_bounds += d*np.array([vlow, vhigh] if d >= 0 else [vhigh, vlow])
        e = paired[(batch, i)]
        np.testing.assert_allclose([ve, vu, max(0., ve+vu-2*covariance)],
            [e['equal_variance_same_probes'], e['allocated_variance_same_probes'], e['paired_difference_variance']], rtol=1e-12, atol=1e-25)
        np.testing.assert_allclose(variance_bounds, e['variance_difference_interval'], rtol=1e-8, atol=1e-12)
        # The deterministic count floor guarantees this for any unknown Bernoulli probabilities.
        assert np.all(1/nu <= 2/ne)
    alpha = np.array([p['target_mass'] for p in plans])
    for arm in cfg['arms']:
        js = []; vs = []; bs = []
        for i in range(len(plans)):
            a, b = measurements[(0, i, arm)], measurements[(1, i, arm)]
            j, v, bounds = (a[0]+b[0])/2, (a[1]+b[1])/4, (a[2]+b[2])/2
            e = result['arms'][arm]['states'][i]
            np.testing.assert_allclose([j, v], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-25)
            np.testing.assert_allclose(bounds, e['bank_simultaneous_interval'], rtol=1e-8, atol=1e-12)
            js.append(j); vs.append(v); bs.append(bounds)
        js, vs, bs = np.array(js), np.array(vs), np.array(bs)
        np.testing.assert_allclose([alpha@js, np.sqrt((alpha*alpha)@vs)],
            [result['arms'][arm]['target_M2'], result['arms'][arm]['target_M2_empirical_MCSE']], rtol=1e-12)
        for name, q in design['state_proposals'].items():
            e = result['arms'][arm]['state_proposal_comparisons'][name]
            d = alpha*(alpha/np.array(q)-1)
            bounds = [np.sum(np.where(d >= 0, d*bs[:, 0], d*bs[:, 1])),
                      np.sum(np.where(d >= 0, d*bs[:, 1], d*bs[:, 0]))]
            np.testing.assert_allclose(bounds, e['M2_difference_interval'], rtol=1e-8, atol=1e-12)
    assert len(seen) == result['unique_native_draws'] == design['unique_native_draws']
    assert logical == result['logical_draws'] == design['logical_draws']
    return {'verified': True, 'summary_sha256': digest(root/'summary.json'),
            'artifact_hashes': len(result['artifact_hashes']), 'unique_native_draws': len(seen),
            'logical_draws': logical, 'max_coefficient_correction_error': correction_error,
            'precision': {arm: [precision_record(s) for s in result['arms'][arm]['states']]
                          for arm in cfg['arms']},
            'scope': 'Independent arithmetic/lock/schedule audit; not independent native data.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/highd_v47_state_moments_v2.json')
    parser.add_argument('--run', default='outputs/v47_state_moments_v2')
    args = parser.parse_args(); report = audit(args.config, args.run)
    print({k: v for k, v in report.items() if k != 'precision'})
    print({'precise_states': {arm: sum(s['precision_supported'] for s in rows)
                              for arm, rows in report['precision'].items()}})
