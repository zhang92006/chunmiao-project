"""Read-only reconstruction and same-probe variance diagnostics; no native run."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .highd_v46_assets import read, digest
from .highd_v46_collision_coverage import audit_sample
from .highd_v47_state_moments import inputs


def implied_direct_variance(strata, probes, j_hat, j_variance, direct_n):
    """Unbiased direct-Q_A variance estimate from the SAME independent probes.

    T=E_QA[Y*w_A^4]=sum_h C_h E_Gh[Y*w_A^2].
    E[J_hat^2 - V_hat(J_hat)]=J^2, so subtract that, not J_hat^2 alone.
    A finite estimate can be negative; retain it and suppress the ratio.
    """
    fourth = 0.
    for h in strata:
        rows = [r for r in probes if r['stratum_id'] == h['stratum_id']]
        fourth += h['coefficient_mass']*float(np.mean([
            r['event_result']*(r['P_action']/r['target_QA_action'])**2 for r in rows]))
    return (fourth - j_hat*j_hat + j_variance)/direct_n


def _bounds(k, n, error):
    return (0. if k == 0 else float(beta.ppf(error/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-error/2, k+1, n-k)))


def audit(config, directory):
    cfg, _, _, _, plans, designs, measurements, proposals, expected = inputs(config)
    root = Path(directory); summary = read(root/'summary.json')
    assert summary['complete'] and summary['config'] == cfg
    assert all(summary['lock'].get(k) == v for k, v in expected.items())
    for name, sha in summary['artifact_hashes'].items():
        assert Path(name).name == name and digest(root/name) == sha
    saved = read(root/'measurement_plan.json')
    assert saved == {'states': measurements, 'state_proposals': proposals}
    total_cells = sum(len(m['strata']) for m in measurements)
    errors = []; reports = []; estimates = []
    for i, m in enumerate(measurements):
        raw = read(root/f'state_{i:03d}.json'); direct, probes = raw['direct'], raw['probes']
        assert raw['state_id'] == i and raw['plan'] == plans[i]
        n = cfg['draws_per_state_per_arm']
        assert len(direct) == len(probes) == n
        assert [r['seed'] for r in direct] == list(range(cfg['direct_seed_start']+i*n, cfg['direct_seed_start']+(i+1)*n))
        p = np.array(designs[i]['P']); qa = np.array(m['target_QA'])
        z = []
        for r in direct:
            audit_sample(r, cfg['epsilon'])
            a = tuple(r['initial_likelihood_audit']['action_indices'])
            lw = np.log(p[a]/qa[a])
            errors.append(abs(lw-r['generation_log_weight']))
            z.append(float(r['event_result'])*np.exp(2*lw))
        old = summary['direct_state_estimates'][i]
        np.testing.assert_allclose([np.mean(z), np.var(z, ddof=1)/n], [old['J'], old['estimator_variance']], rtol=1e-10, atol=0)
        for r, scheduled in zip(probes, m['schedule']):
            assert all(r[k] == v for k, v in scheduled.items())
            assert r['event_result'] == r['native']['cav_first_collision']
            a = tuple(r['action_indices'])
            assert r['P_action'] == p[a] and r['target_QA_action'] == qa[a]
            np.testing.assert_allclose(p[a]*(p[a]/qa[a])/r['measurement_G_conditional'], r['constant_moment_weight'], rtol=1e-13)
        j_hat = variance = known_j = remaining_j = 0.
        local = np.zeros(2); simultaneous = np.zeros(2)
        for h in m['strata']:
            rows = [r for r in probes if r['stratum_id'] == h['stratum_id']]
            assert len(rows) == h['draws']
            count, events, c = len(rows), sum(r['event_result'] for r in rows), h['coefficient_mass']
            rate = events/count; contribution = c*rate
            j_hat += contribution; variance += c*c*rate*(1-rate)/(count-1)
            if h['kind'] == 'historical_event_action': known_j += contribution
            else: remaining_j += contribution
            local += c*np.array(_bounds(events, count, (1-cfg['confidence'])/len(m['strata'])))
            simultaneous += c*np.array(_bounds(events, count, (1-cfg['confidence'])/total_cells))
        e = summary['state_estimates'][i]
        np.testing.assert_allclose([j_hat, variance], [e['J'], e['estimator_variance']], rtol=1e-12, atol=0)
        np.testing.assert_allclose(local, e['state_interval'], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(simultaneous, e['bank_simultaneous_interval'], rtol=1e-8, atol=1e-12)
        estimates.append((j_hat, variance, simultaneous))
        direct_from_probes = implied_direct_variance(m['strata'], probes, j_hat, variance, n)
        reports.append({'state_id': i, 'source_seed': plans[i]['source_seed'],
                        'direct_J': old['J'], 'stratified_J': j_hat,
                        'direct_empirical_relative_SE': float(np.sqrt(old['estimator_variance'])/old['J']) if old['J'] > 0 else None,
                        'stratified_empirical_relative_SE': float(np.sqrt(variance)/j_hat) if j_hat > 0 else None,
                        'stratified_estimator_variance': variance,
                        'implied_direct_variance_from_same_probes': direct_from_probes,
                        'same_probe_variance_ratio_point_only': variance/direct_from_probes if direct_from_probes > 0 else None,
                        'known_action_J_contribution': known_j, 'remaining_action_J_contribution': remaining_j,
                        'state_interval': local.tolist(), 'precision_supported': e['precision_supported'],
                        'direct_events': sum(r['event_result'] for r in direct),
                        'probe_events': sum(r['event_result'] for r in probes)})
    j = np.array([e[0] for e in estimates]); v = np.array([e[1] for e in estimates])
    bounds = np.array([e[2] for e in estimates]); alpha = np.array(proposals['target'])
    for name, result in summary['common_measurement_comparisons'].items():
        q = np.array(proposals[name] if name in proposals else result['state_Q'])
        d = alpha*alpha/q-alpha
        lo = np.sum(np.where(d >= 0, d*bounds[:, 0], d*bounds[:, 1]))
        hi = np.sum(np.where(d >= 0, d*bounds[:, 1], d*bounds[:, 0]))
        np.testing.assert_allclose([alpha@j, (alpha*alpha/q)@j, d@j, lo, hi],
            [result['target_iid_M2'], result['candidate_iid_M2'], result['M2_difference'], *result['simultaneous_M2_difference_interval']], rtol=1e-10, atol=1e-15)
        assert result['certified_improvement'] == (hi < 0)
        if name in proposals:
            assert np.isclose(result['difference_empirical_MCSE'], np.sqrt((d*d)@v))
    assert max(errors) < 1e-12
    return {'summary_sha256': digest(root/'summary.json'), 'verified_artifacts': len(summary['artifact_hashes']),
            'verified_native_records': 2*len(plans)*cfg['draws_per_state_per_arm'],
            'max_direct_log_weight_error': max(errors), 'states': reports,
            'post_hoc_scope': 'Same-probe direct-variance reconstruction is an explanatory point diagnostic, not a preregistered variance-ratio interval or additional experiment.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/highd_v47_state_moments_v1.json')
    p.add_argument('--run', default='outputs/v47_state_moments_v1')
    a = p.parse_args()
    print(audit(a.config, a.run))
