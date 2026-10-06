"""Read-only reconstruction of native branching roots and protected allocation."""
import argparse
import json
from pathlib import Path

import numpy as np

from .highd_v46_assets import digest, read
from .highd_v47_continuation_moments_audit import reconstruct_path
from .highd_v47_branching_moments import inputs


def root_values(row):
    error = reconstruct_path(row['prefix'])
    if row['terminal_prefix']:
        assert not row['branches'] and row['selected_branch'] is None
        assert row['prefix']['collision_pairs'] or row['prefix']['elapsed_s'] >= 4-1e-9
        return row['prefix']['weighted_event'], row['prefix']['weighted_event'], 0., error
    eta = float(np.exp(-np.logaddexp(0., row['prefix']['full_log_t']-row['prefix']['full_log_p'])))
    np.testing.assert_allclose(eta, row['posterior_natural_fraction'], rtol=1e-11)
    assert len(row['branches']) == 4 and 0 < eta < 1
    values = []
    for i, b in enumerate(row['branches']):
        assert b['component'] == ('P' if i < 2 else 'T')
        assert b['fork_state_sha256'] == row['fork_state_sha256']
        assert b['path']['steps'][:len(row['prefix']['steps'])] == row['prefix']['steps']
        assert b['path']['initial_cancelled_log_conditional'] == row['prefix']['initial_cancelled_log_conditional']
        assert b['suffix_steps'] == len(b['path']['steps'])-len(row['prefix']['steps'])
        error = max(error, reconstruct_path(b['path'])); values.append(b['path']['weighted_event'])
    selected = 0 if row['control_uniform'] < eta else 2
    assert selected == row['selected_branch']
    p, t = np.array(values[:2]), np.array(values[2:])
    mean_p, mean_t = p.sum()/2, t.sum()/2
    s_p, s_t = float(((p-mean_p)**2).sum()), float(((t-mean_t)**2).sum())
    v_branch = (eta**2*s_p+(1-eta)**2*s_t)/2
    v_single = eta*s_p+(1-eta)*s_t+eta*(1-eta)*((mean_p-mean_t)**2-(s_p+s_t)/2)
    branch = float(eta*mean_p+(1-eta)*mean_t)
    np.testing.assert_allclose([row['branching_value'], row['single_value'], row['removed_conditional_variance_unbiased']],
                               [branch, values[selected], v_single-v_branch], rtol=1e-10, atol=1e-25)
    assert 0 <= branch <= 2+1e-12
    return float(values[selected]), branch, float(v_single-v_branch), error


def reconstruct_estimate(strata, rows, arm, error):
    j = v = lo = hi = removed = 0.
    for h in strata:
        g = [r for r in rows if r['stratum_id'] == h['stratum_id']]; n = len(g)
        assert n == h['draws'] and {r['within_stratum_draw'] for r in g} == set(range(n))
        z = np.array([r[arm+'_value'] for r in g]); mean = z.sum()/n
        s2 = float(((z-mean)**2).sum()/(n-1)); log = np.log(4/error)
        radius = np.sqrt(2*s2*log/n)+14*log/(3*(n-1))
        bounds = [min(1., max(0., mean-radius)), min(1., mean+radius)]
        c = h['coefficient_mass']; j += c*mean; v += c*c*s2/n
        lo += c*bounds[0]; hi += c*bounds[1]
        removed += c*c*sum(r['removed_conditional_variance_unbiased'] for r in g)/(n*n)
    return float(j), float(v), [lo, hi], float(removed)


def audit(config, directory, replay=False):
    cfg, physical, assets, plans, ads, design, expected = inputs(config)
    root = Path(directory); summary = read(root/'summary.json')
    assert summary['complete'] and summary['config'] == cfg
    assert all(summary['lock'].get(k) == v for k, v in expected.items())
    assert read(root/'measurement_plan.json') == design
    for name, sha in summary['artifact_hashes'].items():
        assert Path(name).name == name and digest(root/name) == sha
    error = (1-cfg['confidence'])/(4*sum(len(s['strata']) for s in design['states']))
    estimates = {(e['batch'], e['state_id'], e['arm']): e for e in summary['state_batch_estimates']}
    measurements = {}; seeds = set(); max_error = 0.; suffixes = decisions = replays = 0
    if replay:
        from . import highd_v47_branching_moments as native
        native.parent.parent._init_worker(str(assets))
    for schedule in design['schedules']:
        b, i = schedule['batch'], schedule['state_id']; stored = read(root/f'batch_{b}_state_{i:03d}.json')
        assert len(stored['rows']) == len(schedule['rows']) == cfg['roots_per_state']
        for plan, row in zip(schedule['rows'], stored['rows']):
            assert all(row[k] == v for k, v in plan.items())
            ss = [row['prefix_seed']]+row['suffix_seeds']; assert not (set(ss)&seeds); seeds.update(ss)
            values = root_values(row); max_error = max(max_error, values[3]); suffixes += len(row['branches'])
            decisions += len(row['prefix']['steps'])+sum(x['suffix_steps'] for x in row['branches'])
            assert row['future_decisions'] == len(row['prefix']['steps'])+sum(x['suffix_steps'] for x in row['branches'])
        for arm in ('single', 'branching'):
            e = estimates[(b, i, arm)]; rebuilt = reconstruct_estimate(design['states'][i]['strata'], stored['rows'], arm, error)
            np.testing.assert_allclose(rebuilt[:2], [e['J'], e['estimator_variance']], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose(rebuilt[2], e['bank_simultaneous_interval'], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose(rebuilt[3], e['removed_conditional_variance_unbiased'], rtol=1e-10, atol=1e-25)
            measurements[(b, i, arm)] = rebuilt
        if replay:
            index = (17*i+31*b) % cfg['roots_per_state']
            recreated = native.root_continuation(native.parent.parent._MODEL, assets, physical, ads[i]['state'],
                plans[i]['actor_ids'], schedule['rows'][index], cfg)
            assert json.loads(json.dumps(recreated)) == {k: v for k, v in stored['rows'][index].items() if k not in schedule['rows'][index]}
            replays += 1
    a = np.array(design['target_masses']); info = {}
    for arm in ('single', 'branching'):
        js = []; vs = []
        for i in range(len(plans)):
            x, y = measurements[(0, i, arm)], measurements[(1, i, arm)]
            j, v = (x[0]+y[0])/2, (x[1]+y[1])/4; es = summary['arms'][arm]['states'][i]
            np.testing.assert_allclose([j, v], [es['J'], es['estimator_variance']], rtol=1e-12, atol=1e-30)
            np.testing.assert_allclose((np.array(x[2])+y[2])/2, es['bank_simultaneous_interval'], rtol=1e-12, atol=1e-30)
            js.append(j); vs.append(v)
        np.testing.assert_allclose([a@js, (a*a)@vs], [summary['arms'][arm]['target_M2'], summary['arms'][arm]['target_measurement_variance']], rtol=1e-12)
        info[arm] = dict(target_M2=float(a@js), variance=float((a*a)@vs))
    allocation = read(root/'batch0_allocation_before_batch1.json')
    assert allocation == summary['uncertainty_aware_allocation']
    q = np.array(allocation['upper_envelope_candidate']); assert np.isclose(q.sum(), 1) and np.all(q >= a/1.1-1e-14)
    train = np.array([measurements[(0, i, 'branching')][2] for i in range(len(plans))])
    d = a*(a/q-1); upper = float(np.where(d>=0, d*train[:, 1], d*train[:, 0]).sum())
    np.testing.assert_allclose(upper, allocation['worst_case_M2_difference_upper'], rtol=1e-12, atol=1e-30)
    evaluation = np.array([measurements[(1, i, 'branching')][0] for i in range(len(plans))])
    cmp = summary['independent_batch_allocation_comparison']
    np.testing.assert_allclose([a@evaluation, (a*a/q)@evaluation], [cmp['target_M2'], cmp['candidate_M2']], rtol=1e-12)
    # Guard is distribution-free for all J>=0; do not infer that data proved gain.
    assert float(np.max(a/q)) <= 1.1+1e-14
    if not allocation['strict_improvement_certified']:
        np.testing.assert_allclose(allocation['recommended_q'], a, rtol=0, atol=0)
    assert suffixes == summary['suffixes_simulated'] and decisions == summary['future_native_decisions']
    assert len(seeds) == 5*summary['independent_roots']
    return dict(verified=True, summary_sha256=digest(root/'summary.json'),
                artifact_hashes_verified=len(summary['artifact_hashes']), independent_roots=summary['independent_roots'],
                suffixes=suffixes, future_native_decisions=decisions, native_root_replays_matched=replays,
                max_full_factor_log_error=max_error, arms=info, allocation=allocation)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/highd_v47_branching_moments_v1.json')
    p.add_argument('--run',default='outputs/v47_branching_moments_v1'); p.add_argument('--replay',action='store_true')
    a=p.parse_args(); print(audit(a.config,a.run,a.replay))
