"""Bounded, action-stratified measurement of fixed-Q_A state second moments."""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import binomtest

from . import highd_v47_state_allocation as bank
from . import highd_v47_state_allocation_update as update
from .highd_v46_assets import digest, read, save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_cross_unit import proposal_factors
from .highd_v46_d2rl import identity
from .highd_v46_pair_witness import continue_state

DEFAULT = 'configs/highd_v47_state_moments_v1.json'


def action_strata(p, qa, risk, risk_edges, lr_edges, budget, singleton_cells=()):
    """Exact c=P^2/Q_A masses; sample c/C within each nonempty stratum."""
    p, qa, risk = (np.asarray(x, float) for x in (p, qa, risk))
    if (p.ndim != 2 or qa.shape != p.shape or risk.shape != p.shape
            or any(not np.all(np.isfinite(x)) for x in (p, qa, risk))
            or np.any(p < 0) or np.any(qa < 0) or not np.isclose(p.sum(), 1)
            or not np.isclose(qa.sum(), 1) or np.any(qa[p > 0] <= 0)
            or np.any((risk < 0) | (risk > 1))):
        raise ValueError('Invalid action law or risk')
    for edges in (risk_edges, lr_edges):
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError('Invalid stratum boundaries')
    support = np.flatnonzero(p.ravel() > 0)
    pp, qq = p.ravel()[support], qa.ravel()[support]
    lr = pp / qq
    c = pp * lr  # Do not square tiny P before dividing.
    if np.any(c <= 0) or not np.all(np.isfinite(c)):
        raise ValueError('Supported coefficient underflow/overflow; cannot omit cells')
    risk_bin = np.searchsorted(risk_edges, risk.ravel()[support], side='right')
    lr_bin = np.searchsorted(lr_edges, np.log10(lr), side='right')
    labels = risk_bin * (len(lr_edges) + 1) + lr_bin
    singleton_cells = sorted(set(singleton_cells))
    if any(cell not in support for cell in singleton_cells):
        raise ValueError('Historical singleton outside natural support')
    singleton_offset = (len(risk_edges)+1)*(len(lr_edges)+1)
    for j, cell in enumerate(singleton_cells):
        labels[support == cell] = singleton_offset+j
    keys = np.unique(labels)
    if budget != int(budget) or budget < 2 * len(keys):
        raise ValueError('At least two draws per nonempty stratum required')
    quotient, remainder = divmod(int(budget), len(keys))
    result = []
    for j, key in enumerate(keys):
        mask = labels == key
        mass = float(c[mask].sum())
        result.append({'stratum_id': int(key), 'kind': 'historical_event_action' if key >= singleton_offset else 'remaining_support',
                       'cells': support[mask].tolist(),
                       'coefficient_mass': mass, 'measurement_probabilities': (c[mask] / mass).tolist(),
                       'draws': quotient + int(j < remainder)})
    return result


def historical_cells(plans, designs, pilot_root, development_root, updated_root):
    """Only past training outcomes define strata; retain all complementary cells."""
    cells = [set() for _ in plans]
    observations = {}; count = 0
    def add(i, sample):
        nonlocal count
        audit_sample(sample, [.1, .1])
        actions = tuple(sample['initial_likelihood_audit']['action_indices'])
        signature = (bool(sample['event_result']), actions, sample['generation_log_weight'])
        key = (i, sample['seed'])
        if key in observations and observations[key] != signature:
            raise ValueError('Inconsistent repeated historical native draw')
        observations[key] = signature; count += 1
        if sample['event_result']:
            cells[i].add(int(np.ravel_multi_index(actions, np.array(designs[i]['P']).shape)))
    for i in range(len(plans)):
        for sample in read(pilot_root/f'case_{i:03d}_labels.json')['sequences']:
            add(i, sample)
    for root in (development_root, updated_root):
        for arm in ('target_iid', 'allocated_iid', 'equal_state_stratified'):
            for sample in read(root/f'{arm}_sequences.json')['sequences']:
                add(sample['state_id'], sample)
    return [sorted(c) for c in cells], {'historical_rows': count, 'unique_state_seed_rows': len(observations),
                                      'distinct_event_action_cells': sum(len(c) for c in cells)}


def binomial_bounds(successes, n, error):
    if not 0 < error < 1 or n < 1 or int(n) != n or not 0 <= successes <= n or int(successes) != successes:
        raise ValueError('Invalid binomial measurement')
    ci = binomtest(int(successes), int(n)).proportion_ci(confidence_level=1-error, method='exact')
    return float(ci.low), float(ci.high)


def estimate_state(strata, observations, confidence, total_cells):
    """CP union bounds distinguish zero observed variance from known zero risk."""
    if not 0 < confidence < 1 or total_cells < len(strata):
        raise ValueError('Invalid simultaneous interval design')
    expected = {(h['stratum_id'], j) for h in strata for j in range(h['draws'])}
    if len(observations) != len(expected) or {(r['stratum_id'], r['within_stratum_draw']) for r in observations} != expected:
        raise ValueError('Incomplete or duplicate stratum schedule')
    cells = []; moment = variance = lo = hi = bank_lo = bank_hi = 0.
    for h in strata:
        ys = [r['event_result'] for r in observations if r['stratum_id'] == h['stratum_id']]
        if any(y not in (True, False, 0, 1) for y in ys):
            raise ValueError('Non-Bernoulli endpoint')
        n, k, mass = len(ys), int(sum(ys)), h['coefficient_mass']
        phat = k/n
        lower, upper = binomial_bounds(k, n, (1-confidence)/len(strata))
        bl, bu = binomial_bounds(k, n, (1-confidence)/total_cells)
        moment += mass*phat
        variance += mass*mass*phat*(1-phat)/(n-1)
        lo += mass*lower; hi += mass*upper
        bank_lo += mass*bl; bank_hi += mass*bu
        cells.append({'stratum_id': h['stratum_id'], 'draws': n, 'events': k,
                      'coefficient_mass': mass, 'conditional_event_rate': phat,
                      'state_simultaneous_probability_interval': [lower, upper],
                      'bank_simultaneous_probability_interval': [bl, bu]})
    return {'J': moment, 'estimator_variance': variance, 'empirical_MCSE': float(np.sqrt(variance)),
            'state_interval': [lo, hi], 'bank_simultaneous_interval': [bank_lo, bank_hi],
            'events': int(sum(r['event_result'] for r in observations)), 'strata': cells,
            'zero_event_strata': sum(x['events'] == 0 for x in cells),
            'precision_supported': lo > 0 and (hi-lo)/2 <= .5*moment}


def moment_comparison(alpha, q, estimates):
    """Common measurements, not separate rare-tail realizations for each Q_S."""
    alpha, q = np.asarray(alpha, float), np.asarray(q, float)
    if (alpha.ndim != 1 or q.shape != alpha.shape or len(estimates) != len(alpha)
            or not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(q))
            or np.any(alpha <= 0) or np.any(q <= 0)
            or not np.isclose(alpha.sum(), 1) or not np.isclose(q.sum(), 1)):
        raise ValueError('Invalid state laws')
    j = np.array([e['J'] for e in estimates]); v = np.array([e['estimator_variance'] for e in estimates])
    bounds = np.array([e['bank_simultaneous_interval'] for e in estimates])
    if (bounds.shape != (len(alpha), 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0) or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError('Invalid simultaneous moment bounds')
    coefficient = alpha*alpha/q
    d = coefficient-alpha
    upper = float(np.sum(np.where(d >= 0, d*bounds[:, 1], d*bounds[:, 0])))
    lower = float(np.sum(np.where(d >= 0, d*bounds[:, 0], d*bounds[:, 1])))
    return {'target_iid_M2': float(alpha@j), 'candidate_iid_M2': float(coefficient@j),
            'M2_difference': float(d@j),
            'difference_empirical_MCSE': float(np.sqrt((d*d)@v)),
            'simultaneous_M2_difference_interval': [lower, upper],
            'certified_improvement': upper < 0,
            'decision': 'eligible_for_separate_evaluation' if upper < 0 else 'retain_target_allocation_insufficient_evidence'}


def inputs(config):
    cfg = read(config)
    if (cfg['version'] != 1 or cfg['draws_per_state_per_arm'] != 128 or cfg['confidence'] != .95
            or cfg['risk_boundaries'] != [.2, .4, .6, .8] or cfg['log10_action_lr_boundaries'] != [-6, -3, -1, 0]
            or cfg['action_schedule_seed'] != 5000017 or cfg['direct_seed_start'] != 5100000
            or cfg['stratified_seed_start'] != 5200000 or cfg['epsilon'] != [.1, .1] or cfg['horizon_s'] != 4):
        raise ValueError('Version protocol before changing the moment measurement')
    old_cfg, base, physical, assets, plans, designs, initial, old_lock = bank.inputs(cfg['bank_config'])
    updated_cfg, _, _, _, _, updated_plans, _, updated, updated_lock = update.inputs(cfg['updated_config'])
    if updated_plans != plans:
        raise ValueError('State target mismatch')
    updated_root = Path(cfg['completed_update_path'])
    if digest(updated_root/'summary.json') != cfg['completed_update_summary_sha256']:
        raise ValueError('Completed training update changed')
    completed = read(updated_root/'summary.json')
    if not completed['complete'] or any(completed['lock'].get(k) != v for k, v in updated_lock.items()):
        raise ValueError('Completed training update lock differs')
    for name, sha in completed['artifact_hashes'].items():
        if Path(name).name != name or digest(updated_root/name) != sha:
            raise ValueError('Completed training update artifact changed')
    singletons, history = historical_cells(plans, designs, Path(old_cfg['teacher_path']),
                                          Path(updated_cfg['development_path']), updated_root)
    rng = np.random.default_rng(cfg['action_schedule_seed']); measurements = []
    for i, d in enumerate(designs):
        p = np.array(d['P']); *_, q1, q2 = proposal_factors(p, np.array(d['H']), cfg['epsilon'])
        qa = q1[:, None]*q2
        strata = action_strata(p, qa, d['risk'], cfg['risk_boundaries'], cfg['log10_action_lr_boundaries'],
                               cfg['draws_per_state_per_arm'], singletons[i])
        schedule = []; offset = i*cfg['draws_per_state_per_arm']
        for h in strata:
            cells = rng.choice(h['cells'], size=h['draws'], p=h['measurement_probabilities'])
            for j, cell in enumerate(cells):
                row, col = np.unravel_index(cell, p.shape)
                g = h['measurement_probabilities'][h['cells'].index(int(cell))]
                schedule.append({'stratum_id': h['stratum_id'], 'within_stratum_draw': j,
                                 'action_indices': [int(row), int(col)],
                                 'native_seed': cfg['stratified_seed_start']+offset+len(schedule),
                                 'P_action': float(p[row, col]), 'target_QA_action': float(qa[row, col]),
                                 'measurement_G_conditional': g, 'constant_moment_weight': h['coefficient_mass']})
        measurements.append({'state_id': i, 'plan': plans[i], 'target_QA': qa.tolist(),
                             'strata': strata, 'schedule': schedule})
    proposals = {'target': initial['target_masses'], 'first_allocation': initial['state_Q'], 'updated_allocation': updated['state_Q']}
    dependency_files = ['highd_v47_state_allocation.py', 'highd_v47_state_allocation_update.py',
                        'highd_v46_pair_witness.py', 'highd_v46_cross_unit.py', 'highd_v46_multistate_q.py']
    lock = {'config_sha256': digest(config), 'implementation_sha256': digest(__file__),
            'bank_lock': old_lock, 'updated_lock': updated_lock,
            'completed_update_summary_sha256': cfg['completed_update_summary_sha256'],
            'historical_design': history,
            'dependency_sha256': {name: digest(Path(__file__).with_name(name)) for name in dependency_files},
            'measurement_sha256': identity(measurements), 'proposals_sha256': identity(proposals)}
    return cfg, base, physical, assets, plans, designs, measurements, proposals, lock


def freeze(config, output):
    *_, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(output, lock)
    return lock


def run(config, lock_path, output):
    cfg, base, physical, assets, plans, designs, measurements, proposals, expected = inputs(config)
    lock = read(lock_path)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen measurement changed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out/'opened_protocol.json', {'config': cfg, 'lock': lock})
    save(out/'measurement_plan.json', {'states': measurements, 'state_proposals': proposals})
    model = bank.PairModelQueries(assets/'model', assets/'lateral', **bank.SINGLE_OPTIONS)
    total_cells = sum(len(m['strata']) for m in measurements)
    estimates = []; direct_estimates = []; started = time.perf_counter()
    for i, (d, m, plan) in enumerate(zip(designs, measurements, plans)):
        direct = []; probes = []
        for j in range(cfg['draws_per_state_per_arm']):
            sample = bank.sample_continuation(model, assets, physical, d['state'],
                {'ids': plan['actor_ids'], 'h': np.array(d['H'])}, cfg['epsilon'],
                cfg['direct_seed_start']+i*cfg['draws_per_state_per_arm']+j, cfg, base['proxy']['clearance_scale_m'])
            audit_sample(sample, cfg['epsilon']); direct.append(sample)
        for action in m['schedule']:
            forced = dict(zip(plan['actor_ids'], action['action_indices']))
            native = continue_state(model, assets, physical, d['state'], forced,
                                    action['native_seed'], cfg['horizon_s'], base['proxy']['clearance_scale_m'])
            probes.append(dict(action, event_result=bool(native['cav_first_collision']), native=native))
        save(out/f'state_{i:03d}.json', {'state_id': i, 'plan': plan, 'direct': direct, 'probes': probes})
        estimate = estimate_state(m['strata'], probes, cfg['confidence'], total_cells)
        estimate.update(state_id=i, source_seed=plan['source_seed']); estimates.append(estimate)
        z = np.array([s['event_result']*np.exp(2*s['generation_log_weight']) for s in direct])
        direct_estimates.append({'state_id': i, 'source_seed': plan['source_seed'], 'J': float(z.mean()),
                                 'estimator_variance': float(z.var(ddof=1)/len(z)),
                                 'events': int(sum(s['event_result'] for s in direct)),
                                 'zero_empirical_variance_is_not_precision': True})
        print({'states_completed': i+1, 'total': len(plans)}, flush=True)
    alpha = np.array(proposals['target'])
    comparisons = {name: moment_comparison(alpha, q, estimates) for name, q in proposals.items()}
    # New fitted Q is only a diagnostic candidate; simultaneous bounds protect the data-dependent decision.
    fitted, _ = bank.state_allocation(alpha, [e['J'] for e in estimates], .5, .01)
    comparisons['measurement_fitted_allocation'] = moment_comparison(alpha, fitted, estimates)
    comparisons['measurement_fitted_allocation']['state_Q'] = fitted.tolist()
    comparisons['measurement_fitted_allocation']['difference_empirical_MCSE'] = None
    comparisons['measurement_fitted_allocation']['data_dependent_candidate'] = True
    result = {'complete': True, 'config': cfg, 'lock': lock,
              'state_estimates': estimates, 'direct_state_estimates': direct_estimates,
              'common_measurement_comparisons': comparisons,
              'native_draws': 2*len(plans)*cfg['draws_per_state_per_arm'], 'nonempty_strata': total_cells,
              'total_wall_s': time.perf_counter()-started, 'production_changed': False,
              'source_generalization_validated': False, 'ppo_ready': False,
              'artifact_hashes': {p.name: digest(p) for p in sorted(out.glob('*.json'))}}
    save(out/'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'run']); p.add_argument('--config', default=DEFAULT)
    p.add_argument('--lock'); p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.mode == 'freeze':
        print(freeze(a.config, a.output))
    else:
        print(run(a.config, a.lock, a.output)['common_measurement_comparisons'])


if __name__ == '__main__':
    main()
