"""Read-only factor-support audit of a checksum-locked validation collection."""
import argparse
import math
from pathlib import Path
import platform

from .highd_v46_assets import digest, read, save

DEFAULT_CONFIG = 'configs/highd_v46_collision_coverage_v1.json'
CATEGORIES = ('H1_positive_H2_positive', 'H1_zero_H2_positive',
              'H1_positive_H2_zero', 'H1_zero_H2_zero', 'natural_fallback')


def support_category(h):
    """H2 is the implemented conditional, including P2 fallback at zero H1."""
    return ('H1_zero' if h[0] == 0 else 'H1_positive') + '_' + (
        'H2_zero' if h[1] == 0 else 'H2_positive')


def audit_sample(sample, epsilon):
    log = sample['initial_likelihood_audit']
    lw = sample['generation_log_weight']
    if not math.isfinite(lw) or not math.isclose(
            log['full_log_p'] - log['full_log_q'], lw, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError('Full likelihood differs from recorded weight')
    if type(sample['event_result']) is not bool:
        raise ValueError('Expected a Boolean CAV event')
    if log.get('abstained', False):
        if lw != 0 or any(k in log for k in ('h_factors', 'p_factors', 'q_factors')):
            raise ValueError('Invalid natural fallback record')
        return {'category': 'natural_fallback', 'weight': 1., 'factor_lr': [1., 1.]}
    p, h, q = (log[k] for k in ('p_factors', 'h_factors', 'q_factors'))
    if (log['generation_epsilon'] != epsilon or any(len(x) != 2 for x in (p, h, q, epsilon))
            or any(not math.isfinite(v) for x in (p, h, q, epsilon) for v in x)
            or any(not 0 < v <= 1+1e-12 for x in (p, q) for v in x)
            or any(not 0 <= v <= 1+1e-12 for v in h)
            or any(not .05 <= e <= 1 for e in epsilon)):
        raise ValueError('Invalid factor probabilities or epsilon')
    if any(not math.isclose(qi, e*pi+(1-e)*hi, rel_tol=1e-10, abs_tol=1e-15)
           for pi, hi, qi, e in zip(p, h, q, epsilon)):
        raise ValueError('Logged Q differs from its factor mixture')
    if h[0] == 0 and not math.isclose(h[1], p[1], rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError('Zero-H1 row must use natural H2 fallback')
    ratios = [pi/qi for pi, qi in zip(p, q)]
    if not math.isclose(sum(math.log(x) for x in ratios), lw, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError('Factor product differs from full likelihood')
    return {'category': support_category(h), 'weight': math.exp(lw), 'factor_lr': ratios}


def summarize(cases):
    """One case/source and equal n: sum(I*w^2)/(K*n) retains all zero cases."""
    sources = [c['source_seed'] for c in cases]
    if len(set(sources)) != len(sources) or len({c['trials'] for c in cases}) > 1:
        raise ValueError('Expected one equally sampled case per selected source')
    events = [e for c in cases for e in c['events']]
    trials = sum(c['trials'] for c in cases)
    total = math.fsum(e['weight']**2 for e in events)
    by_source = {c['source_seed']: math.fsum(e['weight']**2 for e in c['events']) for c in cases}
    shares = [v/total for v in by_source.values()] if total else []
    categories = {}
    for key in CATEGORIES:
        selected = [e for e in events if e['category'] == key]
        value = math.fsum(e['weight']**2 for e in selected)
        categories[key] = {
            'samples': sum(c['category_samples'][key] for c in cases),
            'collision_events': len(selected), 'event_sources': sorted({e['source_seed'] for e in selected}),
            'sum_Iw2': value, 'conditional_M2_contribution': value/trials if trials else None,
            'second_moment_share': value/total if total else None,
            'factor_lr_range': [[min(e['factor_lr'][i] for e in selected), max(e['factor_lr'][i] for e in selected)]
                                for i in (0, 1)] if selected else None}
    largest = max(events, key=lambda e: e['weight']) if events else None
    sensitivity = None
    if largest:
        remaining = [e for e in events if e is not largest]
        residual = math.fsum(e['weight']**2 for e in remaining)
        residual_sources = {seed: math.fsum(e['weight']**2 for e in remaining if e['source_seed'] == seed)
                            for seed in sources}
        sensitivity = {'excluded_observation': {k: largest[k] for k in ('case_id', 'source_seed', 'seed')},
                       'remaining_sum_Iw2': residual,
                       'remaining_maximum_source_share': max(residual_sources.values())/residual if residual else None,
                       'scope': 'Leave-one-observation-out sensitivity only; not a corrected IS estimate. No data removed.'}
    return {'selected_sources': len(cases), 'continuations': trials, 'collision_events': len(events),
            'event_sources': sorted({e['source_seed'] for e in events}),
            'sum_Iw2': total, 'conditional_second_moment': total/trials if trials else None,
            'maximum_source_share': max(shares) if shares else None,
            'effective_second_moment_sources': 1/math.fsum(x*x for x in shares) if shares else None,
            'categories': categories,
            'largest_event': dict(largest, second_moment_share=largest['weight']**2/total) if largest else None,
            'sensitivity_only': sensitivity}


def replay_proxy_witness(cfg, summary, case, event):
    """Post-hoc deterministic proxy reconstruction; no new native continuation."""
    from types import SimpleNamespace
    import numpy as np
    from .highd_v46_assets import ensure_assets
    from .highd_v46_source_selector_validation import check_lock, DEFAULT_CONFIG as SOURCE_CONFIG
    from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
    from .highd_v46_pair_synergy import pair_marginal, response_risk
    from .highd_v46_cross_unit import positive_interaction, proposal_factors

    _, _, base, physical, _ = check_lock(SOURCE_CONFIG, cfg['source_rule_lock'])
    if digest(SOURCE_CONFIG) != summary['config_sha256'] or digest(cfg['source_rule_lock']) != summary['rule_lock_sha256']:
        raise ValueError('Proxy witness source protocol differs')
    state = read(Path(cfg['collection'])/'state_bank.json')['states'][case['state_index']]
    assets, _ = ensure_assets()
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    cars = [SimpleNamespace(**c) for c in state['vehicles']]
    road = SimpleNamespace(vehicles=cars, cav=next(c for c in cars if c.id == state['cav_id']),
                           model=model, config=physical)
    p, _ = pair_marginal([{'ids': u['ids'], 'p': np.asarray(u['p'])} for u in state['units']], case['actor_ids'])
    risk = response_risk(road, case['actor_ids'], {int(k): np.asarray(v) for k, v in state['pmfs'].items()}, base['proxy'], reactive=True)
    interaction = positive_interaction(p, risk); mass = float((p*interaction).sum())
    h = p*interaction/mass if mass > 0 else p.copy()
    factors = proposal_factors(p, h, event['epsilon']); a, b = event['action_indices']
    actual = [[float(factors[i][a]), float(factors[i+1][a, b])] for i in (0, 2, 4)]
    for value, key in zip(actual, ('p_factors', 'h_factors', 'q_factors')):
        if not np.allclose(value, event[key], atol=1e-12, rtol=1e-12):
            raise ValueError('Reconstructed proxy proposal differs from logged factors')
    a0, b0 = np.unravel_index(np.argmax(p), p.shape)
    return {'source_seed': case['source_seed'], 'case_id': case['case_id'], 'state_index': case['state_index'],
            'time_s': state['time_s'], 'actor_ids': case['actor_ids'], 'action_indices': [a, b],
            'baseline_action_indices': [int(a0), int(b0)],
            'four_corner_risk': {'R11': float(risk[a, b]), 'R10': float(risk[a, b0]),
                                 'R01': float(risk[a0, b]), 'R00': float(risk[a0, b0])},
            'raw_interaction': float(risk[a, b]-risk[a, b0]-risk[a0, b]+risk[a0, b0]),
            'positive_interaction': float(interaction[a, b]),
            'all_corners_naturally_supported': bool(all(p[x, y] > 0 for x, y in ((a, b), (a, b0), (a0, b), (a0, b0)))),
            'first_action_is_baseline': bool(a == a0), 'second_action_is_baseline': bool(b == b0),
            'reproduced_p_h_q_factors': actual,
            'scope': 'Largest observed B dual event contribution selected post hoc; proxy clearance score is not collision probability.'}


def run(config_path, output):
    cfg = read(config_path); root = Path(cfg['collection'])
    if cfg['pool_arms'] or cfg['fit_on_validation']:
        raise ValueError('Explanatory audit cannot pool arms or fit validation')
    if digest(root/'summary.json') != cfg['summary_sha256']:
        raise ValueError('Locked validation summary differs')
    summary = read(root/'summary.json')
    if not summary['complete'] or summary['split'] != 'validation' or summary['config']['environment'] != cfg['environment']:
        raise ValueError('Expected completed pinned validation collection')
    for name, expected in summary['artifacts'].items():
        if Path(name).name != name or digest(root/name) != expected:
            raise ValueError('Validation artifact differs: '+name)
    plan = read(root/'case_plan.json'); cases = plan['cases']
    seeds = list(range(cfg['source_seed_range'][0], cfg['source_seed_range'][1]+1))
    if (plan['source_seeds'] != seeds or plan['selection_uses_native_outcomes']
            or len(cases) != summary['case_count'] or len({c['case_id'] for c in cases}) != len(cases)):
        raise ValueError('Source/case identity or outcome-blind plan differs')
    arms = summary['config']['arms']; n = summary['config']['replicates_per_case_arm']
    native_records = {}; checked = 0
    for c in cases:
        records = {}
        for arm, eps in arms.items():
            raw = read(root/f'case_{c["case_id"]:03d}_{arm}.json')
            first_seed = summary['config']['continuation_seed_start'] + (c['source_seed']-seeds[0])*n
            samples = raw['sequences']
            if (raw['plan'] != c or raw['epsilon'] != eps
                    or [s['seed'] for s in samples] != list(range(first_seed, first_seed+n))):
                raise ValueError('Native case/arm/schedule differs')
            record = {'source_seed': c['source_seed'], 'case_id': c['case_id'], 'trials': n,
                      'category_samples': dict.fromkeys(CATEGORIES, 0), 'events': []}
            for sample in samples:
                audit = audit_sample(sample, eps); checked += 1
                if (audit['category'] == 'natural_fallback') != c['natural_fallback']:
                    raise ValueError('Fallback differs from the outcome-blind plan')
                record['category_samples'][audit['category']] += 1
                if sample['event_result']:
                    log = sample['initial_likelihood_audit']
                    record['events'].append({**audit, 'case_id': c['case_id'], 'source_seed': c['source_seed'],
                        'seed': sample['seed'], 'epsilon': eps,
                        **{k: log.get(k) for k in ('action_indices', 'p_factors', 'h_factors', 'q_factors')}})
            records[arm] = record
        native_records[c['case_id']] = records
    reports = {}
    for method in ('A', 'B'):
        selected = [c for c in cases if method in c['selected_for']]
        reports[method] = {arm: summarize([native_records[c['case_id']][arm] for c in selected]) for arm in arms}
        for arm, report in reports[method].items():
            old = summary['selectors'][method]['arms'][arm]['native']
            if (report['collision_events'] != old['cav_events'] or report['continuations'] != old['continuations']
                    or report['event_sources'] != sorted(r['source_seed'] for r in old['by_source'] if r['cav_events'])
                    or not math.isclose(report['conditional_second_moment'], old['conditional_second_moment'], rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError('Audit does not recover original native moments')
        reports[method]['event_source_union_across_arms'] = sorted(set().union(*(set(reports[method][arm]['event_sources']) for arm in arms)))
    witness = None
    largest = reports['B']['dual']['largest_event']
    if cfg['replay_largest_dual_proxy'] and largest is not None and largest['category'] != 'natural_fallback':
        witness = replay_proxy_witness(cfg, summary, next(c for c in cases if c['case_id'] == largest['case_id']), largest)
    result = {'scope': cfg['scope'], 'post_hoc': True, 'config_sha256': digest(config_path),
              'implementation_sha256': digest(__file__), 'source_summary_sha256': cfg['summary_sha256'],
              'source_config_sha256': summary['config_sha256'], 'natural_target_sha256': summary['natural_target_sha256'],
              'environment_of_source_run': cfg['environment'], 'audit_python': platform.python_version(),
              'verified_artifact_hashes': len(summary['artifacts']), 'proxy_witness': witness,
              'unique_physical_sequences_checked': checked, 'selectors': reports,
              'zero_semantics': 'Exact logged effective H factors. H1=0 uses H2=P2; original H conditional is undefined in that row. Natural fallback is separate.',
              'weighting': 'Arms and A/B targets reported separately; shared physical cases read once; all zero-event cases retained.',
              'proposal_changed': False, 'new_continuations': 0, 'ppo_trained': False}
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({'audited_sequences': result['unique_physical_sequences_checked'],
           'dual_categories_B': result['selectors']['B']['dual']['categories'], 'proposal_changed': False})


if __name__ == '__main__':
    main()
