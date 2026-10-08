"""Whole-source paired uncertainty and fixed-state Monte Carlo diagnostics."""
import numpy as np


def summarize_sources(records, arm, readiness, stats):
    selected = [r for r in records if r['case_id'] is not None]
    effective = [r for r in selected if min(r['arm_factor_tv'][arm]) >= stats['minimum_effective_factor_tv']]
    result = {'all_source_episodes': len(records), 'candidate_source_episodes': len(selected),
              'effective_pair_source_episodes': len(effective),
              'candidate_source_fraction': len(selected)/len(records),
              'effective_pair_source_fraction': len(effective)/len(records),
              'native': None}
    if not selected:
        return result
    events = np.asarray([r['arms'][arm]['events'] for r in selected], int)
    trials = np.asarray([r['arms'][arm]['trials'] for r in selected], int)
    means = np.asarray([r['arms'][arm]['mean'] for r in selected], float)
    m2 = np.asarray([r['arms'][arm]['second_moment'] for r in selected], float)
    k = len(selected); total = float(m2.sum())
    contributions = m2/k
    fractions = m2/total if total > 0 else None
    count = int(np.count_nonzero(events))
    q2 = np.asarray([r['arm_factor_tv'][arm][1] for r in selected])
    effective_event_sources = sum(r['arms'][arm]['events'] > 0 for r in effective)
    result['native'] = {'continuations': int(trials.sum()), 'cav_events': int(events.sum()),
        'raw_event_rate': float(events.sum()/trials.sum()),
        'source_weighted_event_rate': float(np.mean(events/trials)),
        'event_source_episodes': count, 'observed_event_source_yield': count/len(records),
        'effective_pair_event_source_episodes': effective_event_sources,
        'conditional_is_mean': float(means.mean()), 'conditional_second_moment': float(m2.mean()),
        'fixed_state_mean_mcse': float(np.sqrt(sum(r['arms'][arm]['mean_mc_variance'] for r in selected))/k),
        'q_second_factor_tv': {'minimum': float(q2.min()), 'mean': float(q2.mean()), 'maximum': float(q2.max())},
        'maximum_source_contribution': float(fractions.max()) if fractions is not None else None,
        'effective_second_moment_sources_descriptive': float(1/np.sum(fractions**2)) if fractions is not None else None,
        'by_source': [{'source_seed': r['source_seed'], 'case_id': r['case_id'],
                       'cav_events': int(e), 'source_second_moment': float(x),
                       'weighted_second_moment_contribution': float(c),
                       'contribution_fraction': float(f) if fractions is not None else None}
                      for r,e,x,c,f in zip(selected,events,m2,contributions,
                                          fractions if fractions is not None else [None]*k)],
        'event_interval_guard': count >= stats['minimum_event_sources_for_intervals'] and int(events.sum()) >= stats['minimum_events_for_intervals'],
        'source_diversity_guard': effective_event_sources >= readiness['minimum_event_sources'],
        'concentration_guard': bool(fractions is not None and fractions.max() <= readiness['maximum_source_second_moment_fraction']
                                    and 1/np.sum(fractions**2) >= readiness['minimum_effective_second_moment_sources']),
        'scope': 'Conditional on selector-specific states; observed event-source yield uses all-source denominator, not zero-risk imputation for abstentions'}
    return result


def interval(values, confidence, permitted=True, bounded_fraction=False):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    alpha = (1-confidence)/2
    boundary = bounded_fraction and len(valid) > 0 and (np.all(valid == 0) or np.all(valid == 1))
    return {'interval': np.quantile(valid, [alpha, 1-alpha]).tolist() if permitted and len(valid) and not boundary else None,
            'valid_replicates': len(valid), 'undefined_replicates': len(values)-len(valid),
            'status': 'insufficient_event_sources' if not permitted else 'degenerate_observed_boundary' if boundary else 'reported' if len(valid) else 'undefined'}


def cluster_bootstrap(groups, summaries, stats):
    """Resample complete source clusters once; A/B share the same source draws.

    Continuation samples stay inside their source. Conditional-on-bank MC
    uncertainty is computed separately, rather than pooled as independent sources.
    """
    names = list(groups)
    seeds = [r['source_seed'] for r in groups[names[0]]]
    if not seeds or len(set(seeds)) != len(seeds) or any([r['source_seed'] for r in groups[n]] != seeds for n in names):
        raise ValueError('Cluster groups need identical ordered, unique source episodes')
    rng = np.random.default_rng(stats['bootstrap_seed'])
    draws = rng.integers(0, len(seeds), size=(stats['bootstrap_repeats'], len(seeds)))
    metrics = {}
    for name, records in groups.items():
        present = np.asarray([r['case_id'] is not None for r in records])
        effective = np.asarray([r['case_id'] is not None and min(r['factor_tv']) >= stats['minimum_effective_factor_tv'] for r in records])
        event_source = np.asarray([r['case_id'] is not None and r['arms']['dual']['events'] > 0 for r in records])
        matrix = {'candidate_source_fraction': present[draws].mean(1),
                  'effective_pair_source_fraction': effective[draws].mean(1),
                  'observed_event_source_yield': event_source[draws].mean(1)}
        for key, field in [('source_weighted_event_rate', None), ('conditional_is_mean', 'mean'), ('conditional_second_moment', 'second_moment')]:
            values = np.asarray([(r['arms']['dual'][field] if field else r['arms']['dual']['events']/r['arms']['dual']['trials'])
                                 if r['case_id'] is not None else np.nan for r in records])
            sampled = values[draws]; counts = np.isfinite(sampled).sum(1)
            matrix[key] = np.divide(np.nansum(sampled, axis=1), counts,
                                    out=np.full(len(draws), np.nan), where=counts > 0)
        metrics[name] = matrix
    results = {}
    for name, matrix in metrics.items():
        native = summaries[name]['arms']['dual']['native']
        guarded = bool(native and native['event_interval_guard'])
        results[name] = {k: interval(v, stats['confidence'], k in ('candidate_source_fraction', 'effective_pair_source_fraction') or guarded,
                                   k in ('candidate_source_fraction','effective_pair_source_fraction','observed_event_source_yield'))
                         for k,v in matrix.items()}
    paired = {}
    if names == ['A', 'B']:
        permitted = all(summaries[n]['arms']['dual']['native'] and summaries[n]['arms']['dual']['native']['event_interval_guard'] for n in names)
        for key in ('candidate_source_fraction', 'effective_pair_source_fraction', 'observed_event_source_yield', 'source_weighted_event_rate'):
            paired[key+'_B_minus_A'] = interval(metrics['B'][key]-metrics['A'][key], stats['confidence'],
                key in ('candidate_source_fraction','effective_pair_source_fraction') or permitted)
    return {'unit': 'complete source episode', 'paired_source_draws': True,
            'replicates': stats['bootstrap_repeats'], 'selectors': results, 'paired': paired,
            'scope': 'Source generalization for a selector-specific state law; no same-target IS variance or population acceleration comparison'}


def summarize_confirmation(records, minimum):
    measured = [r for r in records if r['status'] == 'measured']
    means = np.asarray([r['native_interaction_mean'] for r in measured])
    return {'planned_source_episodes': len(records), 'measured_source_episodes': len(measured),
            'confirmation_count': int(np.count_nonzero(means >= minimum)),
            'confirmation_rate_among_measured': float(np.mean(means >= minimum)) if len(means) else None,
            'confirmed_source_yield_in_planned_subset': float(np.count_nonzero(means >= minimum)/len(records)) if records else None,
            'unmeasured_reasons': {s: sum(r['status'] == s for r in records) for s in sorted({r['status'] for r in records if r['status'] != 'measured'})},
            'scope': 'Preselected supported four-corner native clearance-risk confirmations, separate from real-Q event estimates'}
