"""Audit online epsilon application and summarize ALL rollout outcomes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys


def summarize(root):
    root = Path(root)
    run = json.loads((root / 'manifest_run_summary.json').read_text(encoding='utf-8'))
    failures = []
    episodes = []
    episode_rows = []
    status_counts = Counter()
    checked_terms = 0
    inferred = 0
    raw_weight_status_counts = Counter()
    max_q_error = max_weight_error = max_obs_error = 0.0
    intervention_budget = run.get('online_intervention_budget')
    likelihood_ratio_limit = run.get('online_max_proposal_likelihood_ratio')
    guarded_actor_ids = run.get('online_likelihood_ratio_guard_actor_ids')
    guarded_actor_set = None if guarded_actor_ids is None else set(guarded_actor_ids)
    likelihood_ratio_adjusted_steps = 0
    likelihood_ratio_adjusted_actors = Counter()
    for result in run['results']:
        episode_id = result['episode']
        if result['status'] != 'ok':
            failures.append(f'episode {episode_id}: simulation failed')
            continue
        files = [root / folder / f'{episode_id}.json' for folder in ('crash', 'tested_and_safe', 'rejected')]
        files = [path for path in files if path.is_file()]
        if len(files) != 1:
            failures.append(f'episode {episode_id}: expected exactly one outcome, found {len(files)}')
            continue
        data = json.loads(files[0].read_text(encoding='utf-8'))
        episodes.append(data)
        episode_rows.append((str(Path(result.get('template', ''))), data))
        log_terms = data.get('log_probability_step_info', {})
        total_log_weight = sum(float(x['log_importance_weight']) for x in log_terms.values())
        if not math.isfinite(total_log_weight) or abs(total_log_weight-float(data['log_importance_weight'])) > 1e-7:
            failures.append(f'episode {episode_id}: inconsistent accumulated log weight')
        raw_weight = float(data['weight_episode'])
        if total_log_weight >= math.log(sys.float_info.min):
            raw_weight_status_counts['normal_range'] += 1
            if raw_weight <= 0 or abs(math.log(raw_weight)-total_log_weight) > 1e-6:
                failures.append(f'episode {episode_id}: raw weight disagrees with logged probability product')
        elif raw_weight > 0:
            # Below the minimum normal float, exp/log is no longer invertible:
            # many exact values map to the same subnormal number. The per-step
            # log ledger remains authoritative and is checked independently.
            raw_weight_status_counts['subnormal_quantized'] += 1
        else:
            raw_weight_status_counts['underflowed_zero'] += 1
        if float(data.get('initial_weight', 1.0)) != 1.0:
            failures.append(f'episode {episode_id}: initial-state weight requires explicit handling')
        episode_inferred = 0
        for time, step in data.get('online_policy_step_info', {}).items():
            status_counts[step['status']] += 1
            if step['status'] == 'inferred':
                inferred += 1
                episode_inferred += 1
                if intervention_budget is not None and (
                    step.get('intervention_budget') != intervention_budget
                    or step.get('decisions_used_after') != step.get('decisions_used_before', 0) + 1
                    or step.get('decisions_used_after') > intervention_budget
                ):
                    failures.append(f'episode {episode_id} time {time}: invalid intervention budget transition')
                ids = data.get('controlled_bv_ids_step_info', {}).get(time)
                logged = data.get('real_epsilon_step_info', {}).get(time)
                if ids is not None:
                    if ids != step['actor_ids'] or logged is None or any(
                        abs(float(value) - step['epsilon_by_bv_id'][actor]) > 1e-7
                        for actor, value in zip(ids, logged)
                    ) or len(logged) != len(ids):
                        failures.append(f'episode {episode_id} time {time}: action/actor mismatch')
                    obs = data.get('drl_obs_step_info', {}).get(time, {}).get('joint')
                    if obs is not None:
                        max_obs_error = max(max_obs_error, max(abs(a-b) for a,b in zip(obs, step['observation'])))
                if likelihood_ratio_limit is not None:
                    requested = step.get('requested_epsilon_by_bv_id')
                    applied = step.get('epsilon_by_bv_id')
                    guard = step.get('likelihood_ratio_guard')
                    if (not isinstance(requested, dict) or not isinstance(applied, dict)
                            or set(requested) != set(step['actor_ids'])
                            or set(applied) != set(step['actor_ids'])
                            or not isinstance(guard, dict)
                            or abs(float(guard.get('maximum_proposal_ratio', math.nan))
                                   - float(likelihood_ratio_limit)) > 1e-7
                            or guard.get('guarded_actor_ids') != guarded_actor_ids):
                        failures.append(
                            f'episode {episode_id} time {time}: invalid likelihood-ratio guard metadata'
                        )
                    else:
                        adjusted_ids = set(guard.get('adjusted_actor_ids', []))
                        if any(float(applied[actor]) + 1e-12 < float(requested[actor])
                               for actor in step['actor_ids']):
                            failures.append(
                                f'episode {episode_id} time {time}: likelihood guard reduced epsilon'
                            )
                        if adjusted_ids:
                            likelihood_ratio_adjusted_steps += 1
                            likelihood_ratio_adjusted_actors.update(adjusted_ids)
                        by_actor = guard.get('by_actor', {})
                        if (not isinstance(by_actor, dict)
                                or not adjusted_ids.issubset(by_actor)
                                or (guarded_actor_set is not None
                                    and not set(by_actor).issubset(guarded_actor_set))):
                            failures.append(
                                f'episode {episode_id} time {time}: invalid guarded actor set'
                            )
                            by_actor = {}
                        for diagnostics in by_actor.values():
                            if (
                                float(diagnostics['applied_maximum_proposal_ratio'])
                                > float(likelihood_ratio_limit) * (1.0 + 1e-7)
                            ):
                                failures.append(
                                    f'episode {episode_id} time {time}: likelihood-ratio bound exceeded'
                                )
            elif step['status'] == 'budget_exhausted_naturalistic':
                epsilon_by_actor = step.get('epsilon_by_bv_id', {})
                if (intervention_budget is None
                        or step.get('intervention_budget') != intervention_budget
                        or step.get('decisions_used_before') != intervention_budget
                        or step.get('decisions_used_after') != intervention_budget
                        or set(epsilon_by_actor) != set(step['actor_ids'])
                        or any(abs(float(value)-1.0) > 1e-7 for value in epsilon_by_actor.values())):
                    failures.append(f'episode {episode_id} time {time}: invalid budget exhaustion fallback')
            sampled = step.get('sampled_terms', {})
            sampled_log_weight = 0.0
            for actor, terms in sampled.items():
                checked_terms += 1
                epsilon = terms['epsilon']
                q = epsilon * terms['p'] + (1-epsilon) * terms['c']
                max_q_error = max(max_q_error, abs(q-terms['q']))
                if terms['q'] <= 0 or terms['p'] <= 0:
                    failures.append(f'episode {episode_id}: sampled zero-probability action')
                    continue
                if (likelihood_ratio_limit is not None
                        and (guarded_actor_set is None or actor in guarded_actor_set)
                        and float(terms['q']) / float(terms['p'])
                        > float(likelihood_ratio_limit) * (1.0 + 1e-7)):
                    failures.append(
                        f'episode {episode_id} time {time}: sampled proposal ratio exceeded bound'
                    )
                max_weight_error = max(max_weight_error, abs(terms['weight']-terms['p']/terms['q']))
                sampled_log_weight += math.log(terms['p']) - math.log(terms['q'])
                if actor not in step.get('epsilon_by_bv_id', {}) or abs(
                    epsilon-step['epsilon_by_bv_id'][actor]
                ) > 1e-7:
                    failures.append(f'episode {episode_id}: proposal overwrote policy epsilon')
            if sampled:
                recorded = log_terms.get(time)
                if recorded is None:
                    failures.append(f'episode {episode_id} time {time}: sampled actions missing from probability ledger')
                elif abs(sampled_log_weight-float(recorded['log_importance_weight'])) > 1e-7:
                    failures.append(f'episode {episode_id} time {time}: sampled actions disagree with probability ledger')
        if intervention_budget is not None and episode_inferred > intervention_budget:
            failures.append(f'episode {episode_id}: intervention budget exceeded')
    if max_q_error > 1e-7 or max_weight_error > 1e-7 or max_obs_error > 1e-7:
        failures.append('Probability or observation contract exceeded tolerance')
    if run.get('online_policy') is not None and inferred == 0:
        failures.append('No online inference occurred; interface acceptance is inconclusive')
    collision = [d for d in episodes if d['collision_result'] and 'CAV' in (d.get('collision_id') or [])]
    crash_logs = [float(d['log_importance_weight']) for d in collision]
    n = len(episodes)
    weighted_mean = 0.0
    ess = 0.0
    if crash_logs:
        peak = max(crash_logs)
        scaled = [math.exp(value-peak) for value in crash_logs]
        weighted_log_mean = peak + math.log(sum(scaled)) - math.log(n)
        weighted_mean = math.exp(weighted_log_mean)
        ess = sum(scaled)**2 / sum(value**2 for value in scaled)
    else:
        weighted_log_mean = None
    stratified = run.get('stratified_allocation')
    stratified_summary = None
    if stratified is not None:
        planned = {
            str(Path(path)): int(count)
            for path, count in stratified.get('rollouts_by_template', {}).items()
        }
        grouped = {template: [] for template in planned}
        unknown_templates = Counter()
        for template, data in episode_rows:
            if template not in grouped:
                unknown_templates[template] += 1
                continue
            is_crash = bool(
                data['collision_result'] and 'CAV' in (data.get('collision_id') or [])
            )
            grouped[template].append(
                math.exp(float(data['log_importance_weight'])) if is_crash else 0.0
            )
        if unknown_templates:
            failures.append(f'Unplanned templates in stratified run: {dict(unknown_templates)}')
        actual = {template: len(values) for template, values in grouped.items()}
        if actual != planned:
            failures.append(
                f'Stratified rollout counts disagree with plan: actual={actual}, planned={planned}'
            )
        if grouped:
            stratum_means = {
                template: math.fsum(values) / len(values) if values else 0.0
                for template, values in grouped.items()
            }
            template_count = len(grouped)
            stratified_mean = math.fsum(stratum_means.values()) / template_count
            variance_terms = {}
            estimator_contributions = []
            raw_rates = {}
            for template, values in grouped.items():
                sample_count = len(values)
                raw_rates[template] = (
                    sum(value > 0.0 for value in values) / sample_count
                    if sample_count else None
                )
                if sample_count > 1:
                    center = stratum_means[template]
                    sample_variance = math.fsum(
                        (value - center) ** 2 for value in values
                    ) / (sample_count - 1)
                    variance_terms[template] = sample_variance / sample_count
                else:
                    variance_terms[template] = None
                if sample_count:
                    estimator_contributions.extend(
                        value / (template_count * sample_count) for value in values
                    )
            variance_estimable = all(value is not None for value in variance_terms.values())
            stratified_variance = (
                math.fsum(variance_terms.values()) / (template_count ** 2)
                if variance_estimable else None
            )
            contribution_sum = math.fsum(estimator_contributions)
            contribution_square_sum = math.fsum(
                value * value for value in estimator_contributions
            )
            stratified_ess = (
                contribution_sum ** 2 / contribution_square_sum
                if contribution_square_sum > 0.0 else 0.0
            )
            standard_error = (
                math.sqrt(max(0.0, stratified_variance))
                if stratified_variance is not None else None
            )
            stratified_summary = {
                'estimator': 'equal-template stratified mean',
                'template_count': template_count,
                'rollouts_by_template': actual,
                'raw_cav_collision_mean_across_templates': (
                    math.fsum(raw_rates.values()) / template_count
                    if all(value is not None for value in raw_rates.values()) else None
                ),
                'weighted_cav_collision_mean': stratified_mean,
                'log_weighted_cav_collision_mean': (
                    math.log(stratified_mean) if stratified_mean > 0.0 else None
                ),
                'estimated_variance': stratified_variance,
                'estimated_standard_error': standard_error,
                'estimated_95pct_relative_half_width': (
                    1.96 * standard_error / stratified_mean
                    if standard_error is not None and stratified_mean > 0.0 else None
                ),
                'crash_contribution_ess': stratified_ess,
                'per_template_weighted_means': stratum_means,
            }
            weighted_mean = stratified_mean
            weighted_log_mean = (
                math.log(stratified_mean) if stratified_mean > 0.0 else None
            )
            ess = stratified_ess
    return {
        'attempted': run['attempted'], 'complete_episode_count': n,
        'raw_cav_crashes': len(collision), 'raw_cav_collision_rate': len(collision)/n if n else None,
        'conditional_weighted_cav_collision_mean': weighted_mean if not failures else None,
        'log_conditional_weighted_cav_collision_mean': weighted_log_mean if not failures else None,
        'crash_contribution_ess': ess if not failures else None,
        'stratified_estimation': stratified_summary if not failures else None,
        'online_step_status_counts': dict(status_counts), 'checked_sampled_bv_actions': checked_terms,
        'online_intervention_budget': intervention_budget,
        'online_max_proposal_likelihood_ratio': likelihood_ratio_limit,
        'online_likelihood_ratio_guard_actor_ids': guarded_actor_ids,
        'likelihood_ratio_adjusted_steps': likelihood_ratio_adjusted_steps,
        'likelihood_ratio_adjusted_actor_counts': dict(likelihood_ratio_adjusted_actors),
        'raw_weight_status_counts': dict(raw_weight_status_counts),
        'max_q_reconstruction_error': max_q_error, 'max_weight_reconstruction_error': max_weight_error,
        'max_online_vs_logged_observation_error': max_obs_error,
        'audit_passed': not failures, 'failures': failures,
        'scope': 'Conditional on this fixed scenario/template mixture; not a real-world SHRP2 crash probability. Zero crashes is not evidence of zero risk.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_path')
    args = parser.parse_args()
    result = summarize(args.experiment_path)
    (Path(args.experiment_path) / 'closed_loop_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['audit_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
