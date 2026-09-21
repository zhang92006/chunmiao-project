"""Audit online epsilon application and summarize ALL rollout outcomes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def summarize(root):
    root = Path(root)
    run = json.loads((root / 'manifest_run_summary.json').read_text(encoding='utf-8'))
    failures = []
    episodes = []
    status_counts = Counter()
    checked_terms = 0
    inferred = 0
    max_q_error = max_weight_error = max_obs_error = 0.0
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
        log_terms = data.get('log_probability_step_info', {})
        total_log_weight = sum(float(x['log_importance_weight']) for x in log_terms.values())
        if not math.isfinite(total_log_weight) or abs(total_log_weight-float(data['log_importance_weight'])) > 1e-7:
            failures.append(f'episode {episode_id}: inconsistent accumulated log weight')
        raw_weight = float(data['weight_episode'])
        if raw_weight > 0 and abs(math.log(raw_weight)-total_log_weight) > 1e-6:
            failures.append(f'episode {episode_id}: raw weight disagrees with logged probability product')
        if float(data.get('initial_weight', 1.0)) != 1.0:
            failures.append(f'episode {episode_id}: initial-state weight requires explicit handling')
        for time, step in data.get('online_policy_step_info', {}).items():
            status_counts[step['status']] += 1
            if step['status'] != 'inferred':
                continue
            inferred += 1
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
            for actor, terms in step.get('sampled_terms', {}).items():
                checked_terms += 1
                epsilon = terms['epsilon']
                q = epsilon * terms['p'] + (1-epsilon) * terms['c']
                max_q_error = max(max_q_error, abs(q-terms['q']))
                if terms['q'] <= 0:
                    failures.append(f'episode {episode_id}: sampled zero-probability action')
                    continue
                max_weight_error = max(max_weight_error, abs(terms['weight']-terms['p']/terms['q']))
                if abs(epsilon-step['epsilon_by_bv_id'][actor]) > 1e-7:
                    failures.append(f'episode {episode_id}: proposal overwrote policy epsilon')
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
    return {
        'attempted': run['attempted'], 'complete_episode_count': n,
        'raw_cav_crashes': len(collision), 'raw_cav_collision_rate': len(collision)/n if n else None,
        'conditional_weighted_cav_collision_mean': weighted_mean if not failures else None,
        'log_conditional_weighted_cav_collision_mean': weighted_log_mean if not failures else None,
        'crash_contribution_ess': ess if not failures else None,
        'online_step_status_counts': dict(status_counts), 'checked_sampled_bv_actions': checked_terms,
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
