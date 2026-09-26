"""Read-only checkpoint diagnosis on its training pool, not a policy update.

Compare actual deterministic epsilons with constant controls and the legacy
Beta-mean representable range. No SUMO rollout, NDD changes, or label changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from d2rl_training.highd_event_contract import event_indicator
from .highd_sequence_io import digest, read_json, write_new


def beta_mean_bounds(low, high, small=1e-6):
    cmin, cmax = 1 + np.log1p(small), 1 + np.log1p(1 / small)
    return np.asarray([low + (high - low) * cmin / (cmin + cmax),
                       low + (high - low) * cmax / (cmin + cmax)])


def loss_at_actions(sequence, actions, log_scale):
    """Vectorized copy of the already audited conditional-chain replay math."""
    actions = np.asarray(actions, dtype=float).reshape(-1, 2)
    if len(actions) != len(sequence["steps"]) or not np.isfinite(actions).all() or (actions <= 0).any() or (actions > 1).any():
        raise ValueError("One finite bounded epsilon pair per retained state")
    if not event_indicator(sequence):
        return 0.
    p = np.asarray([s["ndd_record"]["per_agent"] for s in sequence["steps"]]).reshape(-1, 2)
    h = np.asarray([s["weight_record"]["per_agent_critical_probability"] for s in sequence["steps"]]).reshape(-1, 2)
    q = actions * p + (1 - actions) * h
    log_weight = float(np.sum(np.log(p) - np.log(q)))
    return float(np.exp(sequence["generation_log_weight"] + log_weight - log_scale))


def action_stats(values):
    values = np.asarray(values).reshape(-1, 2)
    if not len(values):
        return None
    return {"decision_count": len(values), "minimum": values.min(0).tolist(),
            "median": np.median(values, axis=0).tolist(), "maximum": values.max(0).tolist(),
            "mean": values.mean(0).tolist()}


def pool_diagnostic(env, policy, bounds, policy_bounds=None):
    sequences = env.sequences
    actions = [[np.asarray(policy(s["observation"]), dtype=float).tolist() for s in seq["steps"]] for seq in sequences]
    losses = np.asarray([loss_at_actions(s, a, env.log_scale) for s, a in zip(sequences, actions)])
    baseline_losses = np.asarray([loss_at_actions(s, [[.1, .1]] * len(s["steps"]), env.log_scale) for s in sequences])
    by_source, rows = defaultdict(list), []
    for seq, eps, trained, baseline in zip(sequences, actions, losses, baseline_losses):
        by_source[seq["source_scenario_id"]].extend(eps)
        if event_indicator(seq):
            rows.append({"scenario": seq["source_scenario_id"], "seed": seq["seed"],
                         "critical_steps": len(seq["steps"]), "epsilon": action_stats(eps),
                         "fixed_loss": float(baseline), "trained_loss": float(trained),
                         "trained_to_fixed_ratio": float(trained / baseline) if baseline > 0 else None})
    policy_bounds = bounds if policy_bounds is None else policy_bounds
    grid = []
    grid_values = sorted(set((.05, .1, float(bounds[0]), .2, .4, .6, .8, float(bounds[1]), 1.,
                              float(policy_bounds[0]), float(policy_bounds[1]))))
    for first in grid_values:
        for second in grid_values:
            objective = float(np.mean([loss_at_actions(s, [[first, second]] * len(s["steps"]), env.log_scale) for s in sequences]))
            grid.append({"epsilon": [first, second], "empirical_scaled_second_moment": objective,
                         "representable_by_legacy_deterministic_mean": bool(bounds[0] <= first <= bounds[1] and bounds[0] <= second <= bounds[1]),
                         "representable_by_policy_deterministic_mean": bool(policy_bounds[0] <= first <= policy_bounds[1] and policy_bounds[0] <= second <= policy_bounds[1])})
    grid.sort(key=lambda r: r["empirical_scaled_second_moment"])
    return {"restored_empirical_scaled_second_moment": float(losses.mean()),
            "fixed_empirical_scaled_second_moment": float(baseline_losses.mean()),
            "actual_epsilon_all_steps": action_stats([a for seq in actions for a in seq]),
            "actual_epsilon_by_source": {s: action_stats(a) for s, a in by_source.items()},
            "event_sequence_losses": sorted(rows, key=lambda r: -r["fixed_loss"]),
            "constant_grid": grid,
            "best_training_grid_constant": grid[0],
            "best_representable_training_grid_constant": next(r for r in grid if r["representable_by_legacy_deterministic_mean"]),
            "best_current_policy_training_grid_constant": next(r for r in grid if r["representable_by_policy_deterministic_mean"]),
            "scope": "Training-pool diagnostic only. Grid minima are not held-out results or deployed policies."}


def diagnose(training_root, output):
    from d2rl_training.highd_critical_sequence_env import HighDCriticalSequenceEnv
    from .highd_mean_precision_policy import register_legacy_action_distribution
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.utils.numpy import SMALL_NUMBER
    from ray.tune.registry import register_env

    root, output = Path(training_root), Path(output)
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    protocol, summary = read_json(root / "training_protocol.json"), read_json(root / "training_summary.json")
    config = protocol["ppo_config"]
    distribution_name = config["model"]["custom_action_dist"]
    if distribution_name not in ("d2rl_bounded_beta", "highd_mean_precision_beta_v1") or not config["normalize_actions"]:
        raise ValueError("Unsupported normalized two-epsilon policy")
    if digest(config["env_config"]["sequence_manifest"]) != protocol["sequence_manifest_sha256"]:
        raise ValueError("Training data changed")
    env = HighDCriticalSequenceEnv(config["env_config"])
    bounds = beta_mean_bounds(float(env.action_space.low[0]), float(env.action_space.high[0]), SMALL_NUMBER)
    policy_bounds = bounds
    policy_code = Path(__file__).with_name("highd_mean_precision_policy.py")
    if distribution_name == "highd_mean_precision_beta_v1":
        from .highd_mean_precision_policy import register_mean_precision_policy, MEAN_MARGIN
        register_mean_precision_policy()
        low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
        policy_bounds = np.asarray([low + (high - low) * MEAN_MARGIN, high - (high - low) * MEAN_MARGIN])
        policy_code = Path(__file__).with_name("highd_mean_precision_policy.py")
    if "policy_code_sha256" in protocol and digest(policy_code) != protocol["policy_code_sha256"]:
        raise ValueError("Policy implementation changed since training")
    register_env(config["env"], lambda cfg: HighDCriticalSequenceEnv(cfg))
    register_legacy_action_distribution()
    trainer = None
    ray.init(num_cpus=1, num_gpus=0, include_dashboard=False, ignore_reinit_error=True)
    try:
        trainer = PPOTrainer(config=config)
        trainer.restore(summary["checkpoint"])
        report = pool_diagnostic(env, lambda obs: trainer.compute_single_action(obs, explore=False), bounds, policy_bounds)
        if not np.isclose(report["restored_empirical_scaled_second_moment"], summary["trained_deterministic_empirical_scaled_second_moment"], atol=1e-10, rtol=1e-7):
            raise ValueError("Restored checkpoint or fast replay differs from original training evaluation")
        report.update(checkpoint_sha256=digest(summary["checkpoint"]),
                      sequence_manifest_sha256=protocol["sequence_manifest_sha256"],
                      natural_target_sha256=protocol["natural_target_sha256"],
                      experiment_target_sha256=protocol["experiment_target_sha256"],
                      legacy_beta_deterministic_epsilon_bounds=bounds.tolist(),
                      policy_deterministic_epsilon_bounds=policy_bounds.tolist(),
                      policy_distribution=distribution_name,
                      baseline_epsilon_representable=bool(policy_bounds[0] <= .1 <= policy_bounds[1]),
                      checkpoint_updated=False, ndd_changed=False, formal_performance_evidence=False)
        # Convert NumPy scalar booleans from grid comparisons to JSON bools.
        for row in report["constant_grid"]:
            row["representable_by_legacy_deterministic_mean"] = bool(row["representable_by_legacy_deterministic_mean"])
        write_new(output / "policy_diagnostic.json", report)
        return report
    finally:
        if trainer is not None:
            trainer.stop()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args.training_root, args.output)
    keys = ("restored_empirical_scaled_second_moment", "fixed_empirical_scaled_second_moment",
            "legacy_beta_deterministic_epsilon_bounds", "actual_epsilon_all_steps", "best_training_grid_constant",
            "best_representable_training_grid_constant", "best_current_policy_training_grid_constant", "scope")
    print(json.dumps({k: result[k] for k in keys}, indent=2), flush=True)


if __name__ == "__main__":
    main()
