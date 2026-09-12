"""Evidence-preserving trajectory screening, NOT a SUMO/Autoware planner benchmark.

Backgrounds follow measured highD paths; after an observed prefix, the CAV uses
lane-fixed longitudinal IDM. Axis-aligned boxes approximate vehicle footprints.
No interaction model, tyre dynamics, calibrated likelihood or global optimum is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time as clock

import numpy as np
import scipy
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline, PchipInterpolator

from .highd import file_hash, write_json


def validate_benchmark_config(config):
    for name in ("budget_per_search", "completion_samples"):
        if isinstance(config[name], bool) or int(config[name]) != config[name] or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must lie in [0, 2**32)")
    positive = ("max_longitudinal_offset_m", "max_lateral_offset_m", "max_speed_mps",
                "max_abs_longitudinal_accel_mps2", "max_abs_lateral_accel_mps2", "max_abs_jerk_mps3")
    for name in positive:
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("observed_prefix_s", "intervention_penalty"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not all(np.isfinite(v) and v > 0 for v in config["idm"].values()):
        raise ValueError("IDM parameters must be finite and positive")
    gap, spread = np.asarray(config["missing_interval_s"]), np.asarray(config["completion_spread_m"])
    if gap.shape != (2,) or not np.isfinite(gap).all() or not 0 <= gap[0] < gap[1]:
        raise ValueError("Missing interval must be a finite increasing pair")
    if spread.shape != (2,) or not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError("Completion spreads must be a finite positive pair")


def validate_scene(scene):
    time = np.asarray(scene["time"], dtype=float)
    if time.ndim != 1 or len(time) < 4 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Scene must have finite, increasing sample times")
    actors = scene["actors"]
    if len(actors) < 2 or sum(actor["role"] == "CAV" for actor in actors) != 1 or actors[0]["role"] != "CAV":
        raise ValueError("Scene requires a unique first CAV and at least one BV")
    ids = [actor["id"] for actor in actors]
    if len(set(ids)) != len(ids) or scene["changer_id"] not in ids[1:]:
        raise ValueError("Invalid actor IDs or lane changer")
    for actor in actors:
        for field in ("xy", "velocity", "acceleration"):
            values = np.asarray(actor[field], dtype=float)
            if values.shape != (len(time), 2) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {field} trajectory for {actor['id']}")
        if not all(np.isfinite(actor[key]) and actor[key] > 0 for key in ("length", "width")):
            raise ValueError("Vehicle dimensions must be finite and positive")
    boundaries = np.asarray(scene["road_boundaries_y"])
    if len(boundaries) < 2 or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0):
        raise ValueError("Invalid road boundaries")


def deformation(time, prefix, params):
    """C2 join: prefix positions, velocities and accelerations are unchanged."""
    time = np.asarray(time)
    duration = time[-1] - prefix
    if duration <= 0 or prefix < time[0]:
        raise ValueError("Observed prefix must lie inside the scene")
    u = np.clip((time - prefix) / duration, 0, 1)
    smoothstep = Polynomial([0, 0, 0, 10, -15, 6])
    params = np.asarray(params, dtype=float)
    if params.shape != (2,) or not np.isfinite(params).all():
        raise ValueError("Expected finite longitudinal/lateral offsets")
    return tuple(smoothstep.deriv(order)(u)[:, None] * params / duration**order for order in range(3))


def deform_scene(scene, params, config):
    xy = np.array([actor["xy"] for actor in scene["actors"]], dtype=float)
    velocity = np.array([actor["velocity"] for actor in scene["actors"]], dtype=float)
    acceleration = np.array([actor["acceleration"] for actor in scene["actors"]], dtype=float)
    index = next(i for i, actor in enumerate(scene["actors"]) if actor["id"] == scene["changer_id"])
    delta = deformation(scene["time"], config["observed_prefix_s"], params)
    xy[index] += delta[0]
    velocity[index] += delta[1]
    acceleration[index] += delta[2]
    return xy, velocity, acceleration


def box_clearance(first, second, first_size, second_size):
    """Signed box separation: negative means overlap; nonnegative is distance."""
    gap = np.abs(np.asarray(first) - np.asarray(second)) - (np.asarray(first_size) + second_size) / 2
    return np.linalg.norm(np.maximum(gap, 0), axis=-1) + np.minimum(np.max(gap, axis=-1), 0)


def rollout_idm(scene, xy, velocity, config):
    """Only CAV reacts; BV replay is open-loop. Preserve ALL observed prefix states."""
    xy, velocity = xy.copy(), velocity.copy()
    time = np.asarray(scene["time"])
    actors, idm = scene["actors"], config["idm"]
    lengths = np.array([actor["length"] for actor in actors])
    widths = np.array([actor["width"] for actor in actors])
    controls = []
    for i in range(1, len(time)):
        if time[i] <= config["observed_prefix_s"] + 1e-9:
            continue
        dt = time[i] - time[i - 1]
        speed = max(0.0, velocity[0, i - 1, 0])
        ego_xy = xy[0, i - 1]
        gaps = xy[1:, i - 1, 0] - ego_xy[0] - (lengths[1:] + lengths[0]) / 2
        overlap_y = np.abs(xy[1:, i - 1, 1] - ego_xy[1]) < (widths[1:] + widths[0]) / 2
        ahead = xy[1:, i - 1, 0] > ego_xy[0]
        candidates = np.where(overlap_y & ahead)[0]
        accel = idm["accel_mps2"] * (1 - (speed / idm["desired_speed_mps"])**4)
        if len(candidates):
            j = candidates[np.argmin(gaps[candidates])]
            desired_gap = idm["min_gap_m"] + max(0, speed * idm["time_headway_s"] +
                speed * (speed - velocity[j + 1, i - 1, 0]) / (2 * np.sqrt(idm["accel_mps2"] * idm["brake_mps2"])))
            accel -= idm["accel_mps2"] * (desired_gap / max(gaps[j], 0.1))**2
        accel = float(np.clip(accel, -idm["brake_mps2"], idm["accel_mps2"]))
        # Integrate only until the stopping instant, so braking cannot move backwards.
        active_dt = min(dt, speed / -accel) if accel < 0 else dt
        xy[0, i] = ego_xy + [speed * active_dt + 0.5 * accel * active_dt**2, 0]
        velocity[0, i] = [max(0, speed + accel * dt), 0]
        controls.append(accel)
    jerk = np.diff(controls) / np.median(np.diff(time)) if len(controls) > 1 else np.array([0.0])
    return xy, velocity, float(np.sqrt(np.mean(jerk**2)))


def physical_checks(scene, xy, velocity, acceleration, config):
    """Screen BVs with native highD filtered v/a plus analytic perturbation derivatives.

    This is NOT a dynamics proof: measured x/v/a have a consistency residual.
    CAV response is separately measured; rejecting it would hide planner failures.
    """
    actors = scene["actors"]
    widths = np.array([actor["width"] for actor in actors[1:]])[:, None]
    road = scene["road_boundaries_y"]
    jerk = np.gradient(acceleration[1:], np.asarray(scene["time"]), axis=1)
    checks = {
        "negative_or_excess_speed": bool(np.any(velocity[1:, :, 0] < 0) or np.any(np.linalg.norm(velocity[1:], axis=-1) > config["max_speed_mps"])),
        "longitudinal_acceleration": bool(np.any(np.abs(acceleration[1:, :, 0]) > config["max_abs_longitudinal_accel_mps2"])),
        "lateral_acceleration": bool(np.any(np.abs(acceleration[1:, :, 1]) > config["max_abs_lateral_accel_mps2"])),
        "jerk": bool(np.any(np.abs(jerk) > config["max_abs_jerk_mps3"])),
        "offroad": bool(np.any(xy[1:, :, 1] - widths / 2 < road[0]) or np.any(xy[1:, :, 1] + widths / 2 > road[-1])),
        "initial_overlap": False,
        "background_collision": False,
    }
    for i, first in enumerate(actors):
        for j in range(i + 1, len(actors)):
            separation = box_clearance(xy[i], xy[j], [first["length"], first["width"]],
                                       np.array([actors[j]["length"], actors[j]["width"]]))
            checks["initial_overlap"] |= bool(separation[0] <= 0)
            if i > 0:
                checks["background_collision"] |= bool(np.any(separation <= 0))
    return [name for name, failed in checks.items() if failed]


def risk_metrics(scene, xy, velocity):
    ego = scene["actors"][0]
    clearances, ttcs = [], []
    for j, actor in enumerate(scene["actors"][1:], 1):
        clearances.append(box_clearance(xy[0], xy[j], [ego["length"], ego["width"]],
                                       np.array([actor["length"], actor["width"]])))
        gap = xy[j, :, 0] - xy[0, :, 0] - (ego["length"] + actor["length"]) / 2
        closing = velocity[0, :, 0] - velocity[j, :, 0]
        overlap_y = np.abs(xy[j, :, 1] - xy[0, :, 1]) < (ego["width"] + actor["width"]) / 2
        valid = (gap > 0) & (closing > 1e-6) & overlap_y
        ttcs.extend((gap[valid] / closing[valid]).tolist())
    per_time = np.min(clearances, axis=0)
    collisions = np.where(per_time <= 0)[0]
    return {"collision": bool(len(collisions)), "min_clearance_m": float(np.min(per_time)),
            "min_ttc_s": float(min(ttcs)) if ttcs else None,
            "first_collision_time_s": scene["time"][int(collisions[0])] if len(collisions) else None}


def evaluate(scene, params, config):
    xy, velocity, acceleration = deform_scene(scene, params, config)
    violations = physical_checks(scene, xy, velocity, acceleration, config)
    replay_xy = xy.copy()
    xy, velocity, comfort = rollout_idm(scene, xy, velocity, config)
    metrics = risk_metrics(scene, xy, velocity)
    original = np.asarray([actor["xy"] for actor in scene["actors"]])
    metrics.update(params=list(map(float, params)), feasible=not violations, violations=violations,
                   intervention_l2_m=float(np.linalg.norm(params)),
                   background_ade_m=float(np.mean(np.linalg.norm(replay_xy[1:] - original[1:], axis=-1))),
                   cav_jerk_rms_mps3=comfort)
    return metrics


def ranking(result, config):
    if result["collision"]:
        return (0, result["intervention_l2_m"])
    return (1, result["min_clearance_m"] + config["intervention_penalty"] * result["intervention_l2_m"])


def search(scene, method, config, seed):
    """Same bounded parameter space and max evaluation budget; no hidden GT scoring."""
    budget = int(config["budget_per_search"])
    bounds = np.array([config["max_longitudinal_offset_m"], config["max_lateral_offset_m"]])
    if budget < 1 or np.any(bounds <= 0) or not np.isfinite(bounds).all():
        raise ValueError("Search budget and offset bounds must be positive")
    started, trials, best = clock.perf_counter(), [], None

    def trial(params):
        nonlocal best
        result = evaluate(scene, params, config)
        trials.append(result)
        if result["feasible"] and (best is None or ranking(result, config) < ranking(best, config)):
            best = result

    trial(np.zeros(2))  # Common replay reference; no candidate is forced to be dangerous.
    if method == "uniform":
        for params in np.random.default_rng(seed).uniform(-bounds, bounds, size=(budget - 1, 2)):
            trial(params)
    elif method == "constrained_search":
        step, anchor, seen = bounds / 2, np.zeros(2), {(0.0, 0.0)}
        while len(trials) < budget and np.max(step / bounds) > 1e-5:
            old_best = best
            for dimension, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
                params = anchor.copy()
                params[dimension] += sign * step[dimension]
                params = np.clip(params, -bounds, bounds)
                key = tuple(np.round(params, 10))
                if key in seen:
                    continue
                seen.add(key)
                trial(params)
                if len(trials) >= budget:
                    break
            if best is old_best:
                step /= 2
            if best is not None:
                anchor = np.array(best["params"])
    elif method != "replay":
        raise ValueError(f"Unknown baseline: {method}")
    return {"method": method, "seed": seed, "evaluations": len(trials),
            "feasible_candidates": sum(item["feasible"] for item in trials),
            "violation_counts": dict(Counter(reason for item in trials for reason in item["violations"])),
            "selected": best, "elapsed_s": clock.perf_counter() - started,
            "trials": trials}


def reconstruct(time, observed_xy, method):
    """The function receives NaNs instead of hidden truth. Interior gaps only."""
    time, observed_xy = np.asarray(time), np.asarray(observed_xy, dtype=float)
    observed = np.isfinite(observed_xy).all(axis=1)
    if not observed[0] or not observed[-1] or observed.sum() < 4:
        raise ValueError("Need at least four observations and both endpoints; no extrapolation")
    if method == "linear":
        result = np.column_stack([np.interp(time, time[observed], observed_xy[observed, d]) for d in range(2)])
    elif method == "cubic":
        result = CubicSpline(time[observed], observed_xy[observed], bc_type="natural")(time)
    elif method == "pchip":
        result = PchipInterpolator(time[observed], observed_xy[observed])(time)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
    result[observed] = observed_xy[observed]
    return result


def reconstruction_pilot(scene, config, seed):
    time = np.asarray(scene["time"])
    actor = next(actor for actor in scene["actors"] if actor["id"] == scene["changer_id"])
    truth = np.asarray(actor["xy"])
    low, high = config["missing_interval_s"]
    hidden = (time >= low) & (time <= high)
    if not hidden.any() or hidden[0] or hidden[-1]:
        raise ValueError("Missing interval must hide an interior part of the trajectory")
    evidence = truth.copy()
    evidence[hidden] = np.nan
    results, completions = {}, {}
    for method in ("linear", "cubic", "pchip"):
        completion = reconstruct(time, evidence, method)
        error = np.linalg.norm(completion[hidden] - truth[hidden], axis=-1)
        results[method] = {"hidden_ade_m": float(np.mean(error)), "hidden_last_error_m": float(error[-1]),
                           "observed_max_error_m": float(np.max(np.abs(completion[~hidden] - truth[~hidden])))}
        completions[method] = completion
    # An uncalibrated sensitivity envelope, NOT posterior uncertainty or a new method.
    left = time[np.where(hidden)[0][0] - 1]
    right = time[np.where(hidden)[0][-1] + 1]
    u = np.clip((time - left) / (right - left), 0, 1)
    bump = 64 * u**3 * (1 - u)**3
    bump[~hidden] = 0
    offsets = np.random.default_rng(seed).normal(size=(config["completion_samples"], 2)) * config["completion_spread_m"]
    samples = completions["cubic"][None] + offsets[:, None, :] * bump[None, :, None]
    q_low, q_high = np.quantile(samples[:, hidden], [0.05, 0.95], axis=0)
    covered = np.all((truth[hidden] >= q_low) & (truth[hidden] <= q_high), axis=1)
    results["uncalibrated_envelope"] = {"joint_pointwise_coverage": float(np.mean(covered)),
        "mean_width_xy_m": np.mean(q_high - q_low, axis=0).tolist(),
        "warning": "Hand-chosen Gaussian offsets; no physical filtering or calibrated posterior claim"}
    return results


def run_benchmark(manifest_path, output, config, split="validation"):
    validate_benchmark_config(config)
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from .highd import validate_config
    validate_config(manifest["config"])
    records = [item for item in manifest["scenes"] if item["split"] == split]
    if not records:
        raise ValueError(f"No scenes in split {split}")
    results = []
    for item in records:
        path = (manifest_path.parent / item["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Scene path escapes dataset directory")
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Scene checksum mismatch: {item['scene_id']}")
        scene = json.loads(path.read_text(encoding="utf-8"))
        validate_scene(scene)
        allowed_recordings = {f"{int(rec):02d}" for rec in manifest["config"]["splits"][split]}
        if (scene["split"] != split or scene["recording_id"] not in allowed_recordings
                or scene["scene_id"] != item["scene_id"] or scene["recording_id"] != item["recording_id"]):
            raise ValueError("Scene split disagrees with recording manifest")
        seed = (config["seed"] + int(hashlib.sha256(scene["scene_id"].encode()).hexdigest()[:8], 16)) % 2**32
        result = {"scene_id": scene["scene_id"], "recording_id": scene["recording_id"], "split": split}
        result["baselines"] = [search(scene, method, config, seed) for method in ("replay", "uniform", "constrained_search")]
        result["reconstruction"] = reconstruction_pilot(scene, config, seed)
        xy, velocity = np.asarray([a["xy"] for a in scene["actors"]]), np.asarray([a["velocity"] for a in scene["actors"]])
        residual = np.gradient(xy, np.asarray(scene["time"]), axis=1) - velocity
        result["source_velocity_consistency_rmse_mps"] = float(np.sqrt(np.mean(residual**2)))
        results.append(result)
        print(f"{len(results)}/{len(records)}: {scene['scene_id']}", flush=True)
    summary = {"split": split, "scene_count": len(results), "recording_count": len({r["recording_id"] for r in results}),
               "config": config, "manifest_sha256": file_hash(manifest_path),
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "implementation_sha256": file_hash(__file__),
               "scope": "Kinematic screening: open-loop BVs, lane-fixed IDM CAV, sampled axis-aligned box collisions",
               "baselines": {}, "reconstruction": {}}
    for method in ("replay", "uniform", "constrained_search"):
        runs = [next(b for b in result["baselines"] if b["method"] == method) for result in results]
        chosen = [run["selected"] for run in runs if run["selected"] is not None]
        summary["baselines"][method] = {
            "attempted_scenes": len(runs), "scenes_with_feasible_candidate": len(chosen),
            "collision_scenes": sum(bool(item["collision"]) for item in chosen),
            "candidate_evaluations": sum(run["evaluations"] for run in runs),
            "feasible_candidates": sum(run["feasible_candidates"] for run in runs),
            "mean_selected_clearance_m": float(np.mean([item["min_clearance_m"] for item in chosen])) if chosen else None,
            "mean_selected_intervention_l2_m": float(np.mean([item["intervention_l2_m"] for item in chosen])) if chosen else None,
            "elapsed_s": sum(run["elapsed_s"] for run in runs),
            "violation_counts": dict(sum((Counter(run["violation_counts"]) for run in runs), Counter())),
        }
    for method in ("linear", "cubic", "pchip"):
        summary["reconstruction"][method] = {"mean_hidden_ade_m": float(np.mean([r["reconstruction"][method]["hidden_ade_m"] for r in results]))}
    envelopes = [r["reconstruction"]["uncalibrated_envelope"] for r in results]
    summary["reconstruction"]["uncalibrated_envelope"] = {
        "mean_joint_pointwise_coverage": float(np.mean([r["joint_pointwise_coverage"] for r in envelopes])),
        "mean_width_xy_m": np.mean([r["mean_width_xy_m"] for r in envelopes], axis=0).tolist(),
        "warning": "Sensitivity diagnostic only; not a calibrated confidence interval"}
    summary["source_velocity_consistency_rmse_mps_mean"] = float(np.mean([r["source_velocity_consistency_rmse_mps"] for r in results]))
    write_json(output / "results.json", results)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/trajectory_baselines.json")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()
    summary = run_benchmark(args.manifest, args.output,
                            json.loads(Path(args.config).read_text(encoding="utf-8")), args.split)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
