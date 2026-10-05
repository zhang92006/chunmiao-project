"""Conditional outcome gate and paired empirical shape/time transition kernel.

No safety mask. Censored observations contribute survival, not outcome labels.
Observed donors retain peak, endpoint, time and peak timing together. The kernel
is shared by P and Q, conditional on the actual sampled longitudinal action.
This is an initiation-conditioned candidate, NOT online lateral replanning.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, log_ndtr, ndtri

from .highd_lane_motion import OUTCOMES, category


FEATURES = ("speed", "own_front_gap", "own_front_rr", "previous_acceleration",
            "target_rear_present", "target_rear_gap", "target_rear_rr", "target_rear_acceleration",
            "target_front_present", "target_front_gap", "target_front_rr",
            "target_alongside_present", "target_alongside_dx", "target_alongside_rr", "command_acceleration")
KERNEL = "conditional_paired_motion_v1"


def context_features(state, direction, acceleration):
    state = np.asarray(state, dtype=float)
    if state.shape != (31,) or direction not in (1, 2) or not np.isfinite(acceleration):
        raise ValueError("Motion requires geometry-onset state and an actual lateral/longitudinal action")
    side = direction - 1
    value = np.r_[state[:4], state[8 + 4 * side:12 + 4 * side], state[19 + 6 * side:25 + 6 * side], acceleration]
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite motion context")
    return value


def _standardize(context, model):
    return (np.atleast_2d(context) - model["mean"]) / model["scale"]


def _donor_weights(z, model, code, excluded_keys=None):
    donors = model["donors"][code]
    dz = _standardize(np.array([d["motion_context"] for d in donors]), model)
    distance = np.mean((z[:, None, :] - dz[None, :, :]) ** 2, axis=2)
    logw = -distance / (2 * model["bandwidth"] ** 2)
    mask = np.ones_like(logw, dtype=bool)
    if excluded_keys is not None:
        # All starts from the same vehicle track are excluded in training-score
        # density estimates, not merely the identical donor row.
        mask = np.array([[key != (d["recording"], d["vehicle_id"]) for d in donors] for key in excluded_keys])
        if not mask.any(axis=1).all():
            raise ValueError("Need donors from more than one vehicle per outcome")
    logw = np.where(mask, logw, -np.inf)
    weights = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
    # Declared smoothing over training donors, not a post-draw action rejection.
    prior = mask / mask.sum(axis=1, keepdims=True)
    return (1 - model["donor_prior_mass"]) * weights + model["donor_prior_mass"] * prior


def _emission(events, model, exclude_tracks=False):
    z = _standardize([e["motion_context"] for e in events], model)
    codes = np.array([category(e["status"]) for e in events])
    times = np.array([e["duration_s"] for e in events])
    if (times <= 0).any():
        raise ValueError("Positive event time required")
    excluded = [(e["recording"], e["vehicle_id"]) for e in events] if exclude_tracks else None
    columns = []
    sigma = model["time_log_sigma"]
    for k in range(3):
        w = _donor_weights(z, model, k, excluded)
        centers = np.log([d["duration_s"] for d in model["donors"][k]])
        residual = (np.log(times)[:, None] - centers) / sigma
        logpdf = -.5 * residual ** 2 - np.log(times[:, None] * sigma) - .5 * np.log(2 * np.pi)
        with np.errstate(divide="ignore"):
            logw = np.log(w)
        density = logsumexp(logw + logpdf, axis=1)
        survival = logsumexp(logw + log_ndtr(-residual), axis=1)
        columns.append(np.where(codes < 0, survival, np.where(codes == k, density, -np.inf)))
    return np.column_stack(columns)


def fit(events, config):
    context = np.asarray([e["motion_context"] for e in events])
    if context.ndim != 2 or context.shape[1] != len(FEATURES) or not np.isfinite(context).all():
        raise ValueError("Missing causal motion contexts")
    mean, scale = context.mean(axis=0), context.std(axis=0)
    scale[scale < 1e-6] = 1.
    donors = [[e for e in events if category(e["status"]) == k] for k in range(3)]
    if any(len(d) < 2 for d in donors):
        raise ValueError("Insufficient observed outcomes for a three-outcome process")
    model = {"schema_version": 2, "kernel": KERNEL, "feature_names": list(FEATURES),
             "mean": mean.tolist(), "scale": scale.tolist(), "donors": donors,
             "bandwidth": config["bandwidth"], "donor_prior_mass": config["donor_prior_mass"],
             "time_log_sigma": config["time_log_sigma"]}
    if model["bandwidth"] <= 0 or model["time_log_sigma"] <= 0 or not 0 < model["donor_prior_mass"] < 1:
        raise ValueError("Invalid declared kernel regularization")
    x = np.column_stack([np.ones(len(events)), _standardize(context, model)])
    emission = _emission(events, model, exclude_tracks=True)
    def objective(flat):
        beta = flat.reshape(x.shape[1], 2)
        logits = np.column_stack([np.zeros(len(x)), x @ beta])
        logpi = logits - logsumexp(logits, axis=1, keepdims=True)
        total = logpi + emission
        ll = logsumexp(total, axis=1)
        posterior = np.exp(total - ll[:, None])
        gradient = x.T @ (np.exp(logpi)[:, 1:] - posterior[:, 1:]) / len(x)
        penalty = beta.copy()
        penalty[0] = 0
        return float(-ll.mean() + .5 * config["l2"] * np.square(penalty).sum()), (gradient + config["l2"] * penalty).ravel()
    result = minimize(objective, np.zeros(x.shape[1] * 2), jac=True, method="L-BFGS-B",
                      options={"maxiter": config["max_iterations"], "gtol": 1e-7, "ftol": 1e-12})
    model["outcome_coefficients"] = result.x.reshape(x.shape[1], 2).tolist()
    model["optimization"] = {"converged": bool(result.success), "iterations": int(result.nit),
                             "objective": float(result.fun), "message": str(result.message)}
    model["training_counts"] = dict(Counter(e["status"] for e in events))
    model["scope"] = ("Initiation-conditioned outcomes; context-weighted paired observed peak/endpoint/time donors. "
                      "Censor-aware outcome/time likelihood assumes noninformative censoring. "
                      "No learned ongoing lateral response or alongside acceleration dependence. "
                      "Finite donor support and physical-time conditioning are explicit approximations.")
    return model


def outcome_probabilities(context, model):
    x = np.column_stack([np.ones(len(np.atleast_2d(context))), _standardize(context, model)])
    logits = np.column_stack([np.zeros(len(x)), x @ np.asarray(model["outcome_coefficients"])])
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def evaluate(events, model, training=False):
    context = [e["motion_context"] for e in events]
    pi = outcome_probabilities(context, model)
    emission = _emission(events, model, exclude_tracks=training)
    def group(mask):
        n = int(mask.sum())
        if not n:
            return {"events": 0}
        return {"events": n, "status_counts": dict(Counter(e["status"] for e, use in zip(events, mask) if use)),
                "mean_outcome_probability": dict(zip(OUTCOMES, pi[mask].mean(axis=0).tolist())),
                "outcome_time_nll": float(-logsumexp(np.log(pi[mask]) + emission[mask], axis=1).mean())}
    parallel = np.array([e["motion_context"][11] > .5 for e in events])
    return {"all": group(np.ones(len(events), dtype=bool)), "alongside_present": group(parallel),
            "no_alongside": group(~parallel), "training_density_leave_vehicle_out": training,
            "note": "Marginal outcome/time score; does not validate sampled endpoint/shape realism or significance."}


def sample(model, rng, lane_spacing, remaining_distance, acceleration, context):
    context = np.asarray(context, dtype=float)
    pi = outcome_probabilities(context, model)[0]
    k = int(rng.choice(3, p=pi))
    w = _donor_weights(_standardize(context, model), model, k)[0]
    j = int(rng.choice(len(w), p=w))
    donor = model["donors"][k][j]
    # A real adjustment need not return to its initial point. Preserve its
    # paired endpoint instead of inventing a symmetric out-and-back maneuver.
    endpoint = abs(remaining_distance) if k == 0 else lane_spacing * donor["endpoint_displacement_ratio"]
    peak = endpoint if k == 0 else max(endpoint, lane_spacing * donor["peak_displacement_ratio"])
    points = [peak]
    if abs(endpoint - peak) > .03:
        points.append(endpoint)
    distances = np.abs(np.diff([0., *points]))
    if (distances <= 0).any() or acceleration <= 0:
        raise ValueError("Invalid paired motion geometry")
    minimum_legs = 2 * np.sqrt(distances / acceleration) + .04
    minimum = float(minimum_legs.sum() + .24)
    mu, sigma = np.log(donor["duration_s"]), model["time_log_sigma"]
    zmin = (np.log(minimum) - mu) / sigma
    logsurvival = float(log_ndtr(-zmin))
    duration = float(np.exp(mu - sigma * ndtri(np.exp(logsurvival) * max(float(rng.random()), np.finfo(float).tiny))))
    if not np.isfinite(duration) or duration < minimum:
        raise ValueError("Conditional duration draw lost support")
    fraction = np.array([1.]) if len(points) == 1 else np.array([donor["peak_time_fraction"], 1 - donor["peak_time_fraction"]])
    leg_times = minimum_legs + (duration - minimum) * fraction
    z = (np.log(duration) - mu) / sigma
    logdensity = -np.log(duration * sigma) - .5 * np.log(2 * np.pi) - .5 * z * z - logsurvival
    return {"outcome": OUTCOMES[k], "requested_duration_s": duration, "legs": len(points),
            "outbound_distance_m": float(peak), "endpoint_distance_m": float(endpoint),
            "waypoint_distances_m": list(map(float, points)), "leg_durations_s": leg_times.tolist(),
            "minimum_feasible_duration_s": minimum, "removed_duration_mass": float(-np.expm1(logsurvival)),
            "outcome_probability": float(pi[k]), "conditional_outcome_probabilities": pi.tolist(),
            "donor_index": j, "donor_probability": float(w[j]), "donor_key": [donor["recording"], donor["vehicle_id"], donor["decision_frame"]],
            "donor_duration_s": donor["duration_s"], "conditional_duration_log_density": float(logdensity),
            "motion_context": context.tolist(), "kernel": KERNEL, "log_importance_ratio": 0.,
            "scope": "Shared P/Q post-action kernel; no adversarial change of process outcomes or time."}


def audit_sample(record, model):
    """Reconstruct the common transition density; never infer it from P/Q=1."""
    context = np.asarray(record["motion_context"], dtype=float)
    k = OUTCOMES.index(record["outcome"])
    j = record["donor_index"]
    pi = outcome_probabilities(context, model)[0]
    w = _donor_weights(_standardize(context, model), model, k)[0]
    donor = model["donors"][k][j]
    mu, sigma = np.log(donor["duration_s"]), model["time_log_sigma"]
    t, minimum = record["requested_duration_s"], record["minimum_feasible_duration_s"]
    if t < minimum:
        raise ValueError("Motion is below the logged physical support")
    z = (np.log(t) - mu) / sigma
    logdensity = -np.log(t * sigma) - .5 * np.log(2 * np.pi) - .5 * z * z - log_ndtr(-(np.log(minimum) - mu) / sigma)
    for key, value in (("outcome_probability", pi[k]), ("donor_probability", w[j]),
                       ("conditional_duration_log_density", logdensity), ("log_importance_ratio", 0.)):
        if not np.isclose(record[key], value, rtol=1e-10, atol=1e-12):
            raise ValueError("Common motion density mismatch: " + key)
    return float(np.log(pi[k]) + np.log(w[j]) + logdensity)
