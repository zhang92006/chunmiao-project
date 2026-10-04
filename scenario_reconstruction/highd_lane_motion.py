"""Censor-aware three-outcome motion-time candidate and common P/Q execution.

This is a pooled parametric reference, not a claim of learned lateral paths.
Unresolved observations contribute survival likelihood, never fake completions.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, log_ndtr, ndtr, ndtri


OUTCOMES = ("completed", "within_lane_adjustment", "returned")


def category(status):
    if status.startswith("returned_"):
        return 2
    return OUTCOMES.index(status) if status in OUTCOMES else -1


def parameters(vector):
    logits, mu, log_sigma = np.split(vector, 3)
    return logits - logsumexp(logits), mu, np.exp(log_sigma)


def log_likelihood(events, vector):
    t = np.asarray([event["duration_s"] for event in events])
    code = np.asarray([category(event["status"]) for event in events])
    log_pi, mu, sigma = parameters(vector)
    logt = np.log(np.maximum(t, 1e-12))
    z = (logt[:, None] - mu) / sigma
    logpdf = -logt[:, None] - np.log(sigma) - .5 * np.log(2 * np.pi) - z * z / 2
    result = logsumexp(log_pi + log_ndtr(-z), axis=1)
    observed = code >= 0
    result[observed] = (logpdf + log_pi)[np.flatnonzero(observed), code[observed]]
    return result


def fit(events):
    if any(not np.isfinite(e["duration_s"]) or e["duration_s"] < 0 for e in events):
        raise ValueError("Invalid motion observation duration")
    codes = np.asarray([category(e["status"]) for e in events])
    counts = np.bincount(codes[codes >= 0], minlength=3)
    if (counts < 2).any():
        raise ValueError("Need at least two observed events of each pooled outcome")
    times = np.asarray([e["duration_s"] for e in events])
    initial = np.r_[np.log(counts / counts.sum()),
                    [np.log(times[codes == k]).mean() for k in range(3)],
                    [np.log(max(.2, np.log(times[codes == k]).std())) for k in range(3)]]
    result = minimize(lambda x: -log_likelihood(events, x).mean(), initial, method="L-BFGS-B",
                      bounds=[(-15, 15)] * 3 + [(-5, 5)] * 3 + [(np.log(.1), np.log(2))] * 3,
                      options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-7})
    log_pi, mu, sigma = parameters(result.x)
    peaks = [[float(e["peak_displacement_ratio"]) for e in events if category(e["status"]) == k] for k in range(3)]
    return {"schema_version": 1, "outcomes": list(OUTCOMES), "parameters": result.x.tolist(),
            "outcome_probability": np.exp(log_pi).tolist(), "log_duration_mean": mu.tolist(),
            "log_duration_sigma": sigma.tolist(), "peak_displacement_ratio_by_outcome": peaks,
            "optimization": {"converged": bool(result.success), "iterations": int(result.nit),
                             "message": str(result.message), "negative_log_likelihood": float(result.fun)},
            "training_counts": dict(Counter(e["status"] for e in events)),
            "scope": "Pooled lognormal outcome/time mixture; completed, within-lane and returns retained. Right/other-lane censoring contributes survival. Assumes noninformative censoring; not validated. At execution, time law is explicitly conditioned on physical minimum duration. Same process kernel under P and Q."}


def evaluate(events, model):
    if not events:
        return {"event_count": 0}
    return {"event_count": len(events), "status_counts": dict(Counter(e["status"] for e in events)),
            "negative_log_likelihood": float(-log_likelihood(events, np.asarray(model["parameters"])).mean()),
            "observed_duration_quantiles": {status: np.quantile([e["duration_s"] for e in events if e["status"] == status], [.1, .5, .9]).tolist()
                                            for status in sorted(set(e["status"] for e in events))}}


def sample(model, rng, lane_spacing, remaining_distance, acceleration):
    """Draw one common transition kernel AFTER joint action; its P/Q cancels.

    Keep outcome and time densities auditable, including physical truncation.
    There is no traffic-gap safety mask and no forced avoidance of collision.
    """
    k = int(rng.choice(3, p=model["outcome_probability"]))
    fraction_pool = model["peak_displacement_ratio_by_outcome"][k]
    peak_index = int(rng.integers(len(fraction_pool))) if k else None
    distance = abs(remaining_distance) if k == 0 else lane_spacing * fraction_pool[peak_index]
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError("Motion displacement must be positive")
    legs = 1 if k == 0 else 2
    # Account for acceleration/deceleration and the observed stable-end test.
    minimum = legs * (2 * np.sqrt(distance / acceleration) + .04) + .24
    mu, sigma = model["log_duration_mean"][k], model["log_duration_sigma"][k]
    zmin = (np.log(minimum) - mu) / sigma
    log_survival = float(log_ndtr(-zmin))
    # Inverse survival is numerically stable even for small tail mass.
    survival_draw = np.exp(log_survival) * max(float(rng.random()), np.finfo(float).tiny)
    duration = float(np.exp(mu - sigma * ndtri(survival_draw)))
    if not np.isfinite(duration) or duration < minimum:
        raise ValueError("Motion duration sampling lost numerical support")
    z = (np.log(duration) - mu) / sigma
    log_density = -np.log(duration * sigma) - .5 * np.log(2 * np.pi) - .5 * z * z - log_survival
    return {"outcome": OUTCOMES[k], "requested_duration_s": duration,
            "outbound_distance_m": float(distance), "legs": legs,
            "outcome_probability": model["outcome_probability"][k],
            "peak_index": peak_index, "peak_probability": 1. if k == 0 else 1 / len(fraction_pool),
            "minimum_feasible_duration_s": float(minimum), "removed_duration_mass": float(ndtr(zmin)),
            "conditional_duration_log_density": float(log_density),
            "log_importance_ratio": 0., "kernel": "shared_natural_motion_v1"}
