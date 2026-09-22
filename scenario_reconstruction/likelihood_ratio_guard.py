"""Bound factorized proposal-to-naturalistic likelihood ratios before sampling."""

from __future__ import annotations

import math

import numpy as np


def constrain_epsilon(
    requested_epsilon: float,
    naturalistic_pdf,
    critical_pdf,
    maximum_proposal_ratio: float,
) -> tuple[float, dict]:
    """Increase epsilon just enough to guarantee q(a)/p(a) <= the limit.

    The proposal is q = epsilon * p + (1 - epsilon) * c.  The operation changes
    the proposal before sampling and therefore preserves exact p/q accounting.
    """
    epsilon = float(requested_epsilon)
    limit = float(maximum_proposal_ratio)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("requested_epsilon must lie between zero and one")
    if not math.isfinite(limit) or limit <= 1.0:
        raise ValueError("maximum_proposal_ratio must be finite and greater than one")
    naturalistic = np.asarray(naturalistic_pdf, dtype=float)
    critical = np.asarray(critical_pdf, dtype=float)
    if naturalistic.shape != critical.shape or naturalistic.ndim != 1:
        raise ValueError("naturalistic and critical PDFs must be matching vectors")
    if (not np.all(np.isfinite(naturalistic))
            or not np.all(np.isfinite(critical))
            or np.any(naturalistic < 0.0)
            or np.any(critical < 0.0)):
        raise ValueError("proposal PDFs must be finite and non-negative")
    p_total = float(np.sum(naturalistic))
    c_total = float(np.sum(critical))
    if p_total <= 0.0 or c_total <= 0.0:
        raise ValueError("proposal PDFs must have positive mass")
    naturalistic = naturalistic / p_total
    critical = critical / c_total

    supported = naturalistic > 0.0
    unsupported_critical_mass = float(np.sum(critical[~supported]))
    if unsupported_critical_mass > 0.0:
        applied = 1.0
        maximum_critical_ratio = math.inf
    else:
        maximum_critical_ratio = float(np.max(critical[supported] / naturalistic[supported]))
        requested_maximum = epsilon + (1.0 - epsilon) * maximum_critical_ratio
        if requested_maximum <= limit:
            applied = epsilon
        elif maximum_critical_ratio <= 1.0:
            applied = epsilon
        else:
            required = (maximum_critical_ratio - limit) / (maximum_critical_ratio - 1.0)
            applied = max(epsilon, min(1.0, required))

    proposal = applied * naturalistic + (1.0 - applied) * critical
    if np.any(proposal[~supported] > 1e-15):
        applied = 1.0
        proposal = naturalistic.copy()
    applied_maximum = float(np.max(proposal[supported] / naturalistic[supported]))
    if applied_maximum > limit * (1.0 + 1e-10):
        raise ValueError("likelihood-ratio guard failed to satisfy its bound")
    requested_proposal = epsilon * naturalistic + (1.0 - epsilon) * critical
    requested_maximum = (
        math.inf if np.any(requested_proposal[~supported] > 0.0)
        else float(np.max(requested_proposal[supported] / naturalistic[supported]))
    )
    return applied, {
        "requested_epsilon": epsilon,
        "applied_epsilon": applied,
        "adjusted": applied > epsilon + 1e-12,
        "maximum_proposal_ratio_limit": limit,
        "requested_maximum_proposal_ratio": requested_maximum,
        "applied_maximum_proposal_ratio": applied_maximum,
        "maximum_critical_to_naturalistic_ratio": maximum_critical_ratio,
        "unsupported_critical_mass": unsupported_critical_mass,
    }
