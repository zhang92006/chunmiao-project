"""Numerically stable importance-sampling records and diagnostics."""
from __future__ import annotations

from math import exp, isfinite, log
from typing import Iterable, Mapping


def probability_record(
    proposal_type: str,
    naturalistic_probability: float,
    proposal_probability: float,
) -> dict[str, float | str]:
    """Create one auditable p/q record for an executed action sample.

    Both probabilities must describe the *same* sampled action (or action pair).
    The raw ratio is retained for backwards compatibility, while the logarithmic
    quantities are the authoritative representation for episode accumulation.
    """
    naturalistic = float(naturalistic_probability)
    proposal = float(proposal_probability)
    if not proposal_type:
        raise ValueError("proposal_type must be non-empty")
    if not isfinite(naturalistic) or naturalistic <= 0.0:
        raise ValueError("naturalistic_probability must be finite and positive")
    if not isfinite(proposal) or proposal <= 0.0:
        raise ValueError("proposal_probability must be finite and positive")
    log_naturalistic = log(naturalistic)
    log_proposal = log(proposal)
    return {
        "proposal_type": str(proposal_type),
        "naturalistic_probability": naturalistic,
        "proposal_probability": proposal,
        "log_naturalistic_probability": log_naturalistic,
        "log_proposal_probability": log_proposal,
        "log_importance_weight": log_naturalistic - log_proposal,
        "importance_weight": naturalistic / proposal,
    }


def stable_weight_diagnostics(
    records: Iterable[Mapping[str, object]],
) -> dict[str, float | int | None]:
    """Summarize episode log weights without exponentiating tiny values first."""
    log_weights = [
        float(record["log_importance_weight"])
        for record in records
        if record.get("log_importance_weight") is not None
        and isfinite(float(record["log_importance_weight"]))
    ]
    if not log_weights:
        return {
            "episode_count": 0,
            "effective_sample_size": 0.0,
            "effective_sample_size_ratio": 0.0,
            "largest_normalized_weight": None,
            "minimum_log_importance_weight": None,
            "maximum_log_importance_weight": None,
        }

    maximum = max(log_weights)
    shifted = [exp(value - maximum) for value in log_weights]
    total = sum(shifted)
    normalized = [value / total for value in shifted]
    effective_sample_size = 1.0 / sum(value * value for value in normalized)
    count = len(log_weights)
    return {
        "episode_count": count,
        "effective_sample_size": effective_sample_size,
        "effective_sample_size_ratio": effective_sample_size / count,
        "largest_normalized_weight": max(normalized),
        "minimum_log_importance_weight": min(log_weights),
        "maximum_log_importance_weight": maximum,
    }
