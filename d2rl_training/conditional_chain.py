"""Exact replay of an ordered correlated two-BV mixture (no simulator imports)."""
import numpy as np


def epsilon_pair(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all() or (result <= 0).any() or (result > 1).any():
        raise ValueError("Conditional-chain epsilon must contain two values in (0, 1]")
    return result


def replay_weight(weight_record, epsilon, ndd_record, generation_epsilon=None):
    """p1(u1) p2(u2|u1) / [q1(u1) q2(u2|u1)] at NEW policy epsilon.

    Per-agent fields here are conditional-chain factors, NOT independent
    marginals. Store H explicitly: it cannot be recovered when generation e=1.
    """
    if weight_record.get("proposal_type") != "conditional_chain_v1":
        raise ValueError("Expected a conditional-chain record")
    ids = weight_record["actor_ids"]
    if (len(ids) != 2 or ids[0] == ids[1] or ndd_record.get("actor_ids") != ids
            or ndd_record.get("factorization") != "p_first_times_p_second_given_first"):
        raise ValueError("Conditional probabilities require the same ordered actor IDs")
    p = np.asarray(ndd_record["per_agent"], dtype=float)
    h = np.asarray(weight_record["per_agent_critical_probability"], dtype=float)
    q = np.asarray(weight_record["per_agent_proposal_probability"], dtype=float)
    w = np.asarray(weight_record["per_agent"], dtype=float)
    for value in (p, h, q):
        if value.shape != (2,) or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Invalid conditional component probabilities")
    if (p <= 0).any() or (q <= 0).any() or w.shape != (2,) or not np.isfinite(w).all():
        raise ValueError("Sampled action must have positive P and Q")
    gen = epsilon_pair(weight_record["generation_epsilon"])
    if generation_epsilon is not None and not np.allclose(gen, epsilon_pair(generation_epsilon), atol=1e-14, rtol=0):
        raise ValueError("Generation epsilon order or values disagree")
    # Relative checks matter for rare probabilities; a large absolute tolerance
    # would silently accept wrong products near zero.
    def same(actual, expected):
        if not np.allclose(actual, expected, rtol=1e-9, atol=0):
            raise ValueError("Conditional-chain logged probability product disagrees")
    same(q, gen * p + (1 - gen) * h)
    same(w, p / q)
    same(ndd_record["joint"], p.prod())
    same(weight_record["joint_naturalistic_probability"], p.prod())
    same(weight_record["joint_proposal_probability"], q.prod())
    same(weight_record["joint"], w.prod())
    eps = epsilon_pair(epsilon)
    return float(np.prod(p / (eps * p + (1 - eps) * h)))
