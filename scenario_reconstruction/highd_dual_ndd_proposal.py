"""Two trainable mixture parameters over an explicitly correlated natural P."""
from __future__ import annotations

import numpy as np

from d2rl_training.conditional_chain import epsilon_pair, replay_weight
from .highd_dual_ndd import probability


def conditional_parts(joint):
    joint = probability(joint, axis=(0, 1))
    if joint.ndim != 2:
        raise ValueError("Expected an action matrix")
    first = joint.sum(axis=1)
    second = np.divide(joint, first[:, None], out=np.zeros_like(joint), where=first[:, None] > 0)
    return first, second


class ConditionalChainProposal:
    """q1=e1*p1+(1-e1)*h1, q2=e2*p2(.|u1)+(1-e2)*h2(.|u1).

    H is supplied before sampling and kept fixed when replaying a new epsilon.
    H may emphasize rare actions but must not create actions outside P support.
    This conservative support restriction also preserves the legal-lane mask.
    """
    def __init__(self, natural, critical, epsilon):
        self.p = probability(natural, axis=(0, 1)).copy()
        self.h = probability(critical, axis=(0, 1)).copy()
        if self.p.shape != self.h.shape or ((self.p == 0) & (self.h > 0)).any():
            raise ValueError("H must use the same action encoding and stay within P support")
        self.epsilon = epsilon_pair(epsilon)
        self.p1, self.p2 = conditional_parts(self.p)
        self.h1, self.h2 = conditional_parts(self.h)
        # H's zero-mass first-action rows have no defined conditional. Use P's
        # conditional there explicitly, so mixed Q is still a normalized law.
        self.h2[self.h1 == 0] = self.p2[self.h1 == 0]
        self.q1 = self.epsilon[0] * self.p1 + (1 - self.epsilon[0]) * self.h1
        self.q2 = self.epsilon[1] * self.p2 + (1 - self.epsilon[1]) * self.h2
        self.matrix = self.q1[:, None] * self.q2
        probability(self.matrix, axis=(0, 1))
        if ((self.p > 0) & (self.matrix <= 0)).any():
            raise ValueError("Q lost natural support")

    def sample(self, distribution, rng):
        if not np.array_equal(distribution.natural, self.p):
            # Roundoff normalization of a supplied P is allowed, not another P.
            if not np.allclose(distribution.natural, self.p, atol=0, rtol=1e-13):
                raise ValueError("Proposal constructed for another natural target")
        draw = distribution.sample(rng, self.matrix)
        i, j = [action["action_index"] for action in draw["actions"]]
        p = np.array([self.p1[i], self.p2[i, j]])
        h = np.array([self.h1[i], self.h2[i, j]])
        q = np.array([self.q1[i], self.q2[i, j]])
        ids = list(distribution.actor_ids)
        weight = {"proposal_type": "conditional_chain_v1", "actor_ids": ids,
                  "model_id": distribution.model_id,
                  "action_encoding": "acceleration_index*3+lateral_choice" if distribution.structured else "acceleration_index",
                  "generation_epsilon": self.epsilon.tolist(), "per_agent": (p / q).tolist(),
                  "per_agent_critical_probability": h.tolist(), "per_agent_proposal_probability": q.tolist(),
                  "joint_naturalistic_probability": float(p.prod()), "joint_proposal_probability": float(q.prod()),
                  "joint": float((p / q).prod())}
        ndd = {"factorization": "p_first_times_p_second_given_first", "actor_ids": ids,
               "per_agent": p.tolist(), "joint": float(p.prod())}
        replay_weight(weight, self.epsilon, ndd, self.epsilon)
        draw.update(weight_record=weight, ndd_record=ndd)
        return draw

    def components(self):
        return {"type": "conditional_chain_v1", "epsilon": self.epsilon.tolist(),
                "critical_first": self.h1.tolist(), "critical_second_given_first": self.h2.tolist()}


def tilted_joint(natural, log_score):
    """Make H proportional to P*exp(score), without changing P or adding floors."""
    p = probability(natural, axis=(0, 1))
    score = np.asarray(log_score, dtype=float)
    if score.shape != p.shape or not np.isfinite(score).all():
        raise ValueError("Expected one finite pre-sampling score per joint action")
    h = p * np.exp(score - score[p > 0].max())
    return probability(h / h.sum(), axis=(0, 1))
