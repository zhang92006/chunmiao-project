"""Offline dual-BV probability building blocks; no SUMO controller replacement.

Role order is explicit. Longitudinal actions are 31 acceleration bins, not the
legacy 33-action IDs. Lateral initiation can be composed conditionally on the
sampled accelerations. During a locked maneuver, lateral continuation is
deterministic but longitudinal actions may still change on every decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from . import highd_conditional_lane_start as start
from . import highd_lane_phase_fine as fine
from . import highd_lane_phase_longitudinal as phase
from .highd_pair_data_inventory import digest, read_json
from .highd_pair_model_fit import FrozenMarginal, old_modifier, project_coarse


PERMUTATIONS = np.asarray(list(permutations(range(3))), dtype=int)
PERMUTATION_KERNELS = 3.0 * np.eye(3)[PERMUTATIONS]


def probability(value, axis=-1):
    """Validate first; normalize only floating-point sum roundoff."""
    p = np.asarray(value, dtype=float)
    total = p.sum(axis=axis, keepdims=True)
    if (not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(total, 1, atol=1e-10, rtol=0)):
        raise ValueError("Expected finite, nonnegative normalized probabilities")
    return p / total


def rank_fractions(p):
    """Fraction of each discrete action's CDF interval inside each tercile.

    Splitting a categorical atom across boundaries avoids assigning a whole
    high-probability acceleration to a single latent rank bin. No random PIT
    draw or observed action is needed to query the joint distribution.
    """
    p = probability(p)
    low = np.minimum(np.cumsum(p, axis=-1) - p, np.nextafter(1.0, 0.0))
    edges = np.linspace(0, 1, 4)
    width = np.where(p > 0, p, 1)[..., None]
    right = np.clip((edges[1:] - low[..., None]) / width, 0, 1)
    left = np.clip((edges[:-1] - low[..., None]) / width, 0, 1)
    result = right - left
    # Last endpoint is exactly the end of the probability interval, including
    # tiny final atoms whose addition to a CDF is lost in floating point.
    result[..., -1] += np.where(p > 0, 1 - result.sum(axis=-1), 0)
    return np.where((p > 0)[..., None], result, 0)


def rank_kernel(weights, independent_mass):
    weights = probability(weights)
    if weights.shape[-1] != 6 or not 0 < independent_mass <= 1:
        raise ValueError("Expected six permutation weights and positive independent mass")
    kernel = np.einsum("...k,kuv->...uv", weights, PERMUTATION_KERNELS)
    return independent_mass + (1 - independent_mass) * kernel


def rank_joint(first, second, weights, independent_mass):
    """31x31 (or general categorical) joint preserving BOTH input marginals.

    Every rank kernel has row/column sums of three. Mixing with independence
    preserves support and the marginals; it is part of natural P, not a clip
    applied to importance weights afterwards.
    """
    first, second = probability(first), probability(second)
    left = first[..., :, None] * rank_fractions(first)
    right = second[..., :, None] * rank_fractions(second)
    return np.einsum("...iu,...uv,...jv->...ij", left,
                     rank_kernel(weights, independent_mass), right)


def observed_components(first, second, a, b):
    """Six observed joint/independent ratios; used by a convex mixture fit."""
    rows = np.arange(len(first))
    f = rank_fractions(first)[rows, a]
    r = rank_fractions(second)[rows, b]
    return np.einsum("nu,kuv,nv->nk", f, PERMUTATION_KERNELS, r)


def lift_coarse(first, second, groups, modifier):
    """Preserve the existing following candidate and its original marginals."""
    first, second = probability(first), probability(second)
    masses1 = np.stack([first[:, groups == g].sum(axis=1) for g in range(3)], axis=1)
    masses2 = np.stack([second[:, groups == g].sum(axis=1) for g in range(3)], axis=1)
    coarse, _ = project_coarse(masses1, masses2, modifier)
    ratio = coarse / (masses1[:, :, None] * masses2[:, None, :])
    return first[:, :, None] * second[:, None, :] * ratio[:, groups[:, None], groups[None, :]]


def locked_lateral():
    """Choice zero means continue the existing lateral command while locked."""
    result = np.zeros((31, 3))
    result[:, 0] = 1
    return result


def compose_lateral(longitudinal_joint, first_conditional, second_conditional):
    """Return [a1*3+d1, a2*3+d2], i.e. a structured 93x93 distribution.

    pi1*pi2 is the DECLARED lateral conditional-independence baseline. This
    function does not claim that lateral residual dependence has been fitted.
    Legal-lane masks must already be in pi, never rejection-sampled afterwards.
    """
    joint = probability(longitudinal_joint, axis=(0, 1))
    first, second = probability(first_conditional), probability(second_conditional)
    if joint.shape != (31, 31) or first.shape != (31, 3) or second.shape != (31, 3):
        raise ValueError("Expected a 31x31 joint and two 31x3 conditionals")
    return np.einsum("ij,id,je->idje", joint, first, second).reshape(93, 93)


@dataclass
class PairActionDistribution:
    """One disjoint selected pair, with auditable actual joint P and Q."""

    actor_ids: tuple
    roles: tuple
    natural: np.ndarray
    model_id: str
    structured: bool = False

    def __post_init__(self):
        if len(self.actor_ids) != 2 or self.actor_ids[0] == self.actor_ids[1] or len(self.roles) != 2:
            raise ValueError("Expected two different vehicles in explicit role order")
        self.natural = probability(self.natural, axis=(0, 1)).copy()
        width = 93 if self.structured else 31
        if self.natural.shape != (width, width):
            raise ValueError("Action matrix does not match the declared action encoding")

    def sample(self, rng, proposal=None):
        q = self.natural if proposal is None else probability(proposal, axis=(0, 1))
        if q.shape != self.natural.shape or ((self.natural > 0) & (q <= 0)).any():
            raise ValueError("Proposal must cover the natural target support")
        flat = int(rng.choice(q.size, p=q.ravel()))
        a, b = np.unravel_index(flat, q.shape)
        logp = float(np.log(self.natural[a, b])) if self.natural[a, b] > 0 else -np.inf
        logq = float(np.log(q[a, b]))
        actions = []
        for actor, role, code in zip(self.actor_ids, self.roles, (a, b)):
            acceleration = code // 3 if self.structured else code
            actions.append({"actor_id": actor, "role": role, "action_index": int(code),
                            "acceleration_index": int(acceleration),
                            "acceleration_mps2": float(-4 + .2 * acceleration),
                            "lateral_choice": int(code % 3) if self.structured else None})
        return {"model_id": self.model_id, "actions": actions,
                "natural_joint_probability": float(self.natural[a, b]),
                "proposal_joint_probability": float(q[a, b]),
                "log_natural_joint_probability": logp, "log_proposal_joint_probability": logq,
                "log_importance_ratio": logp - logq}


class FrozenDualNDD:
    """Versioned offline facade for following, lane-process and quiet onset.

    This is not a simulator state builder. Callers must satisfy the existing
    causal feature contracts and select each actor at most once per decision.
    Unsupported pair relations must explicitly use the independent baseline.
    """

    def __init__(self, config):
        self.config = config
        candidate = read_json(config["lane_reference_contract"])
        self.phase_config = read_json(candidate["config"])
        self.marginal = FrozenMarginal(read_json(candidate["reference_ndd"]), [-.5, .5])
        self.models = {}
        with self._load(candidate["model"]) as artifact:
            for key in artifact.files:
                role, name = key.split("__")
                self.models.setdefault(role, {})[name] = artifact[key]
        with self._load(config["following_model"]) as artifact:
            self.following_model = {k: artifact[k] for k in artifact.files}
        with self._load(config["onset_model"]) as artifact:
            self.onset_model = {k: artifact[k] for k in artifact.files}
        expected = candidate["provenance"]
        if (digest(fine.__file__) != expected["code_sha256"]
                or digest(phase.__file__) != expected["phase_code_sha256"]
                or digest(start.__file__) != expected["clock_code_sha256"]):
            raise ValueError("Locked probability/feature implementation changed")

    @staticmethod
    def _load(spec):
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError("Model checksum mismatch: " + spec["path"])
        return np.load(spec["path"], allow_pickle=False)

    def reference_probability(self, state):
        """Query RAW float64 [speed,gap,rr,free] before feature compression.

        Never reconstruct a table index from stored float32 model features:
        rounding around half-bin boundaries can select another NDD table row.
        """
        state = np.asarray(state, dtype=float)
        speed, gap, rr, free = (state[:, k] for k in range(4))
        vi = self.marginal.index(speed, "speed")
        p = self.marginal.ff[vi].copy()
        cf = free == 0
        if cf.any():
            gi = self.marginal.index(gap[cf], "gap")
            ri = self.marginal.index(rr[cf], "range_rate")
            p[cf] = self.marginal.cf[gi, ri, vi[cf]]
        return p

    def lane_marginal(self, state, role, reference_probability):
        if role not in self.models:
            raise ValueError("Unknown lane-process role")
        state = np.asarray(state, dtype=float)
        if state.ndim != 2 or state.shape[1] != len(phase.FEATURES) or not np.isfinite(state).all():
            raise ValueError("Expected finite lane-process feature rows")
        # Feature matrices were intentionally stored as float32 during fitting;
        # the reference table was queried from raw float64 observations first.
        # Keep these two representations separate at inference as well.
        p = probability(reference_probability)
        if p.shape != (len(state), 31):
            raise ValueError("Expected reference probabilities queried before feature compression")
        model = self.models[role]
        x = fine.model_design(state, model, "history")
        logp, _ = fine.log_probability(x, np.log(p), model["history"],
                                       fine.action_basis(self.marginal.axes["acceleration"]),
                                       self.phase_config["reference_mixture_mass"])
        return np.exp(logp)

    def following(self, follower, leader):
        if (len(follower) != len(leader)
                or not np.array_equal(follower.precedingId.to_numpy(), leader.id.to_numpy())
                or not np.array_equal(follower.laneId.to_numpy(), leader.laneId.to_numpy())
                or not np.array_equal(follower.frame.to_numpy(), leader.frame.to_numpy())
                or not np.array_equal(follower.direction.to_numpy(), leader.direction.to_numpy())):
            raise ValueError("Following branch requires synchronous same-lane immediate follower/leader")
        def p(own):
            speed = np.abs(own.xVelocity.to_numpy())
            free = (own.precedingId.to_numpy() == 0) | (own.dhw.to_numpy() > 115)
            state = np.column_stack([speed, np.where(free, 115, own.dhw),
                                     np.where(free, 0, np.abs(own.precedingXVelocity) - speed), free])
            return self.reference_probability(state)
        # Old dependence is applied ONLY to the old longitudinal reference.
        return lift_coarse(p(follower), p(leader), self.marginal.groups,
                           old_modifier(follower, leader, self.following_model))

    def quiet_lateral(self, state, permitted):
        # Divide out a strictly positive dummy marginal to query pi(d|a,h).
        dummy = np.full((len(state), 31), 1 / 31)
        return start.joint_action_probabilities(state, dummy, permitted, self.onset_model) * 31
