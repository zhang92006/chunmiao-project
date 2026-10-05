"""Offline meta-policy replay over FULL dual-BV critical-state sequences.

The recorded vehicle path does not change when epsilon changes in replay.
For behavior path law Q_b, use the empirical second-moment objective
    E_Qb[I_crash * (P/Q_b) * (P/Q_epsilon)].
It is not an online SUMO environment, nor the legacy single-critical reward.
"""
import hashlib
import json
from pathlib import Path

from gym import Env, spaces
import numpy as np

from .conditional_chain import epsilon_pair, replay_weight
from .highd_event_contract import (
    FIRST_EVENT, FIRST_EVENT_DEFINITION, FIRST_SEQUENCE, FULL_SEQUENCE, event_indicator, target_digest,
)


def step_log_weight(step, epsilon):
    # This validates the conditional-chain generation probabilities as well.
    return float(np.log(replay_weight(step["weight_record"], epsilon, step["ndd_record"])))


def sequence_log_weight(sequence, epsilons):
    if len(epsilons) != len(sequence["steps"]):
        raise ValueError("One ordered epsilon pair per critical state is required")
    return sum(step_log_weight(s, e) for s, e in zip(sequence["steps"], epsilons))


def scaled_loss(sequence, log_weight, log_scale):
    if not event_indicator(sequence):
        return 0.
    exponent = sequence["generation_log_weight"] + log_weight - log_scale
    if exponent > np.log(np.finfo(float).max) - 2:
        raise FloatingPointError("Second moment overflow; do not silently clip the training objective")
    return float(np.exp(exponent))


class HighDCriticalSequenceEnv(Env):
    """Two epsilon decisions at every retained state; one terminal IS loss.

    Nonempty paths are drawn uniformly, including safe ones. Multiplying their
    reward by n_nonempty/n_all preserves the full collection denominator.
    Empty paths have no trainable decision and contribute a constant, tracked
    separately. A positive fixed reward scale changes units, not the objective.
    """
    def __init__(self, config):
        manifest = json.loads(Path(config["sequence_manifest"]).read_text(encoding="utf-8"))
        raw = Path(manifest["data_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
            raise ValueError("Sequence dataset changed")
        self.dataset = json.loads(raw)
        contract = self.dataset["contract"]
        if contract not in (FULL_SEQUENCE, FIRST_SEQUENCE):
            raise ValueError("Wrong replay contract")
        if contract == FIRST_SEQUENCE:
            spec = self.dataset["experiment_target"]
            if (self.dataset["event_definition"] != FIRST_EVENT_DEFINITION or
                    spec["event_definition"] != FIRST_EVENT_DEFINITION or
                    spec["natural_target_sha256"] != self.dataset["natural_target_sha256"] or
                    target_digest(spec) != self.dataset["experiment_target_sha256"]):
                raise ValueError("Event/experiment target contract changed")
        if self.dataset["observation_contract"] != "highd_dual_kinematic14_v1":
            raise ValueError("Cannot load an old 33-action observation/checkpoint contract")
        self.sequences = self.dataset["sequences"]
        self.eligible = [i for i, s in enumerate(self.sequences) if s["steps"]]
        if not self.eligible or not any(event_indicator(s) for s in self.sequences):
            raise ValueError("Need nonempty sequences and at least one CAV crash for a smoke test")
        for sequence in self.sequences:
            if sequence.get("event_contract") != (FIRST_EVENT if contract == FIRST_SEQUENCE else None):
                raise ValueError("Cannot mix full-horizon and first-collision labels")
            event_indicator(sequence)
            if sequence["natural_target_sha256"] != self.dataset["natural_target_sha256"]:
                raise ValueError("Mixed natural targets")
            ticks = [s["raw_tick"] for s in sequence["steps"]]
            if ticks != sorted(set(ticks)):
                raise ValueError("Critical sequence is not strictly ordered")
            value = sequence_log_weight(sequence, [s["weight_record"]["generation_epsilon"] for s in sequence["steps"]])
            if not np.isclose(value, sequence["generation_log_weight"], atol=1e-8, rtol=1e-10):
                raise ValueError("Densification lost a path weight")
        self.log_scale = max(2 * s["generation_log_weight"] for s in self.sequences if event_indicator(s))
        self.sampling_correction = len(self.eligible) / len(self.sequences)
        self.constant_empty_loss = sum(scaled_loss(s, 0., self.log_scale) for s in self.sequences if not s["steps"]) / len(self.sequences)
        minimum = float(config.get("epsilon_min", .05))
        if not 0 < minimum < 1:
            raise ValueError("epsilon_min must lie in (0,1)")
        self.action_space = spaces.Box(low=minimum, high=1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5., high=5., shape=(14,), dtype=np.float32)
        self.seed(config.get("seed", 7))
        self.done = True

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self, episode_index=None):
        i = int(self.rng.choice(self.eligible)) if episode_index is None else int(episode_index)
        if i not in self.eligible:
            raise ValueError("Empty episodes are exposure, not fabricated RL decisions")
        self.sequence = self.sequences[i]
        self.position, self.log_weight, self.done = 0, 0., False
        return np.asarray(self.sequence["steps"][0]["observation"], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset before stepping a finished sequence")
        eps = epsilon_pair(action)
        if (eps < self.action_space.low).any():
            raise ValueError("Epsilon below configured action bounds")
        item = self.sequence["steps"][self.position]
        self.log_weight += step_log_weight(item, eps)
        self.position += 1
        self.done = self.position == len(self.sequence["steps"])
        reward = -self.sampling_correction * scaled_loss(self.sequence, self.log_weight, self.log_scale) if self.done else 0.
        next_item = self.sequence["steps"][min(self.position, len(self.sequence["steps"]) - 1)]
        info = {"raw_tick": item["raw_tick"], "delta_time_s": item["delta_time_s"]}
        if self.done:
            info.update(collision_result=self.sequence["collision_result"], critical_steps=self.position,
                        event_result=event_indicator(self.sequence), event_type=self.sequence.get("event_type", "full_horizon"),
                        generation_log_weight=self.sequence["generation_log_weight"],
                        replay_log_weight=self.log_weight, log_reward_scale=self.log_scale)
        return np.asarray(next_item["observation"], dtype=np.float32), float(reward), self.done, info

    def empirical_objective(self, policy):
        losses = []
        for sequence in self.sequences:
            weight = sequence_log_weight(sequence, [policy(s["observation"]) for s in sequence["steps"]])
            losses.append(scaled_loss(sequence, weight, self.log_scale))
        return float(np.mean(losses))
