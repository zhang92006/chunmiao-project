"""Opt-in two-epsilon policy head; does not change BV natural/action P or Q.

Unlike Ray 1.11 TorchBeta's capped alpha/beta logits, the mean is parameterized
directly. RLlib samples in [-1, 1] and maps to the environment's epsilon bounds.
The two independent Beta variables describe *meta-policy exploration*, not an
independence assumption about the two background vehicles' natural actions.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchBeta, TorchDistributionWrapper


ACTION_DIST_NAME = "highd_mean_precision_beta_v1"
MODEL_NAME = "highd_baseline_mean_precision_fc_v1"
# Explicit numerical interior for the mean, not clipping sampled actions or IS
# weights. For physical [.05, 1], deterministic means cover [.050095, .999905].
MEAN_MARGIN = 1e-4
SHAPE_FLOOR = 2.0
INITIAL_EPSILON = .1
INITIAL_PRECISION = 80.0
LEGACY_ACTION_DIST_NAME = "d2rl_bounded_beta"


def initial_head_bias(epsilon_low, epsilon_high, epsilon=INITIAL_EPSILON,
                      precision=INITIAL_PRECISION):
    low, high = np.asarray(epsilon_low, float), np.asarray(epsilon_high, float)
    if low.shape != (2,) or high.shape != (2,) or np.any(high <= low):
        raise ValueError("Expected the physical two-epsilon action space")
    mean = (epsilon - low) / (high - low)
    sigmoid_mean = (mean - MEAN_MARGIN) / (1 - 2 * MEAN_MARGIN)
    if np.any(sigmoid_mean <= 0) or np.any(sigmoid_mean >= 1):
        raise ValueError("Initial epsilon must be inside the mean parameterization")
    excess = precision - SHAPE_FLOOR / np.minimum(mean, 1 - mean)
    if np.any(excess <= 0):
        raise ValueError("Initial precision is below the unimodal shape floor")
    # Stable inverse softplus, including large positive concentration logits.
    raw_precision = excess + np.log(-np.expm1(-excess))
    return np.concatenate([np.log(sigmoid_mean) - np.log1p(-sigmoid_mean), raw_precision])


class HighDMeanPrecisionBeta(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        raw_mean, raw_precision = torch.chunk(self.inputs, 2, dim=-1)
        self.mean = MEAN_MARGIN + (1 - 2 * MEAN_MARGIN) * torch.sigmoid(raw_mean)
        # Preserve that mean exactly while keeping both shapes >= 2, avoiding
        # singular boundary densities during normalized-action PPO sampling.
        floor = SHAPE_FLOOR / torch.minimum(self.mean, 1 - self.mean)
        self.precision = floor + torch.nn.functional.softplus(raw_precision)
        self.dist = torch.distributions.Beta(self.mean * self.precision,
                                             (1 - self.mean) * self.precision)

    def deterministic_sample(self):
        self.last_sample = 2 * self.mean - 1
        return self.last_sample

    def sample(self):
        self.last_sample = 2 * self.dist.rsample() - 1
        return self.last_sample

    def logp(self, actions):
        # Density on normalized [-1, 1], including the affine Jacobian.
        return (self.dist.log_prob((actions + 1) / 2) - math.log(2)).sum(-1)

    def entropy(self):
        return (self.dist.entropy() + math.log(2)).sum(-1)

    def kl(self, other):
        return torch.distributions.kl_divergence(self.dist, other.dist).sum(-1)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return int(np.prod(action_space.shape)) * 2


class LegacyBoundedBeta(TorchBeta):
    """Small-package copy of the established Ray 1.11 baseline distribution."""

    def __init__(self, inputs, model):
        super().__init__(inputs, model, low=-1.0, high=1.0)

    def entropy(self):
        return super().entropy().sum(-1)

    def kl(self, other):
        return super().kl(other).sum(-1)


class HighDBaselineMeanPrecisionFC(FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_options):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        if num_outputs != 4 or self._logits is None or self.free_log_std:
            raise ValueError("Baseline initialization requires a four-logit linear policy head")
        # RLlib may pass a normalized action space to the model. Read the
        # physical bounds explicitly from the serialized custom configuration.
        cfg = dict(model_config["custom_model_config"], **custom_options)
        bias = initial_head_bias(cfg["physical_epsilon_low"], cfg["physical_epsilon_high"],
                                 cfg["initial_epsilon"], cfg["initial_precision"])
        heads = [layer for layer in self._logits.modules() if isinstance(layer, torch.nn.Linear)]
        if len(heads) != 1:
            raise ValueError("Unexpected RLlib final-head layout")
        with torch.no_grad():
            heads[0].weight.zero_()
            heads[0].bias.copy_(torch.as_tensor(bias, dtype=heads[0].bias.dtype))
        # Value head and hidden layers retain the ordinary seeded FCNet init.
        # A subsequent trainer.restore overwrites these initial weights normally.


def register_mean_precision_policy():
    ModelCatalog.register_custom_action_dist(ACTION_DIST_NAME, HighDMeanPrecisionBeta)
    ModelCatalog.register_custom_model(MODEL_NAME, HighDBaselineMeanPrecisionFC)


def register_legacy_action_distribution():
    ModelCatalog.register_custom_action_dist(LEGACY_ACTION_DIST_NAME, LegacyBoundedBeta)


def policy_model_config(epsilon_low, epsilon_high, initial_epsilon=INITIAL_EPSILON,
                        initial_precision=INITIAL_PRECISION):
    return {"custom_action_dist": ACTION_DIST_NAME, "custom_model": MODEL_NAME,
            "fcnet_hiddens": [64, 64],
            "custom_model_config": {
                "physical_epsilon_low": list(map(float, epsilon_low)),
                "physical_epsilon_high": list(map(float, epsilon_high)),
                "initial_epsilon": float(initial_epsilon), "initial_precision": float(initial_precision)}}
