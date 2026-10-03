import importlib.util
import math
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("gym") and importlib.util.find_spec("ray"),
                     "Use D2RLTrain39 for PPO policy tests")
class MeanPrecisionPolicyTests(unittest.TestCase):
    def test_baseline_mean_and_precision(self):
        import torch
        from scenario_reconstruction.highd_mean_precision_policy import (
            HighDMeanPrecisionBeta, initial_head_bias,
        )
        bias = initial_head_bias([.05, .05], [1., 1.])
        dist = HighDMeanPrecisionBeta(torch.tensor(bias[None, :], dtype=torch.float32), None)
        physical = (dist.deterministic_sample().numpy() + 1) * .95 / 2 + .05
        np.testing.assert_allclose(physical, [[.1, .1]], rtol=0, atol=1e-7)
        np.testing.assert_allclose(dist.precision.numpy(), [[80, 80]], atol=1e-5)
        # Lower-than-legacy means and high natural-mixture values are attainable.
        for epsilon in (.06, .1, .98):
            bias = initial_head_bias([.05, .05], [1., 1.], epsilon, 500.)
            dist = HighDMeanPrecisionBeta(torch.tensor(bias[None, :], dtype=torch.float32), None)
            physical = (dist.deterministic_sample().numpy() + 1) * .95 / 2 + .05
            np.testing.assert_allclose(physical, epsilon, rtol=0, atol=1e-7)

    def test_density_entropy_kl_and_gradients(self):
        import torch
        from scenario_reconstruction.highd_mean_precision_policy import HighDMeanPrecisionBeta, initial_head_bias
        logits = torch.tensor(np.tile(initial_head_bias([.05] * 2, [1.] * 2), (16, 1)),
                              dtype=torch.float32, requires_grad=True)
        dist = HighDMeanPrecisionBeta(logits, None)
        action = dist.sample().detach()
        self.assertEqual(action.shape, (16, 2))
        self.assertTrue(torch.all((action > -1) & (action < 1)))
        expected = (dist.dist.log_prob((action + 1) / 2) - math.log(2)).sum(-1)
        torch.testing.assert_close(dist.logp(action), expected)
        torch.testing.assert_close(dist.sampled_action_logp(), dist.logp(dist.last_sample))
        torch.testing.assert_close(dist.entropy(), (dist.dist.entropy() + math.log(2)).sum(-1))
        torch.testing.assert_close(dist.kl(dist), torch.zeros(16), atol=1e-5, rtol=0)
        self.assertEqual(dist.entropy().shape, (16,))
        self.assertTrue(torch.isfinite(dist.logp(action)).all())
        (-dist.logp(action).mean() - .001 * dist.entropy().mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_finite_extreme_parameters_without_old_mean_cap(self):
        import torch
        from scenario_reconstruction.highd_mean_precision_policy import HighDMeanPrecisionBeta
        logits = torch.tensor([[-1000., 1000., -1000., 1000.]], requires_grad=True)
        dist = HighDMeanPrecisionBeta(logits, None)
        self.assertLess(float(dist.mean[0, 0]), .001)
        self.assertGreater(float(dist.mean[0, 1]), .999)
        value = dist.logp(dist.deterministic_sample().detach()) + dist.entropy()
        self.assertTrue(torch.isfinite(value).all())
        value.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_model_initialization_preserves_hidden_value_network_and_restores(self):
        import copy
        import gym
        import torch
        from ray.rllib.models import MODEL_DEFAULTS
        from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
        from scenario_reconstruction.highd_mean_precision_policy import (
            HighDBaselineMeanPrecisionFC, HighDMeanPrecisionBeta, policy_model_config,
        )
        config = dict(copy.deepcopy(MODEL_DEFAULTS), **policy_model_config([.05] * 2, [1.] * 2))
        obs = gym.spaces.Box(-5., 5., (14,), dtype=np.float32)
        normalized_action = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
        torch.manual_seed(7)
        ordinary = FullyConnectedNetwork(obs, normalized_action, 4, config, "ordinary")
        torch.manual_seed(7)
        model = HighDBaselineMeanPrecisionFC(obs, normalized_action, 4, config, "baseline", **config["custom_model_config"])
        for name, parameter in ordinary.state_dict().items():
            if not name.startswith("_logits."):
                torch.testing.assert_close(model.state_dict()[name], parameter)
        inputs = {"obs_flat": torch.rand(40, 14) * 10 - 5}
        logits, _ = model.forward(inputs, [], None)
        physical = (HighDMeanPrecisionBeta(logits, model).deterministic_sample() + 1) * .95 / 2 + .05
        torch.testing.assert_close(physical, torch.full((40, 2), .1), atol=1e-7, rtol=0)
        # Restored trained head must not be replaced by baseline initialization.
        trained_weights = copy.deepcopy(model.state_dict())
        bias_key = next(key for key in trained_weights if key.startswith("_logits.") and key.endswith("bias"))
        trained_weights[bias_key] += .2
        model.load_state_dict(trained_weights)
        restored, _ = model.forward(inputs, [], None)
        torch.testing.assert_close(restored, logits + .2)


if __name__ == "__main__":
    unittest.main()
