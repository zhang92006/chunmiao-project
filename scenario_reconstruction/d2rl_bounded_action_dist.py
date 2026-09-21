"""Bounded Torch action distribution for legacy RLlib PPO."""

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchBeta


ACTION_DIST_NAME = "d2rl_bounded_beta"


class D2RLBoundedBeta(TorchBeta):
    """Independent Beta actions on RLlib's normalized [-1, 1] interval."""

    def __init__(self, inputs, model):
        super().__init__(inputs, model, low=-1.0, high=1.0)

    def entropy(self):
        return super().entropy().sum(-1)

    def kl(self, other):
        return super().kl(other).sum(-1)


def register_bounded_action_distribution() -> None:
    ModelCatalog.register_custom_action_dist(ACTION_DIST_NAME, D2RLBoundedBeta)
