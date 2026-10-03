"""PPO density on the active epsilon factors of a V46 decision."""
import math
import torch
from .highd_mean_precision_policy import HighDMeanPrecisionBeta


class V46MaskedBeta(HighDMeanPrecisionBeta):
    def __init__(self,inputs,model):
        super().__init__(inputs,model)
        self.factor_mask=model.active_factor_mask.detach()
        if self.factor_mask.shape!=self.mean.shape:
            raise ValueError('PPO active-factor mask shape mismatch')

    def logp(self,actions):
        return ((self.dist.log_prob((actions+1)/2)-math.log(2))*self.factor_mask).sum(-1)

    def entropy(self):
        return ((self.dist.entropy()+math.log(2))*self.factor_mask).sum(-1)

    def kl(self,other):
        if not torch.equal(self.factor_mask,other.factor_mask):
            raise ValueError('KL requires the same state-dependent active factors')
        return (torch.distributions.kl_divergence(self.dist,other.dist)*self.factor_mask).sum(-1)
