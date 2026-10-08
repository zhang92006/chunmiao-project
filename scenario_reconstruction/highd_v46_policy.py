"""Portable deterministic inference for freshly trained native V46 epsilon heads."""
from pathlib import Path
import numpy as np
from .highd_v46_assets import read, digest
from .highd_v46_contract import OBSERVATION_DIMENSIONS


class NumpyPolicy:
    def __init__(self,root):
        root=Path(root);self.metadata=read(root/'policy.json');path=root/'policy_weights.npz'
        if digest(path)!=self.metadata['weights_sha256']:raise ValueError('Policy checksum mismatch')
        if self.metadata['observation_contract'] not in OBSERVATION_DIMENSIONS:raise ValueError('Incompatible policy observation')
        with np.load(path) as data:self.weights={k:data[k] for k in data.files}
        if self.metadata.get('policy_type')=='fixed_epsilon':
            epsilon=self.weights['epsilon']
            if epsilon.shape!=(self.metadata['action_dimension'],) or not np.isfinite(epsilon).all() or np.any(epsilon<.05) or np.any(epsilon>1):raise ValueError('Invalid fixed policy epsilon')

    def __call__(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        if x.shape!=(OBSERVATION_DIMENSIONS[self.metadata['observation_contract']],) or not np.isfinite(x).all():raise ValueError('Policy observation shape/values differ')
        if self.metadata.get('policy_type')=='fixed_epsilon':return self.weights['epsilon'].tolist()
        for i in range(self.metadata['hidden_layers']):x=np.tanh(self.weights[f'w{i}']@x+self.weights[f'b{i}'])
        logits=self.weights['head_w']@x+self.weights['head_b']
        raw=logits[:self.metadata['action_dimension']]
        mean=.0001+.9998*np.exp(-np.logaddexp(0.,-raw))
        return (.05+.95*mean).tolist()
