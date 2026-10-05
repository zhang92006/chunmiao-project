"""Portable deterministic inference for freshly trained native V46 epsilon heads."""
from pathlib import Path
import numpy as np
from .highd_v46_assets import read, digest


class NumpyPolicy:
    def __init__(self,root):
        root=Path(root);self.metadata=read(root/'policy.json');path=root/'policy_weights.npz'
        if digest(path)!=self.metadata['weights_sha256']:raise ValueError('Policy checksum mismatch')
        if self.metadata['observation_contract']!='native_cav6_bv4x2_v1':raise ValueError('Incompatible policy observation')
        with np.load(path) as data:self.weights={k:data[k] for k in data.files}

    def __call__(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        for i in range(self.metadata['hidden_layers']):x=np.tanh(self.weights[f'w{i}']@x+self.weights[f'b{i}'])
        logits=self.weights['head_w']@x+self.weights['head_b']
        raw=logits[:self.metadata['action_dimension']]
        mean=.0001+.9998/(1+np.exp(-raw))
        return (.05+.95*mean).tolist()
