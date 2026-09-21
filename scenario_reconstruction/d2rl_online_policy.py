"""Portable deterministic epsilon inference, with no Ray dependency in SUMO."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


class OnlineEpsilonPolicy:
    def __init__(self, model_path):
        path = Path(model_path)
        self.metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.metadata.get('model_sha256') != digest:
            raise ValueError('Online model checksum mismatch')
        if self.metadata.get('observation_size') != 14 or self.metadata.get('action_size') != 2:
            raise ValueError('Online policy requires 14 observations and 2 epsilons')
        self.model = torch.jit.load(str(path), map_location='cpu').eval()

    def compute_action(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (14,) or not np.all(np.isfinite(obs)):
            raise ValueError('Online policy requires a finite 14-dimensional observation')
        with torch.no_grad():
            action = np.asarray(self.model(torch.tensor(obs.tolist(), dtype=torch.float32).unsqueeze(0)).tolist()[0])
        if action.shape != (2,) or not np.all(np.isfinite(action)) or np.any(action < .001 - 1e-7) or np.any(action > .999 + 1e-7):
            raise ValueError('Online epsilon output violates the exported action contract')
        return action.tolist()
