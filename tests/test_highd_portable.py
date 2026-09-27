import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_portable import PortableAssets, policy_provider_class
from scenario_reconstruction.highd_dual_ndd import PairActionDistribution
from scenario_reconstruction.highd_dual_ndd_runtime import source_rows
from scenario_reconstruction.highd_dual_ndd_criticality import CAVCriticalityProvider


class PortableTests(unittest.TestCase):
    def test_known_windows_paths_are_rebound_without_rewriting_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            folder = root / "portable_assets"
            (folder / "files/configs").mkdir(parents=True)
            model = folder / "files/configs/model.json"
            model.write_bytes(b'{"x": 3}\r\n')
            expected = hashlib.sha256(model.read_bytes()).hexdigest()
            manifest = {"source_root": "G:/original/repo", "files": {
                "configs/model.json": {"sha256": expected}}, "frozen_source_hashes": {}}
            payload = json.dumps(manifest).encode()
            (folder / "manifest.json").write_bytes(payload)
            (root / "configs/highd_portable_assets.lock.json").write_text(json.dumps({
                "private_manifest_sha256": hashlib.sha256(payload).hexdigest()}))
            assets = PortableAssets(root)
            for original in ("G:/original/repo/configs/model.json", "g:\\original\\repo\\configs\\model.json", "configs/model.json"):
                self.assertEqual(Path(assets.resolve(original)), model)
                self.assertEqual(assets.read(original), {"x": 3})
            self.assertEqual(assets.translate("not a path"), "not a path")
            self.assertEqual(assets.verify()["assets_verified"], 1)
            self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), expected)
            model.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                assets.verify()

    def scene(self, far=False):
        rows = [{"id": i, "center_x": x, "center_y": lane * 3.2, "lane": lane,
                 "length": 5., "width": 1.8, "speed": speed, "acceleration": 0., "lateral_speed": 0.}
                for i, x, lane, speed in [(100, 50, 0, 30), (1, 300 if far else 40, 0, 32), (2, 350, 1, 30)]]
        latest = pd.DataFrame(source_rows(rows, 0, [0, 3.2]))
        return SimpleNamespace(latest=latest, _own=lambda i: latest[latest.id == i].reset_index(drop=True),
                               cav_id=100, centers={2: 0., 3: 3.2}, last_decision=0)

    def test_online_adapter_changes_only_epsilon_not_p_or_risk_gate(self):
        natural = np.zeros((93, 93))
        natural[::3, ::3] = 1 / (31 * 31)
        distribution = PairActionDistribution((1, 2), ("first", "second"), natural, "test", True)
        calls = []
        def policy(obs):
            calls.append(obs)
            return np.array([.2, .4])
        provider = policy_provider_class(policy)()
        baseline = CAVCriticalityProvider()(self.scene(), {}, distribution)
        actual = provider(self.scene(), {}, distribution)
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual.p, baseline.p)
        np.testing.assert_allclose(actual.h, baseline.h, atol=1e-15)
        np.testing.assert_array_equal(actual.epsilon, [.2, .4])
        self.assertEqual(actual.training_observation, baseline.training_observation)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(provider(self.scene(far=True), {}, distribution))
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(actual.matrix.sum(), 1.)


if __name__ == "__main__":
    unittest.main()
