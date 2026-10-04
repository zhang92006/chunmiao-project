import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.highd_dynamic_handoff import DynamicAssets, validate_policy_target


class DynamicHandoffTests(unittest.TestCase):
    def test_old_target_is_rejected_even_for_smoke(self):
        for smoke in (False, True):
            with self.assertRaisesRegex(ValueError, "natural target differs"):
                validate_policy_target({"natural_target_sha256": "old", "experiment_target_sha256": "scene"}, "new", "scene", smoke)

    def test_subset_must_be_explicitly_labeled_smoke(self):
        protocol = {"natural_target_sha256": "new", "experiment_target_sha256": "full"}
        self.assertTrue(validate_policy_target(protocol, "new", "full"))
        with self.assertRaisesRegex(ValueError, "mixture differs"):
            validate_policy_target(protocol, "new", "subset")
        self.assertFalse(validate_policy_target(protocol, "new", "subset", True))

    def test_two_private_roots_resolve_without_modifying_recorded_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            base = root / "portable_assets"
            dynamic = root / "dynamic_assets"
            for part in (base, dynamic):
                (part / "files/outputs").mkdir(parents=True)
            raw = b'{"data_path": "G:/dynamic/repo/outputs/sequence.json"}\r\n'
            (dynamic / "files/outputs/manifest.json").write_bytes(raw)
            (dynamic / "files/outputs/sequence.json").write_bytes(b'{}')
            base_manifest = {"source_root": "G:/base/repo", "files": {}, "frozen_source_hashes": {}}
            dyn_manifest = {"source_root": "G:/dynamic/repo", "files": {
                "outputs/manifest.json": {"sha256": hashlib.sha256(raw).hexdigest()},
                "outputs/sequence.json": {"sha256": hashlib.sha256(b'{}').hexdigest()}},
                "sequence_manifest": "outputs/manifest.json", "training_root": "outputs/train",
                "natural_target_sha256": "new"}
            for part, manifest, lock_name in ((base, base_manifest, "highd_portable_assets.lock.json"),
                                             (dynamic, dyn_manifest, "highd_dynamic_assets.lock.json")):
                payload = json.dumps(manifest).encode()
                (part / "manifest.json").write_bytes(payload)
                (root / "configs" / lock_name).write_text(json.dumps({"private_manifest_sha256": hashlib.sha256(payload).hexdigest()}))
            (root / "configs/highd_dynamic_pairs_freeze_v1.json").write_text(json.dumps({"pins": {}, "natural_target_sha256": "new"}))
            assets = DynamicAssets(root)
            self.assertEqual(assets.verify()["dynamic_assets_verified"], 2)
            for path in ("G:/dynamic/repo/outputs/manifest.json", "g:\\dynamic\\repo\\outputs\\manifest.json", "outputs/manifest.json"):
                self.assertEqual(assets.read(path)["data_path"], str(dynamic / "files/outputs/sequence.json"))
                self.assertEqual(Path(assets.resolve(path)).read_bytes(), raw)
            self.assertEqual(assets.translate("G:/base/repo/public.py"), str(root / "public.py"))
            (dynamic / "files/outputs/sequence.json").write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                assets.verify()


if __name__ == "__main__":
    unittest.main()
