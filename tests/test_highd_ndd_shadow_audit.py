import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scenario_reconstruction.highd_ndd_shadow_audit import audit_shadow


class HighDShadowAuditTests(unittest.TestCase):
    def test_audits_normalised_read_only_record(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "tested_and_safe"
            folder.mkdir()
            pdf = [1 / 33] * 33
            episode = {
                "scenario_metadata": {"highd_ndd_shadow": {
                    "mode": "read_only_shadow",
                    "runtime_actions_changed": False,
                    "importance_weights_changed": False,
                }},
                "weight_episode": 1,
                "log_importance_weight": 0,
                "highd_ndd_shadow_step_info": {"0": {"BV": {
                    "original_pdf": pdf, "highd_shadow_pdf": pdf,
                    "fallback": False,
                    "longitudinal_source": "highd_free_flow",
                    "lateral_source": "original_structure_no_current_leader",
                    "l1_distance": 0, "kl_original_to_highd": 0,
                    "original_lane_change_probability": 0,
                    "highd_lane_change_probability": 0,
                }}},
            }
            (folder / "0.json").write_text(json.dumps(episode), encoding="utf-8")

            result = audit_shadow(root)

            self.assertTrue(result["audit_passed"])
            self.assertEqual(result["record_count"], 1)
            self.assertEqual(result["fallback_rate"], 0)

    def test_rejects_invalid_fallback_threshold(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                audit_shadow(temporary, maximum_fallback_rate=1.1)


if __name__ == "__main__":
    unittest.main()
