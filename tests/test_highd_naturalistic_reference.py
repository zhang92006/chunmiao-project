import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scenario_reconstruction.highd_naturalistic_reference import support_ids
from scenario_reconstruction.highd_naturalistic_compare import (
    compare_native, describe_pairs, range_matched,
)


class NaturalisticReferenceTests(unittest.TestCase):
    def test_support_is_initial_same_direction_and_does_not_need_future(self):
        scene = pd.DataFrame({"id": [1, 2, 3, 4, 5], "x": [0, 90, 205, 210, 400],
                              "direction": [1, 1, 1, 2, 1], "xVelocity": [-30]*5})
        config = {"support_range_m": 120, "speed_range_mps": [20, 40]}
        selected, excluded = support_ids(scene, [1, 2], config)
        self.assertEqual(selected, [3])
        self.assertEqual(excluded["support_speed_outside_domain"], 0)
        scene.loc[2, "xVelocity"] = -15
        selected, excluded = support_ids(scene, [1, 2], config)
        self.assertEqual(selected, [])
        self.assertEqual(excluded["support_speed_outside_domain"], 1)

    def row(self, actor="BV_core", gap=10, time=.1):
        return {"vehicle_id": actor, "time": time, "lane_index": 0, "speed_mps": 25,
                "acceleration_mps2": 0, "gap_m": gap,
                "headway_s": gap/25 if gap is not None else None, "ttc_s": None,
                "leader_id": "BV_lead" if gap is not None else None}

    def test_range_matching_preserves_raw_and_boundary(self):
        raw = self.row(gap=116)
        masked = range_matched(raw, 115)
        self.assertIsNone(masked["gap_m"])
        self.assertIsNone(masked["headway_s"])
        self.assertIsNone(masked["leader_id"])
        self.assertEqual(raw["gap_m"], 116)
        self.assertEqual(range_matched(self.row(gap=115), 115)["gap_m"], 115)

    def test_gap_common_finite_metrics_use_identical_pairs(self):
        sim = [self.row(gap=10), self.row(gap=None), self.row(gap=100)]
        ref = [self.row(gap=12), self.row(gap=110), self.row(gap=None)]
        result = describe_pairs(sim, ref)["gap_m"]
        self.assertEqual(result["simulation"]["count"], 2)
        self.assertEqual(result["common_finite_pair_count"], 1)
        self.assertEqual(result["common_finite_wasserstein_distance"], 2)

    def test_comparison_excludes_support_and_detects_missing_core(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = root / "template.json"
            template.write_text(json.dumps({"template_id": "sample", "bridge_metadata": {}}))
            ref = root / "reference.json"
            ref.write_text(json.dumps({"evaluation_actor_ids": ["BV_core"], "snapshots": [self.row()]}))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [{"template_path": str(template),
                                                          "reference_path": str(ref)}]}))
            episode_dir = root / "episode_0000"
            episode_dir.mkdir()
            (episode_dir / "naturalistic_audit.json").write_text(json.dumps({
                "probability_audit_passed": True, "model_reconstruction_checked": True}))
            path = episode_dir / "naturalistic_episode.json"
            episode = {"metadata": {"template_id": "sample"},
                       "summary": {"probability_audit_passed": True},
                       "snapshots": [self.row(), self.row("BV_support", gap=110)]}
            path.write_text(json.dumps(episode))
            result = compare_native(manifest, root, root / "comparison.json")
            self.assertEqual(result["paired_state_count"], 1)
            self.assertEqual(result["excluded_support_bv_windows"], 1)
            self.assertEqual(result["metrics"]["gap_m"]["simulation"]["mean"], 10)
            episode["snapshots"] = [self.row("BV_support")]
            path.write_text(json.dumps(episode))
            with self.assertRaisesRegex(ValueError, "lost an evaluated"):
                compare_native(manifest, root, root / "comparison.json")


if __name__ == "__main__":
    unittest.main()
