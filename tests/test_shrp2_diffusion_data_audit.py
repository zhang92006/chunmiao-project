import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.shrp2_diffusion_data_audit import audit_dataset, build_pair_window


class DiffusionDataAuditTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/shrp2_diffusion_data_audit.json").read_text(encoding="utf-8"))
        self.meta = {"impact_timestamp": 4000, "ego_length": 4.5, "ego_width": 1.8, "target_length": 4.5, "target_width": 1.8}
        times = np.arange(0, 4.01, .1)
        self.rows = pd.DataFrame({"event_id": 1, "target_id": 2, "time": times, "x_ego": times, "y_ego": 0., "v_ego": 1., "psi_ego": 0., "x_sur": times + 20., "y_sur": 0., "v_sur": 1., "psi_sur": 0.})

    def test_noncolliding_window_is_usable_and_keeps_front_bumper_reference(self):
        window = build_pair_window(self.rows, self.meta, self.config)
        self.assertTrue(window["training_window_ready"])
        self.assertFalse(window["geometry_diagnostic"]["collision_in_reference_window"])
        self.assertEqual(window["position_reference"][1], "BV_front_bumper")
        self.assertEqual(np.asarray(window["states"]).shape, (41, 2, 4))
        self.assertAlmostEqual(window["states"][0][1][0], 20.)

    def test_missing_segment_is_not_silently_interpolated(self):
        rows = self.rows[(self.rows.time < 1.0) | (self.rows.time > 2.0)]
        with self.assertRaisesRegex(ValueError, "source_gap_exceeds_limit"):
            build_pair_window(rows, self.meta, self.config)

    def test_stationary_heading_is_masked_without_discarding_position(self):
        rows = self.rows.copy()
        rows["x_sur"] = 20.
        rows["v_sur"] = 0.
        window = build_pair_window(rows, self.meta, self.config)
        self.assertTrue(window["training_window_ready"])
        self.assertTrue(all(not m[1][3] for m in window["state_mask"]))

    def test_cross_category_duplicates_are_not_counted_as_independent_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for category in ("Crash", "NearCrash"):
                folder = root / "Export_Online" / "SafetyCriticalTestSet" / category
                folder.mkdir(parents=True)
                pd.DataFrame({"event_id": [1, 2], "conflict": ["leading", "leading"]}).to_csv(folder / "event_meta.csv", index=False)
            summary = audit_dataset(root, root / "audit", self.config)
        self.assertEqual(summary["category_event_count_sum"], 4)
        self.assertEqual(summary["unique_event_count"], 2)
        self.assertEqual(summary["cross_category_duplicate_event_count"], 2)
        self.assertIsNone(summary["quality_pass_unique_event_count"])
        self.assertEqual(summary["inventory"][0]["events_by_split"], summary["inventory"][1]["events_by_split"])

    @unittest.skipUnless(importlib.util.find_spec("tables"), "HDF5 integration requires PyTables")
    def test_full_scan_exports_measured_nearcrash_with_source_category(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "Export_Online" / "SafetyCriticalTestSet" / "NearCrash"
            folder.mkdir(parents=True)
            meta = {**self.meta, "event_id": 1, "event_category": "NearCrash", "severity_first": 2., "conflict": "leading", "duration_enough": True}
            pd.DataFrame([meta]).to_csv(folder / "event_meta.csv", index=False)
            self.rows.to_hdf(folder / "event_data.h5", key="data")
            summary = audit_dataset(root, root / "audit", self.config, scan_trajectories=True, export_windows=True)
            files = list((root / "audit").glob("*/windows.jsonl"))
            self.assertEqual(len(files), 1)
            window = json.loads(files[0].read_text(encoding="utf-8").strip())
        self.assertEqual(summary["quality_pass_unique_event_count"], 1)
        self.assertEqual(window["condition"]["event_category"], "NearCrash")
        self.assertFalse(window["geometry_diagnostic"]["collision_in_reference_window"])


if __name__ == "__main__":
    unittest.main()
