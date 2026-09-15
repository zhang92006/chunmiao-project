import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.shrp2_diffusion_data_audit import build_pair_window
from scenario_reconstruction.shrp2_multibv_seed_export import build_multibv_seed


class MultiBVSeedExportTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            Path("configs/shrp2_diffusion_data_audit.json").read_text(encoding="utf-8")
        )
        self.meta = {
            "impact_timestamp": 4000,
            "ego_length": 4.5,
            "ego_width": 1.8,
            "target_length": 4.5,
            "target_width": 1.8,
        }
        times = np.arange(0, 4.01, 0.1)
        common = {
            "event_id": 1,
            "time": times,
            "x_ego": times,
            "y_ego": np.zeros_like(times),
            "v_ego": np.full_like(times, 1.0),
            "psi_ego": np.zeros_like(times),
        }
        primary = pd.DataFrame({
            **common,
            "target_id": 2,
            "x_sur": times + 20.0,
            "y_sur": np.zeros_like(times),
            "v_sur": np.full_like(times, 1.0),
            "psi_sur": np.zeros_like(times),
        })
        context = pd.DataFrame({
            **common,
            "target_id": 3,
            "x_sur": times + 30.0,
            "y_sur": np.full_like(times, 2.0),
            "v_sur": np.full_like(times, 1.5),
            "psi_sur": np.zeros_like(times),
        })
        self.rows = pd.concat([primary, context], ignore_index=True)
        pair = build_pair_window(primary, self.meta, self.config)
        pair["source"] = {
            "category": "NearCrash",
            "event_id": 1,
            "target_id": 2,
            "split": "train",
            "conflict": "leading",
            "observed_target_count": 2,
        }
        self.pair = pair

    def test_exports_primary_and_nearest_context_in_source_frame(self):
        seed = build_multibv_seed(self.pair, self.rows, self.meta, self.config, bv_count=2)
        self.assertEqual(seed["record_type"], "shrp2_measured_multibv_seed_v1")
        self.assertEqual(np.asarray(seed["states"]).shape, (41, 3, 4))
        self.assertEqual(seed["source"]["context_target_ids"], [3])
        self.assertFalse(seed["drl_training_ready"])
        self.assertEqual(seed["condition"]["mapping_status"], "source_frame_only; lane and route mapping pending")

    def test_requires_a_context_target(self):
        with self.assertRaisesRegex(ValueError, "insufficient_context_targets"):
            build_multibv_seed(
                self.pair,
                self.rows[self.rows.target_id == 2],
                self.meta,
                self.config,
                bv_count=2,
            )

    def test_anchor_only_mode_accepts_context_without_full_history(self):
        short_context = self.rows[(self.rows.target_id == 3) & (self.rows.time >= 3.8)]
        short_rows = pd.concat([
            self.rows[self.rows.target_id == 2], short_context
        ], ignore_index=True)
        seed = build_multibv_seed(
            self.pair, short_rows, self.meta, self.config,
            bv_count=2, context_mode="anchor_only",
        )
        self.assertEqual(seed["condition"]["context_mode"], "anchor_only")
        self.assertEqual(seed["time_s"], [4.0])
        self.assertEqual(np.asarray(seed["states"]).shape, (1, 3, 4))
        with self.assertRaisesRegex(ValueError, "insufficient_context_history"):
            build_multibv_seed(
                self.pair, short_rows, self.meta, self.config,
                bv_count=2, context_mode="full",
            )


if __name__ == "__main__":
    unittest.main()
