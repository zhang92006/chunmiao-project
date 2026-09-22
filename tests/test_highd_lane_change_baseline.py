import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scenario_reconstruction.highd_lane_change_baseline import (
    _decision_rows,
    fit_lane_change_baseline,
)


class HighDLaneChangeBaselineTests(unittest.TestCase):
    def _write_recording(self, root: Path, recording_id: str, direction: int):
        data = root / "data"
        data.mkdir(exist_ok=True)
        frames = list(range(1, 61))
        lane = [2 if frame < 40 else 3 for frame in frames]
        pd.DataFrame({
            "frame": frames,
            "id": [1] * len(frames),
            "xVelocity": [30 if direction == 2 else -30] * len(frames),
            "dhw": [20] * len(frames),
            "precedingXVelocity": [29 if direction == 2 else -29] * len(frames),
            "precedingId": [2] * len(frames),
            "laneId": lane,
        }).to_csv(data / f"{recording_id}_tracks.csv", index=False)
        pd.DataFrame({"id": [1], "drivingDirection": [direction]}).to_csv(
            data / f"{recording_id}_tracksMeta.csv", index=False
        )

    def test_labels_one_direction_relative_lane_change_decision(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, "01", direction=1)

            rows = _decision_rows(
                root, "01", source_hz=25, target_hz=10, decision_lead_s=1.0
            )

            self.assertEqual(int((rows["action_index"] == 0).sum()), 1)
            self.assertEqual(int((rows["action_index"] == 2).sum()), 0)

    def test_excludes_execution_tail_after_boundary_crossing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, "01", direction=1)

            without_tail = _decision_rows(
                root, "01", source_hz=25, target_hz=10,
                decision_lead_s=0.5, execution_tail_s=0.0,
            )
            with_tail = _decision_rows(
                root, "01", source_hz=25, target_hz=10,
                decision_lead_s=0.5, execution_tail_s=0.5,
            )

            self.assertGreater(len(without_tail), len(with_tail))
            self.assertEqual(int((with_tail["action_index"] == 0).sum()), 1)

    def test_fits_and_evaluates_recording_split(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, "01", direction=1)
            self._write_recording(root, "02", direction=1)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"recordings": [
                {"recording_id": "01", "split": "train"},
                {"recording_id": "02", "split": "validation"},
            ]}), encoding="utf-8")
            config = {
                "train_split": "train", "source_frequency_hz": 25,
                "target_frequency_hz": 10, "decision_lead_s": 1.0,
                "laplace_alpha": 0.5,
                "grid": {
                    "speed": [20, 40, 1], "gap": [0, 115, 1],
                    "range_rate": [-10, 8, 1],
                },
            }

            summary = fit_lane_change_baseline(
                root, manifest, root / "output", config
            )

            self.assertEqual(summary["evaluation"]["train"]["left_actions"], 1)
            self.assertEqual(
                summary["evaluation"]["validation"]["state_coverage"], 1.0
            )

    def test_rejects_unknown_evaluation_split(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, "01", direction=1)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"recordings": [
                {"recording_id": "01", "split": "train"},
            ]}), encoding="utf-8")
            config = {
                "train_split": "train", "evaluation_splits": ["test"],
                "source_frequency_hz": 25, "target_frequency_hz": 10,
                "decision_lead_s": 0.5, "laplace_alpha": 0.5,
                "grid": {
                    "speed": [20, 40, 1], "gap": [0, 115, 1],
                    "range_rate": [-10, 8, 1],
                },
            }

            with self.assertRaisesRegex(ValueError, "Unknown evaluation splits"):
                fit_lane_change_baseline(root, manifest, root / "output", config)


if __name__ == "__main__":
    unittest.main()
