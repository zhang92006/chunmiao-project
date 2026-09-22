import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scenario_reconstruction.highd_ndd_baseline import fit_highd_baseline


class HighDNddBaselineTests(unittest.TestCase):
    def _write_recording(self, root: Path, recording_id: str, rows):
        data = root / "data"
        data.mkdir(exist_ok=True)
        columns = [
            "frame", "xVelocity", "xAcceleration", "dhw",
            "precedingXVelocity", "precedingId",
        ]
        with (data / f"{recording_id}_tracks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_fits_direction_normalized_longitudinal_probabilities(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"frame": 5, "xVelocity": 30, "xAcceleration": 0.2, "dhw": 20,
                 "precedingXVelocity": 28, "precedingId": 2},
                {"frame": 10, "xVelocity": -30, "xAcceleration": -0.2, "dhw": 20,
                 "precedingXVelocity": -28, "precedingId": 3},
                {"frame": 15, "xVelocity": 31, "xAcceleration": 0.0, "dhw": 0,
                 "precedingXVelocity": 0, "precedingId": 0},
            ]
            self._write_recording(root, "01", rows)
            self._write_recording(root, "02", rows)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"recordings": [
                {"recording_id": "01", "split": "train"},
                {"recording_id": "02", "split": "validation"},
            ]}), encoding="utf-8")

            summary = fit_highd_baseline(
                root, manifest, root / "output",
                {"frame_stride": 5, "chunk_size": 2, "laplace_alpha": 0.5},
            )

            self.assertEqual(summary["car_following_observations"], 2)
            self.assertEqual(summary["free_flow_observations"], 1)
            self.assertEqual(summary["evaluation"]["validation"]["state_coverage"], 1.0)
            with np.load(root / "output" / "highd_longitudinal_ndd_v1.npz") as model:
                self.assertAlmostEqual(float(model["acceleration_axis"][21]), 0.2)
                self.assertEqual(int(model["car_following_counts"].sum()), 2)

    def test_compares_reference_ndd_support(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{
                "frame": 5, "xVelocity": 30, "xAcceleration": 0.2, "dhw": 20,
                "precedingXVelocity": 28, "precedingId": 2,
            }]
            self._write_recording(root, "01", rows)
            self._write_recording(root, "02", rows)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"recordings": [
                {"recording_id": "01", "split": "train"},
                {"recording_id": "02", "split": "validation"},
            ]}), encoding="utf-8")
            reference = root / "reference"
            (reference / "CF").mkdir(parents=True)
            (reference / "FF").mkdir()
            np.save(reference / "CF" / "Optimized_CF_pdf_array.npy", np.zeros((116, 19, 21, 31)))
            np.save(reference / "FF" / "Optimized_FF_pdf_array.npy", np.full((21, 31), 1 / 31))

            summary = fit_highd_baseline(
                root, manifest, root / "output", {"frame_stride": 5},
                reference_ndd_root=reference,
            )

            validation = summary["evaluation"]["validation"]
            self.assertEqual(validation["reference_ndd_zero_probability_rate"], 1.0)
            self.assertGreater(validation["reference_ndd_mean_negative_log_likelihood"], 20.0)


if __name__ == "__main__":
    unittest.main()
