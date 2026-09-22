import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_longitudinal_calibration import calibrate, empirical_probabilities, score_counts


class LongitudinalCalibrationTests(unittest.TestCase):
    def test_prior_keeps_positive_support_and_favors_observed_actions(self):
        cf = np.zeros((2, 2, 2, 3))
        cf[..., 1] = 10
        cf[0, 0, 0] = 0
        ff = np.array([[0, 100, 0], [0, 100, 0]])
        cp, fp = empirical_probabilities(cf, ff, 30)
        np.testing.assert_allclose(cp.sum(axis=-1), 1)
        np.testing.assert_allclose(fp.sum(axis=-1), 1)
        self.assertTrue(np.all(cp > 0))
        self.assertGreater(cp[0, 0, 0, 1], .9)
        self.assertEqual(cf[0, 0, 0].sum(), 0)  # Empty-state runtime gate remains unchanged.

    def test_score_excludes_unseen_runtime_states(self):
        cf = np.zeros((2, 1, 1, 3))
        cf[0, 0, 0, 1] = 10
        ff = np.array([[0, 100, 0]])
        observed_cf = cf.copy()
        observed_cf[1, 0, 0, 0] = 100
        pdf = empirical_probabilities(cf, ff, 30)
        result = score_counts(observed_cf, ff, cf, ff, pdf)
        self.assertEqual(result["eligible_observations"], 210)
        self.assertEqual(result["runtime_covered_observations"], 110)
        self.assertLess(result["covered_nll"], .1)

    def test_rejects_invalid_prior_strength(self):
        with self.assertRaises(ValueError):
            empirical_probabilities(np.ones((2, 2, 2, 3)), np.ones((2, 3)), 0)

    def test_freezes_selection_before_validation_and_never_reads_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, manifest, output = root / "model.npz", root / "manifest.json", root / "out"
            cf, ff = np.ones((2, 2, 2, 3)), np.ones((2, 3))
            np.savez(model, car_following_counts=cf, free_flow_counts=ff,
                     car_following_probability=cf/3, free_flow_probability=ff/3,
                     gap_axis=[0, 115], range_rate_axis=[-10, 8], speed_axis=[20, 40],
                     acceleration_axis=[-4, -1, 2])
            manifest.write_text(json.dumps({"recordings": [
                {"recording_id": r, "split": s} for r, s in
                (("01", "train"), ("03", "calibration"), ("18", "validation"), ("60", "test"))]}))
            config = {"train_recordings": ["01"], "source_frequency_hz": 25,
                      "target_frequency_hz": 10, "chunk_size": 100,
                      "parent_concentration": 100, "prior_concentrations": [1, 10],
                      "source_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
            read = []

            def rows(source_root, recording, **kwargs):
                read.append(recording)
                if recording == "18":
                    self.assertTrue((output / "selection.json").exists())
                    self.assertTrue((output / "highd_longitudinal_ndd_calibrated_v2.npz").exists())
                return [pd.DataFrame({"xVelocity": [30], "xAcceleration": [0],
                    "dhw": [20], "precedingXVelocity": [29], "precedingId": [2]})]

            with patch("scenario_reconstruction.highd_longitudinal_calibration._recording_rows", rows):
                calibrate(root, manifest, model, config, output)
            self.assertEqual(read, ["03", "18"])


if __name__ == "__main__":
    unittest.main()
