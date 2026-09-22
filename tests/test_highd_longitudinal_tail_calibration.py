import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_longitudinal_tail_calibration import (
    calibrate, gap_preserving_probabilities, select_candidate, support_report,
)


class TailTests(unittest.TestCase):
    def test_gap_prior_preserves_local_braking_and_support(self):
        counts = np.zeros((2, 1, 2, 3))
        counts[0, 0, 0, 0] = 1000
        counts[1, 0, 0, 2] = 1000
        p = gap_preserving_probabilities(counts, 100)
        np.testing.assert_allclose(p.sum(axis=-1), 1)
        self.assertTrue(np.all(p > 0))
        self.assertGreater(p[0, 0, 1, 0], .9)
        self.assertGreater(p[1, 0, 1, 2], .9)
        self.assertEqual(counts[0, 0, 1].sum(), 0)
        for strength in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                gap_preserving_probabilities(counts, strength)

    def test_selection_cannot_trade_tail_quality_for_average(self):
        def candidate(name, overall, tail):
            return {'name': name, 'overall': {'covered_nll': overall, 'covered_brier': overall},
                    'tail': {'covered_nll': tail, 'covered_brier': tail}}
        base, bad, good = candidate('base', 2, 3), candidate('bad', 1, 4), candidate('good', 1.5, 2.5)
        config = {'minimum_covered_tail_frames': 100, 'minimum_tail_recordings': 3}
        self.assertIs(select_candidate([base, bad, good], True, 100, 3, config)[0], good)
        self.assertIsNone(select_candidate([base, good], False, 100, 3, config)[0])
        self.assertIsNone(select_candidate([base, good], True, 99, 3, config)[0])
        self.assertIsNone(select_candidate([base, good], True, 100, 2, config)[0])

    def test_support_counts_frames_not_cells(self):
        train = np.array([0, 2, 8, 30, 200])[:, None, None, None]
        observed = np.ones_like(train) * 10
        report = support_report(observed, train)
        self.assertEqual(sum(x['observed_frames'] for x in report.values()), 50)
        self.assertEqual(report['n1_2']['training_cells'], 1)

    def test_partial_scan_never_reads_validation_or_exports_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, manifest = root/'model.npz', root/'manifest.json'
            cf = np.ones((2, 2, 2, 3), dtype=np.uint32)
            ff = np.ones((2, 3), dtype=np.uint32)
            np.savez(model, car_following_counts=cf, free_flow_counts=ff,
                car_following_probability=cf/3, free_flow_probability=ff/3,
                gap_axis=[0, 115], range_rate_axis=[-10, 8], speed_axis=[20, 40], acceleration_axis=[-4, -1, 2])
            manifest.write_text(json.dumps({'recordings': [
                {'recording_id': r, 'split': s} for r, s in
                [('01','train'),('03','calibration'),('06','calibration'),('18','validation'),('60','test')]]}))
            config = {'train_recordings': ['01'], 'source_model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
                'chunk_size': 100, 'source_frequency_hz': 25, 'target_frequency_hz': 10,
                'tail_ttc_s': 3, 'parent_concentration': 100, 'prior_concentrations': [1,10],
                'minimum_covered_tail_frames': 100, 'minimum_tail_recordings': 3}
            read = []
            def rows(source, recording, **kwargs):
                read.append(recording)
                yield pd.DataFrame({'xVelocity':[30], 'xAcceleration':[-1], 'dhw':[5],
                                    'precedingXVelocity':[25], 'precedingId':[2]})
            with patch('scenario_reconstruction.highd_longitudinal_tail_calibration._recording_rows', rows):
                result = calibrate(root, manifest, model, config, root/'out', ['03'])
            self.assertEqual(read, ['03'])
            self.assertEqual(result['per_recording']['03']['tail_frames'], 1)
            self.assertEqual(result['status'], 'partial_calibration_diagnostic_only')
            self.assertIsNone(result['selected'])
            self.assertFalse(list((root/'out').glob('*.npz')))
            for recording in ['18','60']:
                with self.assertRaises(ValueError):
                    calibrate(root, manifest, model, config, root/'rejected', [recording])
