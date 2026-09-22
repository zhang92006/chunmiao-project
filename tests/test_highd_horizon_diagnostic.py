import copy
import json
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.highd_horizon_diagnostic import extended_template, check_prefix
from scenario_reconstruction.highd_naturalistic_compare import compare_native


class HorizonTests(unittest.TestCase):
    def test_extension_preserves_source_and_marks_scope(self):
        original = {'duration': 4, 'bridge_metadata': {'source_split': 'validation'}}
        new = extended_template(original, 8)
        self.assertEqual(original['duration'], 4)
        self.assertNotIn('horizon_diagnostic', original['bridge_metadata'])
        self.assertEqual(new['duration'], 8)
        self.assertEqual(new['bridge_metadata']['horizon_diagnostic']['original_duration_s'], 4)
        for duration in [3, 4, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                extended_template(original, duration)

    def test_prefix_detects_dropped_or_changed_history(self):
        row = {'time': 4.0, 'vehicle_id': 'BV_1', 'speed_mps': 30}
        original = {'decisions': [row], 'snapshots': [row]}
        new = copy.deepcopy(original)
        new['snapshots'].append(dict(row, time=4.1))
        self.assertTrue(check_prefix(original, new)['passed'])
        new['snapshots'][0]['speed_mps'] = 31
        self.assertFalse(check_prefix(original, new)['passed'])
        new = copy.deepcopy(original)
        new['decisions'] = []
        self.assertFalse(check_prefix(original, new)['passed'])
        new = copy.deepcopy(original)
        new['snapshots'].append(dict(row, vehicle_id='BV_extra'))
        self.assertFalse(check_prefix(original, new)['passed'])

    def test_extended_output_cannot_enter_reference_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'records': []}))
            episode_dir = root / 'episode_0000'
            episode_dir.mkdir()
            (episode_dir / 'naturalistic_episode.json').write_text(json.dumps({
                'metadata': {'horizon_diagnostic': {'original_duration_s': 4}}}))
            with self.assertRaisesRegex(ValueError, 'not measured-reference'):
                compare_native(manifest, root, root / 'comparison.json')
            self.assertFalse((root / 'comparison.json').exists())
