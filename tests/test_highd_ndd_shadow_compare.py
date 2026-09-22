import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scenario_reconstruction.highd_ndd_shadow_compare import compare_shadow_roots


class HighDShadowCompareTests(unittest.TestCase):
    def _write_episode(self, root: Path, lane_probability: float, source: str):
        path = root / "tested_and_safe" / "0.json"
        path.parent.mkdir(parents=True)
        original = np.full(33, 1 / 33).tolist()
        episode = {
            "weight_episode": 1.0,
            "log_importance_weight": 0.0,
            "scenario_metadata": {
                "template_id": "paired",
                "source_event_id": 1,
                "source_split": "train",
                "simulation_seed": 7,
            },
            "highd_ndd_shadow_step_info": {
                "0.1": {
                    "BV": {
                        "original_pdf": original,
                        "highd_lane_change_probability": lane_probability,
                        "lateral_source": source,
                    }
                }
            },
        }
        path.write_text(json.dumps(episode), encoding="utf-8")

    def test_compares_identical_states_by_leader_presence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            strict = root / "strict"
            extension = root / "extension"
            self._write_episode(
                strict, 0.0, "original_structure_no_current_leader"
            )
            self._write_episode(extension, 0.001, "highd_adjacent_context")

            result = compare_shadow_roots(strict, extension)

            self.assertTrue(result["paired_inputs_and_weights_passed"])
            self.assertEqual(result["paired_record_count"], 1)
            group = result["by_current_leader_presence"]["no_current_leader"]
            self.assertEqual(group["record_count"], 1)
            self.assertAlmostEqual(
                group["extension_mean_lane_change_probability"], 0.001
            )

    def test_rejects_different_original_probabilities(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            strict = root / "strict"
            extension = root / "extension"
            self._write_episode(strict, 0.0, "original_structure_no_current_leader")
            self._write_episode(extension, 0.001, "highd_adjacent_context")
            path = extension / "tested_and_safe" / "0.json"
            episode = json.loads(path.read_text(encoding="utf-8"))
            episode["highd_ndd_shadow_step_info"]["0.1"]["BV"]["original_pdf"][0] += 0.1
            path.write_text(json.dumps(episode), encoding="utf-8")

            result = compare_shadow_roots(strict, extension)

            self.assertFalse(result["paired_inputs_and_weights_passed"])


if __name__ == "__main__":
    unittest.main()
