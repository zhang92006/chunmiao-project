import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scenario_reconstruction.highd_joint_action_audit import audit_joint_actions


class HighDJointActionAuditTests(unittest.TestCase):
    def _write_recording(self, root: Path, recording_id: str):
        data = root / "data"
        data.mkdir(exist_ok=True)
        rows = []
        for frame in range(1, 101):
            acceleration = -1.0 if frame % 2 else 1.0
            rows.extend([
                {"frame": frame, "id": 1, "xVelocity": 30,
                 "xAcceleration": acceleration, "dhw": 20,
                 "precedingXVelocity": 30, "precedingId": 2, "laneId": 2},
                {"frame": frame, "id": 2, "xVelocity": 30,
                 "xAcceleration": acceleration, "dhw": 0,
                 "precedingXVelocity": 0, "precedingId": 0, "laneId": 2},
            ])
        pd.DataFrame(rows).to_csv(
            data / f"{recording_id}_tracks.csv", index=False
        )

    def test_detects_synchronized_action_dependence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, "01")
            self._write_recording(root, "02")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"recordings": [
                {"recording_id": "01", "split": "train"},
                {"recording_id": "02", "split": "validation"},
            ]}), encoding="utf-8")
            config = {
                "train_split": "train", "source_frequency_hz": 25,
                "target_frequency_hz": 25, "laplace_alpha": 0.5,
                "acceleration_range": [-4, 2],
                "action_thresholds": [-0.5, 0.5],
                "minimum_state_count_for_cmi": 10,
                "material_nll_penalty": 0.01,
                "state_grid": {
                    "speed": [20, 40, 5], "gap": [0, 120, 20],
                    "range_rate": [-10, 10, 4],
                },
            }

            summary = audit_joint_actions(
                root, manifest, root / "output", config
            )

            validation = summary["evaluation"]["validation"]
            self.assertGreater(validation["factorization_nll_penalty"], 0.1)
            self.assertFalse(summary["factorized_assumption_supported"])
            self.assertGreater(summary["conditional_mutual_information_nats"], 0.1)


if __name__ == "__main__":
    unittest.main()
