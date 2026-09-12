import json
from pathlib import Path
import unittest

import numpy as np

from scenario_reconstruction.highd import center, make_scene, validate_config


class HighDTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/highd_pilot.json").read_text())
        self.config.update(seconds_before=1, seconds_after=1, frame_stride=1)

    def test_recording_split_leakage_rejected(self):
        self.config["splits"]["test"] = [1]
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_config(self.config)

    def test_center_uses_box_dimensions(self):
        np.testing.assert_equal(center(dict(x=10, y=5, width=4, height=2)), [12, 6])

    def test_both_travel_directions_and_complete_window(self):
        for direction in (1, 2):
            sign = 1 if direction == 2 else -1
            metadata = {"frameRate": "2", "locationId": "1",
                        "upperLaneMarkings": "0;4;8", "lowerLaneMarkings": "0;4;8"}
            meta = {i: {"drivingDirection": str(direction), "width": "4", "height": "2"} for i in (1, 2)}
            tracks = {i: {f: dict(x=sign * (f * 2 + (10 if i == 2 else 0)), y=2,
                                  width=4, height=2, xVelocity=sign * 4, yVelocity=0,
                                  xAcceleration=0, yAcceleration=0, followingId=1, laneId=2)
                          for f in range(1, 6)} for i in (1, 2)}
            scene = make_scene("01", "train", metadata, meta, tracks, 2, 3, self.config)
            self.assertEqual(scene["actors"][0]["role"], "CAV")
            self.assertEqual(scene["actors"][0]["xy"][0], [0, 0])
            self.assertEqual(scene["actors"][0]["xy"][-1][0], 8)
            self.assertEqual(scene["actors"][1]["xy"][0][0], 10)
            self.assertEqual(scene["time"], [0, 0.5, 1, 1.5, 2])
            tracks[1][1]["laneId"] = 3
            with self.assertRaisesRegex(ValueError, "changes_lane"):
                make_scene("01", "train", metadata, meta, tracks, 2, 3, self.config)
            del tracks[1][1]
            with self.assertRaisesRegex(ValueError, "incomplete"):
                make_scene("01", "train", metadata, meta, tracks, 2, 3, self.config)


if __name__ == "__main__":
    unittest.main()
