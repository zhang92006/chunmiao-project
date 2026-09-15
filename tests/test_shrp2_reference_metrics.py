import json
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.shrp2_reference_metrics import compare_selected, observation_tracks


class ReferenceMetricsTests(unittest.TestCase):
    @staticmethod
    def episode(last_time):
        return {"av_obs": {str(t): {"Ego": {"position": [400 + t, 46], "heading": 90, "velocity": 1}, "Lead": {"veh_id": "BV_primary", "position": [410 + t, 46], "heading": 90, "velocity": 1}} for t in range(last_time + 1)}}

    def test_front_bumper_and_sumo_angle_are_converted_to_center_and_cartesian(self):
        tracks = observation_tracks(self.episode(1), 5.)
        self.assertEqual(tracks["CAV"]["position"][0].tolist(), [397.5, 46.])
        self.assertEqual(tracks["BV_front"]["position"][0].tolist(), [410., 46.])
        self.assertEqual(tracks["BV_center"]["position"][0].tolist(), [407.5, 46.])
        self.assertAlmostEqual(tracks["CAV"]["heading"][0], 0.)

    def test_all_candidates_use_shared_observed_window_without_extrapolation(self):
        cav = {"time_s": [0, 1, 2], "xy_m": [[0, 0], [1, 0], [2, 0]], "reported_speed_mps": [1, 1, 1], "reported_heading_rad": [0, 0, 0]}
        bv = {**cav, "xy_m": [[10, 0], [11, 0], [12, 0]], "source_front_bumper_xy_m": [[12.5, 0], [13.5, 0], [14.5, 0]]}
        ref = {"source": {"event_id": 1, "source_split": "train"}, "actors": {"CAV": cav, "BV_primary": bv}, "collision_audit": {"first_sampled_contact_time_s": 2}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = root / "template.json"
            template.write_text(json.dumps({"ego": {"position": 400}}), encoding="utf-8")
            selected = []
            for i, end in enumerate((1, 2)):
                path = root / f"episode{i}.json"
                path.write_text(json.dumps(self.episode(end)), encoding="utf-8")
                selected.append({"candidate_index": i, "episode_path": str(path), "template_path": str(template), "gap_offset_m": 0, "end_time_s": end, "collision_time_error_s": end - 2})
            summary = {"selected": selected, "executed_count": 2, "outcomes": selected}
            report = compare_selected(ref, summary)
        self.assertEqual(report["common_window_s"], [0., 1.])
        self.assertEqual(report["common_sample_count"], 2)
        for outcome in report["outcomes"]:
            self.assertAlmostEqual(outcome["common_window_metrics"]["BV_front"]["ade_m"], 0.)


if __name__ == "__main__":
    unittest.main()
