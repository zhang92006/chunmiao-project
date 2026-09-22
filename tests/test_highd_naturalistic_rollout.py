import copy
from types import SimpleNamespace
import unittest

import numpy as np

from scenario_reconstruction.highd_naturalistic_rollout import (
    actor_rng, execution_pdf, _summary, validate_template_for_ndd,
)


class NaturalisticTests(unittest.TestCase):
    def test_core_rng_is_independent_of_added_support_actors(self):
        expected = actor_rng(7, "BV_core").random(30)
        actor_rng(7, "BV_support").random(1000)
        np.testing.assert_array_equal(expected, actor_rng(7, "BV_core").random(30))
        self.assertFalse(np.array_equal(expected, actor_rng(8, "BV_core").random(30)))

    def observation(self):
        return {"Ego": {"could_drive_adjacent_lane_left": True,
                        "could_drive_adjacent_lane_right": False}, "Lead": None}

    def record(self):
        candidate = np.full(33, 1 / 33)
        pdf = execution_pdf(candidate, self.observation())
        return {"candidate_pdf": candidate.tolist(), "executed_pdf": pdf.tolist(),
                "p_action": float(pdf[0]), "q_action": float(pdf[0]), "action_id": 0,
                "observation": self.observation(), "applied": True, "fallback": False,
                "speed_boundary_adjusted": False, "longitudinal_source": "highd_free_flow"}

    def test_masks_before_sampling_and_preserves_free_flow_left(self):
        original = np.full(33, 1 / 33)
        pdf = execution_pdf(original, self.observation())
        self.assertEqual(pdf[1], 0)
        self.assertAlmostEqual(pdf[0], 1 / 32)
        self.assertAlmostEqual(pdf.sum(), 1)
        self.assertGreater(original[1], 0)

    def test_audit_detects_probability_tampering(self):
        row = self.record()
        self.assertTrue(_summary([row], [])["probability_audit_passed"])
        row["p_action"] *= .5
        self.assertFalse(_summary([row], [])["probability_audit_passed"])

    def test_audit_detects_unapplied_command_and_wrong_transform(self):
        row = self.record()
        row["applied"] = False
        self.assertIn("sampled command was not applied", _summary([row], [])["failures"])
        row = self.record()
        row["candidate_pdf"][0] *= 2
        self.assertIn("executed PDF disagrees with legality transform",
                      _summary([row], [])["failures"])

    def test_boundary_crossings_and_observation_exposure(self):
        rows = []
        for time, lane in ((.1, 0), (.2, 0), (.3, 1)):
            rows.append({"time": time, "vehicle_id": "BV_test", "lane_index": lane,
                         "speed_mps": 25, "acceleration_mps2": 0,
                         "headway_s": None, "ttc_s": None, "gap_m": None})
        summary = _summary([self.record()], rows)
        self.assertEqual(summary["observed_lane_crossings"], 1)
        self.assertAlmostEqual(summary["bv_observed_seconds"], .2)

    def test_rejects_faults_and_test_inputs(self):
        template = SimpleNamespace(events=[], bridge_metadata={"source_split": "train"},
            tags=[], ego=SimpleNamespace(id="CAV", speed=30),
            actors=[SimpleNamespace(id="BV_test", speed=30)])
        validate_template_for_ndd(template)
        invalid = copy.deepcopy(template)
        invalid.events = ["forced action"]
        with self.assertRaises(ValueError):
            validate_template_for_ndd(invalid)
        template.bridge_metadata["source_split"] = "test"
        with self.assertRaises(ValueError):
            validate_template_for_ndd(template)

    def test_explicit_lengths_must_cover_every_actor(self):
        template = SimpleNamespace(events=[], bridge_metadata={
            "source_split": "train", "vehicle_lengths_m": {"CAV": 4.5, "BV_test": 12.0}},
            tags=[], ego=SimpleNamespace(id="CAV", speed=30),
            actors=[SimpleNamespace(id="BV_test", speed=30)])
        validate_template_for_ndd(template)
        template.bridge_metadata["vehicle_lengths_m"]["BV_test"] = -1
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_template_for_ndd(template)


if __name__ == "__main__":
    unittest.main()
