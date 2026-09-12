import json
from pathlib import Path
import unittest

import numpy as np

from scenario_reconstruction.trajectory_benchmark import (
    box_clearance, conformal_radius, deformation, deform_scene, evaluate,
    make_masked_evidence, reconstruct, risk_metrics, rollout_idm, run_benchmark,
    search, validate_scene, validate_benchmark_config,
)


def synthetic_scene():
    time = np.linspace(0, 4, 101)
    offset, velocity, acceleration = deformation(time, 0, [0, 4])
    ego_xy = np.column_stack([25 * time, np.zeros(len(time))])
    bv_xy = np.column_stack([20 + 23 * time, np.full(len(time), -4)]) + offset
    return {"scene_id": "synthetic", "time": time.tolist(), "changer_id": "2",
            "road_boundaries_y": [-6, -2, 2], "actors": [
                {"id": "1", "role": "CAV", "length": 4.5, "width": 2,
                 "xy": ego_xy.tolist(), "velocity": np.tile([25, 0], (len(time), 1)).tolist(),
                 "acceleration": np.zeros_like(ego_xy).tolist()},
                {"id": "2", "role": "BV", "length": 4.5, "width": 2,
                 "xy": bv_xy.tolist(), "velocity": (velocity + [23, 0]).tolist(),
                 "acceleration": acceleration.tolist()}]}


class TrajectoryBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/trajectory_baselines.json").read_text())
        self.scene = synthetic_scene()

    def test_scene_validation_and_observed_prefix_fixed(self):
        validate_scene(self.scene)
        observed = np.asarray(self.scene["time"]) <= 1
        generated = deform_scene(self.scene, [-8, 0.5], self.config)
        for field, values in zip(("xy", "velocity", "acceleration"), generated):
            original = np.asarray([actor[field] for actor in self.scene["actors"]])
            np.testing.assert_equal(values[:, observed], original[:, observed])
        self.scene["time"][1] = self.scene["time"][0]
        with self.assertRaises(ValueError):
            validate_scene(self.scene)

    def test_invalid_config_rejected(self):
        self.config["max_abs_jerk_mps3"] = float("nan")
        with self.assertRaisesRegex(ValueError, "jerk"):
            validate_benchmark_config(self.config)

    def test_masking_noise_and_conformal_use_recording_scores(self):
        time = np.linspace(0, 4, 101)
        truth = np.column_stack([time, time * 0])
        protocol = self.config["missing_protocols"][0]
        evidence, hidden = make_masked_evidence(time, truth, protocol, 7)
        self.assertTrue(np.isnan(evidence[hidden]).all())
        np.testing.assert_equal(evidence[~hidden], truth[~hidden])
        radius, rank = conformal_radius(list(range(1, 11)), 0.1)
        self.assertEqual((radius, rank), (10, 10))
        with self.assertRaisesRegex(ValueError, "Too few"):
            conformal_radius(list(range(1, 9)), 0.1)

    def test_test_split_locked_by_default(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            run_benchmark("missing.json", "unused", self.config, split="test")

    def test_box_geometry_and_adjacent_lane_ttc(self):
        self.assertLess(box_clearance([0, 0], [2, 0], [4, 2], np.array([4, 2])), 0)
        self.assertGreater(box_clearance([0, 0], [0, 4], [4, 2], np.array([4, 2])), 0)
        xy, velocity, _ = deform_scene(self.scene, [0, 0], self.config)
        xy[1, :, 1] = -4
        metrics = risk_metrics(self.scene, xy, velocity)
        self.assertIsNone(metrics["min_ttc_s"])
        self.assertFalse(metrics["collision"])

    def test_constraints_and_reproducible_search(self):
        self.assertTrue(evaluate(self.scene, [0, 0], self.config)["feasible"])
        self.assertIn("offroad", evaluate(self.scene, [0, 10], self.config)["violations"])
        first = search(self.scene, "uniform", self.config, 7)
        second = search(self.scene, "uniform", self.config, 7)
        self.assertEqual(first["trials"], second["trials"])
        self.assertEqual(first["evaluations"], self.config["budget_per_search"])
        optimized = search(self.scene, "constrained_search", self.config, 7)
        self.assertLessEqual(optimized["evaluations"], self.config["budget_per_search"])
        self.assertTrue(optimized["selected"]["feasible"])

    def test_cav_reacts_after_prefix(self):
        xy, velocity, _ = deform_scene(self.scene, [0, 0], self.config)
        close = xy.copy()
        close[1, :, 1] = 0
        far = xy.copy()
        far[1, :, 1] = -4
        close_result, _, _ = rollout_idm(self.scene, close, velocity, self.config)
        far_result, _, _ = rollout_idm(self.scene, far, velocity, self.config)
        observed = np.asarray(self.scene["time"]) <= 1
        np.testing.assert_equal(close_result[:, observed], close[:, observed])
        self.assertLess(close_result[0, -1, 0], far_result[0, -1, 0])

    def test_no_hidden_truth_input_or_endpoint_extrapolation(self):
        time = np.linspace(0, 4, 101)
        truth = np.column_stack([2 * time, np.zeros(len(time))])
        hidden = (time > 1) & (time < 3)
        evidence = truth.copy()
        evidence[hidden] = np.nan
        for method in ("linear", "cubic", "pchip", "smoothing_spline"):
            completion = reconstruct(time, evidence, method)
            np.testing.assert_allclose(completion, truth, atol=1e-12)
            np.testing.assert_equal(completion[~hidden], evidence[~hidden])
        evidence[0] = np.nan
        with self.assertRaises(ValueError):
            reconstruct(time, evidence, "linear")

    def test_smoothing_spline_uses_declared_noise_scale(self):
        time = np.linspace(0, 4, 101)
        truth = np.column_stack([time, 0.2 * time**2])
        evidence = truth.copy()
        evidence[30:70] = np.nan
        evidence[::2, 1] += 0.1
        interpolated = reconstruct(time, evidence, "cubic")
        smoothed = reconstruct(time, evidence, "smoothing_spline", [0.2, 0.1])
        self.assertFalse(np.array_equal(smoothed[np.isfinite(evidence).all(axis=1)],
                                        evidence[np.isfinite(evidence).all(axis=1)]))
        self.assertLess(np.mean(np.linalg.norm(smoothed[30:70] - truth[30:70], axis=1)),
                        np.mean(np.linalg.norm(interpolated[30:70] - truth[30:70], axis=1)))


if __name__ == "__main__":
    unittest.main()
