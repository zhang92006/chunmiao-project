import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.shrp2_collision import (
    associate_primary_target,
    front_bumper_to_center,
    project_split,
    signed_box_clearance,
    simulate_collision,
    validate_config,
)


class SHRP2CollisionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema_version": 1,
            "dataset_doi": "10.15787/VTT1/T7UUC1",
            "seed": 7,
            "source_category": "Crash",
            "source_split": "train",
            "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "conflict_types": {"leading": "rear_end"},
            "maximum_target_alignment_dt_s": 0.2,
            "maximum_source_box_clearance_m": 8.0,
            "duration_s": 6.0,
            "impact_time_s": 4.0,
            "sample_hz": 10,
            "collision_substeps": 10,
            "contact_penetration_m": 0.05,
            "variants": [{"name": "base", "ego_speed_scale": 1.0, "target_speed_scale": 1.0}],
        }

    def test_geometry_and_front_bumper_conversion(self):
        np.testing.assert_allclose(front_bumper_to_center(10, 0, 0, 4), [8, 0])
        self.assertLessEqual(signed_box_clearance([0, 0], 0, 4, 2, [3.9, 0], 0, 4, 2), 0)
        self.assertAlmostEqual(signed_box_clearance([0, 0], 0, 4, 2, [5, 0], 0, 4, 2), 1.0)

    def test_target_association_prefers_box_clearance_near_impact(self):
        rows = pd.DataFrame([
            {"event_id": 1, "target_id": 10, "time": 2.0, "x_ego": 0, "y_ego": 0, "v_ego": 5, "psi_ego": 0,
             "x_sur": 4.0, "y_sur": 0, "v_sur": 1, "psi_sur": 0},
            {"event_id": 1, "target_id": 20, "time": 2.0, "x_ego": 0, "y_ego": 0, "v_ego": 5, "psi_ego": 0,
             "x_sur": 20.0, "y_sur": 0, "v_sur": 1, "psi_sur": 0},
        ])
        meta = {"impact_timestamp": 2000, "ego_length": 4.0, "ego_width": 2.0,
                "target_length": 4.0, "target_width": 2.0}
        result = associate_primary_target(rows, meta, 0.2)
        self.assertEqual(result["target_id"], 10)
        self.assertLessEqual(result["source_box_clearance_m"], 0)

    def test_split_is_reproducible_and_config_rejects_bad_values(self):
        self.assertEqual(project_split(123, 7, self.config["split_fractions"]),
                         project_split(123, 7, self.config["split_fractions"]))
        validate_config(self.config)
        bad = {**self.config, "impact_time_s": 7.0}
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_simulation_starts_separate_and_reaches_collision(self):
        source = {
            "event_id": 1, "target_id": 10, "source_conflict": "leading", "simulation_type": "rear_end",
            "source_split": "train", "source_sample_dt_s": 0.0, "source_box_clearance_m": 0.0,
            "ego_center_m": [0.0, 0.0], "target_center_m": [4.0, 0.0],
            "ego_heading_rad": 0.0, "target_heading_rad": 0.0,
            "ego_speed_mps": 10.0, "target_speed_mps": 5.0,
            "ego_length_m": 4.5, "ego_width_m": 1.8,
            "target_length_m": 4.5, "target_width_m": 1.8,
        }
        result = simulate_collision(source, self.config["variants"][0], self.config)
        self.assertTrue(result["collision_audit"]["collision"])
        self.assertFalse(result["collision_audit"]["initial_collision"])
        self.assertAlmostEqual(
            result["collision_audit"]["first_collision_time_s"],
            self.config["impact_time_s"], delta=0.02,
        )


if __name__ == "__main__":
    unittest.main()
