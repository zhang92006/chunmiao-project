import unittest

from scenario_reconstruction.shrp2_reference_audit import _is_reconstruction_ready


class SHRP2ReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.config = {"history_s": 4.0, "maximum_collision_time_error_s": 0.5}
        self.quality = {
            "CAV": {
                "position_speed_consistent": True,
                "heading_usable": True,
            },
            "BV_primary": {
                "source_front_bumper_diagnostics": {
                    "reported_vs_path_speed_rmse_mps": 0.2,
                    "reported_vs_path_heading_mae_rad": 0.1,
                },
            },
        }

    def test_ready_requires_collision_free_start_and_timed_contact(self):
        collision = {
            "initial_collision": False,
            "collision_in_reference_window": True,
            "first_sampled_contact_time_s": 3.6,
        }
        self.assertTrue(_is_reconstruction_ready(self.quality, collision, self.config))

        initial_overlap = {**collision, "initial_collision": True}
        self.assertFalse(
            _is_reconstruction_ready(self.quality, initial_overlap, self.config)
        )

    def test_ready_rejects_contact_outside_tolerance(self):
        collision = {
            "initial_collision": False,
            "collision_in_reference_window": True,
            "first_sampled_contact_time_s": 3.2,
        }
        self.assertFalse(_is_reconstruction_ready(self.quality, collision, self.config))


if __name__ == "__main__":
    unittest.main()
