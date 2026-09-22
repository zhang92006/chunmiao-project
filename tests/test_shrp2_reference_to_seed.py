import unittest

from scenario_reconstruction.shrp2_reference_to_seed import reference_to_seed


class SHRP2ReferenceToSeedTests(unittest.TestCase):
    def test_conversion_preserves_initial_state_and_declares_projection(self):
        reference = {
            "record_type": "shrp2_preimpact_soft_reference",
            "source": {
                "dataset": "SHRP2",
                "doi": "10.15787/VTT1/T7UUC1",
                "event_id": 1,
                "source_split": "train",
                "associated_target_id": 2,
            },
            "alignment": {"reference_time_s": [0.0, 4.0]},
            "collision_audit": {"first_sampled_contact_time_s": 3.6},
            "actors": {
                "CAV": {
                    "length_m": 4.5,
                    "width_m": 1.8,
                    "xy_m": [[0.0, 0.0]],
                    "reported_speed_mps": [2.0],
                    "reported_heading_rad": [0.0],
                },
                "BV_primary": {
                    "source_target_id": 2,
                    "length_m": 4.5,
                    "width_m": 1.8,
                    "xy_m": [[15.0, 0.0]],
                    "reported_speed_mps": [1.0],
                    "reported_heading_rad": [0.0],
                },
            },
        }
        seed = reference_to_seed(reference, duration_s=1.0, sample_hz=10)
        self.assertEqual(seed["actors"][0]["xy_m"][0], [0.0, 0.0])
        self.assertEqual(seed["actors"][1]["xy_m"][0], [15.0, 0.0])
        self.assertEqual(seed["bridge_policy"]["uses"], "initial positions, speeds and headings only")
        self.assertEqual(seed["impact_conditioning"]["requested_impact_time_s"], 3.6)
        self.assertEqual(
            seed["impact_conditioning"]["source"],
            "collision_audit.first_sampled_contact_time_s",
        )


if __name__ == "__main__":
    unittest.main()
