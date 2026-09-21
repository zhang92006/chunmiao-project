import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from scenario_reconstruction.songdo_adapter import (
    REQUIRED_COLUMNS,
    audit_member,
    iter_records,
    load_segmentation_pairs,
    split_group_for_member,
)


def _row(vehicle_id, time, x, lane, speed="18.0"):
    return {
        "Vehicle_ID": str(vehicle_id),
        "Local_Time": time,
        "Drone_ID": "4",
        "Ortho_X": str(100 + x),
        "Ortho_Y": "200.0",
        "Local_X": str(x),
        "Local_Y": "20.0",
        "Latitude": "37.0",
        "Longitude": "126.0",
        "Vehicle_Length": "4.8",
        "Vehicle_Width": "2.0",
        "Vehicle_Class": "0",
        "Vehicle_Speed": speed,
        "Vehicle_Acceleration": "0.0",
        "Road_Section": "1_1",
        "Lane_Number": str(lane),
        "Visibility": "1",
    }


class SongdoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.archive_path = root / "2022-10-04_G.zip"
        self.member = "2022-10-04_G_AM1.csv"
        rows = [
            _row(1, "07:00:00.000", 0.000, 1),
            _row(1, "07:00:00.033", 0.165, 1),
            _row(1, "07:00:00.066", 0.330, 2),
            _row(2, "07:00:00.000", 10.000, 1),
            _row(2, "07:00:00.033", 10.165, 1),
            _row(2, "07:00:00.066", 10.330, 1),
        ]
        csv_path = root / self.member
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.write(csv_path, arcname=self.member)
        self.segmentation_path = root / "G.csv"
        self.segmentation_path.write_text(
            "Section,Lane,tlx,tly,blx,bly,brx,bry,trx,try\n"
            "1_1,1,0,0,0,1,1,1,1,0\n"
            "1_1,2,1,0,1,1,2,1,2,0\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalizes_speed_and_split_group(self):
        records = list(iter_records(self.archive_path, self.member))
        self.assertEqual(len(records), 6)
        self.assertAlmostEqual(records[0].speed_mps, 5.0)
        self.assertEqual(records[0].split_group, "2022-10-04_G_AM1")
        self.assertEqual(split_group_for_member(self.member), "2022-10-04_G_AM1")

    def test_audits_synchronization_continuity_and_lane_candidates(self):
        pairs = load_segmentation_pairs(self.segmentation_path)
        summary = audit_member(
            self.archive_path,
            self.member,
            segmentation_pairs=pairs,
            window_seconds=0.05,
        )
        self.assertEqual(summary["row_count"], 6)
        self.assertEqual(summary["vehicle_count"], 2)
        self.assertEqual(
            summary["synchronization"]["timestamps_with_at_least_2_vehicles"], 3
        )
        self.assertEqual(summary["continuity"]["tracks_with_gap"], 0)
        self.assertEqual(summary["continuity"]["usable_track_count"], 2)
        self.assertEqual(
            summary["kinematics"]["speed_domain_row_counts"]["5_to_below_20_mps"], 6
        )
        self.assertEqual(summary["lane"]["invalid_segmentation_pair_rows"], 0)
        self.assertEqual(summary["lane"]["same_section_lane_change_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
