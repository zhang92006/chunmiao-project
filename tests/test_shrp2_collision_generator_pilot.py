import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scenario_reconstruction.shrp2_collision_generator_pilot import (
    build_collision_generator_pilot,
)


class CollisionGeneratorPilotTests(unittest.TestCase):
    def test_positive_controls_and_diverse_candidates_are_selected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates = root / "templates"
            templates.mkdir()
            records = []
            for event_id, conflict, ttc, blocking in (
                (1, "leading", 4.0, False),
                (2, "leading", 2.5, True),
                (3, "adjacent_lane", 3.0, False),
                (4, "none", 2.0, False),
            ):
                path = templates / f"{event_id}.json"
                path.write_text(json.dumps({
                    "bridge_metadata": {
                        "source_event_id": event_id,
                        "source_category": "Crash",
                        "source_conflict": conflict,
                        "source_adaptive_critical_window": {
                            "primary_ttc_s": ttc,
                            "primary_gap_m": 10.0,
                            "context_blocking_potential": blocking,
                        },
                    }
                }), encoding="utf-8")
                records.append({
                    "event_id": event_id,
                    "split": "train",
                    "category": "Crash",
                    "status": "template_created",
                    "template_path": str(path),
                })
            bridge = root / "bridge.json"
            bridge.write_text(json.dumps({"records": records}), encoding="utf-8")
            rollout = root / "rollout"
            (rollout / "crash").mkdir(parents=True)
            (rollout / "tested_and_safe").mkdir()
            (rollout / "crash" / "0.json").write_text(json.dumps({
                "scenario_metadata": {"source_event_id": 1},
                "ttc_step_info": {"0": [1.0, 2.0]},
                "distance_step_info": {"0": [3.0]},
                "criticality_step_info": {"0": 0.8},
                "log_importance_weight": -4.0,
            }), encoding="utf-8")
            for event_id in (2, 3, 4):
                (rollout / "tested_and_safe" / f"{event_id}.json").write_text(
                    json.dumps({
                        "scenario_metadata": {"source_event_id": event_id},
                        "ttc_step_info": {"0": [2.0]},
                    }),
                    encoding="utf-8",
                )

            result = build_collision_generator_pilot(
                bridge, rollout, root / "output", pilot_count=3
            )

            selected = result["selected_records"]
            self.assertEqual(selected[0]["source_event_id"], 1)
            self.assertEqual(selected[0]["pilot_role"], "positive_control")
            self.assertEqual({item["conflict"] for item in selected[1:]}, {
                "adjacent_lane", "none"
            })
            manifest = json.loads(
                Path(result["pilot_manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["records"]), 3)


if __name__ == "__main__":
    unittest.main()
