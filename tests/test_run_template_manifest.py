import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scenario_reconstruction.run_template_manifest import run_template_manifest


class RunTemplateManifestTests(unittest.TestCase):
    def test_repeats_assign_unique_episode_ids_and_preserve_repeat_number(self):
        manifest = {
            "records": [
                {"status": "template_created", "split": "train", "template_path": "one.json"},
                {"status": "template_created", "split": "train", "template_path": "two.json"},
            ]
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "scenario_reconstruction.run_template_manifest.run_template",
                return_value=1.0,
            ) as run:
                summary = run_template_manifest(
                    manifest_path,
                    root / "episodes",
                    split="train",
                    repeats=3,
                )

        self.assertEqual(summary["attempted"], 6)
        self.assertEqual(summary["repeats"], 3)
        self.assertEqual([item["episode"] for item in summary["results"]], list(range(6)))
        self.assertEqual(
            [item["repeat"] for item in summary["results"]], [0, 0, 1, 1, 2, 2]
        )
        self.assertEqual(run.call_count, 6)

    def test_initial_primary_ttc_filter_keeps_only_closing_same_lane_templates(self):
        manifest = {
            "records": [
                {"status": "template_created", "split": "train", "template_path": "eligible.json"},
                {"status": "template_created", "split": "train", "template_path": "wrong_lane.json"},
                {"status": "template_created", "split": "train", "template_path": "large_ttc.json"},
            ]
        }
        template = {
            "ego": {"lane_index": 1, "position": 400.0, "speed": 30.0},
            "actors": [{"id": "BV_primary", "lane_index": 1, "position": 420.0, "speed": 25.0}],
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "eligible.json").write_text(json.dumps(template), encoding="utf-8")
            wrong_lane = json.loads(json.dumps(template))
            wrong_lane["actors"][0]["lane_index"] = 0
            (root / "wrong_lane.json").write_text(json.dumps(wrong_lane), encoding="utf-8")
            large_ttc = json.loads(json.dumps(template))
            large_ttc["actors"][0]["position"] = 500.0
            (root / "large_ttc.json").write_text(json.dumps(large_ttc), encoding="utf-8")
            for record in manifest["records"]:
                record["template_path"] = str(root / record["template_path"])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "scenario_reconstruction.run_template_manifest.run_template",
                return_value=1.0,
            ) as run:
                summary = run_template_manifest(
                    manifest_path,
                    root / "episodes",
                    split="train",
                    max_initial_primary_ttc_s=5.0,
                )

        self.assertEqual(summary["records_after_split"], 3)
        self.assertEqual(summary["records_after_initial_primary_ttc_filter"], 1)
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(run.call_count, 1)

    def test_context_blocking_filter_requires_a_declared_source_flag(self):
        manifest = {
            "records": [
                {"status": "template_created", "split": "train", "template_path": "block.json"},
                {"status": "template_created", "split": "train", "template_path": "free.json"},
            ]
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            for name, blocking in (("block.json", True), ("free.json", False)):
                (root / name).write_text(json.dumps({
                    "bridge_metadata": {
                        "source_adaptive_critical_window": {
                            "context_blocking_potential": blocking
                        }
                    }
                }), encoding="utf-8")
            for record in manifest["records"]:
                record["template_path"] = str(root / record["template_path"])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "scenario_reconstruction.run_template_manifest.run_template",
                return_value=1.0,
            ) as run:
                summary = run_template_manifest(
                    manifest_path, root / "episodes", split="train",
                    require_context_blocking=True,
                )

        self.assertEqual(summary["records_after_split"], 2)
        self.assertEqual(summary["records_after_context_blocking_filter"], 1)
        self.assertTrue(summary["require_context_blocking"])
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(run.call_count, 1)

    def test_source_event_filter_uses_declared_bridge_metadata(self):
        manifest = {
            "records": [
                {"status": "template_created", "split": "train", "template_path": "one.json"},
                {"status": "template_created", "split": "train", "template_path": "two.json"},
            ]
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            for name, event_id in (("one.json", 101), ("two.json", 202)):
                (root / name).write_text(json.dumps({
                    "bridge_metadata": {"source_event_id": event_id}
                }), encoding="utf-8")
            for record in manifest["records"]:
                record["template_path"] = str(root / record["template_path"])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "scenario_reconstruction.run_template_manifest.run_template",
                return_value=1.0,
            ) as run:
                summary = run_template_manifest(
                    manifest_path,
                    root / "episodes",
                    split="train",
                    source_event_ids={202},
                )

        self.assertEqual(summary["records_after_split"], 2)
        self.assertEqual(summary["source_event_ids"], [202])
        self.assertEqual(summary["records_after_source_event_filter"], 1)
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(Path(run.call_args.args[0]).name, "two.json")

    def test_manifest_forwards_the_selected_proposal_mode(self):
        manifest = {
            "records": [
                {"status": "template_created", "split": "train", "template_path": "one.json"},
            ]
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "scenario_reconstruction.run_template_manifest.run_template",
                return_value=1.0,
            ) as run:
                summary = run_template_manifest(
                    manifest_path,
                    root / "episodes",
                    proposal_mode="factorized",
                )

        self.assertEqual(summary["proposal_mode"], "factorized")
        self.assertEqual(run.call_args.kwargs["proposal_mode"], "factorized")


if __name__ == "__main__":
    unittest.main()
