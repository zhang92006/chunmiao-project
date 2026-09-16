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


if __name__ == "__main__":
    unittest.main()
