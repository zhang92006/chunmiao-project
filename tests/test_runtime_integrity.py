from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest

from scenario_reconstruction.highd import write_json
from scenario_reconstruction.likelihood import has_supported_likelihoods
from scenario_reconstruction.prepare_training_data import prepare_safe_weight_dict
from scenario_reconstruction.templates import EventSpec, load_template


TEMPLATE = "scenario_reconstruction/templates/autoware_cut_in.yaml"


class RuntimeIntegrityTests(unittest.TestCase):
    def test_schema_faults_cannot_silently_run(self):
        for kind in ("perception_dropout", "perception_delay", "perception_position_bias", "control_delay"):
            template = load_template(TEMPLATE)
            template.events = [EventSpec(kind, "BV_cut_in", 0, 0.5)]
            template.validate()
            with self.assertRaisesRegex(ValueError, "schema-only"):
                template.validate_runtime()

    def test_nonfinite_and_fake_probabilities_rejected(self):
        template = load_template(TEMPLATE)
        template.ego.speed = float("nan")
        with self.assertRaises(ValueError):
            template.validate()
        template = load_template(TEMPLATE)
        template.events[0].params["ndd_possi"] = 1e-8
        with self.assertRaisesRegex(ValueError, "fabricated"):
            template.validate_runtime()

    def test_old_and_new_heuristic_data_excluded(self):
        cases = [{"scenario_metadata": {"likelihood_valid": False}},
                 {"weight_step_info": {"forced_0.1": 0.01}},
                 {"ndd_step_info": {"0.1": {"joint": 0.001}}}]
        for case in cases:
            self.assertFalse(has_supported_likelihoods(case))
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder) / "tested_and_safe/0.json", cases[0])
            self.assertEqual(prepare_safe_weight_dict(folder), {})


@unittest.skipUnless(os.environ.get("RUN_SUMO_TESTS") == "1", "Opt-in SUMO runtime test")
class SumoIntegrityTests(unittest.TestCase):
    def test_seed_duration_all_safe_logging_and_no_overwrite(self):
        from scenario_reconstruction.run_template import run_template
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = load_template(TEMPLATE)
            template.duration = 1.0
            template.events = []
            template.ego.position, template.ego.speed = 100, 25
            for index, actor in enumerate(template.actors):
                actor.position, actor.speed = 250 + 100 * index, 25
            write_json(root / "safe.json", asdict(template))
            logs = []
            for run in ("a", "b"):
                self.assertIsNone(run_template(str(root / "safe.json"), 0, str(root / run), seed=7))
                logs.append(json.loads((root / run / "tested_and_safe/0.json").read_text()))
            self.assertEqual(logs[0], logs[1])
            self.assertEqual(logs[0]["collision_result"], 0)
            self.assertAlmostEqual(logs[0]["episode_info"]["elapsed_time"], 1.0)
            self.assertIn("5", logs[0]["termination_reason"])
            self.assertFalse(logs[0]["scenario_metadata"]["likelihood_valid"])
            with self.assertRaises(FileExistsError):
                run_template(str(root / "safe.json"), 0, str(root / "a"), seed=7)

    def test_forced_action_is_logged_without_inventing_training_steps(self):
        from scenario_reconstruction.run_template import run_template
        with tempfile.TemporaryDirectory() as directory:
            run_template(TEMPLATE, 0, directory, seed=7)
            paths = list(Path(directory).glob("crash/*.json")) + list(Path(directory).glob("tested_and_safe/*.json"))
            self.assertEqual(len(paths), 1)
            log = json.loads(paths[0].read_text())
            self.assertTrue(any(event["applied"] for event in log["scenario_events"]))
            self.assertFalse(any(str(key).startswith("forced_") for key in log["weight_step_info"]))
            self.assertFalse(has_supported_likelihoods(log))


if __name__ == "__main__":
    unittest.main()
