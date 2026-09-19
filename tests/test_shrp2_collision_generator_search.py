from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scenario_reconstruction.shrp2_collision_generator_search import run_epsilon_grid


class CollisionGeneratorSearchTests(unittest.TestCase):
    def test_grid_uses_named_epsilons_and_marks_output_search_only(self):
        def fake_run(_manifest, experiment_path, **kwargs):
            path = Path(experiment_path)
            (path / "crash").mkdir(parents=True)
            (path / "tested_and_safe").mkdir()
            primary = kwargs["epsilon"]["BV_primary"]
            if primary < 0.001:
                (path / "crash" / "0.json").write_text("{}", encoding="utf-8")
            else:
                (path / "tested_and_safe" / "0.json").write_text("{}", encoding="utf-8")
            return {
                "attempted": 1,
                "successful_runs": 1,
                "training_ready_crashes": int(primary < 0.001),
                "importance_weight_diagnostics": None,
            }

        with TemporaryDirectory() as temporary, patch(
            "scenario_reconstruction.shrp2_collision_generator_search.run_template_manifest",
            side_effect=fake_run,
        ) as run:
            result = run_epsilon_grid(
                "pilot.json", temporary, [0.0001, 0.01], [0.0001], repeats=2
            )

            self.assertEqual(run.call_count, 2)
            self.assertEqual(result["trial_count"], 2)
            self.assertEqual(
                result["recommended_frozen_candidate"]["epsilon_primary"], 0.0001
            )
            self.assertTrue(
                (Path(temporary) / "SEARCH_ONLY_DO_NOT_TRAIN.json").is_file()
            )
            first_call = run.call_args_list[0]
            self.assertEqual(first_call.kwargs["proposal_mode"], "factorized")
            self.assertEqual(
                first_call.kwargs["epsilon"],
                {"BV_primary": 0.0001, "BV_context": 0.0001},
            )


if __name__ == "__main__":
    unittest.main()
