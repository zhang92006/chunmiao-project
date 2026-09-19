import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scenario_reconstruction.prepare_training_data import _is_training_ready_episode
from scenario_reconstruction.shrp2_collision_action_cem import (
    build_search_template,
    run_collision_action_cem,
)
from scenario_reconstruction.templates import load_template


def _source_template():
    return {
        "template_id": "source",
        "description": "source",
        "map": "2Lane",
        "route": "route_0",
        "duration": 12.0,
        "tags": ["shrp2", "multibv", "train"],
        "ego": {
            "id": "CAV", "role": "CAV", "route": "route_0",
            "lane_index": 1, "position": 400.0, "speed": 30.0,
        },
        "actors": [
            {
                "id": "BV_primary", "role": "BV", "route": "route_0",
                "lane_index": 1, "position": 420.0, "speed": 28.0,
            },
            {
                "id": "BV_context", "role": "BV", "route": "route_0",
                "lane_index": 0, "position": 405.0, "speed": 29.0,
            },
        ],
        "events": [],
        "perturbations": [],
        "bridge_metadata": {
            "source_event_id": 101,
            "source_split": "train",
        },
    }


class CollisionActionCEMTests(unittest.TestCase):
    def test_search_template_is_executable_but_permanently_not_training_ready(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "candidate.json"
            source.write_text(json.dumps(_source_template()), encoding="utf-8")
            candidate = build_search_template(source, output, "candidate_1", {
                "start_time_s": 1.0,
                "context_delay_s": 0.5,
                "primary_duration_s": 1.0,
                "context_duration_s": 1.5,
                "primary_action_id": 2,
                "context_action_id": 0,
            })

            loaded = load_template(output)
            self.assertEqual(len(loaded.events), 2)
            self.assertTrue(candidate["events"][0]["params"]["search_only"])
            self.assertTrue(candidate["bridge_metadata"]["not_for_d2rl_training"])
            episode = {
                "scenario_metadata": {
                    "collision_search_only": True,
                    "not_for_d2rl_training": True,
                },
                "weight_step_info": {"0": {"joint": 0.1, "per_agent": [0.2, 0.5]}},
                "drl_obs_step_info": {"0": {"joint": [0.0] * 14, "per_agent": [[0.0] * 10] * 2}},
                "criticality_step_info": {"0": 1.0},
                "ndd_step_info": {"0": {"joint": 0.01, "per_agent": [0.1, 0.1]}},
            }
            self.assertFalse(_is_training_ready_episode(episode, multi_bv=True))

    def test_cem_runs_each_source_and_persists_search_marker(self):
        space = {
            "start_time_s": [1.0],
            "context_delay_s": [0.0],
            "primary_duration_s": [1.0],
            "context_duration_s": [1.0],
            "primary_action_id": [2],
            "context_action_id": [0],
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(_source_template()), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [{
                "event_id": 101,
                "template_path": str(source),
                "pilot_role": "search_candidate",
            }]}), encoding="utf-8")

            def fake_run(_template, episode, experiment_path, **_kwargs):
                crash = Path(experiment_path) / "crash"
                safe = Path(experiment_path) / "tested_and_safe"
                rejected = Path(experiment_path) / "rejected"
                crash.mkdir(parents=True, exist_ok=True)
                safe.mkdir(parents=True, exist_ok=True)
                rejected.mkdir(parents=True, exist_ok=True)
                (crash / f"{episode}.json").write_text(json.dumps({
                    "collision_result": 1,
                    "collision_id": ["CAV", "BV_primary"],
                    "ttc_step_info": {"0": 0.2},
                    "distance_step_info": {"0": -0.1},
                }), encoding="utf-8")
                return 1.0

            output = root / "search"
            with patch(
                "scenario_reconstruction.shrp2_collision_action_cem.run_template",
                side_effect=fake_run,
            ) as run:
                result = run_collision_action_cem(
                    manifest,
                    output,
                    generations=1,
                    population=2,
                    rollouts_per_candidate=1,
                    search_space=space,
                )

            self.assertEqual(run.call_count, 2)
            self.assertEqual(result["attempted_rollouts"], 2)
            self.assertEqual(result["sources_with_target_collision"], 1)
            self.assertTrue((output / "SEARCH_ONLY_DO_NOT_TRAIN.json").is_file())
            self.assertTrue((output / "cem_search_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
