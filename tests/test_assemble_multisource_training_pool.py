import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scenario_reconstruction.assemble_multisource_training_pool import (
    assemble_training_pool,
)


class AssembleTrainingPoolTests(unittest.TestCase):
    def test_merges_external_indexes_and_counts_sources_without_copying(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiments = []
            for index, source in enumerate((101, 202)):
                experiment = root / f"experiment_{index}"
                crash = experiment / "crash"
                crash.mkdir(parents=True)
                episode = crash / "0.json"
                episode.write_text(json.dumps({
                    "scenario_metadata": {
                        "source_event_id": source,
                        "source_split": "train",
                    },
                    "log_importance_weight": -float(index + 1),
                    "weight_episode": math.exp(-float(index + 1)),
                    "log_probability_step_info": {
                        "0.0": {"log_importance_weight": -float(index + 1)}
                    },
                    "weight_step_info": {
                        "0.0": {"joint": math.exp(-float(index + 1))}
                    },
                }), encoding="utf-8")
                key = episode.as_posix()
                (experiment / "crash_weight_dict.json").write_text(
                    json.dumps({key: [0.1, 0.1]}), encoding="utf-8"
                )
                (experiment / "crash_log_weight_dict.json").write_text(
                    json.dumps({"log_weights": {key: -float(index + 1)}}),
                    encoding="utf-8",
                )
                experiments.append(experiment)

            output = root / "pool"
            result = assemble_training_pool(experiments, output)

            self.assertEqual(result["crash_episode_count"], 2)
            self.assertEqual(result["source_event_count"], 2)
            self.assertEqual(result["crash_count_by_source_event"], {
                "101": 1, "202": 1
            })
            self.assertFalse((output / "crash").exists())

    def test_rejects_search_only_episode(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            crash = experiment / "crash"
            crash.mkdir(parents=True)
            episode = crash / "0.json"
            episode.write_text(json.dumps({
                "scenario_metadata": {
                    "source_event_id": 101,
                    "source_split": "train",
                    "collision_search_only": True,
                },
                "log_importance_weight": -1.0,
                "weight_episode": 0.36787944117144233,
                "log_probability_step_info": {
                    "0.0": {"log_importance_weight": -1.0}
                },
                "weight_step_info": {
                    "0.0": {"joint": 0.36787944117144233}
                },
            }), encoding="utf-8")
            key = episode.as_posix()
            (experiment / "crash_weight_dict.json").write_text(
                json.dumps({key: [0.1, 0.1]}), encoding="utf-8"
            )
            (experiment / "crash_log_weight_dict.json").write_text(
                json.dumps({"log_weights": {key: -1.0}}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Search-only"):
                assemble_training_pool([experiment], root / "pool")

    def test_can_explicitly_skip_invalid_probability_ledger(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            crash = experiment / "crash"
            crash.mkdir(parents=True)
            episode = crash / "0.json"
            episode.write_text(json.dumps({
                "scenario_metadata": {
                    "source_event_id": 101,
                    "source_split": "train",
                },
                "log_importance_weight": -1.0,
                "weight_episode": 0.0,
                "log_probability_step_info": {
                    "0.0": {"log_importance_weight": -1.0}
                },
                "weight_step_info": {"0.0": {"joint": 0.0}},
            }), encoding="utf-8")
            key = episode.as_posix()
            (experiment / "crash_weight_dict.json").write_text(
                json.dumps({key: [0.0, 0.0]}), encoding="utf-8"
            )
            (experiment / "crash_log_weight_dict.json").write_text(
                json.dumps({"log_weights": {key: -1.0}}), encoding="utf-8"
            )

            result = assemble_training_pool(
                [experiment],
                root / "pool",
                skip_invalid_probability_ledger=True,
            )

            self.assertEqual(result["crash_episode_count"], 0)
            self.assertEqual(
                result["excluded_invalid_probability_ledger_count"], 1
            )


if __name__ == "__main__":
    unittest.main()
