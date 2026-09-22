import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nadeinfoextractor import NADEInfoExtractor
from scenario_reconstruction.importance import (
    probability_record,
    stable_weight_diagnostics,
)
from scenario_reconstruction.prepare_training_data import (
    write_importance_weight_diagnostics,
)
from scenario_reconstruction.multibv import build_multibv_joint_obs


class ImportanceLoggingTests(unittest.TestCase):
    def test_probability_record_and_episode_log_accumulate_in_log_domain(self):
        record = probability_record("joint_pair", 1e-6, 2e-3)
        extractor = NADEInfoExtractor.__new__(NADEInfoExtractor)
        extractor.episode_log = extractor._new_episode_log()

        extractor._record_log_probability("0.0", record)
        extractor._record_log_probability("0.1", record)

        self.assertAlmostEqual(
            extractor.episode_log["log_importance_weight"],
            2 * record["log_importance_weight"],
        )
        self.assertEqual(extractor.episode_log["proposal_modes"], ["joint_pair"])
        self.assertEqual(len(extractor.episode_log["log_probability_step_info"]), 2)

    def test_stable_diagnostics_avoid_tiny_weight_underflow(self):
        summary = stable_weight_diagnostics([
            {"log_importance_weight": -1000.0},
            {"log_importance_weight": -1001.0},
        ])

        self.assertEqual(summary["episode_count"], 2)
        self.assertGreater(summary["effective_sample_size"], 1.0)
        self.assertLess(summary["largest_normalized_weight"], 1.0)

    def test_training_diagnostics_group_crashes_by_source_event(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps({
                "weight_episode": 1e-20,
                "log_importance_weight": -40.0,
                "proposal_modes": ["joint_pair"],
                "scenario_metadata": {"source_event_id": 101},
            }), encoding="utf-8")
            second.write_text(json.dumps({
                "weight_episode": 1e-30,
                "log_importance_weight": -50.0,
                "proposal_modes": ["factorized"],
                "scenario_metadata": {"source_event_id": 202},
            }), encoding="utf-8")

            summary = write_importance_weight_diagnostics(root, {
                first.as_posix(): [1e-20, 1e-20],
                second.as_posix(): [1e-30, 1e-30],
            })

            self.assertEqual(summary["overall"]["episode_count"], 2)
            self.assertEqual(set(summary["by_source_event"]), {"101", "202"})
            self.assertEqual(summary["proposal_mode_counts"], {
                "joint_pair": 1,
                "factorized": 1,
            })

    def test_preparation_writes_log_weight_sidecar(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crash = root / "crash"
            crash.mkdir()
            episode_path = crash / "episode.json"
            step_weight = 1e-100
            step_log_weight = -230.25850929940458
            episode_path.write_text(json.dumps({
                "weight_episode": 0.0,
                "log_importance_weight": 4 * step_log_weight,
                "weight_step_info": {
                    str(index): {"joint": step_weight, "per_agent": [step_weight, 1.0]}
                    for index in range(4)
                },
                "log_probability_step_info": {
                    str(index): {"log_importance_weight": step_log_weight}
                    for index in range(4)
                },
                "drl_obs_step_info": {
                    str(index): {"joint": list(range(14)), "per_agent": [list(range(10)), list(range(10))]}
                    for index in range(4)
                },
                "criticality_step_info": {str(index): 1.0 for index in range(4)},
                "ndd_step_info": {
                    str(index): {"joint": 0.01, "per_agent": [0.01, 1.0]}
                    for index in range(4)
                },
            }), encoding="utf-8")

            from scenario_reconstruction.prepare_training_data import prepare_crash_weight_dict
            result = prepare_crash_weight_dict(root, multi_bv=True, agent_num=2)
            sidecar = json.loads((root / "crash_log_weight_dict.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            sidecar["log_weights"][episode_path.as_posix()], 4 * step_log_weight
        )

    def test_joint_observation_uses_log_weight_after_raw_underflow(self):
        full_obs = {
            "CAV": {"position": [400.0, 46.0], "velocity": 25.0},
            "BV_primary": {"position": [410.0, 46.0], "velocity": 20.0},
            "BV_context": {"position": [405.0, 42.0], "velocity": 21.0},
        }
        observation = build_multibv_joint_obs(
            full_obs,
            ["BV_primary", "BV_context"],
            episode_weight=0.0,
            log_episode_weight=-100.0,
        )

        self.assertEqual(len(observation), 14)
        self.assertGreater(observation[3], -5.0)


if __name__ == "__main__":
    unittest.main()
