import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from d2rl_training.d2rl_training_env import D2RLTrainingEnv
from scenario_reconstruction.environment import ScenarioNADE, _duration_reached
from scenario_reconstruction.templates import EventSpec


class MultiBVCompatibilityTests(unittest.TestCase):
    def test_joint_value_supports_legacy_and_multibv_records(self):
        self.assertEqual(D2RLTrainingEnv._joint_value(0.25), 0.25)
        self.assertEqual(
            D2RLTrainingEnv._joint_value({"joint": 0.125, "per_agent": [0.25, 0.5]}),
            0.125,
        )

    def test_primary_observation_projects_first_agent(self):
        primary = list(range(10))
        record = {
            "joint": list(range(14)),
            "per_agent": [primary, list(range(10, 20))],
        }
        self.assertEqual(D2RLTrainingEnv._primary_observation(record), primary)

    def test_joint_training_uses_joint_observation_and_vector_action(self):
        env = D2RLTrainingEnv.__new__(D2RLTrainingEnv)
        env.multi_bv_training = True
        env.multi_bv_num = 2
        env.observation_dim = 14
        env.action_dim = 2
        record = {
            "joint": list(range(14)),
            "per_agent": [list(range(10)), list(range(10, 20))],
        }

        self.assertEqual(env._training_observation(record), list(range(14)))
        self.assertTrue(
            np.allclose(env._normalize_action([0.2, 0.8]), [0.2, 0.8])
        )
        with self.assertRaisesRegex(ValueError, "Expected 2-D action"):
            env._normalize_action([0.2])

    def test_joint_importance_weight_multiplies_per_agent_terms(self):
        result = D2RLTrainingEnv._joint_epsilon_weight(
            {"joint": 0.125, "per_agent": [0.5, 0.25]},
            [0.5, 0.25],
            {"joint": 0.02, "per_agent": [0.1, 0.2]},
        )
        self.assertAlmostEqual(result, (0.1 / (1 - 0.5)) * (0.2 / (1 - 0.25)))

    def test_joint_pair_importance_weight_uses_the_correlated_proposal(self):
        result = D2RLTrainingEnv._joint_epsilon_weight(
            {
                "proposal_type": "joint_pair",
                "joint": 0.25,
                "per_agent": [0.5, 0.5],
                "joint_naturalistic_probability": 0.02,
                "joint_proposal_probability": 0.08,
            },
            [0.001, 0.001],
            {"proposal_type": "joint_pair", "joint": 0.02, "per_agent": [0.1, 0.2]},
        )
        self.assertAlmostEqual(result, 0.25)

    def test_joint_training_env_reset_and_step_keep_two_actions(self):
        episode = {
            "collision_result": 1,
            "weight_step_info": {
                "forced_0.100000": {"joint": 0.125, "per_agent": [0.5, 0.25]}
            },
            "drl_obs_step_info": {
                "forced_0.100000": {
                    "joint": list(range(14)),
                    "per_agent": [list(range(10)), list(range(10, 20))],
                }
            },
            "drl_epsilon_step_info": {"forced_0.100000": [0.5, 0.5]},
            "real_epsilon_step_info": {"forced_0.100000": [0.5, 0.5]},
            "criticality_step_info": {"forced_0.100000": 1.0},
            "ndd_step_info": {
                "forced_0.100000": {"joint": 0.02, "per_agent": [0.1, 0.2]}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path = root / "episode.json"
            episode_path.write_text(json.dumps(episode), encoding="utf-8")
            (root / "crash_weight_dict.json").write_text(
                json.dumps({str(episode_path): [0.125, 0.125]}), encoding="utf-8"
            )
            env = D2RLTrainingEnv(
                {
                    "root_folder": "",
                    "data_folders": [str(root)],
                    "data_folder_weights": [1],
                    "clip_reward_threshold": 100,
                    "multi_bv_training": True,
                    "multi_bv_num": 2,
                }
            )
            observation = env.reset()
            _, _, done, _ = env.step(np.array([0.2, 0.8], dtype=np.float32))

        self.assertEqual(env.observation_space.shape, (14,))
        self.assertEqual(env.action_space.shape, (2,))
        self.assertEqual(len(observation), 14)
        self.assertTrue(done)
        self.assertTrue(
            np.allclose(
                env.episode_data["drl_epsilon_step_info"]["forced_0.100000"],
                [0.2, 0.8],
            )
        )

    def test_log_weight_sidecar_preserves_relative_sampling_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = (root / "first.json").as_posix()
            second = (root / "second.json").as_posix()
            sidecar = root / "crash_log_weight_dict.json"
            sidecar.write_text(json.dumps({
                "schema_version": 1,
                "log_weights": {first: -1000.0, second: -1001.0},
            }), encoding="utf-8")

            weights = D2RLTrainingEnv._stable_sampling_weights(
                [first, second], sidecar
            )

        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], np.exp(-1.0))

    def test_single_critical_mode_keeps_one_latest_maximum_criticality_step(self):
        def joint_record(value):
            return {"joint": value, "per_agent": [value, value]}

        episode = {
            "collision_result": 1,
            "weight_step_info": {
                "0.0": joint_record(0.5),
                "0.1": joint_record(0.4),
                "0.2": joint_record(0.3),
            },
            "drl_obs_step_info": {
                timestep: {"joint": list(range(14)), "per_agent": [list(range(10)), list(range(10))]}
                for timestep in ("0.0", "0.1", "0.2")
            },
            "drl_epsilon_step_info": {
                timestep: [0.5, 0.5] for timestep in ("0.0", "0.1", "0.2")
            },
            "real_epsilon_step_info": {
                timestep: [0.5, 0.5] for timestep in ("0.0", "0.1", "0.2")
            },
            "criticality_step_info": {"0.0": 1.0, "0.1": 4.0, "0.2": 4.0},
            "ndd_step_info": {
                timestep: joint_record(0.2) for timestep in ("0.0", "0.1", "0.2")
            },
            "controlled_bv_ids_step_info": {
                timestep: ["BV_primary", "BV_context"]
                for timestep in ("0.0", "0.1", "0.2")
            },
        }
        env = D2RLTrainingEnv.__new__(D2RLTrainingEnv)
        env.multi_bv_training = True
        env.multi_bv_decision_mode = "single_critical"
        env.yaml_conf = {"clip_reward_threshold": 100}
        env.total_steps = 0

        selected = env.filter_episode_data(episode)

        self.assertEqual(list(selected["weight_step_info"]), ["0.2"])
        self.assertEqual(selected["d2rl_decision_selection"], {
            "mode": "single_critical",
            "selected_timestep": "0.2",
            "selected_criticality": 4.0,
            "candidate_count": 3,
        })
        self.assertEqual(env.get_multiple_adv_action_num(selected["weight_step_info"]), 1)
        env.episode_data = selected
        self.assertNotEqual(env._get_reward(), 0)

    def test_scenario_duration_boundary_is_inclusive(self):
        self.assertFalse(_duration_reached(5.99, 6.0))
        self.assertTrue(_duration_reached(6.0, 6.0))
        self.assertTrue(_duration_reached(6.01, 6.0))

    def test_low_speed_calibration_action_bypasses_vehicle_action_clamp(self):
        environment = ScenarioNADE.__new__(ScenarioNADE)
        environment.simulator = MagicMock(step_size=0.1)
        vehicle = SimpleNamespace(id="CAV", action_step_size=0.1)
        environment.vehicle_list = {"CAV": vehicle}
        event = EventSpec(
            type="calibration_cav_action",
            actor="CAV",
            start_time=0.0,
            duration=1.0,
            params={"longitudinal": 0.0},
        )

        environment._apply_calibration_longitudinal_action(event)

        environment.simulator.set_vehicle_speedmode.assert_called_once_with("CAV", 0)
        environment.simulator.change_vehicle_speed.assert_called_once_with(
            "CAV", 0.0, 0.1
        )


if __name__ == "__main__":
    unittest.main()
