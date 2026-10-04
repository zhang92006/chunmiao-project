import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scenario_reconstruction.highd_dual_ndd import PairActionDistribution, compose_lateral
from scenario_reconstruction.highd_dual_ndd_proposal import ConditionalChainProposal
from scenario_reconstruction.highd_critical_sequences import (
    audit_collection, censored_diagnostics, extract, prepare, prepare_first_collision, weighted_diagnostics,
)
from d2rl_training.highd_event_contract import (
    FIRST_EVENT_DEFINITION, FIRST_SEQUENCE, event_indicator, target_digest, with_first_event,
)
try:
    from d2rl_training.highd_critical_sequence_env import HighDCriticalSequenceEnv, sequence_log_weight, scaled_loss
except ModuleNotFoundError as exc:
    if exc.name != "gym":
        raise
    # SUMO/export runs in the light D2RL environment; legacy PPO/Gym runs in
    # D2RLTrain39. Do not require installing training dependencies into SUMO.
    HighDCriticalSequenceEnv = None


def raw_fixture():
    longitudinal = np.ones((31, 31)) / 961
    lateral = np.zeros((31, 3))
    lateral[:, 0] = 1
    p = compose_lateral(longitudinal, lateral, lateral)
    pair = PairActionDistribution((1, 2), ("follower", "leader"), p, "fixture", structured=True)
    h = p * np.exp(np.arange(93)[:, None] / 30 + np.arange(93)[None, :] / 60)
    h /= h.sum()
    decisions = []
    for tick in range(3):
        active = tick != 1
        proposal = ConditionalChainProposal(p, h, [.1, .1] if active else [1., 1.])
        draw = proposal.sample(pair, np.random.default_rng(tick + 5))
        unit = {"actor_ids": [1, 2], "relation": "following", "draw": draw,
                "longitudinal_joint": longitudinal.tolist(), "lateral_conditionals": [lateral.tolist()] * 2}
        if active:
            unit.update(proposal_components=proposal.components(), criticality=.1,
                        training_observation={"contract": "highd_dual_kinematic14_v1", "joint": [0.] * 14})
        decisions.append({"decision_tick": tick, "time_s": tick / 10, "units": [unit],
                          "log_natural_probability": draw["log_natural_joint_probability"],
                          "log_proposal_probability": draw["log_proposal_joint_probability"],
                          "log_importance_ratio": draw["log_importance_ratio"]})
    return {"protocol": {"forced_start": False, "seed": 7, "natural_target_sha256": "fixture",
                         "config": {"cav_id": 100, "scenario_id": "fixture", "duration_s": .3,
                                    "actors": [{"id": i} for i in (1, 2, 100)]}},
            "termination": "collision", "failure": {"time_s": .3, "colliding_actor_ids": ["100", "1"]},
            "decisions": decisions}


def dataset_file(folder, sequences, **metadata):
    data = {"contract": "highd_full_critical_sequence_v1", "natural_target_sha256": "fixture",
            "observation_contract": "highd_dual_kinematic14_v1", "sequences": sequences}
    data.update(metadata)
    raw = json.dumps(data).encode()
    path = Path(folder) / "data.json"
    path.write_bytes(raw)
    manifest = Path(folder) / "manifest.json"
    manifest.write_text(json.dumps({"data_path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}))
    return str(manifest)


class SequenceTests(unittest.TestCase):
    @unittest.skipIf(HighDCriticalSequenceEnv is None, "Use D2RLTrain39 for Gym replay tests")
    def test_preserves_all_critical_states_and_full_path_weight(self):
        raw = raw_fixture()
        seq = extract(raw)
        self.assertEqual([s["raw_tick"] for s in seq["steps"]], [0, 2])
        self.assertEqual(seq["steps"][0]["next_raw_tick"], 2)
        self.assertAlmostEqual(seq["steps"][0]["delta_time_s"], .2)
        self.assertAlmostEqual(seq["steps"][-1]["delta_time_s"], .1)
        self.assertAlmostEqual(sequence_log_weight(seq, [[.1, .1]] * 2), seq["generation_log_weight"])
        self.assertNotAlmostEqual(sequence_log_weight(seq, [[.8, .2], [.1, .1]]), seq["generation_log_weight"])

    def test_never_silently_accepts_missing_steps_or_bv_only_collision(self):
        for kind in ("missing", "bv_only", "unbiased_q"):
            raw = raw_fixture()
            if kind == "missing":
                del raw["decisions"][1]
            elif kind == "bv_only":
                raw["failure"]["colliding_actor_ids"] = ["1", "2"]
            else:
                raw["decisions"][1]["units"][0]["draw"]["proposal_joint_probability"] *= .5
            with self.assertRaises(ValueError):
                extract(raw)

    @unittest.skipIf(HighDCriticalSequenceEnv is None, "Use D2RLTrain39 for Gym replay tests")
    def test_sequence_env_ends_only_after_all_steps_and_corrects_sampling(self):
        crash = extract(raw_fixture())
        safe = copy.deepcopy(crash)
        safe["collision_result"] = False
        empty = copy.deepcopy(safe)
        empty.update(steps=[], generation_log_weight=0.)
        with tempfile.TemporaryDirectory() as tmp:
            env = HighDCriticalSequenceEnv({"sequence_manifest": dataset_file(tmp, [crash, safe, empty])})
            env.reset(0)
            _, reward, done, _ = env.step([.1, .1])
            self.assertFalse(done)
            self.assertEqual(reward, 0.)
            _, reward, done, info = env.step([.1, .1])
            self.assertTrue(done)
            self.assertAlmostEqual(reward, -2 / 3)
            self.assertEqual(info["critical_steps"], 2)
            self.assertAlmostEqual(env.empirical_objective(lambda obs: [.1, .1]), 1 / 3)
            with self.assertRaises(RuntimeError):
                env.step([.1, .1])
            with self.assertRaises(ValueError):
                env.reset(2)
            env.reset(1)
            env.step([.1, .1])
            self.assertEqual(env.step([.1, .1])[1], 0.)

    @unittest.skipIf(HighDCriticalSequenceEnv is None, "Use D2RLTrain39 for Gym replay tests")
    def test_offpolicy_second_moment_includes_behavior_weight(self):
        sequence = extract(raw_fixture())
        new_w = sequence_log_weight(sequence, [[.5, .6], [.7, .8]])
        expected = np.exp(sequence["generation_log_weight"] + new_w)
        self.assertAlmostEqual(scaled_loss(sequence, new_w, 0.), expected)
        self.assertNotAlmostEqual(expected, np.exp(2 * new_w))

    def test_empty_exposure_remains_in_weighted_risk_denominator(self):
        crash = {"collision_result": True, "generation_log_weight": np.log(.1), "source_scenario_id": "a", "seed": 7}
        safe = {"collision_result": False, "generation_log_weight": 0.}
        result = weighted_diagnostics([crash, safe, safe, safe])
        self.assertAlmostEqual(result["conditional_weighted_collision_mean"], .025)
        self.assertEqual(result["crash_contribution_ess"], 1.)

    def test_censored_prefix_is_unknown_not_safe_and_still_audited(self):
        raw = raw_fixture()
        raw["failure"]["colliding_actor_ids"] = ["1", "2"]
        seq = extract(raw, allow_censored=True)
        self.assertIsNone(seq["collision_result"])
        self.assertFalse(seq["outcome_complete"])
        self.assertEqual(len(seq["steps"]), 2)
        with self.assertRaises(ValueError):
            weighted_diagnostics([seq])
        raw["decisions"][0]["log_importance_ratio"] += 1
        with self.assertRaises(ValueError):
            extract(raw, allow_censored=True)

    def test_partial_contributions_keep_all_attempts_and_no_full_risk(self):
        crash = extract(raw_fixture())
        crash["generation_log_weight"] = np.log(.1)
        safe = copy.deepcopy(crash)
        safe["collision_result"] = False
        unknown = copy.deepcopy(crash)
        unknown["collision_result"] = None
        stats = censored_diagnostics([crash, safe, unknown, unknown])
        self.assertEqual(stats["attempted"], 4)
        self.assertEqual(stats["complete_outcome_count"], 2)
        self.assertEqual(stats["full_horizon_no_crash_count"], 1)
        self.assertEqual(stats["bv_only_censored_count"], 2)
        self.assertAlmostEqual(stats["observed_crash_weighted_contribution_per_attempt"], .025)
        self.assertIsNone(stats["conditional_weighted_collision_mean"])
        self.assertAlmostEqual(censored_diagnostics([crash, safe])["conditional_weighted_collision_mean"], .05)

    def test_audit_exports_unknowns_but_training_export_stays_strict(self):
        for censored in (False, True):
            with self.subTest(censored=censored), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = raw_fixture()
                raw["protocol"]["proposal_scorer"] = "cav_criticality_v1"
                if censored:
                    raw["failure"]["colliding_actor_ids"] = ["1", "2"]
                episode_path = root / "episode.json"
                episode_path.write_text(json.dumps(raw), encoding="utf-8")
                run = {"mode": "behavior", "scenario_id": "fixture", "seed": 7,
                       "raw_cav_collision": not censored, "episode_path": str(episode_path)}
                (root / "collection_summary.json").write_text(json.dumps({"runs": [run]}), encoding="utf-8")
                (root / "collection_protocol.json").write_text(json.dumps({
                    "seed": 7, "repeats": 1, "modes": ["behavior"], "scenario_ids": ["fixture"]}), encoding="utf-8")
                summary = audit_collection(root, root / "audit")
                self.assertTrue(summary["probability_audit_passed"])
                self.assertEqual(summary["full_horizon_outcomes_complete"], not censored)
                self.assertFalse((root / "audit" / "sequence_manifest.json").exists())
                with self.assertRaises(ValueError):
                    audit_collection(root, root / "audit")
                if censored:
                    with self.assertRaises(ValueError):
                        prepare(root, root / "training")
                    self.assertFalse((root / "training").exists())
                else:
                    self.assertTrue(prepare(root, root / "training")["audit_passed"])

    def test_other_incomplete_episodes_are_not_bv_collision_prefixes(self):
        raw = raw_fixture()
        raw["termination"] = "support_exit"
        with self.assertRaises(ValueError):
            extract(raw, allow_censored=True)

    def test_first_collision_retains_unknown_full_horizon_result(self):
        raw = raw_fixture()
        raw["protocol"]["config"]["duration_s"] = 8.
        raw["failure"]["colliding_actor_ids"] = ["1", "2"]
        original = extract(raw, allow_censored=True)
        derived = with_first_event(original)
        self.assertIsNone(derived["collision_result"])
        self.assertFalse(derived["outcome_complete"])
        self.assertFalse(event_indicator(derived))
        self.assertEqual(derived["event_type"], "bv_only_competing_collision")
        self.assertNotIn("event_result", original)
        derived["event_result"] = True
        with self.assertRaises(ValueError):
            event_indicator(derived)
        raw["failure"]["colliding_actor_ids"] = ["1", "2", "100"]
        self.assertTrue(event_indicator(with_first_event(extract(raw))))

    def test_first_collision_wrong_horizon_and_fake_safe_are_rejected(self):
        with self.assertRaises(ValueError):
            with_first_event(extract(raw_fixture()))
        raw = raw_fixture()
        raw["protocol"]["config"]["duration_s"] = 8.
        raw["failure"]["colliding_actor_ids"] = ["1", "2"]
        sequence = extract(raw, allow_censored=True)
        sequence["collision_result"] = False
        with self.assertRaises(ValueError):
            with_first_event(sequence)

    def test_first_collision_export_preserves_all_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for seed, ids in ((7, ["1", "100"]), (8, ["1", "2"])):
                raw = raw_fixture()
                raw["protocol"].update(seed=seed, proposal_scorer="cav_criticality_v1")
                raw["protocol"]["config"]["duration_s"] = 8.
                raw["failure"]["colliding_actor_ids"] = ids
                path = root / f"episode{seed}.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                runs.append({"mode": "behavior", "scenario_id": "fixture", "seed": seed,
                             "raw_cav_collision": seed == 7, "episode_path": str(path)})
            protocol = {"seed": 7, "repeats": 2, "modes": ["behavior"], "scenario_ids": ["fixture"],
                        "config": {"duration_s": 8., "scenarios": [
                            {"id": "fixture", "actors": raw["protocol"]["config"]["actors"]}]}}
            (root / "collection_summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
            (root / "collection_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            result = prepare_first_collision(root, root / "first")
            stats = result["by_mode"]["behavior"]
            self.assertEqual(stats["attempted"], 2)
            self.assertEqual(stats["endpoint_counts"], {"cav_first_collision": 1, "bv_only_competing_collision": 1})
            data = json.loads((root / "first" / "sequences.json").read_text())
            self.assertEqual(data["contract"], FIRST_SEQUENCE)
            self.assertIsNone(data["sequences"][1]["collision_result"])
            self.assertAlmostEqual(stats["conditional_weighted_cav_first_collision_mean"],
                                   np.exp(data["sequences"][0]["generation_log_weight"]) / 2)

    @unittest.skipIf(HighDCriticalSequenceEnv is None, "Use D2RLTrain39 for Gym replay tests")
    def test_first_collision_replay_keeps_empty_competing_episode_in_denominator(self):
        raw = raw_fixture()
        raw["protocol"]["config"]["duration_s"] = 8.
        crash = with_first_event(extract(raw))
        raw["failure"]["colliding_actor_ids"] = ["1", "2"]
        competing = with_first_event(extract(raw, allow_censored=True))
        competing.update(steps=[], generation_log_weight=0.)
        safe = dict(competing, termination="duration_reached", end_time_s=8., colliding_actor_ids=[],
                    collision_result=False, outcome_complete=True)
        safe = with_first_event(safe)
        spec = {"natural_target_sha256": "fixture", "event_definition": FIRST_EVENT_DEFINITION,
                "initial_distribution": {"kind": "fixture"}}
        metadata = {"contract": FIRST_SEQUENCE, "event_definition": FIRST_EVENT_DEFINITION,
                    "experiment_target": spec, "experiment_target_sha256": target_digest(spec)}
        with tempfile.TemporaryDirectory() as tmp:
            manifest = dataset_file(tmp, [crash, competing, safe], **metadata)
            env = HighDCriticalSequenceEnv({"sequence_manifest": manifest})
            self.assertAlmostEqual(env.sampling_correction, 1 / 3)
            self.assertAlmostEqual(env.empirical_objective(lambda obs: [.1, .1]), 1 / 3)
            env.reset(0)
            env.step([.1, .1])
            _, reward, done, info = env.step([.1, .1])
            self.assertTrue(done)
            self.assertAlmostEqual(reward, -1 / 3)
            self.assertEqual(info["event_type"], "cav_first_collision")
            self.assertEqual(scaled_loss(competing, 0., env.log_scale), 0.)
            with self.assertRaises(ValueError):
                env.reset(1)
            # Repackaging a competing endpoint as legacy full-horizon is rejected.
            with self.assertRaises(ValueError):
                HighDCriticalSequenceEnv({"sequence_manifest": dataset_file(tmp, [crash, competing])})


if __name__ == "__main__":
    unittest.main()
