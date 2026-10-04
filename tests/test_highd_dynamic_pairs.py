import copy
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_dual_ndd import PairActionDistribution, compose_lateral, locked_lateral
from scenario_reconstruction.highd_dual_ndd_proposal import ConditionalChainProposal
from scenario_reconstruction.highd_dual_ndd_runtime import source_rows
from scenario_reconstruction.highd_dynamic_pairs import DynamicSceneNDD, partition, pair_key, audit_dynamic_decision
from scenario_reconstruction.highd_critical_sequences import extract


def actors(extra=False):
    values = [(100, 150, 46, 1), (1, 230, 42, 0), (2, 205, 46, 1), (3, 280, 42, 0), (4, 280, 46, 1)]
    if extra:
        values.append((5, 500, 42, 0))
    return [{"id": i, "center_x": x, "center_y": y, "lane": lane, "speed": 30.,
             "length": 5., "width": 1.8, "acceleration": 0., "lateral_speed": 0.}
            for i, x, y, lane in values]


def edge(ids, separation=10, relation="following", rank=1):
    return {"actor_ids": ids, "relation": relation, "priority": (rank, 10., separation, *ids)}


class MatchingTests(unittest.TestCase):
    def test_multiple_disjoint_pairs_and_odd_single(self):
        candidates = [edge((1, 2)), edge((2, 3), 11), edge((3, 4)), edge((4, 5), 12)]
        selected, _ = partition(candidates)
        self.assertEqual([c["actor_ids"] for c in selected], [(1, 2), (3, 4)])
        self.assertEqual(selected, partition(candidates[::-1])[0])

    def test_valid_relation_persists_but_disappeared_relation_does_not(self):
        old = edge((1, 2), 80)
        other = edge((2, 3), 5)
        self.assertEqual(partition([old, other], [pair_key(old)])[0][0]["actor_ids"], (1, 2))
        self.assertEqual(partition([other], [pair_key(old)])[0][0]["actor_ids"], (2, 3))

    def test_lane_interaction_preempts_retained_following(self):
        old = edge((1, 2))
        new = edge((2, 3), relation="lane_process", rank=0)
        self.assertEqual(partition([old, new], [pair_key(old)])[0][0]["relation"], "lane_process")

    def test_caps_and_invalid_values(self):
        edges = [edge((1, 2)), edge((3, 4))]
        self.assertEqual(len(partition(edges, maximum_pairs=1)[0]), 1)
        self.assertEqual(partition(edges, maximum_pairs=0)[0], [])
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                partition(edges, maximum_pairs=value)


class FixtureBundle:
    manifest = {"interaction_selection": "target_geometry_v1"}
    motion_model = None

    def following_pair(self, first, second):
        pdf = np.ones((31, 31)) + np.eye(31) * 2
        pdf /= pdf.sum()
        return PairActionDistribution((int(first.id.iloc[0]), int(second.id.iloc[0])),
                                      ("follower", "leader"), pdf, "fixture_following")

    def independent_pair(self, ids, first, second):
        return PairActionDistribution(ids, ("focal", "other"), first[:, None] * second, "fixture_independent")

    def with_lateral(self, pair, first, second):
        return PairActionDistribution(pair.actor_ids, pair.roles,
                                      compose_lateral(pair.natural, first, second), pair.model_id, structured=True)


class FixtureScene(DynamicSceneNDD):
    def reference(self, actor):
        return np.ones(31) / 31

    def lateral(self, actor, locked_ids, age):
        return locked_lateral(), "fixture_stay"

    def set_snapshot(self, tick, values):
        self.frame = tick * 25 // 10
        self.latest = pd.DataFrame(source_rows(values, self.frame, self.centers))


class DynamicRuntimeTests(unittest.TestCase):
    def scene(self, extra=False, cap=None):
        scene = FixtureScene(FixtureBundle(), [1, 2, 3, 4, 5] if extra else [1, 2, 3, 4],
                             100, [42, 46], maximum_pairs=cap)
        scene.set_snapshot(0, actors(extra))
        return scene

    def provider(self, scene, scores=None):
        state = copy.deepcopy(scene.rng.bit_generator.state)
        calls = []
        def provide(current, candidate, distribution):
            # Every candidate must be scored BEFORE the first RNG draw.
            self.assertEqual(state, current.rng.bit_generator.state)
            calls.append(tuple(distribution.actor_ids))
            if scores is not None and tuple(distribution.actor_ids) not in scores:
                return None
            p = distribution.natural
            h = p * np.exp(np.arange(93)[:, None] / 90)
            h /= h.sum()
            proposal = ConditionalChainProposal(p, h, [.1, .2])
            proposal.criticality = scores[tuple(distribution.actor_ids)] if scores else 1.
            proposal.training_observation = {"contract": "highd_dual_kinematic14_v1", "joint": [0.] * 14}
            return proposal
        return provide, calls

    def test_all_units_natural_with_unpaired_vehicle(self):
        record = self.scene(extra=True).decide(0)
        self.assertEqual(len(record["selected_pairs"]), 2)
        self.assertEqual(sorted(a["actor_id"] for a in record["actions"]), [1, 2, 3, 4, 5])
        self.assertEqual(len(record["units"]), 3)
        self.assertEqual(record["log_importance_ratio"], 0.)
        self.assertIsNone(record["intervention_actor_ids"])

    def test_highest_critical_pair_only_and_other_pair_cancels(self):
        scene = self.scene()
        provider, calls = self.provider(scene, {(1, 3): .1, (2, 4): .7})
        record = scene.decide(0, proposal_provider=provider)
        self.assertEqual(calls, [(1, 3), (2, 4)])
        self.assertEqual(record["intervention_actor_ids"], [2, 4])
        biased = [u for u in record["units"] if "proposal_components" in u]
        self.assertEqual(len(biased), 1)
        self.assertAlmostEqual(record["log_importance_ratio"], biased[0]["draw"]["log_importance_ratio"])
        self.assertLess(audit_dynamic_decision(record), 1e-12)

    def test_no_critical_candidate_preserves_natural_rng_and_actions(self):
        scene = self.scene()
        provider, _ = self.provider(scene, {})
        actual = scene.decide(0, proposal_provider=provider)
        expected = self.scene().decide(0)
        self.assertEqual(actual["actions"], expected["actions"])
        self.assertIsNone(actual["intervention_actor_ids"])

    def test_tie_is_deterministic(self):
        scene = self.scene()
        provider, _ = self.provider(scene)
        record = scene.decide(0, proposal_provider=provider)
        self.assertEqual(record["intervention_actor_ids"], [1, 3])

    def test_partner_and_relation_change_after_command_and_completion(self):
        scene = self.scene()
        before = scene.decide(0)
        scene.set_snapshot(1, actors())
        during = scene.decide(1, {1: {"target_lane": 1}})
        self.assertEqual(during["selected_pairs"][0]["actor_ids"], (1, 2))
        self.assertEqual(during["selected_pairs"][0]["relation"], "target_rear_independent")
        self.assertIn("independent", during["units"][0]["model_id"])
        values = actors()
        values[1].update(lane=1, center_y=46.)
        scene.set_snapshot(2, values)
        after = scene.decide(2)
        self.assertEqual(after["selected_pairs"][0]["actor_ids"], (2, 1))
        self.assertEqual(after["selected_pairs"][0]["relation"], "following")
        self.assertTrue(after["pair_transitions"]["removed"])
        self.assertNotEqual(before["selected_pairs"], after["selected_pairs"])

    def test_critical_sequence_handles_changing_pairs_without_weight_reset(self):
        scene = self.scene()
        provider, _ = self.provider(scene)
        first = scene.decide(0, proposal_provider=provider)
        values = actors()
        values[1].update(lane=1, center_y=46.)
        scene.set_snapshot(1, values)
        provider, _ = self.provider(scene)
        second = scene.decide(1, proposal_provider=provider)
        for tick, record in enumerate((first, second)):
            record["time_s"] = tick / 10
        episode = {"protocol": {"forced_start": False, "natural_target_sha256": "dynamic_fixture",
                   "seed": 7, "config": {"cav_id": 100, "duration_s": .2, "scenario_id": "test", "actors": actors()}},
                   "termination": "duration_reached", "failure": None, "decisions": [first, second]}
        sequence = extract(episode)
        self.assertEqual(len(sequence["steps"]), 2)
        self.assertNotEqual(sequence["steps"][0]["actor_ids"], sequence["steps"][1]["actor_ids"])
        self.assertAlmostEqual(sequence["generation_log_weight"], first["log_importance_ratio"] + second["log_importance_ratio"])

    def test_rejects_duplicate_time_and_mismatched_selected_unit(self):
        scene = self.scene()
        record = scene.decide(0)
        with self.assertRaises(ValueError):
            scene.decide(0)
        record["intervention_actor_ids"] = [1, 3]
        with self.assertRaises(ValueError):
            audit_dynamic_decision(record)


if __name__ == "__main__":
    unittest.main()
