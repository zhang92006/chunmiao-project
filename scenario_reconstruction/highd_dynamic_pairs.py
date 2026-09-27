"""Opt-in multi-pair natural target; frozen single-pair runtime stays unchanged.

All units are prepared from one pre-action snapshot. Deterministic matching is
part of P; the separately selected single intervention unit changes only Q.
"""
from __future__ import annotations

import numpy as np

from .highd_dual_ndd import probability
from .highd_dual_ndd_runtime import SceneNDD, disjoint_pairs, audit_decision


PARTITION_VERSION = "dynamic_disjoint_relation_continuity_v1"
INTERVENTION_VERSION = "maximum_critical_mass_one_natural_unit_v1"


def pair_key(candidate):
    # Roles/order matter for conditional probabilities and lane-process history.
    return (candidate["relation"], *map(int, candidate["actor_ids"]))


def partition(candidates, previous=(), maximum_pairs=None):
    if maximum_pairs is not None and (type(maximum_pairs) is not int or maximum_pairs < 0):
        raise ValueError("maximum_pairs must be a nonnegative integer or null (all eligible pairs)")
    previous = set(previous)
    ranked = []
    for candidate in candidates:
        item = dict(candidate)
        priority = candidate["priority"]
        # Retain valid same-role relations within a relation class. A stronger
        # class (e.g. ongoing lane interaction) can preempt following immediately.
        # CAV proximity is deliberately NOT the natural matching objective.
        item["priority"] = (priority[0], 0 if pair_key(item) in previous else 1,
                            priority[2], *map(int, item["actor_ids"]), item["relation"])
        ranked.append(item)
    selected = disjoint_pairs(ranked, len(ranked) if maximum_pairs is None else maximum_pairs)
    return selected, sorted(ranked, key=lambda c: c["priority"])


def select_intervention(scene, prepared, provider):
    """Evaluate all natural units before any draw; choose at most one Q != P.

    The supplied provider is a pre-action scorer, not a vehicle sampler. No
    cross-pair regrouping and no random selection probability is introduced.
    """
    proposals, diagnostics = {}, []
    for index, (candidate, pair, lateral, structured) in enumerate(prepared):
        if provider is None:
            break
        first_record = len(getattr(provider, "records", []))
        proposal = provider(scene, candidate, structured)
        added = getattr(provider, "records", [])[first_record:]
        mass = float(proposal.criticality) if proposal is not None else 0.
        if proposal is not None:
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("Intervention selection requires finite positive critical mass")
            proposals[index] = proposal
        diagnostics.append({"actor_ids": list(candidate["actor_ids"]), "relation": candidate["relation"],
                            "eligible": proposal is not None, "critical_mass": mass,
                            "provider_records": added})
    chosen = min(proposals, key=lambda i: (-float(proposals[i].criticality),
                  pair_key(prepared[i][0]))) if proposals else None
    for index, row in enumerate(diagnostics):
        row["selected"] = index == chosen
        for record in row.pop("provider_records"):
            record["selected_for_intervention"] = index == chosen
            if row["eligible"] and index != chosen:
                record["status"] = "critical_not_selected_natural"
    return chosen, proposals.get(chosen), diagnostics


class DynamicSceneNDD(SceneNDD):
    def __init__(self, bundle, controlled_ids, cav_id, lane_centers, maximum_pairs=None,
                 seed=7, relation_mode="bumper_parallel_v2"):
        partition([], maximum_pairs=maximum_pairs)  # validate before starting SUMO decisions
        super().__init__(bundle, controlled_ids, cav_id, lane_centers, maximum_pairs, seed, relation_mode)
        self.previous_pairs = set()

    def _prepare_pair(self, candidate, refs, locked_ids, age):
        ids = candidate["actor_ids"]
        if candidate["relation"] == "following":
            pair = self.bundle.following_pair(self._own(ids[0]), self._own(ids[1]))
        elif candidate["relation"] == "lane_process":
            f, r = candidate["focal_data"], candidate["rear_data"]
            pair = self.bundle.lane_pair(ids, f["state"][0], r["state"][0], refs[ids[0]], refs[ids[1]], candidate["phase"])
        else:
            # Unsupported correlated roles remain explicitly independent. Never
            # pass a new rear/front/alongside identity to the fitted locked-rear model.
            pair = self.bundle.independent_pair(ids, *(self.active_marginal(i, refs[i]) for i in ids))
            pair.roles = ("focal", candidate["relation"].removeprefix("target_").removesuffix("_independent"))
            pair.model_id += ":" + candidate["relation"]
        lateral = [self.lateral(actor, locked_ids, age) for actor in ids]
        pair.model_id += ":" + self.relation_mode
        return candidate, pair, lateral, self.bundle.with_lateral(pair, lateral[0][0], lateral[1][0])

    def decide(self, decision_tick, locked_ids=(), proposal_provider=None):
        if decision_tick <= self.last_decision or self.frame != decision_tick * 25 // 10:
            raise ValueError("Expected a new 10 Hz decision using the latest causal 25 Hz frame")
        self.last_decision = decision_tick
        age = (decision_tick * 25 - self.frame * 10) / 250
        refs = {actor: self.reference(actor) for actor in self.controlled_ids}
        candidates = self.candidates(set(locked_ids), locked_ids if isinstance(locked_ids, dict) else None)
        selected, ranked = partition(candidates, self.previous_pairs, self.maximum_pairs)
        prepared = [self._prepare_pair(c, refs, locked_ids, age) for c in selected]
        chosen, proposal, interventions = select_intervention(self, prepared, proposal_provider)
        # Only now may any unit consume the action RNG.
        assigned, actions, units = set(), [], []
        for index, (candidate, pair, lateral, structured) in enumerate(prepared):
            active = proposal if index == chosen else None
            draw = structured.sample(self.rng) if active is None else active.sample(structured, self.rng)
            unit = {"actor_ids": list(candidate["actor_ids"]), "relation": candidate["relation"],
                    "phase": candidate.get("phase"), "model_id": draw["model_id"],
                    "longitudinal_joint": pair.natural.tolist(),
                    "lateral_conditionals": [p.tolist() for p, _ in lateral],
                    "lateral_status": [s for _, s in lateral], "draw": draw}
            if active is not None:
                unit.update(proposal_components=active.components(),
                            training_observation=active.training_observation, criticality=active.criticality)
            units.append(unit)
            actions.extend(draw["actions"])
            assigned.update(candidate["actor_ids"])
        for actor in self.controlled_ids:
            if actor in assigned:
                continue
            p, source = refs[actor], "frozen_single_reference"
            if actor in self.active:
                p, source = self.active_marginal(actor, refs[actor]), "unpaired_active_focal_history"
            lateral, status = self.lateral(actor, locked_ids, age)
            pdf = probability(p[:, None] * lateral, axis=(0, 1))
            code = int(self.rng.choice(93, p=pdf.ravel()))
            a, d = divmod(code, 3)
            action = {"actor_id": actor, "role": "single", "action_index": code,
                      "acceleration_index": a, "acceleration_mps2": -4 + .2 * a, "lateral_choice": d}
            draw = {"actions": [action], "natural_joint_probability": float(pdf[a, d]),
                    "proposal_joint_probability": float(pdf[a, d]),
                    "log_natural_joint_probability": float(np.log(pdf[a, d])),
                    "log_proposal_joint_probability": float(np.log(pdf[a, d])), "log_importance_ratio": 0.}
            units.append({"actor_ids": [actor], "relation": source, "lateral_status": [status],
                          "longitudinal_probability": p.tolist(), "lateral_conditionals": [lateral.tolist()], "draw": draw})
            actions.append(action)
        if self.bundle.motion_model and self.bundle.motion_model.get("kernel") == "conditional_paired_motion_v1":
            from .highd_lane_motion_context import context_features
            for action in actions:
                if action["lateral_choice"]:
                    state, _ = self.quiet_state(action["actor_id"], age)
                    action["motion_context"] = context_features(state, action["lateral_choice"], action["acceleration_mps2"]).tolist()
        if sorted(a["actor_id"] for a in actions) != list(self.controlled_ids):
            raise ValueError("Every controlled BV must be sampled exactly once")
        current = {pair_key(c) for c in selected}
        record = {"decision_tick": decision_tick, "observation_frame": self.frame, "observation_age_s": age,
                  "partition_version": PARTITION_VERSION, "intervention_version": INTERVENTION_VERSION,
                  "maximum_pairs": self.maximum_pairs,
                  "previous_pairs": sorted(self.previous_pairs),
                  "pair_transitions": {"added": sorted(current - self.previous_pairs),
                                       "removed": sorted(self.previous_pairs - current)},
                  "candidate_pairs": [{k: c[k] for k in ("actor_ids", "relation", "priority")} for c in ranked],
                  "selected_pairs": [{k: c[k] for k in ("actor_ids", "relation", "priority")} for c in selected],
                  "intervention_candidates": interventions,
                  "intervention_actor_ids": list(selected[chosen]["actor_ids"]) if chosen is not None else None,
                  "reference_relations": [{"actor_id": int(r.id), "preceding_id": int(r.precedingId),
                      "parallel_ids": r.parallel_ids, "reference_relation": r.reference_relation,
                      "gap_m": float(r.dhw)} for _, r in self.latest.iterrows() if r.id in self.controlled_ids],
                  "units": units, "actions": actions,
                  "log_natural_probability": sum(u["draw"]["log_natural_joint_probability"] for u in units),
                  "log_proposal_probability": sum(u["draw"]["log_proposal_joint_probability"] for u in units),
                  "log_importance_ratio": sum(u["draw"]["log_importance_ratio"] for u in units)}
        self.previous_pairs = current
        audit_dynamic_decision(record)
        return record


def audit_dynamic_decision(record):
    error = audit_decision(record)
    if record["partition_version"] != PARTITION_VERSION or record["intervention_version"] != INTERVENTION_VERSION:
        raise ValueError("Unknown partition/intervention version")
    selected, _ = partition(record["candidate_pairs"], map(tuple, record["previous_pairs"]), record["maximum_pairs"])
    if [pair_key(c) for c in selected] != [pair_key(c) for c in record["selected_pairs"]]:
        raise ValueError("Partition does not reproduce the deterministic matching rule")
    previous = set(map(tuple, record["previous_pairs"]))
    current = {pair_key(c) for c in selected}
    transitions = record["pair_transitions"]
    if (set(map(tuple, transitions["added"])) != current - previous or
            set(map(tuple, transitions["removed"])) != previous - current):
        raise ValueError("Pair transition log disagrees with history")
    biased = [u for u in record["units"] if "proposal_components" in u]
    if len(biased) > 1 or (biased[0]["actor_ids"] if biased else None) != record["intervention_actor_ids"]:
        raise ValueError("At most one selected natural unit may be biased")
    logged_pairs = [(tuple(u["actor_ids"]), u["relation"]) for u in record["units"] if len(u["actor_ids"]) == 2]
    if logged_pairs != [(tuple(c["actor_ids"]), c["relation"]) for c in record["selected_pairs"]]:
        raise ValueError("Sampled units differ from the natural partition")
    interventions = record["intervention_candidates"]
    if interventions and [pair_key(c) for c in interventions] != [pair_key(c) for c in selected]:
        raise ValueError("Intervention scores must cover the existing natural pairs in order")
    eligible = [c for c in interventions if c["eligible"]]
    winner = min(eligible, key=lambda c: (-c["critical_mass"], pair_key(c))) if eligible else None
    if (list(winner["actor_ids"]) if winner else None) != record["intervention_actor_ids"]:
        raise ValueError("Intervention does not reproduce the deterministic critical-mass selector")
    for candidate in interventions:
        if not np.isfinite(candidate["critical_mass"]) or candidate["critical_mass"] < 0:
            raise ValueError("Invalid logged critical mass")
        if candidate["selected"] != (candidate is winner):
            raise ValueError("Incorrect intervention selection flag")
    for unit in record["units"]:
        if "proposal_components" not in unit and not np.isclose(
                unit["draw"]["log_natural_joint_probability"], unit["draw"]["log_proposal_joint_probability"],
                rtol=0, atol=1e-12):
            raise ValueError("Nonintervened natural pair/single must have Q=P")
    p = sum(u["draw"]["log_natural_joint_probability"] for u in record["units"])
    q = sum(u["draw"]["log_proposal_joint_probability"] for u in record["units"])
    if not np.allclose([p, q, p - q], [record["log_natural_probability"], record["log_proposal_probability"],
                                      record["log_importance_ratio"]], rtol=1e-10, atol=1e-10):
        raise ValueError("Scene probability product disagrees with its units")
    return error
