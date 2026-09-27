"""Causal scene construction and disjoint selection for the offline dual bundle.

Opt-in diagnostic runtime, not an accepted replacement for the D2RL NDE.
Observe on the original 25 Hz clock and decide at 10 Hz. SUMO coordinates are
reflected into the existing highD direction-1 convention without changing the
road: raw x=-SUMO front-x, raw y=SUMO center-y-width/2, lane=SUMO lane+2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import highd_lane_phase_longitudinal as phase
from .highd_dual_ndd import probability, locked_lateral
from .highd_lane_geometry import longitudinal_relation, side_features


class UnsupportedState(ValueError):
    pass


def source_rows(actors, frame, centers, relation_mode="bumper_parallel_v2"):
    """Build current raw observations from a synchronous straight-road snapshot.

    actors: integer id, center_x/y, lane (0/1), length/width, speed,
    acceleration, lateral_speed. Neighbor lookup includes noncontrolled actors.
    """
    rows = []
    for actor in actors:
        if actor["lane"] not in (0, 1):
            raise ValueError("Only the unchanged two-lane road is supported")
        others = [other for other in actors if other["id"] != actor["id"]]
        def neighbor(lane, ahead):
            eligible = [other for other in others if other["lane"] == lane
                        and ((other["center_x"] > actor["center_x"]) if ahead
                             else (other["center_x"] <= actor["center_x"]))
                        and (relation_mode == "legacy_center_v1" or longitudinal_relation(actor, other)[0] == ("front" if ahead else "rear"))]
            return min(eligible, key=lambda other: (abs(other["center_x"] - actor["center_x"]), other["id"])) if eligible else None
        if relation_mode not in ("legacy_center_v1", "bumper_parallel_v2"):
            raise ValueError("Unknown longitudinal relation contract")
        same_lane = [other for other in others if other["lane"] == actor["lane"]]
        parallel = [other for other in same_lane if longitudinal_relation(actor, other)[0] == "alongside"]
        if relation_mode == "bumper_parallel_v2":
            leaders = [other for other in same_lane if longitudinal_relation(actor, other)[0] == "front"]
            lead = min(leaders, key=lambda other: (other["center_x"], other["id"])) if leaders else None
        else:
            lead = neighbor(actor["lane"], True)
        gap = lead["center_x"] - actor["center_x"] - (lead["length"] + actor["length"]) / 2 if lead else 0.0
        rr = lead["speed"] - actor["speed"] if lead else 0.0
        free = lead is None or gap > 115
        # Remove only numerical noise at an executed acceleration-grid endpoint.
        accel = float(actor["acceleration"])
        for boundary in (-4., 2.):
            if abs(accel - boundary) < 1e-8:
                accel = boundary
        row = {"frame": int(frame), "id": int(actor["id"]), "laneId": actor["lane"] + 2,
               "x": -actor["center_x"] - actor["length"] / 2,
               "y": actor["center_y"] - actor["width"] / 2,
               "width": actor["length"], "height": actor["width"],
               "xVelocity": -actor["speed"], "yVelocity": actor["lateral_speed"],
               "acceleration_mps2": accel, "xAcceleration": -accel,
               "speed_mps": actor["speed"], "direction": 1, "travel_sign": -1,
               "precedingId": lead["id"] if lead else 0,
               "precedingXVelocity": -lead["speed"] if lead else 0., "dhw": gap,
               "parallel_ids": [int(other["id"]) for other in parallel],
               "reference_relation": ("following" if lead else "no_front_parallel_present" if parallel else "no_front"),
               "two_lane": True, "supported": (20 <= actor["speed"] <= 40 and -4 <= accel <= 2
                   and (free or (0 <= gap <= 115 and -10 <= rr <= 8)))}
        for side, lane in (("left", actor["lane"] + 1), ("right", actor["lane"] - 1)):
            rear = neighbor(lane, False) if lane in (0, 1) else None
            row[side + "FollowingId"] = rear["id"] if rear else 0
            front = neighbor(lane, True) if lane in (0, 1) else None
            alongside = [other for other in others if other["lane"] == lane
                         and longitudinal_relation(actor, other)[0] == "alongside"] if lane in (0, 1) else []
            nearest = min(alongside, key=lambda other: (abs(other["center_x"] - actor["center_x"]), other["id"])) if alongside else None
            row[side + "PrecedingId"] = front["id"] if front else 0
            row[side + "AlongsideId"] = nearest["id"] if nearest else 0
        rows.append(row)
    return rows


def disjoint_pairs(candidates, maximum_pairs):
    """Candidate priority is (relation priority, CAV distance, separation, IDs).

    Inputs are already causal eligible relations. Deterministic ties make the
    selector part of a well-defined target, not a stochastic unlogged proposal.
    """
    used, selected = set(), []
    for item in sorted(candidates, key=lambda item: item["priority"]):
        ids = tuple(item["actor_ids"])
        if len(set(ids)) != 2:
            raise ValueError("Self-paired actor")
        if len(selected) >= maximum_pairs:
            break
        if used.isdisjoint(ids):
            selected.append(item)
            used.update(ids)
    return selected


class SceneNDD:
    def __init__(self, bundle, controlled_ids, cav_id, lane_centers, maximum_pairs=1, seed=7,
                 relation_mode="bumper_parallel_v2"):
        self.bundle = bundle
        self.controlled_ids = tuple(sorted(controlled_ids))
        if len(set(self.controlled_ids)) != len(self.controlled_ids) or cav_id in self.controlled_ids:
            raise ValueError("CAV must be separate from distinct controlled BVs")
        self.cav_id = cav_id
        self.centers = {k + 2: float(value) for k, value in enumerate(lane_centers)}
        self.maximum_pairs = maximum_pairs
        self.relation_mode = relation_mode
        self.rng = np.random.default_rng(seed)
        self.rows = []
        self.frame = -1
        self.last_decision = -1
        self.latest = None
        self.index = None
        self.active = {}
        self.ended_events = []

    def observe(self, frame, actors):
        if frame <= self.frame:
            raise ValueError("Observation time must increase")
        rows = source_rows(actors, frame, self.centers, self.relation_mode)
        previous = {} if self.latest is None else self.latest.set_index("id").to_dict("index")
        for row in rows:
            old = previous.get(row["id"])
            row["segment_start"] = old["segment_start"] if old and old["frame"] + 1 == frame else frame
        self.rows.extend(rows)
        self.frame = frame
        self.latest = pd.DataFrame(rows)
        history = pd.DataFrame(self.rows)
        self.index = history.set_index(["frame", "id"])
        self.active = {}
        self.ended_events = []
        for actor, track in history.groupby("id", sort=False):
            if actor not in self.controlled_ids or int(track.frame.iloc[-1]) != frame:
                continue
            fields, events = phase.causal_phases(track, self.centers, {2, 3}, self.bundle.interface.phase_config)
            self.ended_events.extend(e for e in events if e["status"] not in ("right_censored", "gap_censored"))
            if fields["phase"][-1] > 0:
                part = track.iloc[-1:].copy()
                for key, values in fields.items():
                    part[key] = values[-1]
                self.active[int(actor)] = part

    def _own(self, actor):
        result = self.latest[self.latest.id == actor].reset_index(drop=True)
        if len(result) != 1:
            raise UnsupportedState(f"Missing actor {actor}")
        return result

    def reference(self, actor):
        own = self._own(actor)
        speed, gap = float(own.speed_mps.iloc[0]), float(own.dhw.iloc[0])
        free = int(own.precedingId.iloc[0]) == 0 or gap > 115
        rr = abs(float(own.precedingXVelocity.iloc[0])) - speed
        try:
            return self.bundle.interface.reference_probability([[speed, 115 if free else gap, 0 if free else rr, int(free)]])[0]
        except ValueError as exc:
            raise UnsupportedState(f"Actor {actor} leaves frozen NDD state support: speed={speed}, gap={gap}, rr={rr}") from exc

    def _lane_rows(self, actor):
        focal = self.active[actor]
        rear = phase.aligned(self.index, focal.frame.to_numpy(), focal.locked_rear_id.to_numpy())
        rear["frame"] = focal.frame.to_numpy()
        rear["id"] = focal.locked_rear_id.to_numpy()
        args = (focal, self.index, self.bundle.interface.marginal, self.bundle.interface.phase_config)
        f = phase.make_role_rows(focal, rear, *args, rear_role=False)
        r = phase.make_role_rows(rear, focal, *args, rear_role=True)
        return f, r

    def quiet(self, actor):
        own = self._own(actor).iloc[0]
        frames = np.arange(self.frame - 5, self.frame + 1)
        history = phase.aligned(self.index, frames, np.full(6, actor))
        return bool(np.isfinite(history.yVelocity).all() and (history.yVelocity.abs() < .2).all()
                    and (history.segment_start == own.segment_start).all())

    def quiet_state(self, actor, age):
        own = self._own(actor)
        row = own.iloc[0]
        prev, missing = phase.lag_acceleration(self.index, own, .2)
        free = row.precedingId <= 0 or row.dhw > 115
        state = [row.speed_mps, 115 if free else row.dhw,
                 0 if free else abs(row.precedingXVelocity) - row.speed_mps, prev[0], row.yVelocity,
                 row.y + row.height / 2 - self.centers[int(row.laneId)],
                 row.laneId < 3, row.laneId > 2]
        for side, delta in (("left", 1), ("right", -1)):
            rear_id = int(row[side + "FollowingId"])
            rear = self.latest[self.latest.id == rear_id]
            valid = len(rear) == 1 and rear.laneId.iloc[0] == row.laneId + delta
            if valid:
                rear = rear.iloc[0]
                # Match the trained bumper-gap feature in reflected coordinates.
                gap = -(row.x + row.width / 2 - rear.x - rear.width / 2) - (row.width + rear.width) / 2
                state.extend([1, np.clip(gap, -20, 115), rear.speed_mps - row.speed_mps, rear.acceleration_mps2])
            else:
                state.extend([0, 115, 0, 0])
        state.extend([0, age, missing[0]])
        if self.bundle.manifest.get("geometry_onset_model"):
            for side, delta in (("left", 1), ("right", -1)):
                front = phase.aligned(self.index, own.frame.to_numpy(), own[side + "PrecedingId"].to_numpy())
                alongside = phase.aligned(self.index, own.frame.to_numpy(), own[side + "AlongsideId"].to_numpy())
                state.extend(side_features(own, front, alongside, delta)[0])
        return np.asarray(state, dtype=np.float32), np.array([True, row.laneId < 3, row.laneId > 2])

    def lateral(self, actor, locked_ids, age):
        if actor in locked_ids or actor in self.active or not self.quiet(actor):
            return locked_lateral(), "locked_or_not_eligible"
        state, permitted = self.quiet_state(actor, age)
        return self.bundle.lateral_conditional(locked=False, quiet_state=state, permitted=permitted), "quiet_conditional_start"

    def candidates(self, locked_ids, command_locks=None):
        candidates = []
        cav = self._own(self.cav_id).iloc[0]
        def priority(ids, relation):
            x = [float(self._own(i).x.iloc[0]) for i in ids]
            return (relation, sum(abs(v - cav.x) for v in x), abs(x[0] - x[1]), *ids)
        geometry_focals = set()
        if self.bundle.manifest.get("interaction_selection") == "target_geometry_v1":
            command_locks = command_locks or {}
            # Use command target before phase confirmation, then retain the
            # observed causal target even across a lane-ID switch.
            for focal in sorted(set(self.active) | set(command_locks)):
                target = (int(command_locks[focal]["target_lane"]) + 2 if focal in command_locks
                          else int(self.active[focal].target_lane.iloc[0]))
                own = self._own(focal).iloc[0]
                neighbors = self.latest[(self.latest.laneId == target) & (self.latest.id != focal)]
                choices = []
                for _, other in neighbors.iterrows():
                    dx = -(other.x + other.width / 2 - own.x - own.width / 2)
                    half = (other.width + own.width) / 2
                    role = "alongside" if abs(dx) < half else "front" if dx >= half else "rear"
                    choices.append((0 if role == "alongside" else 1, max(0., abs(dx) - half), int(other.id), role))
                if not choices:
                    continue
                _, _, partner, role = min(choices)
                if role == "rear" and focal in self.active and partner == int(self.active[focal].locked_rear_id.iloc[0]):
                    continue  # Only this exact relation may use the fitted rear model.
                geometry_focals.add(focal)
                if partner not in self.controlled_ids:
                    # Never replace a geometrically relevant CAV with a farther
                    # BV merely to obtain an eligible fitted joint pair.
                    continue
                ids = (focal, partner)
                candidates.append({"actor_ids": ids, "relation": "target_" + role + "_independent",
                                   "priority": priority(ids, -2 if role == "alongside" else -1),
                                   "phase": int(self.active[focal].phase.iloc[0]) if focal in self.active else None})
        for focal, part in self.active.items():
            if focal in geometry_focals:
                continue
            rear = int(part.locked_rear_id.iloc[0])
            if rear in self.controlled_ids:
                f, r = self._lane_rows(focal)
                if f is not None and r is not None:
                    ids = (focal, rear)
                    candidates.append({"actor_ids": ids, "relation": "lane_process",
                                       "priority": priority(ids, 0), "focal_data": f, "rear_data": r,
                                       "phase": int(part.phase.iloc[0])})
        for follower in self.controlled_ids:
            first = self._own(follower)
            leader = int(first.precedingId.iloc[0])
            ids = (follower, leader)
            if (leader in self.controlled_ids and all(i not in self.active and i not in locked_ids for i in ids)
                    and 0 <= float(first.dhw.iloc[0]) <= 115):
                candidates.append({"actor_ids": ids, "relation": "following", "priority": priority(ids, 1)})
        return candidates

    def active_marginal(self, actor, reference):
        if actor not in self.active:
            return reference
        f, _ = self._lane_rows(actor)
        if f is None:
            raise UnsupportedState(f"Active actor {actor} lacks supported lane-process features")
        return self.bundle.interface.lane_marginal(f["state"], "focal", reference[None, :])[0]

    def decide(self, decision_tick, locked_ids=(), proposal_provider=None):
        if decision_tick <= self.last_decision or self.frame != decision_tick * 25 // 10:
            raise ValueError("Expected a new 10 Hz decision using the latest causal 25 Hz frame")
        self.last_decision = decision_tick
        age = (decision_tick * 25 - self.frame * 10) / 250
        refs = {actor: self.reference(actor) for actor in self.controlled_ids}
        selected = disjoint_pairs(self.candidates(set(locked_ids), locked_ids if isinstance(locked_ids, dict) else None), self.maximum_pairs)
        assigned, actions, units = set(), [], []
        for candidate in selected:
            ids = candidate["actor_ids"]
            if candidate["relation"] == "following":
                pair = self.bundle.following_pair(self._own(ids[0]), self._own(ids[1]))
            elif candidate["relation"] == "lane_process":
                f, r = candidate["focal_data"], candidate["rear_data"]
                pair = self.bundle.lane_pair(ids, f["state"][0], r["state"][0], refs[ids[0]], refs[ids[1]], candidate["phase"])
            else:
                pair = self.bundle.independent_pair(ids, *(self.active_marginal(i, refs[i]) for i in ids))
                pair.roles = ("focal", candidate["relation"].removeprefix("target_").removesuffix("_independent"))
                pair.model_id += ":" + candidate["relation"]
            lateral = [self.lateral(actor, locked_ids, age) for actor in ids]
            pair.model_id += ":" + self.relation_mode
            structured = self.bundle.with_lateral(pair, lateral[0][0], lateral[1][0])
            proposal = proposal_provider(self, candidate, structured) if proposal_provider is not None else None
            draw = structured.sample(self.rng) if proposal is None else proposal.sample(structured, self.rng)
            units.append({"actor_ids": list(ids), "relation": candidate["relation"],
                          "phase": candidate.get("phase"), "model_id": draw["model_id"],
                          "longitudinal_joint": pair.natural.tolist(),
                          "lateral_conditionals": [p.tolist() for p, _ in lateral],
                          "lateral_status": [s for _, s in lateral], "draw": draw})
            if proposal is not None:
                units[-1]["proposal_components"] = proposal.components()
                if hasattr(proposal, "training_observation"):
                    units[-1]["training_observation"] = proposal.training_observation
                    units[-1]["criticality"] = proposal.criticality
            actions.extend(draw["actions"])
            assigned.update(ids)
        for actor in self.controlled_ids:
            if actor in assigned:
                continue
            p, source = refs[actor], "frozen_single_reference"
            if actor in self.active:
                p = self.active_marginal(actor, refs[actor])
                source = "unpaired_active_focal_history"
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
        return {"decision_tick": decision_tick, "observation_frame": self.frame, "observation_age_s": age,
                "reference_relations": [{"actor_id": int(r.id), "preceding_id": int(r.precedingId),
                    "parallel_ids": r.parallel_ids, "reference_relation": r.reference_relation,
                    "gap_m": float(r.dhw)} for _, r in self.latest.iterrows() if r.id in self.controlled_ids],
                "selected_pairs": [{k: c[k] for k in ("actor_ids", "relation", "priority")} for c in selected],
                "units": units, "actions": actions,
                "log_natural_probability": sum(u["draw"]["log_natural_joint_probability"] for u in units),
                "log_proposal_probability": sum(u["draw"]["log_proposal_joint_probability"] for u in units),
                "log_importance_ratio": sum(u["draw"]["log_importance_ratio"] for u in units)}


def audit_decision(record):
    """Reconstruct the actual sampled structured probability of each unit."""
    largest = 0.
    actors = []
    for unit in record["units"]:
        actions = unit["draw"]["actions"]
        indices = [a["acceleration_index"] for a in actions]
        if len(actions) == 2:
            value = unit["longitudinal_joint"][indices[0]][indices[1]]
        else:
            value = unit["longitudinal_probability"][indices[0]]
        for index, action, lateral in zip(indices, actions, unit["lateral_conditionals"]):
            value *= lateral[index][action["lateral_choice"]]
        largest = max(largest, abs(value - unit["draw"]["natural_joint_probability"]))
        if "proposal_components" in unit:
            from .highd_dual_ndd import compose_lateral
            from .highd_dual_ndd_proposal import ConditionalChainProposal
            from d2rl_training.conditional_chain import replay_weight
            spec = unit["proposal_components"]
            p = compose_lateral(np.asarray(unit["longitudinal_joint"]), *map(np.asarray, unit["lateral_conditionals"]))
            h = np.asarray(spec["critical_first"])[:, None] * np.asarray(spec["critical_second_given_first"])
            proposal = ConditionalChainProposal(p, h, spec["epsilon"])
            codes = tuple(a["action_index"] for a in actions)
            q = float(proposal.matrix[codes])
            logged = unit["draw"]
            if not np.isclose(q, logged["proposal_joint_probability"], rtol=1e-10, atol=0):
                raise ValueError("Logged Q disagrees with the actual conditional proposal")
            weight = replay_weight(logged["weight_record"], spec["epsilon"], logged["ndd_record"])
            if not np.isclose(np.log(weight), logged["log_importance_ratio"], rtol=1e-9, atol=1e-12):
                raise ValueError("Logged log P/Q disagrees with conditional components")
            largest = max(largest, abs(q - logged["proposal_joint_probability"]))
        actors.extend(a["actor_id"] for a in actions)
    if len(set(actors)) != len(actors):
        raise ValueError("Actor sampled by overlapping control units")
    return largest
