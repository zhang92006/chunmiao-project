"""Assemble pinned dual-BV offline candidates without changing SUMO defaults."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from . import highd_dual_ndd as dual
from .highd_pair_data_inventory import digest, read_json


class DualNDDBundle:
    def __init__(self, manifest_path="configs/highd_dual_ndd_bundle_v1.json"):
        self.manifest = read_json(manifest_path)
        interface_spec = self.manifest["interface_config"]
        if digest(interface_spec["path"]) != interface_spec["sha256"]:
            raise ValueError("Pinned interface configuration changed")
        self.interface = dual.FrozenDualNDD(read_json(interface_spec["path"]))
        spec = self.manifest["lane_pair_model"]
        report = read_json(spec["train_summary"])
        if digest(spec["path"]) != spec["sha256"] or report["model_sha256"] != spec["sha256"]:
            raise ValueError("Pinned lane-pair artifact changed")
        if report["provenance"]["interface_code_sha256"] != digest(dual.__file__):
            raise ValueError("Frozen marginal/joint probability code changed")
        if report["provenance"]["lane_reference_contract_sha256"] != digest(self.interface.config["lane_reference_contract"]):
            raise ValueError("Lane joint was fitted with another marginal reference")
        if not all(item["converged"] for item in report["optimization"]):
            raise ValueError("Lane-pair fitting did not converge")
        with np.load(spec["path"], allow_pickle=False) as artifact:
            self.weights = artifact["weights"]
            self.independent_mass = float(artifact["independent_mass"])
            if not np.array_equal(artifact["permutation_order"], dual.PERMUTATIONS):
                raise ValueError("Joint rank encoding changed")
        # Onset and longitudinal components together define P. The lane-pair
        # hash alone cannot distinguish a new onset target from the old one.
        self.model_id = self.manifest.get("version", "highd_dual_ndd_bundle_v1") + ":" + digest(manifest_path)
        self.motion_model = None
        if self.manifest.get("motion_model"):
            spec = self.manifest["motion_model"]
            if digest(spec["path"]) != spec["sha256"]:
                raise ValueError("Motion process model changed")
            self.motion_model = read_json(spec["path"])
            contract = self.manifest["runtime_contract"]
            for name in ("geometry", "motion"):
                module = "highd_lane_" + name
                if name == "motion" and self.motion_model.get("kernel") == "conditional_paired_motion_v1":
                    module = "highd_lane_motion_context"
                if digest(Path(__file__).with_name(module + ".py")) != contract[name + "_code_sha256"]:
                    raise ValueError("Frozen lateral process/geometry code changed")
        if self.manifest.get("geometry_onset_model"):
            from .highd_lane_geometry import GEOMETRY_FEATURES
            if (list(GEOMETRY_FEATURES) != self.manifest["geometry_feature_names"]
                    or self.interface.onset_model["mean"].shape != (19 + len(GEOMETRY_FEATURES),)):
                raise ValueError("Geometry onset feature encoding changed")

    def lane_pair(self, actor_ids, first_state, second_state, first_reference, second_reference, phase_code,
                  correlated=True):
        """One active focal / causally locked target-rear pair, at one time.

        Both feature rows use the causal phase of the focal maneuver. Reference
        PDFs must be queried from raw float64 observations before compression.
        """
        if phase_code not in (1, 2, 3):
            raise ValueError("Lane process requires an observed active phase")
        states = [np.asarray(value, dtype=np.float32)[None, :] for value in (first_state, second_state)]
        for state in states:
            if state.shape != (1, 28) or not np.array_equal(state[0, 11:14], np.eye(3)[phase_code - 1]):
                raise ValueError("State/phase mismatch")
        p = self.interface.lane_marginal(states[0], "focal", np.asarray(first_reference)[None, :])[0]
        q = self.interface.lane_marginal(states[1], "rear", np.asarray(second_reference)[None, :])[0]
        joint = (dual.rank_joint(p, q, self.weights[phase_code - 1], self.independent_mass)
                 if correlated else p[:, None] * q[None, :])
        mode = "lane_pair_correlated" if correlated else "lane_pair_independent"
        return dual.PairActionDistribution(tuple(actor_ids), ("focal", "locked_target_rear"), joint,
                                          self.model_id + ":" + mode)

    def following_pair(self, follower, leader):
        if len(follower) != 1 or len(leader) != 1:
            raise ValueError("Expected a single pair at one decision")
        return dual.PairActionDistribution((int(follower.id.iloc[0]), int(leader.id.iloc[0])),
                                          ("follower", "leader"), self.interface.following(follower, leader)[0],
                                          self.model_id + ":following_original_reference")

    def independent_pair(self, actor_ids, first, second):
        p, q = dual.probability(first), dual.probability(second)
        return dual.PairActionDistribution(tuple(actor_ids), ("first", "second"), p[:, None] * q[None, :],
                                          self.model_id + ":explicit_independent")

    def lateral_conditional(self, *, locked, quiet_state=None, permitted=None):
        if locked:
            return dual.locked_lateral()
        if quiet_state is None or permitted is None:
            raise ValueError("Unlocked initiation requires a causal eligible quiet state and legal-lane mask")
        return self.interface.quiet_lateral(np.asarray(quiet_state)[None, :], np.asarray(permitted)[None, :])[0]

    def with_lateral(self, pair, first_conditional, second_conditional):
        if pair.structured:
            raise ValueError("Pair already has lateral choices")
        joint = dual.compose_lateral(pair.natural, first_conditional, second_conditional)
        return dual.PairActionDistribution(pair.actor_ids, pair.roles, joint,
                                          pair.model_id + ":conditional_lateral_product", structured=True)
