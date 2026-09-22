"""Read-only runtime adapter for comparing highD and the original NDD PDFs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .highd_lane_change_context import _bin, hierarchical_probability
from .highd_ndd_baseline import _nearest_index


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HighDShadowNDD:
    """Compute candidate PDFs without sampling actions or changing weights."""

    def __init__(
        self,
        longitudinal_model: str | Path,
        context_model: str | Path,
        context_config: str | Path,
    ) -> None:
        longitudinal_model = Path(longitudinal_model)
        context_model = Path(context_model)
        context_config = Path(context_config)
        with np.load(longitudinal_model, allow_pickle=False) as artifact:
            self.speed_axis = artifact["speed_axis"].astype(float)
            self.gap_axis = artifact["gap_axis"].astype(float)
            self.range_rate_axis = artifact["range_rate_axis"].astype(float)
            self.acceleration_axis = artifact["acceleration_axis"].astype(float)
            self.cf_counts = artifact["car_following_counts"].astype(np.uint64)
            self.ff_counts = artifact["free_flow_counts"].astype(np.uint64)
            self.cf_probability = artifact["car_following_probability"].astype(float)
            self.ff_probability = artifact["free_flow_probability"].astype(float)
        with np.load(context_model, allow_pickle=False) as artifact:
            self.base_total = artifact["base_total"].astype(float)
            self.base_positive = artifact["base_positive"].astype(float)
            self.context_total = artifact["context_total"].astype(float)
            self.context_positive = artifact["context_positive"].astype(float)
            self.base_concentration = float(artifact["selected_base_concentration"])
            self.context_concentration = float(artifact["selected_context_concentration"])
            self.probability_scale = float(artifact["probability_scale"])
        self.config = json.loads(context_config.read_text(encoding="utf-8"))
        self.metadata = {
            "mode": "read_only_shadow",
            "runtime_actions_changed": False,
            "importance_weights_changed": False,
            "longitudinal_model": longitudinal_model.name,
            "longitudinal_model_sha256": _sha256(longitudinal_model),
            "context_model": context_model.name,
            "context_model_sha256": _sha256(context_model),
            "context_config": context_config.name,
            "context_config_sha256": _sha256(context_config),
            "safety_filter_applied_to_highd": False,
        }

    @staticmethod
    def _normalise(pdf: np.ndarray) -> np.ndarray:
        pdf = np.asarray(pdf, dtype=float)
        if pdf.shape != (33,) or not np.isfinite(pdf).all() or np.any(pdf < 0):
            raise ValueError("Original NDD PDF must be a finite nonnegative 33-vector")
        total = float(pdf.sum())
        if total <= 0:
            raise ValueError("Original NDD PDF must have positive mass")
        return pdf / total

    def _longitudinal(self, obs: dict[str, Any]) -> tuple[np.ndarray | None, str]:
        speed = float(obs["Ego"]["velocity"])
        if speed < self.speed_axis[0] or speed > self.speed_axis[-1]:
            return None, "original_fallback_speed_out_of_domain"
        speed_index = int(_nearest_index(np.array([speed]), self.speed_axis)[0])
        leader = obs.get("Lead")
        if leader is None:
            if self.ff_counts[speed_index].sum() == 0:
                return None, "original_fallback_unseen_free_flow_state"
            return self.ff_probability[speed_index], "highd_free_flow"
        gap = float(leader["distance"])
        range_rate = float(leader["velocity"]) - speed
        if (
            gap < self.gap_axis[0] or gap > self.gap_axis[-1]
            or range_rate < self.range_rate_axis[0]
            or range_rate > self.range_rate_axis[-1]
        ):
            return None, "original_fallback_car_following_out_of_domain"
        gap_index = int(_nearest_index(np.array([gap]), self.gap_axis)[0])
        rr_index = int(_nearest_index(np.array([range_rate]), self.range_rate_axis)[0])
        if self.cf_counts[gap_index, rr_index, speed_index].sum() == 0:
            return None, "original_fallback_unseen_car_following_state"
        return (
            self.cf_probability[gap_index, rr_index, speed_index],
            "highd_car_following",
        )

    def _context_indices(self, obs: dict[str, Any], side: str) -> tuple[int, int]:
        config = self.config
        speed = float(obs["Ego"]["velocity"])
        leader = obs.get("Lead")
        gap = float(leader["distance"]) if leader else float(config["maximum_gap_m"])
        rr = float(leader["velocity"]) - speed if leader else 0.0
        base_tuple = (
            int(_bin(np.array([speed]), config["speed_boundaries_mps"])[0]),
            int(_bin(np.array([gap]), config["current_gap_boundaries_m"])[0]),
            int(_bin(np.array([rr]), config["relative_speed_boundaries_mps"])[0]),
        )
        base_shape = (
            len(config["speed_boundaries_mps"]) + 1,
            len(config["current_gap_boundaries_m"]) + 1,
            len(config["relative_speed_boundaries_mps"]) + 1,
        )
        if config.get("include_current_leader_presence", False):
            base_tuple += (int(leader is not None),)
            base_shape += (2,)
        context_values = []
        neighbors = []
        for relation in ("Lead", "Foll"):
            neighbor = obs.get(side.capitalize() + relation)
            neighbors.append(neighbor)
            if neighbor is None:
                context_values.extend((0, 0))
                continue
            distance = max(float(neighbor["distance"]), 0.0)
            relative_speed = (
                float(neighbor["velocity"]) - speed
                if relation == "Lead"
                else speed - float(neighbor["velocity"])
            )
            context_values.extend((
                1 + int(_bin(np.array([distance]), config["target_gap_boundaries_m"])[0]),
                1 + int(_bin(
                    np.array([relative_speed]),
                    config["relative_speed_boundaries_mps"],
                )[0]),
            ))
        alongside = int(any(
            neighbor is not None and float(neighbor["distance"]) < 0
            for neighbor in neighbors
        ))
        context_shape = base_shape + (
            len(config["target_gap_boundaries_m"]) + 2,
            len(config["relative_speed_boundaries_mps"]) + 2,
            len(config["target_gap_boundaries_m"]) + 2,
            len(config["relative_speed_boundaries_mps"]) + 2,
            2,
        )
        return (
            int(np.ravel_multi_index(base_tuple, base_shape)),
            int(np.ravel_multi_index(base_tuple + tuple(context_values) + (alongside,), context_shape)),
        )

    def _lateral(self, obs: dict[str, Any]) -> tuple[float, float, str]:
        config = self.config
        speed = float(obs["Ego"]["velocity"])
        leader = obs.get("Lead")
        if config.get("require_current_leader_for_lateral", False) and leader is None:
            return 0.0, 0.0, "original_structure_no_current_leader"
        gap = float(leader["distance"]) if leader else float(config["maximum_gap_m"])
        rr = float(leader["velocity"]) - speed if leader else 0.0
        if (
            speed < config["speed_range_mps"][0]
            or speed > config["speed_range_mps"][1]
            or gap < 0 or gap > config["maximum_gap_m"]
            or rr < config["relative_speed_range_mps"][0]
            or rr > config["relative_speed_range_mps"][1]
        ):
            return 0.0, 0.0, "original_fallback_lateral_out_of_domain"
        global_probability = (
            self.base_positive.sum() + 0.5
        ) / (self.base_total.sum() + 1.0)
        probabilities = []
        for side in ("left", "right"):
            available = bool(obs["Ego"][f"could_drive_adjacent_lane_{side}"])
            if not available:
                probabilities.append(0.0)
                continue
            base_index, context_index = self._context_indices(obs, side)
            base_probability = (
                self.base_positive[base_index]
                + self.base_concentration * global_probability
            ) / (self.base_total[base_index] + self.base_concentration)
            probability = hierarchical_probability(
                self.context_positive,
                self.context_total,
                np.array([context_index]),
                np.array([base_probability]),
                self.context_concentration,
            )[0]
            probabilities.append(float(probability * self.probability_scale))
        left, right = probabilities
        maximum = float(config["maximum_total_lane_change_probability"])
        if left + right > maximum:
            scale = maximum / (left + right)
            left, right = left * scale, right * scale
        return left, right, "highd_adjacent_context"

    def compare(self, obs: dict[str, Any], original_pdf: np.ndarray) -> dict[str, Any]:
        original = self._normalise(original_pdf)
        longitudinal, longitudinal_source = self._longitudinal(obs)
        left, right, lateral_source = self._lateral(obs)
        if longitudinal is None or lateral_source.startswith("original_fallback"):
            candidate = original.copy()
            fallback = True
        else:
            longitudinal = np.asarray(longitudinal, dtype=float)
            longitudinal = longitudinal / longitudinal.sum()
            stay = 1.0 - left - right
            candidate = np.concatenate(([left, right], stay * longitudinal))
            candidate = candidate / candidate.sum()
            fallback = False
        floor = np.finfo(float).tiny
        kl_original_to_highd = float(np.sum(
            original * (np.log(np.maximum(original, floor))
                        - np.log(np.maximum(candidate, floor)))
        ))
        return {
            "vehicle_id": obs["Ego"].get("veh_id"),
            "fallback": fallback,
            "longitudinal_source": longitudinal_source,
            "lateral_source": lateral_source,
            "original_pdf": original.tolist(),
            "highd_shadow_pdf": candidate.tolist(),
            "l1_distance": float(np.abs(original - candidate).sum()),
            "kl_original_to_highd": kl_original_to_highd,
            "original_lane_change_probability": float(original[0] + original[1]),
            "highd_lane_change_probability": float(candidate[0] + candidate[1]),
        }
