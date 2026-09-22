"""Fit a calibration-selected highD dependence correction for two BV actions.

The fitted table describes synchronized longitudinal actions of a directed
same-lane follower/leader pair.  It is not a universal two-vehicle policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .highd_joint_action_audit import _pair_indices, _pair_rows
from .highd_ndd_baseline import _axis, _load_split_manifest


def factorized_probability(counts: np.ndarray, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    first = counts.sum(axis=-1).astype(float) + alpha
    second = counts.sum(axis=-2).astype(float) + alpha
    first /= first.sum(axis=-1, keepdims=True)
    second /= second.sum(axis=-1, keepdims=True)
    return first[..., :, None] * second[..., None, :]


def shrink_joint_probability(counts: np.ndarray, concentration: float,
                             alpha: float) -> np.ndarray:
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    prior = factorized_probability(counts, alpha)
    total = counts.sum(axis=(-2, -1), keepdims=True).astype(float)
    return (counts + concentration * prior) / (total + concentration)


def score_counts(observed: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    if observed.shape != probability.shape or observed.shape[-2:] != (3, 3):
        raise ValueError("Observed counts and probability shapes disagree")
    selected = observed > 0
    if not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ValueError("Joint probabilities must have full positive support")
    total = int(observed.sum())
    nll = -float(np.sum(observed[selected] * np.log(probability[selected]))) / total if total else None
    return {"pair_actions": total, "mean_negative_log_likelihood": nll}


def _counts_for_recording(source_root: Path, recording: str,
                          axes: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    shape = tuple(len(axes[name]) for name in ("speed", "gap", "range_rate")) + (3, 3)
    result = np.zeros(shape, dtype=np.uint32)
    pairs = _pair_rows(source_root, recording,
                       source_hz=int(config["source_frequency_hz"]),
                       target_hz=int(config["target_frequency_hz"]))
    states, first, second, _ = _pair_indices(
        pairs, axes, config["acceleration_range"], config["action_thresholds"])
    np.add.at(result, states + (first, second), 1)
    return result


def _aggregate_recording_scores(per_recording: dict[str, np.ndarray],
                                probability: np.ndarray) -> dict[str, Any]:
    records = []
    total_count = 0
    total_log_loss = 0.0
    for recording, counts in per_recording.items():
        score = score_counts(counts, probability)
        records.append({"recording_id": recording, **score})
        if score["pair_actions"]:
            total_count += score["pair_actions"]
            total_log_loss += score["pair_actions"] * score["mean_negative_log_likelihood"]
    return {
        "recording_count": len(records),
        "pair_actions": total_count,
        "mean_negative_log_likelihood": total_log_loss / total_count if total_count else None,
        "per_recording": records,
    }


def coarse_groups(acceleration_axis: np.ndarray, thresholds: list[float]) -> np.ndarray:
    lower, upper = map(float, thresholds)
    axis = np.asarray(acceleration_axis, dtype=float)
    return np.where(axis < lower, 0, np.where(axis > upper, 2, 1)).astype(int)


def lift_dependence_to_33(first_pdf: np.ndarray, second_pdf: np.ndarray,
                          coarse_joint: np.ndarray, acceleration_axis: np.ndarray,
                          thresholds: list[float], iterations: int = 200) -> np.ndarray:
    """Lift a 3x3 dependence table while preserving both 33-action marginals."""
    first = np.asarray(first_pdf, dtype=float)
    second = np.asarray(second_pdf, dtype=float)
    coarse = np.asarray(coarse_joint, dtype=float)
    if first.shape != (33,) or second.shape != (33,) or coarse.shape != (3, 3):
        raise ValueError("Expected two 33-vectors and one 3x3 table")
    if (not np.isfinite(first).all() or not np.isfinite(second).all()
            or np.any(first < 0) or np.any(second < 0)
            or not np.isclose(first.sum(), 1) or not np.isclose(second.sum(), 1)
            or not np.isfinite(coarse).all() or np.any(coarse <= 0)
            or not np.isclose(coarse.sum(), 1)):
        raise ValueError("Input marginals must be normalized and nonnegative; the coarse table must have positive support")
    groups = coarse_groups(acceleration_axis, thresholds)
    if groups.shape != (31,):
        raise ValueError("Acceleration axis must match the 31 longitudinal actions")
    coarse_first = coarse.sum(axis=1)
    coarse_second = coarse.sum(axis=0)
    ratio = coarse / np.outer(coarse_first, coarse_second)
    modifier = np.ones((33, 33), dtype=float)
    modifier[2:, 2:] = ratio[groups[:, None], groups[None, :]]
    joint = np.outer(first, second) * modifier
    for _ in range(iterations):
        row_sum = joint.sum(axis=1)
        row_scale = np.divide(first, row_sum, out=np.zeros_like(first), where=row_sum > 0)
        joint *= row_scale[:, None]
        column_sum = joint.sum(axis=0)
        column_scale = np.divide(second, column_sum, out=np.zeros_like(second), where=column_sum > 0)
        joint *= column_scale[None, :]
    if (np.max(np.abs(joint.sum(axis=1) - first)) > 1e-10
            or np.max(np.abs(joint.sum(axis=0) - second)) > 1e-10):
        raise RuntimeError("Marginal-preserving projection did not converge")
    return joint / joint.sum()


def fit(source_root: str | Path, manifest_path: str | Path, config: dict[str, Any],
        output: str | Path) -> dict[str, Any]:
    source_root, manifest_path, output = Path(source_root), Path(manifest_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = _load_split_manifest(manifest_path)
    if set(splits["train"]) & set(splits["calibration"]):
        raise ValueError("Train and calibration recordings overlap")
    axes = {name: _axis(config["state_grid"][name])
            for name in ("speed", "gap", "range_rate")}
    train_by_recording = {r: _counts_for_recording(source_root, r, axes, config)
                          for r in splits["train"]}
    calibration_by_recording = {r: _counts_for_recording(source_root, r, axes, config)
                                for r in splits["calibration"]}
    train = sum(train_by_recording.values())
    alpha = float(config["marginal_alpha"])
    baseline_probability = factorized_probability(train, alpha)
    baseline = _aggregate_recording_scores(calibration_by_recording, baseline_probability)
    candidates = []
    for concentration in config["concentration_candidates"]:
        probability = shrink_joint_probability(train, float(concentration), alpha)
        score = _aggregate_recording_scores(calibration_by_recording, probability)
        baseline_by_recording = {r["recording_id"]: r for r in baseline["per_recording"]}
        improved = sum(
            record["mean_negative_log_likelihood"]
            < baseline_by_recording[record["recording_id"]]["mean_negative_log_likelihood"]
            for record in score["per_recording"] if record["pair_actions"]
        )
        candidates.append({
            "concentration": float(concentration), **score,
            "factorization_nll_penalty": baseline["mean_negative_log_likelihood"]
                                         - score["mean_negative_log_likelihood"],
            "recordings_improved_over_factorized": improved,
        })
    required = int(config["minimum_calibration_recordings_improved"])
    eligible = [item for item in candidates
                if item["factorization_nll_penalty"] > 0
                and item["recordings_improved_over_factorized"] >= required]
    selected = min(eligible, key=lambda item: item["mean_negative_log_likelihood"]) if eligible else None
    model_path = None
    model_hash = None
    if selected is not None:
        selected_probability = shrink_joint_probability(train, selected["concentration"], alpha)
        model_path = output / "highd_joint_longitudinal_v2.npz"
        np.savez_compressed(
            model_path,
            **{f"{name}_axis": axis for name, axis in axes.items()},
            joint_counts=train,
            joint_probability=selected_probability.astype(np.float32),
            factorized_probability=baseline_probability.astype(np.float32),
            action_thresholds=np.asarray(config["action_thresholds"], dtype=float),
            selected_concentration=np.asarray(selected["concentration"]),
        )
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    result = {
        "schema_version": 2,
        "status": "calibration_selected_pending_validation" if selected else "factorized_retained",
        "dataset": "highD-v1.0",
        "pair_definition": "directed same-lane follower and its recorded immediate leader",
        "runtime_applicability": "Use dependence correction only when the controlled BV pair has this relation; otherwise retain factorized natural probabilities.",
        "train_recordings": splits["train"],
        "calibration_recordings": splits["calibration"],
        "splits_not_read": [name for name in ("validation", "test") if name in splits],
        "train_pair_actions": int(train.sum()),
        "factorized_calibration": baseline,
        "candidates": candidates,
        "selected": selected,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": model_hash,
        "scope": "Offline longitudinal dependence calibration only. No validation/test read, no lane-change dependence, no runtime enablement, and no D2RL efficiency claim.",
    }
    (output / "joint_longitudinal_v2_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fit(args.source_root, args.manifest,
                 json.loads(Path(args.config).read_text(encoding="utf-8")), args.output)
    print(json.dumps({"status": result["status"], "train_pair_actions": result["train_pair_actions"],
                      "selected": result["selected"], "model_sha256": result["model_sha256"]},
                     indent=2))


if __name__ == "__main__":
    main()
