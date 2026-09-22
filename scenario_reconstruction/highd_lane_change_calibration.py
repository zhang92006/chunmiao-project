"""Calibrate sparse lane-change tables; preserve the original runtime NDD.

Uses the existing proxy labels for a controlled smoothing ablation. These
labels are not observed driver intentions or an upstream calibration recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .highd_lane_change_baseline import _decision_rows, _state_action
from .highd_ndd_baseline import _load_split_manifest


def empirical_prior(counts):
    totals = counts.reshape(-1, 3).sum(axis=0, dtype=np.float64)
    if totals.sum() <= 0:
        raise ValueError("Training counts must be nonempty")
    return (totals + 0.5) / (totals.sum() + 1.5)


def shrunk_probability(counts, prior, concentration):
    if concentration <= 0 or not np.isfinite(concentration):
        raise ValueError("concentration must be finite and positive")
    return (counts + concentration * prior) / (
        counts.sum(axis=-1, keepdims=True) + concentration
    )


def score_counts(counts, probability):
    if not np.isfinite(probability).all() or (probability <= 0).any():
        raise ValueError("Probabilities must be finite and positive")
    if not np.allclose(probability.sum(axis=-1), 1):
        raise ValueError("Probability rows must sum to one")
    totals = counts.sum(axis=-1)
    n = float(totals.sum())
    if n == 0:
        raise ValueError("Evaluation counts must be nonempty")
    observed = float(counts[..., 0].sum() + counts[..., 2].sum())
    expected = float((totals * (probability[..., 0] + probability[..., 2])).sum())
    return {
        "decision_count": int(n),
        "observed_lane_changes": int(observed),
        "expected_lane_changes": expected,
        "expected_to_observed_ratio": expected / observed if observed else None,
        "mean_nll": float(-(counts * np.log(probability)).sum() / n),
        "mean_brier": float((
            (totals * np.square(probability).sum(axis=-1)).sum()
            - 2 * (counts * probability).sum() + n
        ) / n),
    }


def choose_candidate(scores):
    # Only calibration scores are accepted here, never validation/test data.
    return min(scores, key=lambda name: scores[name]["mean_nll"])


def calibrate(source_root, manifest, model, config, output):
    source_root, manifest, model, output = map(Path, (source_root, manifest, model, output))
    splits = _load_split_manifest(manifest)
    flat = [rid for ids in splits.values() for rid in ids]
    if len(flat) != len(set(flat)):
        raise ValueError("Recording IDs overlap between splits")
    with np.load(model, allow_pickle=False) as artifact:
        train_counts = artifact["counts"].astype(np.float64)
        axes = {name: artifact[name + "_axis"] for name in ("speed", "gap", "range_rate")}
    # Check provenance before using a previously fitted artifact.
    provenance = json.loads((model.parent / "lane_change_summary.json").read_text(encoding="utf-8"))
    if provenance["recordings_by_split"] != splits:
        raise ValueError("Model and evaluation split manifests disagree")
    for key in ("decision_lead_s", "source_frequency_hz", "target_frequency_hz"):
        if provenance[key] != config[key]:
            raise ValueError("Model label configuration disagrees: " + key)
    prior = empirical_prior(train_counts)
    candidates = {
        "legacy_uniform_alpha05": (train_counts + 0.5) / (train_counts.sum(axis=-1, keepdims=True) + 1.5),
        "constant_train_frequency": np.broadcast_to(prior, train_counts.shape),
    }
    for value in config["prior_concentrations"]:
        candidates["empirical_prior_" + str(value)] = shrunk_probability(train_counts, prior, value)
    evaluated = {}
    by_recording = {}
    selected = None
    # Test remains untouched; previous test exposure is documented separately.
    for split in ("calibration", "validation"):
        aggregate = np.zeros_like(train_counts)
        by_recording[split] = {}
        for rid in splits[split]:
            rows = _decision_rows(source_root, rid,
                source_hz=config["source_frequency_hz"], target_hz=config["target_frequency_hz"],
                decision_lead_s=config["decision_lead_s"])
            states, actions, _ = _state_action(rows, axes)
            counts = np.zeros_like(train_counts)
            np.add.at(counts, states + (actions,), 1)
            aggregate += counts
            if split == "validation":
                by_recording[split][rid] = {
                    name: score_counts(counts, candidates[name])
                    for name in (selected, "legacy_uniform_alpha05", "constant_train_frequency")
                }
        evaluated[split] = {name: score_counts(aggregate, prob) for name, prob in candidates.items()}
        if split == "calibration":
            selected = choose_candidate(evaluated[split])
    result = {
        "schema_version": 1, "status": "offline_smoothing_ablation",
        "upstream_exact_reproduction": False,
        "training_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "config": config, "recordings_by_split": splits,
        "train_action_prior": prior.tolist(), "selected_by_calibration_nll": selected,
        "evaluation": evaluated, "validation_per_recording": by_recording["validation"],
        "test_evaluated": False, "runtime_enabled": False,
        "limitations": [
            (
                f"Labels are placed {config['decision_lead_s']} seconds before the "
                "observed lane-boundary crossing and are not observed driver intentions."
            ),
            "Execution-state censoring remains outcome-dependent.",
            "Average 10 Hz source-frame selection is not exact 100 ms interpolation or an action-horizon alignment.",
            "No adjacent-lane state or lane feasibility mask is modeled yet.",
            "No runtime or closed-loop calibration claim is made.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "lane_change_candidate.npz", probability=candidates[selected],
        counts=train_counts, prior=prior, **{k + "_axis": v for k, v in axes.items()})
    (output / "calibration_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_root", "manifest", "model", "config", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = calibrate(args.source_root, args.manifest, args.model,
        json.loads(Path(args.config).read_text(encoding="utf-8")), args.output)
    print(json.dumps({"selected": result["selected_by_calibration_nll"],
        "validation": result["evaluation"]["validation"]}, indent=2))


if __name__ == "__main__":
    main()
