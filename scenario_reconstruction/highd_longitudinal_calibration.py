"""Calibrate highD longitudinal smoothing without changing the action interface.

Train counts define every prior. Calibration NLL on runtime-covered states selects
the concentration (or retains the original model). Validation is evaluated only
after selection. Empty training states still fall back in the runtime adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .highd_ndd_baseline import (
    _add_counts, _empty_counts, _load_split_manifest, _observations, _recording_rows,
)


def empirical_probabilities(cf, ff, concentration, parent_concentration=100.0):
    """Hierarchical train-only priors: global -> speed -> (relative speed, speed)."""
    if concentration <= 0 or parent_concentration <= 0:
        raise ValueError("Prior concentrations must be positive")
    cf, ff = np.asarray(cf, dtype=float), np.asarray(ff, dtype=float)
    if cf.ndim != 4 or ff.shape != cf.shape[2:]:
        raise ValueError("Expected CF[gap,relative_speed,speed,action] and FF[speed,action]")
    if any(not np.isfinite(c).all() or np.any(c < 0) or c.sum() <= 0 for c in (cf, ff)):
        raise ValueError("Expected nonempty finite nonnegative training counts")

    def posterior(counts, prior, strength):
        return (counts + strength * prior) / (counts.sum(axis=-1, keepdims=True) + strength)

    # Tiny global pseudocount gives positive support without a uniform prior per fine state.
    global_cf = cf.sum(axis=(0, 1, 2)) + .5
    global_cf /= global_cf.sum()
    speed_prior = posterior(cf.sum(axis=(0, 1)), global_cf, parent_concentration)
    parent_prior = posterior(cf.sum(axis=0), speed_prior, parent_concentration)
    global_ff = ff.sum(axis=0) + .5
    global_ff /= global_ff.sum()
    return posterior(cf, parent_prior, concentration), posterior(ff, global_ff, concentration)


def score_counts(cf, ff, train_cf, train_ff, probabilities):
    count = covered = 0
    loss = brier = 0.0
    for observed, train, pdf in zip((cf, ff), (train_cf, train_ff), probabilities):
        count += int(observed.sum())
        seen = train.sum(axis=-1) > 0
        values = observed[seen].astype(float)
        p = np.asarray(pdf[seen], dtype=float)
        p = p / p.sum(axis=-1, keepdims=True)
        covered += int(values.sum())
        loss -= float(np.sum(values * np.log(np.maximum(p, 1e-300))))
        brier += float(np.sum(values.sum(axis=-1) * np.sum(p*p, axis=-1)
                              - 2*np.sum(values*p, axis=-1) + values.sum(axis=-1)))
    return {"eligible_observations": count, "runtime_covered_observations": covered,
            "runtime_coverage": covered/count if count else None,
            "covered_nll": loss/covered if covered else None,
            "covered_brier": brier/covered if covered else None}


def calibrate(source_root, manifest_path, model_path, config, output):
    splits = _load_split_manifest(Path(manifest_path))
    if sorted(splits.get("train", [])) != sorted(config["train_recordings"]):
        raise ValueError("Train recordings do not match the pinned model")
    flattened = [r for ids in splits.values() for r in ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Recording splits overlap")
    source_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if source_hash != config["source_model_sha256"]:
        raise ValueError("Source training artifact differs from the pinned model")
    with np.load(model_path, allow_pickle=False) as model:
        axes = {k: model[k + "_axis"] for k in ("speed", "gap", "range_rate", "acceleration")}
        train_cf, train_ff = model["car_following_counts"], model["free_flow_counts"]
        original = (model["car_following_probability"], model["free_flow_probability"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calibration_cf, calibration_ff = _empty_counts(axes)

    def read_recording(recording):
        cf, ff = _empty_counts(axes)
        for chunk in _recording_rows(Path(source_root), recording,
                chunk_size=config["chunk_size"], frame_stride=None,
                source_frequency_hz=config["source_frequency_hz"],
                target_frequency_hz=config["target_frequency_hz"]):
            _add_counts(cf, ff, _observations(chunk, axes), axes)
        print(json.dumps({"recording_completed": recording, "eligible": int(cf.sum()+ff.sum())}), flush=True)
        return cf, ff

    for recording in splits["calibration"]:
        cf, ff = read_recording(recording)
        calibration_cf += cf
        calibration_ff += ff
    candidates = [{"name": "original_uniform_alpha_0.5", "concentration": None,
        **score_counts(calibration_cf, calibration_ff, train_cf, train_ff, original)}]
    best_pdf, best = original, candidates[0]
    if best["covered_nll"] is None:
        raise ValueError("No calibration observations in runtime support")
    for strength in config["prior_concentrations"]:
        pdf = empirical_probabilities(train_cf, train_ff, strength, config["parent_concentration"])
        candidate = {"name": "empirical_hierarchical", "concentration": strength,
            **score_counts(calibration_cf, calibration_ff, train_cf, train_ff, pdf)}
        candidates.append(candidate)
        if candidate["covered_nll"] < best["covered_nll"]:
            best_pdf, best = pdf, candidate
    # Freeze selection BEFORE reading validation.
    (output / "selection.json").write_text(json.dumps({"selected": best, "candidates": candidates}, indent=2))
    best_pdf = tuple(p.astype(np.float32) for p in best_pdf)
    fitted_path = output / "highd_longitudinal_ndd_calibrated_v2.npz"
    np.savez_compressed(fitted_path, **{k + "_axis": v for k, v in axes.items()},
        car_following_counts=train_cf, free_flow_counts=train_ff,
        car_following_probability=best_pdf[0], free_flow_probability=best_pdf[1])
    validation_cf, validation_ff = _empty_counts(axes)
    by_recording = {}
    for recording in splits["validation"]:
        cf, ff = read_recording(recording)
        validation_cf += cf
        validation_ff += ff
        by_recording[recording] = {
            "original": score_counts(cf, ff, train_cf, train_ff, original),
            "selected": score_counts(cf, ff, train_cf, train_ff, best_pdf)}
    result = {"schema_version": 1, "config": config, "source_model_sha256": source_hash,
        "selected_model_sha256": hashlib.sha256(fitted_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "recordings_read": {s: splits[s] for s in ("calibration", "validation")},
        "selection_split": "calibration", "selected": best, "candidates": candidates,
        "validation": {
            "original": score_counts(validation_cf, validation_ff, train_cf, train_ff, original),
            "selected": score_counts(validation_cf, validation_ff, train_cf, train_ff, best_pdf)},
        "validation_by_recording": by_recording,
        "scope": "Conditional instantaneous acceleration NLL on covered states only. Same 10Hz-average frame selection, support, action grid and runtime fallback. Not a temporal-dynamics or safety validation."}
    (output / "longitudinal_calibration_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_root", "manifest", "model", "config", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = calibrate(args.source_root, args.manifest, args.model,
                       json.loads(Path(args.config).read_text(encoding="utf-8")), args.output)
    print(json.dumps({"selected": result["selected"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
