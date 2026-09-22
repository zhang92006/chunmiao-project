"""Fit an auditable highD longitudinal natural-driving baseline.

The output model is deliberately separate from the runtime NDD tables.  It is
used to measure domain coverage before any probability model is replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_GRID = {
    "speed": [20.0, 40.0, 1.0],
    "gap": [0.0, 115.0, 1.0],
    "range_rate": [-10.0, 8.0, 1.0],
    "acceleration": [-4.0, 2.0, 0.2],
}


def _axis(spec: list[float]) -> np.ndarray:
    low, high, step = map(float, spec)
    return np.linspace(low, high, round((high - low) / step) + 1)


def _nearest_index(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    step = float(axis[1] - axis[0])
    return np.rint((values - axis[0]) / step).astype(np.int64)


def _load_split_manifest(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for recording in payload["recordings"]:
        result.setdefault(str(recording["split"]), []).append(
            str(recording["recording_id"]).zfill(2)
        )
    return result


def _recording_rows(
    source_root: Path,
    recording_id: str,
    *,
    chunk_size: int,
    frame_stride: int | None,
    source_frequency_hz: int,
    target_frequency_hz: int | None,
) -> Iterable[pd.DataFrame]:
    path = source_root / "data" / f"{recording_id}_tracks.csv"
    columns = [
        "frame",
        "xVelocity",
        "xAcceleration",
        "dhw",
        "precedingXVelocity",
        "precedingId",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
        if target_frequency_hz is not None:
            frames = chunk["frame"].to_numpy(dtype=np.int64)
            # Select the last source frame before each target-rate boundary.
            # For 25 -> 10 Hz this gives alternating 2/3-frame intervals while
            # preserving a shared global clock for every vehicle.
            selected = (
                ((frames + 1) * target_frequency_hz) // source_frequency_hz
                > (frames * target_frequency_hz) // source_frequency_hz
            )
            chunk = chunk.loc[selected]
        elif frame_stride is not None and frame_stride > 1:
            chunk = chunk.loc[chunk["frame"] % frame_stride == 0]
        if not chunk.empty:
            yield chunk


def _observations(chunk: pd.DataFrame, axes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    direction_sign = np.where(chunk["xVelocity"].to_numpy() >= 0.0, 1.0, -1.0)
    speed = np.abs(chunk["xVelocity"].to_numpy(dtype=float))
    acceleration = chunk["xAcceleration"].to_numpy(dtype=float) * direction_sign
    has_leader = chunk["precedingId"].to_numpy(dtype=np.int64) > 0
    gap = chunk["dhw"].to_numpy(dtype=float)
    leader_speed = np.abs(chunk["precedingXVelocity"].to_numpy(dtype=float))
    range_rate = leader_speed - speed

    finite = np.isfinite(speed) & np.isfinite(acceleration)
    in_speed = (speed >= axes["speed"][0]) & (speed <= axes["speed"][-1])
    in_acceleration = (
        (acceleration >= axes["acceleration"][0])
        & (acceleration <= axes["acceleration"][-1])
    )
    car_following = (
        has_leader
        & (gap >= axes["gap"][0])
        & (gap <= axes["gap"][-1])
        & (range_rate >= axes["range_rate"][0])
        & (range_rate <= axes["range_rate"][-1])
    )
    free_flow = (~has_leader) | (gap > axes["gap"][-1])
    eligible = finite & in_speed & in_acceleration
    return {
        "speed": speed,
        "acceleration": acceleration,
        "gap": gap,
        "range_rate": range_rate,
        "eligible": eligible,
        "car_following": eligible & car_following,
        "free_flow": eligible & free_flow,
        "in_speed": finite & in_speed,
    }


def _empty_counts(axes: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    cf = np.zeros(tuple(len(axes[name]) for name in ("gap", "range_rate", "speed", "acceleration")), dtype=np.uint32)
    ff = np.zeros((len(axes["speed"]), len(axes["acceleration"])), dtype=np.uint32)
    return cf, ff


def _add_counts(
    cf_counts: np.ndarray,
    ff_counts: np.ndarray,
    obs: dict[str, np.ndarray],
    axes: dict[str, np.ndarray],
) -> None:
    cf = obs["car_following"]
    if np.any(cf):
        indices = tuple(
            _nearest_index(obs[name][cf], axes[name])
            for name in ("gap", "range_rate", "speed", "acceleration")
        )
        np.add.at(cf_counts, indices, 1)
    ff = obs["free_flow"]
    if np.any(ff):
        indices = (
            _nearest_index(obs["speed"][ff], axes["speed"]),
            _nearest_index(obs["acceleration"][ff], axes["acceleration"]),
        )
        np.add.at(ff_counts, indices, 1)


def _probabilities(counts: np.ndarray, alpha: float) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    action_count = counts.shape[-1]
    return ((counts + alpha) / (totals + alpha * action_count)).astype(np.float32)


def _evaluate(
    source_root: Path,
    recording_ids: list[str],
    axes: dict[str, np.ndarray],
    cf_counts: np.ndarray,
    ff_counts: np.ndarray,
    *,
    alpha: float,
    chunk_size: int,
    frame_stride: int | None,
    source_frequency_hz: int,
    target_frequency_hz: int | None,
    reference_probabilities: dict[str, np.ndarray] | None = None,
    reference_probability_floor: float = 1e-12,
) -> dict[str, Any]:
    mode_totals = {
        "car_following": cf_counts.sum(axis=-1),
        "free_flow": ff_counts.sum(axis=-1),
    }
    probabilities = {
        "car_following": _probabilities(cf_counts, alpha),
        "free_flow": _probabilities(ff_counts, alpha),
    }
    metrics = Counter()
    log_likelihood = 0.0
    seen_log_likelihood = 0.0
    reference_log_likelihood = 0.0
    brier_sum = 0.0
    reference_brier_sum = 0.0
    for recording_id in recording_ids:
        for chunk in _recording_rows(
            source_root,
            recording_id,
            chunk_size=chunk_size,
            frame_stride=frame_stride,
            source_frequency_hz=source_frequency_hz,
            target_frequency_hz=target_frequency_hz,
        ):
            obs = _observations(chunk, axes)
            metrics["sampled_rows"] += len(chunk)
            metrics["speed_domain_rows"] += int(obs["in_speed"].sum())
            metrics["eligible_rows"] += int(obs["eligible"].sum())
            for mode, names in {
                "car_following": ("gap", "range_rate", "speed", "acceleration"),
                "free_flow": ("speed", "acceleration"),
            }.items():
                mask = obs[mode]
                if not np.any(mask):
                    continue
                indices = tuple(_nearest_index(obs[name][mask], axes[name]) for name in names)
                state_indices = indices[:-1]
                action_indices = indices[-1]
                state_counts = mode_totals[mode][state_indices]
                seen = state_counts > 0
                distribution = probabilities[mode][state_indices]
                selected_probabilities = distribution[
                    np.arange(len(action_indices)), action_indices
                ]
                n = int(mask.sum())
                metrics[f"{mode}_rows"] += n
                metrics[f"{mode}_seen_state_rows"] += int(seen.sum())
                log_likelihood += float(np.log(selected_probabilities).sum())
                seen_log_likelihood += float(np.log(selected_probabilities[seen]).sum())
                metrics["evaluated_rows"] += n
                metrics["seen_state_rows"] += int(seen.sum())
                metrics["top1_correct_rows"] += int(
                    (np.argmax(distribution, axis=-1) == action_indices).sum()
                )
                brier_sum += float(
                    (
                        np.square(distribution).sum(axis=-1)
                        - 2.0 * selected_probabilities
                        + 1.0
                    ).sum()
                )
                if reference_probabilities is not None:
                    reference_distribution = reference_probabilities[mode][state_indices]
                    reference = reference_distribution[
                        np.arange(len(action_indices)), action_indices
                    ]
                    metrics["reference_zero_probability_rows"] += int(
                        (reference <= 0.0).sum()
                    )
                    reference_log_likelihood += float(
                        np.log(np.maximum(reference, reference_probability_floor)).sum()
                    )
                    metrics["reference_top1_correct_rows"] += int(
                        (np.argmax(reference_distribution, axis=-1) == action_indices).sum()
                    )
                    reference_brier_sum += float(
                        (
                            np.square(reference_distribution).sum(axis=-1)
                            - 2.0 * reference
                            + 1.0
                        ).sum()
                    )
    evaluated = metrics["evaluated_rows"]
    seen = metrics["seen_state_rows"]
    result = {
        **dict(metrics),
        "state_coverage": seen / evaluated if evaluated else None,
        "mean_negative_log_likelihood": -log_likelihood / evaluated if evaluated else None,
        "seen_state_mean_negative_log_likelihood": -seen_log_likelihood / seen if seen else None,
        "mean_brier_score": brier_sum / evaluated if evaluated else None,
        "top1_action_accuracy": metrics["top1_correct_rows"] / evaluated if evaluated else None,
    }
    if reference_probabilities is not None:
        zero = metrics["reference_zero_probability_rows"]
        result.update({
            "reference_ndd_mean_negative_log_likelihood": (
                -reference_log_likelihood / evaluated if evaluated else None
            ),
            "reference_ndd_zero_probability_rate": zero / evaluated if evaluated else None,
            "reference_probability_floor": reference_probability_floor,
            "reference_ndd_mean_brier_score": (
                reference_brier_sum / evaluated if evaluated else None
            ),
            "reference_ndd_top1_action_accuracy": (
                metrics["reference_top1_correct_rows"] / evaluated
                if evaluated else None
            ),
        })
    return result


def fit_highd_baseline(
    source_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    reference_ndd_root: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_recordings = _load_split_manifest(manifest_path)
    grid = config.get("grid", DEFAULT_GRID)
    axes = {name: _axis(grid[name]) for name in DEFAULT_GRID}
    alpha = float(config.get("laplace_alpha", 0.5))
    chunk_size = int(config.get("chunk_size", 250_000))
    frame_stride = (
        int(config["frame_stride"]) if config.get("frame_stride") is not None else None
    )
    source_frequency_hz = int(config.get("source_frequency_hz", 25))
    target_frequency_hz = (
        int(config["target_frequency_hz"])
        if config.get("target_frequency_hz") is not None else None
    )
    train_split = str(config.get("train_split", "train"))
    train_ids = split_recordings[train_split]

    reference_probabilities = None
    if reference_ndd_root is not None:
        reference_ndd_root = Path(reference_ndd_root)
        reference_probabilities = {
            "car_following": np.load(
                reference_ndd_root / "CF" / "Optimized_CF_pdf_array.npy"
            ),
            "free_flow": np.load(
                reference_ndd_root / "FF" / "Optimized_FF_pdf_array.npy"
            ),
        }

    cf_counts, ff_counts = _empty_counts(axes)
    train_rows = 0
    for recording_id in train_ids:
        for chunk in _recording_rows(
            source_root,
            recording_id,
            chunk_size=chunk_size,
            frame_stride=frame_stride,
            source_frequency_hz=source_frequency_hz,
            target_frequency_hz=target_frequency_hz,
        ):
            train_rows += len(chunk)
            _add_counts(cf_counts, ff_counts, _observations(chunk, axes), axes)

    model_path = output_dir / "highd_longitudinal_ndd_v1.npz"
    np.savez_compressed(
        model_path,
        **{f"{name}_axis": value for name, value in axes.items()},
        car_following_counts=cf_counts,
        free_flow_counts=ff_counts,
        car_following_probability=_probabilities(cf_counts, alpha),
        free_flow_probability=_probabilities(ff_counts, alpha),
    )
    evaluation = {
        split: _evaluate(
            source_root,
            ids,
            axes,
            cf_counts,
            ff_counts,
            alpha=alpha,
            chunk_size=chunk_size,
            frame_stride=frame_stride,
            source_frequency_hz=source_frequency_hz,
            target_frequency_hz=target_frequency_hz,
            reference_probabilities=reference_probabilities,
            reference_probability_floor=float(
                config.get("reference_probability_floor", 1e-12)
            ),
        )
        for split, ids in split_recordings.items()
    }
    summary = {
        "schema_version": 1,
        "status": "pilot_longitudinal_baseline_not_runtime_enabled",
        "dataset": "highD-v1.0",
        "target_domain": "highway_20_to_40_mps",
        "probability_semantics": "p_highD(longitudinal_acceleration | discretized state)",
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "train_split": train_split,
        "recordings_by_split": split_recordings,
        "training_sampled_rows": train_rows,
        "frame_stride": frame_stride,
        "source_frequency_hz": source_frequency_hz,
        "target_frequency_hz": target_frequency_hz,
        "laplace_alpha": alpha,
        "reference_ndd_compared": reference_probabilities is not None,
        "grid": grid,
        "car_following_observations": int(cf_counts.sum()),
        "free_flow_observations": int(ff_counts.sum()),
        "occupied_car_following_states": int(np.count_nonzero(cf_counts.sum(axis=-1))),
        "possible_car_following_states": int(np.prod(cf_counts.shape[:-1])),
        "occupied_free_flow_states": int(np.count_nonzero(ff_counts.sum(axis=-1))),
        "possible_free_flow_states": int(ff_counts.shape[0]),
        "evaluation": evaluation,
        "limitations": [
            "Longitudinal acceleration only; lane-change probabilities are not fitted.",
            "The model represents the highD highway domain, not SHRP2 event probability.",
            "The two-BV joint baseline is not yet fitted; runtime NDD tables are unchanged.",
        ],
    }
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference_ndd_root")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = fit_highd_baseline(
        args.source_root,
        args.manifest,
        args.output,
        config,
        reference_ndd_root=args.reference_ndd_root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
