"""Fit per-decision lane-motion onset probabilities conditional on acceleration.

0/1/2 mean keep/start-left/start-right. Labels use later motion solely as
supervision; state features use present and previous observations. The current
acceleration is the candidate sampled longitudinal action, never a state input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .highd_pair_data_inventory import digest, in_range, lane_sets, longitudinal_domain, read_json, write_new
from .highd_pair_sample_export import SOURCE_COLUMNS


STATE_FEATURES = (
    "speed", "own_front_gap", "own_front_relative_speed", "previous_acceleration",
    "lateral_velocity", "lane_center_offset", "left_available", "right_available",
    "left_rear_present", "left_rear_gap", "left_rear_relative_speed", "left_rear_acceleration",
    "right_rear_present", "right_rear_gap", "right_rear_relative_speed", "right_rear_acceleration",
    "outside_lane_available", "observation_age_s", "previous_acceleration_missing",
)
ACTION_FEATURES = ("sampled_acceleration", "sampled_acceleration_squared",
                   "acceleration_x_left_rear_gap", "acceleration_x_right_rear_gap")


def decision_clock(frames, fps, hz):
    """Exact k/hz clock using only the latest source frame at or before it."""
    frames = np.asarray(frames, dtype=np.int64)
    if not 0 < hz <= fps or not len(frames):
        raise ValueError("Expected nonempty source and 0 < decision_hz <= fps")
    ticks = np.arange((frames[0] * hz + fps - 1) // fps, frames[-1] * hz // fps + 1)
    anchors = ticks * fps // hz
    ends = (ticks + 1) * fps // hz
    valid = np.isin(anchors, frames)
    # Integer arithmetic avoids boundary rounding errors at large frame IDs.
    age = (ticks * fps - anchors * hz) / (hz * fps)
    return anchors[valid], ends[valid], age[valid]


def onset_labels(frames, velocity, lane, anchors, ends, *, threshold, quiet_frames, sustain_frames,
                 left_lanes, right_lanes):
    """Label (decision_time, decision_time + .1]; -1 is censored/not at risk.

    Velocity is in highD's global y direction; left_lanes[0] maps it into
    driver-relative left. Sustained lateral signals are proxies, not confirmed
    completed lane changes. q/s are counts of consecutive source observations.
    """
    frames = np.asarray(frames, dtype=np.int64)
    velocity = np.asarray(velocity, dtype=float) * left_lanes[0]
    lane = np.asarray(lane, dtype=int)
    anchors = np.asarray(anchors, dtype=np.int64)
    pos = np.searchsorted(frames, anchors)
    if len(pos) and ((pos >= len(frames)).any() or not np.array_equal(frames[pos], anchors)):
        raise ValueError("An anchor is absent from its raw vehicle track")
    ends = np.asarray(ends, dtype=np.int64)
    if ends.shape != anchors.shape or (ends <= anchors).any():
        raise ValueError("Each interval must have one later end frame")
    next_index = np.searchsorted(frames, ends)
    valid_next = (next_index < len(frames))
    valid_next[valid_next] &= frames[next_index[valid_next]] == ends[valid_next]
    q, s = quiet_frames, sustain_frames
    quiet = np.isfinite(velocity) & (np.abs(velocity) < threshold)
    signs = (velocity >= threshold, velocity <= -threshold)
    contiguous = np.r_[0, np.cumsum(np.diff(frames) != 1)]
    quiet_sum = np.r_[0, np.cumsum(quiet)]
    positive = []
    for signal in signs:
        cs = np.r_[0, np.cumsum(signal)]
        possible = np.arange(len(frames)) + s <= len(frames)
        starts = np.flatnonzero(possible)
        sustained = np.zeros(len(frames), dtype=bool)
        sustained[starts] = (cs[starts + s] - cs[starts] == s)
        # A fresh signal must follow a genuinely quiet history, not an active maneuver.
        eligible = starts[starts >= q]
        fresh = np.zeros(len(frames), dtype=bool)
        fresh[eligible] = (sustained[eligible] & (quiet_sum[eligible] - quiet_sum[eligible - q] == q)
                           & (contiguous[eligible + s - 1] == contiguous[eligible - q]))
        positive.append(np.r_[0, np.cumsum(fresh)])
    result = np.full(len(pos), -1, dtype=np.int8)
    risk = pos >= q - 1
    risk[risk] &= (quiet_sum[pos[risk] + 1] - quiet_sum[pos[risk] + 1 - q] == q)
    complete = (valid_next & (next_index + s - 1 < len(frames)) & risk)
    idx = np.flatnonzero(complete)
    if len(idx):
        end = next_index[idx]
        finite_sum = np.r_[0, np.cumsum(np.isfinite(velocity))]
        ok = ((contiguous[end + s - 1] == contiguous[pos[idx] + 1 - q])
              & (finite_sum[end + s] - finite_sum[pos[idx] + 1 - q]
                 == end + s - pos[idx] - 1 + q))
        idx = idx[ok]
        end = next_index[idx]
        left = positive[0][end + 1] - positive[0][pos[idx] + 1]
        right = positive[1][end + 1] - positive[1][pos[idx] + 1]
        if (left > 1).any() or (right > 1).any() or ((left > 0) & (right > 0)).any():
            raise ValueError("Ambiguous signal onset inside one decision interval")
        result[idx] = 0
        result[idx[left > 0]] = 1
        result[idx[right > 0]] = 2
        available = np.column_stack((np.ones(len(idx), dtype=bool),
                                     np.isin(lane[pos[idx]] + left_lanes[0], list(left_lanes[1])),
                                     np.isin(lane[pos[idx]] + right_lanes[0], list(right_lanes[1]))))
        illegal_positive = ((result[idx] == 1) & ~available[:, 1]) | ((result[idx] == 2) & ~available[:, 2])
        result[idx[illegal_positive]] = -1
    return result


def _nearest(indexed, frames, ids, columns):
    lookup = pd.MultiIndex.from_arrays([frames, ids])
    return indexed.reindex(lookup)[list(columns)].reset_index(drop=True)


def build_recording(source, rec, config):
    meta = pd.read_csv(source / "data" / f"{rec}_recordingMeta.csv").iloc[0]
    fps = int(meta.frameRate)
    if fps != 25:
        raise ValueError("Current label contract expects the highD 25 Hz source")
    # The old downsampled export does not retain all causal clock anchors.
    # Read raw observations for both splits; no hindsight transition guards.
    domain = read_json(config["domain_config"])
    obs = pd.read_csv(source / "data" / f"{rec}_tracks.csv", usecols=SOURCE_COLUMNS)
    metadata = pd.read_csv(source / "data" / f"{rec}_tracksMeta.csv")
    lanes = lane_sets(meta)
    obs = obs.sort_values(["id", "frame"]).reset_index(drop=True)
    if obs.duplicated(["frame", "id"]).any():
        raise ValueError("Duplicate source state")
    obs["direction"] = obs.id.map(dict(zip(metadata.id, metadata.drivingDirection)))
    if not obs.direction.isin([1, 2]).all():
        raise ValueError("Unknown travel direction")
    obs["travel_sign"] = np.where(obs.direction == 1, -1, 1)
    obs["speed_mps"] = np.abs(obs.xVelocity)
    obs["acceleration_mps2"] = obs.xAcceleration * obs.travel_sign
    obs["supported"] = (longitudinal_domain(obs, domain)
                        & in_range(obs.acceleration_mps2, domain["acceleration_mps2"])
                        & np.array([int(l) in lanes[int(d)] for l, d in zip(obs.laneId, obs.direction)]))
    data = []
    eligible, unknown, active, positives = 0, 0, 0, [0, 0]
    for vehicle, source_track in obs.groupby("id", sort=False):
        anchors, ends, age = decision_clock(source_track.frame, fps, config["decision_hz"])
        positions = np.searchsorted(source_track.frame, anchors)
        part = source_track.iloc[positions].copy()
        part["interval_end_frame"] = ends
        part["observation_age_s"] = age
        part = part.loc[part.supported]
        if part.empty:
            continue
        direction = int(part.direction.iloc[0])
        delta_left = 1 if direction == 1 else -1
        delta_right = -delta_left
        available = lanes[direction]
        label = onset_labels(source_track.frame, source_track.yVelocity, source_track.laneId,
                             part.frame, part.interval_end_frame,
                             threshold=config["lateral_velocity_threshold_mps"],
                             quiet_frames=config["quiet_frames"], sustain_frames=config["sustain_frames"],
                             left_lanes=(delta_left, available), right_lanes=(delta_right, available))
        active += int((np.abs(part.yVelocity.to_numpy()) >= config["lateral_velocity_threshold_mps"]).sum())
        unknown += int((label < 0).sum())
        eligible += len(label)
        positives[0] += int((label == 1).sum())
        positives[1] += int((label == 2).sum())
        known = label >= 0
        if known.any():
            selected = part.iloc[np.flatnonzero(known)].copy()
            selected["target"] = label[known]
            data.append(selected)
    if not data:
        raise ValueError(f"No known supported decision rows in recording {rec}")
    selected = pd.concat(data, ignore_index=True)
    indexed = obs.set_index(["frame", "id"])
    frame = selected.frame.to_numpy()
    own_id = selected.id.to_numpy()
    previous = _nearest(indexed, frame - round(.2 * fps), own_id, ["acceleration_mps2"])
    own_prev = previous.acceleration_mps2.to_numpy(float)
    own_prev_mask = np.isfinite(own_prev)
    rear = {}
    for side in ("left", "right"):
        ids = selected[side + "FollowingId"].to_numpy(int)
        b = _nearest(indexed, frame, ids, ["x", "width", "speed_mps", "acceleration_mps2", "laneId", "direction"])
        direction = selected.direction.to_numpy(int)
        delta = (1 if side == "left" else -1) * np.where(direction == 1, 1, -1)
        valid = ((ids > 0) & (b.laneId.to_numpy() == selected.laneId.to_numpy() + delta)
                 & (b.direction.to_numpy() == direction))
        distance = ((selected.x.to_numpy() + selected.width.to_numpy() / 2
                     - b.x.to_numpy() - b.width.to_numpy() / 2) * selected.travel_sign.to_numpy()
                    - (selected.width.to_numpy() + b.width.to_numpy()) / 2)
        rear[side] = {"valid": valid, "gap": np.where(valid, np.clip(distance, -20, 115), 115),
                      "rr": np.where(valid, b.speed_mps.to_numpy() - selected.speed_mps.to_numpy(), 0),
                      "acc": np.where(valid, b.acceleration_mps2.to_numpy(), 0)}
    directions = selected.direction.to_numpy(int)
    lane = selected.laneId.to_numpy(int)
    left_delta = np.where(directions == 1, 1, -1)
    right_delta = -left_delta
    left_allowed = np.array([int(l + d) in lanes[int(dr)] for l, d, dr in zip(lane, left_delta, directions)])
    right_allowed = np.array([int(l + d) in lanes[int(dr)] for l, d, dr in zip(lane, right_delta, directions)])
    center = np.empty(len(selected))
    for direction, field in ((1, "upperLaneMarkings"), (2, "lowerLaneMarkings")):
        marks = np.array([float(x) for x in str(meta[field]).split(";")])
        mask = directions == direction
        idx = lane[mask] - min(lanes[direction])
        center[mask] = (marks[idx] + marks[idx + 1]) / 2
    own_free = ((selected.precedingId.to_numpy(int) <= 0) | (selected.dhw.to_numpy() > 115))
    own_front_rr = np.abs(selected.precedingXVelocity.to_numpy()) - selected.speed_mps.to_numpy()
    outside = np.array([len(lanes[int(d)]) > 2 for d in directions])
    state = np.column_stack([
        selected.speed_mps, np.where(own_free, 115, selected.dhw), np.where(own_free, 0, own_front_rr),
        np.where(own_prev_mask, own_prev, 0), selected.yVelocity.to_numpy() * left_delta,
        (selected.y.to_numpy() + selected.height.to_numpy() / 2 - center) * left_delta,
        left_allowed, right_allowed,
        rear["left"]["valid"], rear["left"]["gap"], rear["left"]["rr"], rear["left"]["acc"],
        rear["right"]["valid"], rear["right"]["gap"], rear["right"]["rr"], rear["right"]["acc"],
        outside, selected.observation_age_s, ~own_prev_mask,
    ]).astype(np.float32)
    action = selected.acceleration_mps2.to_numpy(float)
    action_index = np.rint((action + 4) / .2).astype(np.int8)
    permitted = np.column_stack([np.ones(len(selected), dtype=bool), left_allowed, right_allowed])
    observed = selected.target.to_numpy(np.int8)
    good = (np.isfinite(state).all(axis=1) & (action_index >= 0) & (action_index <= 30)
            & permitted[np.arange(len(selected)), observed])
    result = {"state": state[good], "action_index": action_index[good],
              "target": observed[good], "permitted": permitted[good], "two_lane": ~outside[good]}
    return result, {"recording": rec, "sampled_decision_rows": eligible, "unknown_or_active_rows": unknown,
                    "active_speed_threshold_rows": active, "positive_left_rows": positives[0],
                    "positive_right_rows": positives[1], "fitted_rows": int(good.sum()),
                    "removed_for_state_or_action": int((~good).sum()), "source_fps": fps,
                    "maximum_observation_age_s": float(selected.observation_age_s.max()),
                    "previous_acceleration_missing_rows": int((~own_prev_mask).sum()),
                    "source_stamps": {k: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                      for k, p in {
                                          "tracks": source / "data" / f"{rec}_tracks.csv",
                                          "recordingMeta": source / "data" / f"{rec}_recordingMeta.csv",
                                          "tracksMeta": source / "data" / f"{rec}_tracksMeta.csv",
                                      }.items()}}


def make_design(state, action_index, mean, scale, conditional):
    z = (state.astype(float) - mean) / scale
    x = np.column_stack([np.ones(len(state)), z])
    if conditional:
        # a_scaled = (a_mps2 + 1) / 2; the 31 original action IDs are unchanged.
        a = (action_index.astype(float) - 15) / 10
        x = np.column_stack([x, a, a * a, a * z[:, 9], a * z[:, 13]])
    return x


def log_probs(x, coefficient, permitted):
    logits = x @ coefficient
    logits[~permitted] = -np.inf
    return logits - logsumexp(logits, axis=1, keepdims=True)


def joint_action_probabilities(state, longitudinal_probability, permitted, model):
    """Return [row, acceleration_id, lateral_choice] P(a|h) pi(d|a,h).

    Call only for eligible quiet states. The supplied longitudinal distribution
    must already be normalized; this routine does not silently alter its support.
    This is an offline probability query, not a SUMO action adapter.
    """
    p = np.asarray(longitudinal_probability, dtype=float)
    permitted = np.asarray(permitted, dtype=bool)
    if (p.shape != (len(state), 31) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Expected normalized 31-action longitudinal marginals")
    if permitted.shape != (len(state), 3) or not permitted[:, 0].all():
        raise ValueError("Expected three lateral masks including keep")
    result = np.empty((len(state), 31, 3))
    for action in range(31):
        x = make_design(state, np.full(len(state), action), model["mean"], model["scale"], True)
        result[:, action] = p[:, action, None] * np.exp(log_probs(x, model["conditional"], permitted))
    return result


def fit(x, target, permitted, config, initial_coefficient=None):
    counts = np.bincount(target, minlength=3)
    initial = np.zeros((x.shape[1], 3))
    initial[0] = np.log((counts + 1) / (counts.sum() + 3))
    if initial_coefficient is not None:
        initial = np.asarray(initial_coefficient, dtype=float).copy()
        if initial.shape != (x.shape[1], 3) or not np.isfinite(initial).all():
            raise ValueError("Warm-start coefficient shape or values differ")
    rows = np.arange(len(target))
    def objective(flat):
        coefficient = flat.reshape(initial.shape)
        logs = log_probs(x, coefficient, permitted)
        loss = -logs[rows, target].mean() + config["l2"] * np.square(coefficient[1:]).sum() / 2
        error = np.exp(logs)
        error[rows, target] -= 1
        gradient = x.T @ error / len(target)
        gradient[1:] += config["l2"] * coefficient[1:]
        return loss, gradient.ravel()
    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": config["max_iterations"], "gtol": config.get("gtol", 1e-8),
                               "ftol": config.get("ftol", 1e-10), "maxcor": config.get("maxcor", 10)})
    if not np.isfinite(result.fun):
        raise ValueError("Nonfinite training loss")
    return result.x.reshape(initial.shape), {"iterations": int(result.nit), "converged": bool(result.success),
                                            "objective": float(result.fun), "message": str(result.message),
                                            "gradient_max_abs": float(np.max(np.abs(result.jac))),
                                            "warm_started": initial_coefficient is not None}


def evaluate(data, model, split):
    logs = {}
    for kind in ("baseline", "conditional"):
        logs[kind] = np.empty((len(data["target"]), 3))
        for first in range(0, len(data["target"]), 65536):
            section = slice(first, first + 65536)
            x = make_design(data["state"][section], data["action_index"][section],
                            model["mean"], model["scale"], kind == "conditional")
            logs[kind][section] = log_probs(x, model[kind], data["permitted"][section])
    log0, log1 = logs["baseline"], logs["conditional"]
    labels = data["target"]
    idx = np.arange(len(labels))
    gain = log1[idx, labels] - log0[idx, labels]
    score = {"split": split, "rows": len(labels), "label_counts": np.bincount(labels, minlength=3).tolist(),
             "baseline_nll": float(-log0[idx, labels].mean()),
             "conditional_nll": float(-log1[idx, labels].mean()), "gain": float(gain.mean()),
             "observed_start_rate": float((labels > 0).mean()),
             "baseline_predicted_start_rate": float(np.exp(log0[:, 1:]).sum(axis=1).mean()),
             "conditional_predicted_start_rate": float(np.exp(log1[:, 1:]).sum(axis=1).mean()),
             "positive_label_gain": float(gain[labels > 0].mean()) if (labels > 0).any() else None,
             "two_lane_gain": float(gain[data["two_lane"]].mean()) if data["two_lane"].any() else None,
             "three_lane_gain": float(gain[~data["two_lane"]].mean()) if (~data["two_lane"]).any() else None,
             "maximum_probability_sum_error": float(np.max(np.abs(np.exp(log1).sum(axis=1) - 1)))}
    for name, mask in (("two_lane", data["two_lane"]), ("three_lane", ~data["two_lane"])):
        if mask.any():
            score[name] = {"rows": int(mask.sum()), "start_count": int((labels[mask] > 0).sum()),
                           "observed_start_rate": float((labels[mask] > 0).mean()),
                           "conditional_predicted_start_rate": float(np.exp(log1[mask, 1:]).sum(axis=1).mean())}
    score["by_choice"] = {
        name: {"observed_rate": float((labels == choice).mean()),
               "baseline_predicted_rate": float(np.exp(log0[:, choice]).mean()),
               "conditional_predicted_rate": float(np.exp(log1[:, choice]).mean())}
        for choice, name in enumerate(("keep", "left", "right"))}
    return score


def run(output, config, split):
    output = Path(output)
    source = Path(config["source_root"])
    contract = read_json(config["freeze_contract"])
    if config["decision_hz"] != 10 or digest(contract["model"]["path"]) != contract["model"]["sha256"]:
        raise ValueError("Expected fixed decision clock and frozen single reference")
    model_path = output / "conditional_lane_start_v1.npz"
    if split == "train" and model_path.exists():
        raise ValueError("Fitted model exists; use a new output directory")
    if split == "calibration" and not model_path.exists():
        raise ValueError("Train before fixed calibration evaluation")
    if (output / (split + "_summary.json")).exists():
        raise ValueError("Report exists; use a fresh output or inspect the existing result")
    provenance = {"script_sha256": digest(__file__), "domain_sha256": digest(config["domain_config"]),
                  "freeze_contract_sha256": digest(config["freeze_contract"])}
    if split == "calibration":
        training = read_json(output / "train_summary.json")
        if (training["model_sha256"] != digest(model_path) or training["config"] != config
                or training["provenance"] != provenance):
            raise ValueError("Calibration must use the unchanged fitted artifact and label/feature protocol")
    output.mkdir(parents=True, exist_ok=True)
    summaries, pieces = [], []
    for rec in contract[split + "_recordings"]:
        data, summary = build_recording(source, rec, config)
        pieces.append(data)
        summaries.append(summary)
        print(json.dumps({"recording": rec, "rows": summary["fitted_rows"],
                          "left": summary["positive_left_rows"], "right": summary["positive_right_rows"]}), flush=True)
    data = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    quality = {}
    if split == "train":
        mean = data["state"].astype(float).mean(axis=0)
        scale = data["state"].astype(float).std(axis=0)
        scale[scale < 1e-8] = 1
        model = {"mean": mean, "scale": scale}
        previous_model = {}
        if config.get("warm_start_root"):
            previous_root = Path(config["warm_start_root"])
            previous_report = read_json(previous_root / "train_summary.json")
            previous_path = previous_root / "conditional_lane_start_v1.npz"
            if (digest(previous_path) != previous_report["model_sha256"]
                    or previous_report["recordings"] != summaries
                    or previous_report["features"] != {"state": list(STATE_FEATURES), "action": list(ACTION_FEATURES)}):
                raise ValueError("Warm start must use the same train observations and features")
            fit_keys = {"warm_start_root", "max_iterations", "gtol", "ftol", "maxcor", "purpose"}
            if {k: v for k, v in config.items() if k not in fit_keys} != {
                    k: v for k, v in previous_report["config"].items() if k not in fit_keys}:
                raise ValueError("Warm start must preserve the label and regularization configuration")
            with np.load(previous_path, allow_pickle=False) as artifact:
                previous_model = {key: artifact[key] for key in artifact.files}
            if not np.array_equal(mean, previous_model["mean"]) or not np.array_equal(scale, previous_model["scale"]):
                raise ValueError("Warm start must preserve train standardization")
        for kind in ("baseline", "conditional"):
            x = make_design(data["state"], data["action_index"], mean, scale, kind == "conditional")
            model[kind], quality[kind] = fit(x, data["target"], data["permitted"], config,
                                            previous_model.get(kind))
            print(json.dumps({"model": kind, **quality[kind]}), flush=True)
            del x
        np.savez_compressed(model_path, **model)
    else:
        with np.load(model_path, allow_pickle=False) as artifact:
            model = {key: artifact[key] for key in artifact.files}
    per_recording = {row["recording"]: evaluate(piece, model, split) for piece, row in zip(pieces, summaries)}
    report = {"model_sha256": digest(model_path), "config": config, "split": split, "provenance": provenance,
              "features": {"state": STATE_FEATURES, "action": ACTION_FEATURES},
              "optimization": quality, "recordings": summaries, "overall": evaluate(data, model, split),
              "by_recording": per_recording,
              "recordings_with_positive_gain": sum(score["gain"] > 0 for score in per_recording.values()),
              "scope": "Sustained lateral signal onset over the next exact 0.1s decision interval, conditional on observed supported acceleration. Not confirmed lane-change starts. Present/lagged features use source observations no later than the decision (age <= .02s). Six quiet observations span .2s; ten sustained observations span .36s. Fixed calibration recordings have prior development use; repeated frames are not independent experiments. Offline fit only, no runtime changes."}
    write_new(output / (split + "_summary.json"), report)
    print(json.dumps(report["overall"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/highd_conditional_lane_start_v1.json"))
    p.add_argument("--split", choices=("train", "calibration"), default="train")
    args = p.parse_args()
    run(args.output, read_json(args.config), args.split)


if __name__ == "__main__":
    main()
