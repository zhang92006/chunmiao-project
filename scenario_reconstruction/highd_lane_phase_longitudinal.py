"""Causal lateral phases and phase/interaction-conditioned longitudinal NDD.

Whole-event completion is diagnostic metadata only. Every supported active
decision is eligible, including incomplete and returning maneuvers. Models tilt
three acceleration-group masses and retain the frozen 31-action within-group
shape. No future acceleration, completion, or time-to-crossing is an input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .highd_conditional_lane_start import decision_clock
from .highd_pair_data_inventory import digest, in_range, lane_sets, longitudinal_domain, read_json, write_new
from .highd_pair_model_fit import FrozenMarginal
from .highd_pair_sample_export import SOURCE_COLUMNS


HISTORY_FEATURES = ("speed", "own_gap", "own_rr", "free_flow", "a_lag_02", "a_lag_04", "a_lag_08",
                    "missing_lag_02", "missing_lag_04", "missing_lag_08", "a_lag_02_squared")
PHASE_FEATURES = ("before_boundary", "after_boundary", "returning", "elapsed_s", "progress_m",
                  "directed_lateral_velocity", "history_x_before", "history_x_after", "two_lane")
INTERACTION_FEATURES = ("partner_present", "partner_gap", "partner_relative_speed",
                        "partner_a_lag_02", "partner_history_missing", "closing_per_gap",
                        "partner_a_x_before", "partner_a_x_after")
FEATURES = HISTORY_FEATURES + PHASE_FEATURES + INTERACTION_FEATURES
MODEL_FEATURES = {"history": len(HISTORY_FEATURES),
                  "phase": len(HISTORY_FEATURES) + len(PHASE_FEATURES), "interaction": len(FEATURES)}
PHASE_NAMES = {1: "before_boundary", 2: "after_boundary", 3: "returning"}


def causal_phases(track, centers, legal_lanes, config):
    """Track state using only prefixes; retrospective outcome never changes it.

    Onset is detected after a short directed-motion confirmation (not backdated).
    A return before crossing does not prove lane-change intent. Completion is
    detected only after an observed stable run in the destination lane.
    """
    frames = track.frame.to_numpy(int)
    lane = track.laneId.to_numpy(int)
    velocity = track.yVelocity.to_numpy(float)
    y = track.y.to_numpy(float) + track.height.to_numpy(float) / 2
    n = len(track)
    phase = np.zeros(n, dtype=np.int8)
    event_ids = np.full(n, -1, dtype=int)
    rear_ids = np.zeros(n, dtype=int)
    target_lanes = np.zeros(n, dtype=int)
    elapsed = np.zeros(n)
    progress = np.zeros(n)
    directed_velocity = np.zeros(n)
    threshold = config["onset_velocity_mps"]
    q, c, stable_n = (config[k] for k in ("quiet_frames", "onset_confirm_frames", "stable_frames"))
    quiet = np.isfinite(velocity) & (np.abs(velocity) < threshold)
    prefix_q = np.r_[0, np.cumsum(quiet)]
    prefix_pos = np.r_[0, np.cumsum(velocity >= threshold)]
    prefix_neg = np.r_[0, np.cumsum(velocity <= -threshold)]
    events, current = [], None
    segment_start = 0
    stable_count = 0
    reverse_count = 0
    last_quiet_end = -1000000
    direction = int(track.direction.iloc[0])
    vehicle = int(track.id.iloc[0])
    fps = 25

    def finish(index, status):
        event = dict(current)
        event.update(end_frame=int(frames[index]), status=status,
                     duration_s=float((frames[index] - event["start_frame"]) / fps))
        events.append(event)

    for i in range(n):
        if i and frames[i] != frames[i - 1] + 1:
            if current is not None:
                finish(i - 1, "gap_censored")
            current = None
            segment_start = i
            last_quiet_end = -1000000
            stable_count = reverse_count = 0
        if current is None:
            if i - segment_start + 1 >= q and prefix_q[i + 1] - prefix_q[i + 1 - q] == q:
                last_quiet_end = i
            if i - segment_start + 1 < q + c or lane[i] not in legal_lanes:
                continue
            first = i + 1 - c
            if (last_quiet_end < segment_start + q - 1
                    or i - last_quiet_end > round(config["onset_after_quiet_max_s"] * fps)
                    or (lane[last_quiet_end + 1 - q:i + 1] != lane[i]).any()):
                continue
            sign = (1 if prefix_pos[i + 1] - prefix_pos[first] == c else
                    -1 if prefix_neg[i + 1] - prefix_neg[first] == c else 0)
            target = lane[i] + sign
            if not sign or target not in legal_lanes or sign * (y[i] - centers[lane[i]]) < config["minimum_onset_offset_m"]:
                continue
            side = "left" if (direction == 1 and sign > 0) or (direction == 2 and sign < 0) else "right"
            current = {"vehicle_id": vehicle, "event_index": len(events), "start_frame": int(frames[i]),
                       "origin_lane": int(lane[i]), "target_lane": int(target), "side": side,
                       "y_sign": sign, "rear_id": int(track[side + "FollowingId"].iloc[i]),
                       "crossing_frame": None, "maximum_excursion_m": 0.0}
            stable_count = reverse_count = 0
        if lane[i] not in (current["origin_lane"], current["target_lane"]):
            finish(i, "other_lane_censored")
            current = None
            continue
        if current["crossing_frame"] is None and lane[i] == current["target_lane"]:
            current["crossing_frame"] = int(frames[i])
        sign = current["y_sign"]
        reverse_count = reverse_count + 1 if velocity[i] * sign <= -threshold else 0
        returning = (reverse_count >= c or (i > 0 and phase[i - 1] == 3
                     and event_ids[i - 1] == current["event_index"]))
        phase[i] = 3 if returning else (2 if current["crossing_frame"] is not None else 1)
        event_ids[i] = current["event_index"]
        rear_ids[i] = current["rear_id"]
        target_lanes[i] = current["target_lane"]
        elapsed[i] = (frames[i] - current["start_frame"]) / fps
        progress[i] = (y[i] - centers[current["origin_lane"]]) * sign
        directed_velocity[i] = velocity[i] * sign
        current["maximum_excursion_m"] = max(current["maximum_excursion_m"], float(progress[i]))
        stable_now = quiet[i] and abs(y[i] - centers[lane[i]]) <= config["stable_center_tolerance_m"]
        stable_count = stable_count + 1 if stable_now and (i == 0 or lane[i] == lane[i - 1]) else int(stable_now)
        if stable_count >= stable_n:
            if lane[i] == current["target_lane"]:
                status = "completed"
            elif current["crossing_frame"] is not None:
                status = "returned_after_crossing"
            elif current["maximum_excursion_m"] >= config["return_excursion_threshold_m"]:
                status = "returned_before_crossing"
            else:
                status = "within_lane_adjustment"
            finish(i, status)
            current = None
    if current is not None:
        finish(n - 1, "right_censored")
    return {"phase": phase, "event_index": event_ids, "locked_rear_id": rear_ids,
            "target_lane": target_lanes, "elapsed_s": elapsed, "progress_m": progress,
            "directed_velocity": directed_velocity}, events


def aligned(index, frames, ids):
    return index.reindex(pd.MultiIndex.from_arrays([frames, ids])).reset_index(drop=True)


def lag_acceleration(index, own, lag):
    previous = aligned(index, own.frame.to_numpy() - round(25 * lag), own.id.to_numpy())
    valid = ((previous.segment_start.to_numpy() == own.segment_start.to_numpy())
             & np.isfinite(previous.acceleration_mps2.to_numpy()))
    return np.where(valid, previous.acceleration_mps2.to_numpy(), 0), ~valid


def make_role_rows(own, partner, phase_rows, index, marginal, config, *, rear_role):
    valid = own.supported.fillna(False).to_numpy(bool)
    # Rear identity is locked at the causal onset, not selected using an outcome.
    if rear_role:
        valid &= own.laneId.to_numpy() == phase_rows.target_lane.to_numpy()
    own = own.loc[valid].reset_index(drop=True)
    partner = partner.loc[valid].reset_index(drop=True)
    phase_rows = phase_rows.loc[valid].reset_index(drop=True)
    if own.empty:
        return None
    history = [lag_acceleration(index, own, lag) for lag in config["history_lags_s"]]
    free = (own.precedingId.to_numpy() <= 0) | (own.dhw.to_numpy() > 115)
    gap = np.where(free, 115, own.dhw)
    rr = np.where(free, 0, np.abs(own.precedingXVelocity) - own.speed_mps)
    partner_valid = ((partner.direction.to_numpy() == own.direction.to_numpy())
                     & np.isfinite(partner.x.to_numpy()))
    if not rear_role:
        partner_valid &= partner.laneId.to_numpy() == phase_rows.target_lane.to_numpy()
    partner_lag, partner_missing = lag_acceleration(index, partner, .2)
    partner_missing |= ~partner_valid
    partner_lag = np.where(partner_missing, 0, partner_lag)
    distance = ((own.x.to_numpy() + own.width.to_numpy() / 2
                 - partner.x.to_numpy() - partner.width.to_numpy() / 2) * own.travel_sign.to_numpy())
    # Signed center separation supports both focal/rear roles, including overlap.
    distance = np.where(partner_valid, np.clip(distance, -115, 115), 0)
    relative_speed = np.where(partner_valid, partner.speed_mps - own.speed_mps, 0)
    before, after = (phase_rows.phase.to_numpy() == k for k in (1, 2))
    history0 = history[0][0]
    state = np.column_stack([
        own.speed_mps, gap, rr, free, *(h[0] for h in history), *(h[1] for h in history), history0 ** 2,
        before, after, phase_rows.phase == 3, np.minimum(phase_rows.elapsed_s, 15), phase_rows.progress_m,
        phase_rows.directed_velocity, history0 * before, history0 * after, own.two_lane,
        partner_valid, distance, relative_speed, partner_lag, partner_missing,
        relative_speed / np.maximum(np.abs(distance), 5), partner_lag * before, partner_lag * after,
    ]).astype(np.float32)
    p0, masses, group = marginal.query(own)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite current or lagged features")
    result = {"state": state, "log_p0": np.log(p0), "mass": masses, "group": group,
            "phase": phase_rows.phase.to_numpy(np.int8), "two_lane": own.two_lane.to_numpy(bool),
            "vehicle_id": phase_rows.id.to_numpy(int), "frame": own.frame.to_numpy(int)}
    if config.get("full_action_probability", False):
        vi = marginal.index(own.speed_mps, "speed")
        full_p = marginal.ff[vi].copy()
        cf = ~free
        if cf.any():
            gi = marginal.index(own.dhw.to_numpy()[cf], "gap")
            ri = marginal.index(np.asarray(rr)[cf], "range_rate")
            full_p[cf] = marginal.cf[gi, ri, vi[cf]]
        result["log_p0_full"] = np.log(full_p)
        result["action_index"] = marginal.index(own.acceleration_mps2, "acceleration")
    return result


def extract_recording(source, rec, config, marginal):
    root = source / "data"
    meta = pd.read_csv(root / f"{rec}_recordingMeta.csv").iloc[0]
    if int(meta.frameRate) != 25:
        raise ValueError("Expected 25 Hz highD input")
    lanes = lane_sets(meta)
    centers = {}
    for direction, field in ((1, "upperLaneMarkings"), (2, "lowerLaneMarkings")):
        marks = np.array([float(v) for v in str(meta[field]).split(";")])
        centers.update({lane: (marks[i] + marks[i + 1]) / 2 for i, lane in enumerate(sorted(lanes[direction]))})
    obs = pd.read_csv(root / f"{rec}_tracks.csv", usecols=SOURCE_COLUMNS).sort_values(["id", "frame"]).reset_index(drop=True)
    tracks_meta = pd.read_csv(root / f"{rec}_tracksMeta.csv")
    domain = read_json(config["domain_config"])
    obs["direction"] = obs.id.map(dict(zip(tracks_meta.id, tracks_meta.drivingDirection)))
    obs["travel_sign"] = np.where(obs.direction == 1, -1, 1)
    obs["speed_mps"] = np.abs(obs.xVelocity)
    obs["acceleration_mps2"] = obs.xAcceleration * obs.travel_sign
    obs["two_lane"] = np.array([len(lanes[int(d)]) == 2 for d in obs.direction])
    obs["supported"] = (longitudinal_domain(obs, domain)
                        & in_range(obs.acceleration_mps2, domain["acceleration_mps2"]))
    start = (obs.id != obs.id.shift()) | (obs.frame != obs.frame.shift() + 1)
    segment_indices = np.maximum.accumulate(np.where(start.to_numpy(), np.arange(len(obs)), 0))
    obs["segment_start"] = obs.frame.to_numpy()[segment_indices]
    events, pieces = [], []
    crossings, linked_crossings = 0, 0
    for vehicle, track in obs.groupby("id", sort=False):
        fields, these_events = causal_phases(track, centers, lanes[int(track.direction.iloc[0])], config)
        events.extend(these_events)
        edges = np.diff(track.laneId.to_numpy())
        crossings += int(((np.abs(edges) == 1) & (np.diff(track.frame.to_numpy()) == 1)).sum())
        linked_crossings += sum(e["crossing_frame"] is not None for e in these_events)
        anchors, _, _ = decision_clock(track.frame, 25, config["decision_hz"])
        positions = np.searchsorted(track.frame, anchors)
        positions = positions[fields["phase"][positions] > 0]
        if len(positions):
            part = track.iloc[positions].copy()
            for key, values in fields.items():
                part[key] = values[positions]
            pieces.append(part)
    if not pieces:
        raise ValueError(f"No causally detected lateral decisions in {rec}")
    focal = pd.concat(pieces, ignore_index=True)
    index = obs.set_index(["frame", "id"])
    rear = aligned(index, focal.frame.to_numpy(), focal.locked_rear_id.to_numpy())
    # Reindex retains frame/id as index only; restore them for history queries.
    rear["frame"] = focal.frame.to_numpy()
    rear["id"] = focal.locked_rear_id.to_numpy()
    e = make_role_rows(focal, rear, focal, index, marginal, config, rear_role=False)
    r = make_role_rows(rear, focal, focal, index, marginal, config, rear_role=True)
    summary = {"recording": rec, "event_status_counts": dict(Counter(e["status"] for e in events)),
               "detected_motions": len(events), "observed_adjacent_lane_crossings": crossings,
               "motions_with_observed_crossing": linked_crossings,
               "active_decision_rows_before_domain_filter": len(focal),
               "focal_rows": len(e["group"]) if e is not None else 0,
               "rear_rows": len(r["group"]) if r is not None else 0}
    return {"focal": e, "rear": r}, events, summary


def design(state, mean, scale, kind):
    count = MODEL_FEATURES[kind]
    return np.column_stack([np.ones(len(state)), (state[:, :count] - mean[:count]) / scale[:count]])


def group_log_probability(x, mass, coefficient):
    logits = np.log(mass) + x @ coefficient
    return logits - logsumexp(logits, axis=1, keepdims=True)


def tilt_longitudinal_probability(p0, x, coefficient, action_groups):
    """Normalized 31-action distribution with unchanged within-group shape."""
    mass = np.column_stack([p0[:, action_groups == k].sum(axis=1) for k in range(3)])
    group_p = np.exp(group_log_probability(x, mass, coefficient))
    return p0 * (group_p / mass)[:, action_groups]


def fit_correction(x, data, config):
    rows = np.arange(len(x))
    def objective(flat):
        beta = flat.reshape(x.shape[1], 3)
        logp = group_log_probability(x, data["mass"], beta)
        loss = -logp[rows, data["group"]].mean() + config["l2"] * (beta[1:] ** 2).sum() / 2
        error = np.exp(logp)
        error[rows, data["group"]] -= 1
        grad = x.T @ error / len(x)
        grad[1:] += config["l2"] * beta[1:]
        return loss, grad.ravel()
    result = minimize(objective, np.zeros(x.shape[1] * 3), jac=True, method="L-BFGS-B",
                      options={k: config[k] for k in ("gtol", "ftol", "maxcor")} | {"maxiter": config["max_iterations"]})
    return result.x.reshape(x.shape[1], 3), {"converged": bool(result.success), "iterations": int(result.nit),
               "objective": float(result.fun), "gradient_max_abs": float(np.max(np.abs(result.jac))), "message": str(result.message)}


def scores(data, model):
    result = {"frozen": data["log_p0"]}
    rows = np.arange(len(data["group"]))
    for kind in MODEL_FEATURES:
        x = design(data["state"], model["mean"], model["scale"], kind)
        logq = group_log_probability(x, data["mass"], model[kind])
        result[kind] = data["log_p0"] + logq[rows, data["group"]] - np.log(data["mass"][rows, data["group"]])
    return result


def score_summary(data, logs):
    result = {"rows": len(data["group"]), "nll": {k: float(-v.mean()) for k, v in logs.items()}, "by_phase": {}}
    for code, name in PHASE_NAMES.items():
        mask = data["phase"] == code
        if mask.any():
            result["by_phase"][name] = {"rows": int(mask.sum()), "nll": {k: float(-v[mask].mean()) for k, v in logs.items()}}
    mask = data["two_lane"]
    result["two_lane"] = {"rows": int(mask.sum()), "nll": {k: float(-v[mask].mean()) if mask.any() else None for k, v in logs.items()}}
    return result


def cluster_gain(records, prior, candidate, config, two_lane=False):
    blocks = [r["two_lane"] if two_lane else r for r in records]
    blocks = [b for b in blocks if b["rows"]]
    if not blocks:
        return {"recording_count": 0}
    n = np.array([b["rows"] for b in blocks])
    delta = np.array([b["nll"][prior] - b["nll"][candidate] for b in blocks])
    rng = np.random.default_rng(config["bootstrap_seed"])
    draw = rng.integers(0, len(n), (config["bootstrap_repetitions"], len(n)))
    boot = (n[draw] * delta[draw]).sum(axis=1) / n[draw].sum(axis=1)
    return {"recording_count": len(n), "gain": float((n * delta).sum() / n.sum()),
            "positive_recordings": int((delta > 0).sum()), "recording_bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist(),
            "interpretation": "Exploratory recording-cluster interval on reused development data; not confirmatory significance."}


def run(output, config, split):
    output = Path(output)
    report_path = output / (split + "_summary.json")
    if report_path.exists():
        raise ValueError("Report exists; use a fresh output")
    contract = read_json(config["freeze_contract"])
    marginal = FrozenMarginal(contract, config["action_thresholds_mps2"])
    provenance = {"code_sha256": digest(__file__), "clock_code_sha256": digest(Path(__file__).with_name("highd_conditional_lane_start.py")),
                  "domain_sha256": digest(config["domain_config"]), "freeze_contract_sha256": digest(config["freeze_contract"])}
    model_path = output / "lane_phase_longitudinal_v1.npz"
    if split == "train" and model_path.exists():
        raise ValueError("Model exists; use a fresh output")
    if split == "calibration":
        protocol = read_json(output / "train_summary.json")
        if protocol["config"] != config or protocol["provenance"] != provenance or protocol["model_sha256"] != digest(model_path):
            raise ValueError("Fixed model/protocol changed")
    output.mkdir(parents=True, exist_ok=True)
    pieces = {"focal": [], "rear": []}
    recording_rows, all_events = [], []
    for rec in contract[split + "_recordings"]:
        roles, events, summary = extract_recording(Path(config["source_root"]), rec, config, marginal)
        recording_rows.append(summary)
        all_events.extend(dict(recording=rec, **event) for event in events)
        for role in pieces:
            if roles[role] is not None:
                pieces[role].append((rec, roles[role]))
        print(json.dumps(summary), flush=True)
    # Write derived diagnostic tables locally; outcome metadata never feeds fit.
    write_new(output / (split + "_maneuvers.json"), all_events)
    data = {role: {key: np.concatenate([d[key] for _, d in blocks]) for key in blocks[0][1]}
            for role, blocks in pieces.items() if blocks}
    models, optimization = {}, {}
    if split == "train":
        for role, d in data.items():
            mean, scale = d["state"].astype(float).mean(axis=0), d["state"].astype(float).std(axis=0)
            scale[scale < 1e-8] = 1
            models[role] = {"mean": mean, "scale": scale}
            optimization[role] = {}
            for kind in MODEL_FEATURES:
                x = design(d["state"], mean, scale, kind)
                models[role][kind], optimization[role][kind] = fit_correction(x, d, config)
                print(json.dumps({"role": role, "model": kind, **optimization[role][kind]}), flush=True)
        np.savez_compressed(model_path, **{role + "__" + key: value for role, model in models.items() for key, value in model.items()})
    else:
        with np.load(model_path, allow_pickle=False) as artifact:
            for key in artifact.files:
                role, name = key.split("__")
                models.setdefault(role, {})[name] = artifact[key]
    results = {}
    for role, blocks in pieces.items():
        if not blocks:
            continue
        model = models[role]
        records = {rec: score_summary(d, scores(d, model)) for rec, d in blocks}
        results[role] = {"overall": score_summary(data[role], scores(data[role], model)), "by_recording": records,
                         "comparisons": {prior + "_to_" + candidate: {
                             "all": cluster_gain(list(records.values()), prior, candidate, config),
                             "two_lane": cluster_gain(list(records.values()), prior, candidate, config, True)}
                             for prior, candidate in (("frozen", "history"), ("history", "phase"), ("phase", "interaction"))}}
    report = {"config": config, "provenance": provenance, "model_sha256": digest(model_path), "split": split,
              "features": FEATURES, "optimization": optimization, "recordings": recording_rows, "roles": results,
              "scope": "Conditional on causally detected active lateral motion and supported current states/actions. All completion outcomes retained. Three-group residual adjustments preserve within-group frozen shape, not the entire frozen marginal. Separate role conditionals, not a correlated dual-BV joint model. Reused calibration is development only; no validation/test or runtime replacement."}
    write_new(report_path, report)
    print(json.dumps({role: r["overall"] for role, r in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/highd_lane_phase_longitudinal_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration"), default="train")
    args = parser.parse_args()
    run(args.output, read_json(args.config), args.split)


if __name__ == "__main__":
    main()
