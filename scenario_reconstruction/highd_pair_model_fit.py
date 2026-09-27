"""Score frozen-marginal following dependence and fit post-crossing dependence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .highd_pair_data_inventory import digest, read_json, write_new
from .highd_pair_sample_export import clock_mask


class FrozenMarginal:
    def __init__(self, contract, thresholds):
        if digest(contract["model"]["path"]) != contract["model"]["sha256"]:
            raise ValueError("Frozen marginal artifact changed")
        with np.load(contract["model"]["path"], allow_pickle=False) as a:
            self.axes = {k: a[k + "_axis"].astype(float) for k in contract["grid"]}
            self.cf = a["car_following_probability"].astype(float)
            self.ff = a["free_flow_probability"].astype(float)
        self.cf /= self.cf.sum(axis=-1, keepdims=True)
        self.ff /= self.ff.sum(axis=-1, keepdims=True)
        axis = self.axes["acceleration"]
        self.groups = np.where(axis < thresholds[0], 0, np.where(axis > thresholds[1], 2, 1))

    def index(self, values, name):
        axis = self.axes[name]
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (values < axis[0]).any() or (values > axis[-1]).any():
            raise ValueError("Out-of-domain frozen marginal query: " + name)
        return np.rint((values - axis[0]) / (axis[1] - axis[0])).astype(int)

    def query(self, obs):
        speed = np.abs(obs.xVelocity.to_numpy(float))
        speed_index = self.index(speed, "speed")
        gap = obs.dhw.to_numpy(float)
        free = (obs.precedingId.to_numpy(int) == 0) | (gap > self.axes["gap"][-1])
        p = self.ff[speed_index].copy()
        cf = ~free
        if cf.any():
            gi = self.index(gap[cf], "gap")
            rr = np.abs(obs.precedingXVelocity.to_numpy(float)[cf]) - speed[cf]
            ri = self.index(rr, "range_rate")
            p[cf] = self.cf[gi, ri, speed_index[cf]]
        action = self.index(obs.acceleration_mps2.to_numpy(float), "acceleration")
        masses = np.stack([p[:, self.groups == k].sum(axis=1) for k in range(3)], axis=1)
        return p[np.arange(len(p)), action], masses, self.groups[action]


def project_coarse(first, second, modifier, tolerance=1e-10, max_iterations=200):
    """Batched 3x3 IPF; block lifting preserves both frozen 31-action marginals."""
    joint = first[:, :, None] * second[:, None, :] * modifier
    for iteration in range(max_iterations):
        joint *= (first / joint.sum(axis=2))[:, :, None]
        joint *= (second / joint.sum(axis=1))[:, None, :]
        error = max(float(np.max(np.abs(joint.sum(axis=2) - first))),
                    float(np.max(np.abs(joint.sum(axis=1) - second))))
        if error <= tolerance:
            return joint, error
    raise RuntimeError("Marginal projection did not converge")


def score_correction(first, second, first_group, second_group, modifier, config):
    joint, error = project_coarse(first, second, modifier, config["projection_tolerance"], config["projection_max_iterations"])
    ix = np.arange(len(first))
    ratio = joint[ix, first_group, second_group] / (first[ix, first_group] * second[ix, second_group])
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("Invalid observed joint correction")
    return np.log(ratio), error


def aligned(observations, frames, ids):
    keys = pd.MultiIndex.from_arrays([np.asarray(frames), np.asarray(ids)])
    result = observations.reindex(keys)
    if result.xVelocity.isna().any():
        raise ValueError("Missing paired observation")
    return result


def old_modifier(first, second, old):
    values = [np.abs(first.xVelocity.to_numpy()), first.dhw.to_numpy(),
              np.abs(second.xVelocity.to_numpy()) - np.abs(first.xVelocity.to_numpy())]
    states = []
    for value, key in zip(values, ("speed", "gap", "range_rate")):
        axis = old[key + "_axis"]
        if (value < axis[0]).any() or (value > axis[-1]).any():
            raise ValueError("Old following candidate lacks current state")
        states.append(np.rint((value - axis[0]) / (axis[1] - axis[0])).astype(int))
    coarse = old["joint_probability"][tuple(states)].astype(float)
    coarse /= coarse.sum(axis=(1, 2), keepdims=True)
    return coarse / (coarse.sum(axis=2)[:, :, None] * coarse.sum(axis=1)[:, None, :])


def following_score(folder, observations, marginal, old, config, pairs=None):
    if pairs is None:
        pairs = pd.read_csv(folder / "following.csv.gz")
    pairs = pairs.loc[pairs.both_state_and_action_supported]
    loss0, gain, error = 0.0, 0.0, 0.0
    batch = config["batch_size"]
    for start in range(0, len(pairs), batch):
        part = pairs.iloc[start:start + batch]
        first = aligned(observations, part.frame, part.follower_id)
        second = aligned(observations, part.frame, part.leader_id)
        p, f, a = marginal.query(first)
        q, s, b = marginal.query(second)
        delta, err = score_correction(f, s, a, b, old_modifier(first, second, old), config)
        loss0 -= float((np.log(p) + np.log(q)).sum())
        gain += float(delta.sum())
        error = max(error, err)
    n = len(pairs)
    return {"pair_rows": n, "factorized_31x31_nll": loss0 / n,
            "old_candidate_frozen_marginal_31x31_nll": (loss0 - gain) / n,
            "mean_log_likelihood_gain": gain / n, "max_marginal_error": error,
            "scope": "Training-sample comparison; both existing artifacts saw these recordings."}


def response_data(folder, observations, marginal, config, fps, tables=None):
    if tables is None:
        windows = pd.read_csv(folder / "windows.csv.gz")
        events = pd.read_csv(folder / "events.csv.gz")
    else:
        windows, events = tables
    valid = (windows.both_state_supported & windows.both_action_supported
             & (windows.relative_time_to_crossing_s >= 0)
             & (windows.relative_time_to_crossing_s <= config["response_horizon_s"])
             & clock_mask(windows.frame, fps, config["analysis_hz"]))
    rows = windows.loc[valid].merge(events[["event_id", "side", "target_lane"]], on="event_id", validate="many_to_one")
    first = aligned(observations, rows.frame, rows.id).reset_index(drop=True)
    second = aligned(observations, rows.frame, rows.rear_id).reset_index(drop=True)
    rows = rows.reset_index(drop=True)
    # At this instant the selected rear must actually follow the actor that just crossed.
    eligible = ((second.precedingId.to_numpy() == rows.id.to_numpy())
                & (first.laneId.to_numpy() == rows.target_lane.to_numpy())
                & (second.laneId.to_numpy() == rows.target_lane.to_numpy())
                & np.isclose(first.seconds_since_observed_lane_change.to_numpy(), rows.relative_time_to_crossing_s.to_numpy())
                & (second.dhw.to_numpy() >= 0) & (second.dhw.to_numpy() <= config["response_maximum_gap_m"]))
    rows, first, second = [x.loc[eligible].reset_index(drop=True) for x in (rows, first, second)]
    if rows.duplicated(["frame", "id", "rear_id"]).any():
        raise ValueError("Overlapping post-crossing episodes require disambiguation")
    p, f, a = marginal.query(first)
    q, s, b = marginal.query(second)
    state = (rows.side.eq("right").to_numpy(int) * 3
             + np.searchsorted(config["response_gap_cuts_m"], second.dhw.to_numpy(), side="right"))
    weights = 1.0 / rows.groupby("event_id").event_id.transform("size").to_numpy(float)
    observed, expected = np.zeros((6, 3, 3)), np.zeros((6, 3, 3))
    np.add.at(observed, (state, a, b), weights)
    np.add.at(expected, state, weights[:, None, None] * f[:, :, None] * s[:, None, :])
    return {"observed": observed, "expected": expected, "first": f, "second": s, "a": a, "b": b,
            "state": state, "weights": weights, "baseline_nll": -np.log(p) - np.log(q),
            "event_count": int(rows.event_id.nunique()), "row_count": len(rows),
            "events_by_state": [int(rows.loc[state == k, "event_id"].nunique()) for k in range(6)]}


def fit_modifier(observed, expected, shrinkage):
    """Shrink observed/expected joint enrichment towards independence."""
    total = expected.sum(axis=(1, 2), keepdims=True)
    prior = np.divide(expected, total, out=np.full_like(expected, 1.0 / 9), where=total > 0)
    return (observed + shrinkage * prior) / (expected + shrinkage * prior)


def fit_response_modifier(observed, expected, shrinkage, structure):
    if structure == "global":
        ratio = fit_modifier(observed.sum(axis=0, keepdims=True), expected.sum(axis=0, keepdims=True), shrinkage)
        return np.repeat(ratio, 6, axis=0)
    if structure == "side":
        ratio = fit_modifier(observed.reshape(2, 3, 3, 3).sum(axis=1),
                             expected.reshape(2, 3, 3, 3).sum(axis=1), shrinkage)
        return np.repeat(ratio, 3, axis=0)
    if structure == "side_gap":
        return fit_modifier(observed, expected, shrinkage)
    raise ValueError("Unknown response structure")


def response_score(data, modifier, config):
    if not data["row_count"]:
        return {"rows": 0, "events": 0, "factorized_nll": None, "candidate_nll": None, "gain": None, "max_marginal_error": 0.0}
    delta, err = score_correction(data["first"], data["second"], data["a"], data["b"], modifier[data["state"]], config)
    weights = data["weights"]
    baseline = float(np.average(data["baseline_nll"], weights=weights))
    gain = float(np.average(delta, weights=weights))
    return {"rows": data["row_count"], "events": data["event_count"], "factorized_nll": baseline,
            "candidate_nll": baseline - gain, "gain": gain, "max_marginal_error": err}


def run(input_root, output, config):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    contract = read_json(config["freeze_contract"])
    exported = read_json(input_root / "export_summary_10.json")
    if not exported["all_train_recordings_exported"] or set(exported["recording_ids"]) != set(contract["train_recordings"]):
        raise ValueError("Expected frozen train split only")
    marginal = FrozenMarginal(contract, config["action_thresholds_mps2"])
    if digest(config["old_following_model"]) != config["old_following_sha256"]:
        raise ValueError("Old candidate changed")
    with np.load(config["old_following_model"], allow_pickle=False) as artifact:
        old = {k: artifact[k] for k in artifact.files}
    response, following = {}, []
    for rec in exported["recording_ids"]:
        folder = input_root / "recordings" / rec
        meta = read_json(folder / "summary.json")
        obs = pd.read_csv(folder / "observations.csv.gz").set_index(["frame", "id"])
        f = following_score(folder, obs, marginal, old, config)
        following.append({"recording_id": rec, **f})
        response[rec] = response_data(folder, obs, marginal, config, meta["frame_rate"])
        print(json.dumps({"recording": rec, "following_gain": f["mean_log_likelihood_gain"],
                          "response_rows": response[rec]["row_count"], "response_events": response[rec]["event_count"]}), flush=True)
    observed = sum(d["observed"] for d in response.values())
    expected = sum(d["expected"] for d in response.values())
    candidates = []
    modifiers = {}
    for structure in config.get("response_structures", ["side_gap"]):
        modifier = fit_response_modifier(observed, expected, config["response_shrinkage_event_equivalents"], structure)
        modifiers[structure] = modifier
        crossfit, training = [], []
        for rec, data in response.items():
            fold = fit_response_modifier(np.maximum(observed - data["observed"], 0), np.maximum(expected - data["expected"], 0),
                                         config["response_shrinkage_event_equivalents"], structure)
            crossfit.append({"recording_id": rec, **response_score(data, fold, config)})
            training.append({"recording_id": rec, **response_score(data, modifier, config)})
        event_total = sum(x["events"] for x in crossfit)
        candidates.append({"structure": structure,
                           "event_weighted_crossfit_gain": sum(x["gain"] * x["events"] for x in crossfit if x["events"]) / event_total,
                           "recordings_improved": sum(x["gain"] > 0 for x in crossfit if x["events"]),
                           "crossfit": crossfit, "training": training})
    best = max(candidates, key=lambda x: x["event_weighted_crossfit_gain"])
    selected_structure = best["structure"] if best["event_weighted_crossfit_gain"] > 0 else "factorized"
    modifier = modifiers[selected_structure] if selected_structure != "factorized" else np.ones((6, 3, 3))
    crossfit, training = best["crossfit"], best["training"]
    n = sum(x["pair_rows"] for x in following)
    events = sum(x["events"] for x in crossfit)
    aggregate = {"following_pair_rows": n,
                 "following_training_nll_gain": sum(x["mean_log_likelihood_gain"] * x["pair_rows"] for x in following) / n,
                 "following_recordings_improved": sum(x["mean_log_likelihood_gain"] > 0 for x in following),
                 "response_rows": sum(x["rows"] for x in crossfit), "response_events": events,
                 "response_crossfit_event_weighted_gain": sum(x["gain"] * x["events"] for x in crossfit if x["events"]) / events,
                 "response_crossfit_recordings_improved": sum(x["gain"] > 0 for x in crossfit if x["events"]),
                 "response_training_event_weighted_gain": sum(x["gain"] * x["events"] for x in training if x["events"]) / events,
                 "best_fitted_response_structure": best["structure"],
                 "selected_response_structure": selected_structure,
                 "response_metrics_describe": "best fitted candidate; selected factorized has zero gain if all fitted candidates are worse"}
    output.mkdir(parents=True)
    artifact = output / "post_crossing_dependence_v1.npz"
    np.savez_compressed(artifact, modifier=modifier, observed_event_weighted_counts=observed,
                        expected_event_weighted_counts=expected, gap_cuts=np.array(config["response_gap_cuts_m"]),
                        action_thresholds=np.array(config["action_thresholds_mps2"]),
                        shrinkage=np.array(config["response_shrinkage_event_equivalents"]),
                        selected_structure=np.array(selected_structure),
                        **{k + "_modifier": v for k, v in modifiers.items()})
    report = {"schema_version": 1, "config": config, "train_recordings": exported["recording_ids"],
              "aggregate": aggregate, "following_same_sample_scores": following,
              "response_crossfit": crossfit, "response_training": training,
              "response_candidates": candidates,
              "response_events_by_state": {r: d["events_by_state"] for r, d in response.items()},
              "model_sha256": digest(artifact), "frozen_marginal_sha256": contract["model"]["sha256"],
              "script_sha256": digest(__file__), "scope": "Following scores are in-sample. Response correction is leave-one-recording-out, but frozen single marginals were fitted on all train recordings. Not independent holdout validation. Conditional post-crossing longitudinal model only; no lane-change initiation probability or runtime deployment."}
    write_new(output / "model_fit_summary.json", report)
    print(json.dumps(aggregate, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/highd_pair_model_fit_v1.json"))
    args = p.parse_args()
    run(args.input, args.output, read_json(args.config))


if __name__ == "__main__":
    main()
