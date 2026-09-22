"""Descriptive native-highD versus closed-loop metrics on matched BV/time windows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


def range_matched(row, observation_range_m):
    """Keep raw measurements untouched; produce the simulator-visible view."""
    row = dict(row)
    if row["gap_m"] is not None and row["gap_m"] > observation_range_m:
        for key in ("gap_m", "headway_s", "ttc_s", "leader_id"):
            row[key] = None
    return row


def describe_pairs(simulated, measured):
    """Marginal distances and distances on IDENTICAL finite paired states."""
    def distribution(values):
        values = np.asarray(values, dtype=float)
        return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
                "p05": float(np.quantile(values, .05)) if len(values) else None,
                "p95": float(np.quantile(values, .95)) if len(values) else None}

    def finite(value):
        return value is not None and np.isfinite(value)

    metrics = {}
    for key in ("speed_mps", "acceleration_mps2", "gap_m", "headway_s", "ttc_s"):
        sim_values = [r[key] for r in simulated if finite(r[key])]
        ref_values = [r[key] for r in measured if finite(r[key])]
        common = [(a[key], b[key]) for a, b in zip(simulated, measured)
                  if finite(a[key]) and finite(b[key])]
        metrics[key] = {"simulation": distribution(sim_values), "highd": distribution(ref_values),
            "wasserstein_distance": float(wasserstein_distance(sim_values, ref_values))
                if sim_values and ref_values else None,
            "common_finite_pair_count": len(common),
            "common_finite_wasserstein_distance": float(wasserstein_distance(*zip(*common)))
                if common else None,
            "paired_mean_absolute_error": float(np.mean([abs(a-b) for a, b in common]))
                if common else None}
    return metrics


def compare_native(manifest_path, rollout_root, output):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    references = {}
    for record in manifest["records"]:
        template = json.loads(Path(record["template_path"]).read_text(encoding="utf-8"))
        references[template["template_id"]] = (record["reference_path"], template)
    simulated, measured, raw_measured = [], [], []
    crossing_counts = {"simulation": 0, "highd": 0}
    exposure = 0.0
    episode_count = 0
    bv_windows = 0
    visibility = Counter()
    initialization_visibility = Counter()
    excluded_support_windows = 0
    missing_support_observations = 0
    observation_ranges = set()
    termination_counts = Counter()
    sampled_support_count = 0
    for path in sorted(Path(rollout_root).glob("episode_*/naturalistic_episode.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        if episode["metadata"].get("horizon_diagnostic"):
            raise ValueError("Horizon extrapolation is not measured-reference validation")
        audit = json.loads((path.parent / "naturalistic_audit.json").read_text(encoding="utf-8"))
        if not audit.get("probability_audit_passed") or not audit.get("model_reconstruction_checked"):
            raise ValueError("Episode lacks a successful model-probability audit")
        reference_path, template = references[episode["metadata"]["template_id"]]
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        if not episode["summary"]["probability_audit_passed"]:
            raise ValueError("Refusing to compare a failed probability audit")
        observation_range = episode["metadata"].get("observation_range_m", 115.0)
        if observation_range != template["bridge_metadata"].get("observation_range_m", 115.0):
            raise ValueError("Reference and simulator observation ranges disagree")
        observation_ranges.add(observation_range)
        episode_count += 1
        termination_counts.update([str(episode.get("termination", {}).get("reason"))])
        sampled_support_count += len(episode["metadata"].get("support_actor_ids", []))
        present = {r["vehicle_id"] for r in episode["snapshots"] if r["vehicle_id"].startswith("BV_")}
        actors = sorted(reference.get("evaluation_actor_ids", present))
        if not set(actors).issubset(present):
            raise ValueError("Simulation lost an evaluated core actor")
        excluded_support_windows += len(present - set(actors))
        for actor in actors:
            sim = sorted((r for r in episode["snapshots"] if r["vehicle_id"] == actor), key=lambda r: r["time"])
            ref = sorted((r for r in reference["snapshots"] if r["vehicle_id"] == actor), key=lambda r: r["time"])
            if not ref:
                raise ValueError(f"No measured reference for {actor}")
            bv_windows += 1
            # No interpolation across changes of preceding ID: nearest original 25Hz state.
            times = np.array([r["time"] for r in ref])
            paired_ref = [ref[int(np.argmin(abs(times - r["time"])))] for r in sim]
            if any(abs(a["time"] - b["time"]) > .020001 for a, b in zip(sim, paired_ref)):
                raise ValueError("Reference does not cover simulation time")
            visible_ref = [range_matched(r, observation_range) for r in paired_ref]
            simulated.extend(sim)
            measured.extend(visible_ref)
            raw_measured.extend(paired_ref)
            for index, (a, b, raw) in enumerate(zip(sim, visible_ref, paired_ref)):
                state = f"sim_{int(a['gap_m'] is not None)}_highd_{int(b['gap_m'] is not None)}"
                visibility[state] += 1
                if index == 0:
                    initialization_visibility[state] += 1
                if b["gap_m"] is not None and raw.get("source_leader_in_template") is False:
                    missing_support_observations += 1
            exposure += sim[-1]["time"] - sim[0]["time"]
            for key, rows in (("simulation", sim), ("highd", paired_ref)):
                crossing_counts[key] += sum(a["lane_index"] != b["lane_index"]
                                            for a, b in zip(rows, rows[1:]))
    if not simulated:
        raise ValueError("No native closed-loop episodes")

    result = {"schema_version": 2, "episode_count": episode_count,
        "bv_windows": bv_windows, "paired_state_count": len(simulated),
        "excluded_support_bv_windows": excluded_support_windows,
        "initialized_support_bv_windows": sampled_support_count,
        "termination_counts": dict(termination_counts),
        "observation_ranges_m": sorted(observation_ranges),
        "observed_bv_seconds": exposure, "lane_crossings": crossing_counts,
        "lead_observation_fraction": {"simulation": sum(r["gap_m"] is not None for r in simulated)/len(simulated),
            "highd_range_matched": sum(r["gap_m"] is not None for r in measured)/len(measured),
            "highd_raw": sum(r["gap_m"] is not None for r in raw_measured)/len(raw_measured)},
        "lead_visibility_pairs": dict(visibility),
        "initial_lead_visibility_pairs": dict(initialization_visibility),
        "visible_reference_leader_not_initialized_states": missing_support_observations,
        "metrics": describe_pairs(simulated, measured),
        "scope": "Descriptive local-window check with IDM CAV; length mapping is recorded in each episode. No equivalence claim or population crash-rate estimate."}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("rollout_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_native(args.manifest, args.rollout_root, args.output), indent=2))


if __name__ == "__main__":
    main()
