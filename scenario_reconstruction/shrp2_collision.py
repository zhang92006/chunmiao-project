"""Import public SHRP2 crash trajectories and create auditable collision seeds.

The generated records are kinematic, impact-conditioned simulations. They are
not exact crash reconstructions, SUMO closed-loop outcomes, D2RL samples, or
road-crash probability estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation", "test")
META_COLUMNS = {
    "event_id", "event_category", "ego_width", "ego_length",
    "target_width", "target_length", "impact_timestamp", "severity_first",
    "conflict", "duration_enough",
}
DATA_COLUMNS = {
    "target_id", "time", "event_id", "x_ego", "y_ego", "v_ego",
    "psi_ego", "x_sur", "y_sur", "v_sur", "psi_sur",
}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Only SHRP2 collision schema_version 1 is supported")
    if not isinstance(config.get("seed"), int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must lie in [0, 2**32)")
    fractions = config.get("split_fractions", {})
    if set(fractions) != set(SPLITS) or not all(
        np.isfinite(value) and 0 < value < 1 for value in fractions.values()
    ) or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-12):
        raise ValueError("split_fractions must contain positive train/validation/test values summing to one")
    if config.get("source_split") not in SPLITS:
        raise ValueError("source_split must be train, validation, or test")
    conflicts = config.get("conflict_types")
    if not isinstance(conflicts, dict) or not conflicts or any(not key or not value for key, value in conflicts.items()):
        raise ValueError("conflict_types must be a non-empty source-to-simulation mapping")
    positive = (
        "maximum_target_alignment_dt_s", "maximum_source_box_clearance_m",
        "duration_s", "impact_time_s", "sample_hz", "collision_substeps",
        "contact_penetration_m",
    )
    for name in positive:
        value = config.get(name)
        if isinstance(value, bool) or value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config["impact_time_s"] >= config["duration_s"]:
        raise ValueError("impact_time_s must be before duration_s")
    for name in ("sample_hz", "collision_substeps"):
        if int(config[name]) != config[name]:
            raise ValueError(f"{name} must be an integer")
    variants = config.get("variants")
    names = [variant.get("name") for variant in variants or []]
    if not variants or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("variants need unique non-empty names")
    for variant in variants:
        for name in ("ego_speed_scale", "target_speed_scale"):
            value = variant.get(name)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"variant {name} must be finite and positive")


def project_split(event_id: int, seed: int, fractions: dict[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}:{int(event_id)}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < fractions["train"]:
        return "train"
    if value < fractions["train"] + fractions["validation"]:
        return "validation"
    return "test"


def locate_export_online(source_root: str | Path) -> Path:
    source = Path(source_root).resolve()
    candidates = (source, source / "Export_Online", source / "extracted" / "Export_Online")
    for candidate in candidates:
        if (candidate / "SafetyCriticalTestSet").is_dir():
            return candidate
    raise FileNotFoundError("Could not find Export_Online/SafetyCriticalTestSet under source_root")


def load_category(source_root: str | Path, category: str):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("SHRP2 HDF5 import requires pandas and PyTables") from exc
    export_root = locate_export_online(source_root)
    folder = export_root / "SafetyCriticalTestSet" / category
    meta_path, data_path = folder / "event_meta.csv", folder / "event_data.h5"
    if not meta_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing SHRP2 category files under {folder}")
    metadata = pd.read_csv(meta_path)
    try:
        data = pd.read_hdf(data_path, key="data")
    except ImportError as exc:
        raise RuntimeError("SHRP2 HDF5 import requires PyTables (the 'tables' package)") from exc
    missing_meta, missing_data = META_COLUMNS - set(metadata.columns), DATA_COLUMNS - set(data.columns)
    if missing_meta or missing_data:
        raise ValueError(f"SHRP2 schema mismatch: metadata={sorted(missing_meta)}, data={sorted(missing_data)}")
    if metadata["event_id"].duplicated().any():
        raise ValueError("event_meta.csv contains duplicate event_id values")
    return export_root, meta_path, data_path, metadata, data


def front_bumper_to_center(x: float, y: float, heading: float, length: float) -> np.ndarray:
    return np.array([x, y], dtype=float) - 0.5 * float(length) * _heading(heading)


def box_vertices(center, heading: float, length: float, width: float) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    forward, lateral = _heading(heading), np.array([-math.sin(heading), math.cos(heading)])
    return np.asarray([
        center + sx * 0.5 * length * forward + sy * 0.5 * width * lateral
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ])


def signed_box_clearance(center_a, heading_a, length_a, width_a,
                         center_b, heading_b, length_b, width_b) -> float:
    first = box_vertices(center_a, heading_a, length_a, width_a)
    second = box_vertices(center_b, heading_b, length_b, width_b)
    axes = []
    for heading in (heading_a, heading_b):
        axes.extend((_heading(heading), np.array([-math.sin(heading), math.cos(heading)])))
    overlaps = []
    separated = False
    for axis in axes:
        left, right = first @ axis, second @ axis
        overlap = min(left.max(), right.max()) - max(left.min(), right.min())
        overlaps.append(float(overlap))
        separated |= overlap < 0
    if not separated:
        return -min(overlaps)
    distances = []
    for points, polygon in ((first, second), (second, first)):
        for point in points:
            for index in range(4):
                distances.append(_point_segment_distance(point, polygon[index], polygon[(index + 1) % 4]))
    return float(min(distances))


def associate_primary_target(event_rows, metadata_row, max_dt_s: float) -> dict[str, Any] | None:
    impact_s = float(metadata_row["impact_timestamp"]) / 1000.0
    dimensions = [metadata_row[name] for name in ("ego_length", "ego_width", "target_length", "target_width")]
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        return None
    candidates = []
    for target_id, rows in event_rows.groupby("target_id", sort=True):
        finite = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=[
            "time", "x_ego", "y_ego", "v_ego", "psi_ego", "x_sur", "y_sur", "v_sur", "psi_sur"
        ])
        if finite.empty:
            continue
        index = (finite["time"] - impact_s).abs().idxmin()
        row = finite.loc[index]
        dt = float(row["time"] - impact_s)
        if abs(dt) > max_dt_s:
            continue
        ego_center = np.array([row["x_ego"], row["y_ego"]], dtype=float)
        target_center = front_bumper_to_center(
            row["x_sur"], row["y_sur"], row["psi_sur"], metadata_row["target_length"]
        )
        clearance = signed_box_clearance(
            ego_center, row["psi_ego"], metadata_row["ego_length"], metadata_row["ego_width"],
            target_center, row["psi_sur"], metadata_row["target_length"], metadata_row["target_width"],
        )
        candidates.append({
            "target_id": int(target_id), "source_sample_dt_s": dt,
            "source_box_clearance_m": clearance,
            "ego_center_m": ego_center.tolist(), "target_center_m": target_center.tolist(),
            "ego_heading_rad": float(row["psi_ego"]), "target_heading_rad": float(row["psi_sur"]),
            "ego_speed_mps": max(0.0, float(row["v_ego"])),
            "target_speed_mps": max(0.0, float(row["v_sur"])),
            "ego_length_m": float(metadata_row["ego_length"]), "ego_width_m": float(metadata_row["ego_width"]),
            "target_length_m": float(metadata_row["target_length"]), "target_width_m": float(metadata_row["target_width"]),
        })
    return min(candidates, key=lambda item: (item["source_box_clearance_m"], abs(item["source_sample_dt_s"]), item["target_id"])) if candidates else None


def select_source_events(metadata, data, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected, counts = [], {}
    for source_conflict, simulation_type in config["conflict_types"].items():
        candidates = []
        subset = metadata[
            (metadata["conflict"] == source_conflict)
            & (metadata["severity_first"] == 3)
            & (metadata["duration_enough"] == True)  # noqa: E712
        ]
        for _, row in subset.sort_values("event_id").iterrows():
            event_id = int(row["event_id"])
            if project_split(event_id, config["seed"], config["split_fractions"]) != config["source_split"]:
                continue
            association = associate_primary_target(
                data[data["event_id"] == event_id], row, config["maximum_target_alignment_dt_s"]
            )
            if association is None or association["source_box_clearance_m"] > config["maximum_source_box_clearance_m"]:
                continue
            association.update({
                "event_id": event_id, "source_conflict": source_conflict,
                "simulation_type": simulation_type, "source_split": config["source_split"],
                "impact_timestamp": int(row["impact_timestamp"]),
            })
            _, _, shift = _conditioned_impact_geometry(
                association, config["contact_penetration_m"]
            )
            association["impact_conditioning_shift_m"] = shift
            association["impact_conditioning_quality"] = _conditioning_quality(shift)
            candidates.append(association)
        counts[source_conflict] = len(candidates)
        if not candidates:
            raise ValueError(f"No usable {source_conflict} events in project split {config['source_split']}")
        selected.append(min(candidates, key=lambda item: (
            item["impact_conditioning_shift_m"], abs(item["source_sample_dt_s"]), item["event_id"]
        )))
    return selected, counts


def simulate_collision(source: dict[str, Any], variant: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target_impact, target_heading, conditioning_shift = _conditioned_impact_geometry(
        source, config["contact_penetration_m"]
    )
    target_heading = _wrap_angle(source["target_heading_rad"] - source["ego_heading_rad"])
    ego_impact = np.zeros(2)
    ego_speed = source["ego_speed_mps"] * variant["ego_speed_scale"]
    target_speed = source["target_speed_mps"] * variant["target_speed_scale"]
    conditioning_quality = _conditioning_quality(conditioning_shift)
    times = np.arange(0.0, config["duration_s"] + 0.5 / config["sample_hz"], 1.0 / config["sample_hz"])
    ego_velocity = ego_speed * np.array([1.0, 0.0])
    target_velocity = target_speed * _heading(target_heading)
    ego_xy = ego_impact + (times - config["impact_time_s"])[:, None] * ego_velocity
    target_xy = target_impact + (times - config["impact_time_s"])[:, None] * target_velocity
    collision = _collision_audit(
        times, ego_xy, target_xy, 0.0, target_heading,
        source["ego_length_m"], source["ego_width_m"],
        source["target_length_m"], source["target_width_m"],
        config["collision_substeps"],
    )
    return {
        "schema_version": 1,
        "scenario_id": f"shrp2_{source['event_id']}_{source['simulation_type']}_{variant['name']}",
        "status": "SHRP2-conditioned kinematic collision seed; not exact replay, SUMO, D2RL, or a probability estimate",
        "source": {
            "dataset": "SHRP2 NDS public bird's-eye trajectory reconstruction",
            "doi": config["dataset_doi"], "event_id": source["event_id"],
            "source_split": source["source_split"], "source_conflict": source["source_conflict"],
            "associated_target_id": source["target_id"],
            "association_rule": "minimum oriented-box clearance near annotated impact",
            "source_sample_dt_s": source["source_sample_dt_s"],
            "source_box_clearance_m": source["source_box_clearance_m"],
            "target_coordinate_conversion": "front bumper minus half target length along target heading",
        },
        "simulation_type": source["simulation_type"], "variant": variant,
        "impact_conditioning": {
            "requested_impact_time_s": config["impact_time_s"],
            "contact_penetration_m": config["contact_penetration_m"],
            "target_center_shift_from_source_at_impact_m": conditioning_shift,
            "quality": conditioning_quality,
            "quality_rule": "high <= 1 m; medium <= 3 m; low > 3 m",
        },
        "time_s": times.tolist(),
        "actors": [
            {
                "id": "CAV", "role": "CAV", "length_m": source["ego_length_m"], "width_m": source["ego_width_m"],
                "heading_rad": 0.0, "speed_mps": ego_speed, "xy_m": ego_xy.tolist(),
            },
            {
                "id": "BV_primary", "role": "BV", "source_target_id": source["target_id"],
                "length_m": source["target_length_m"], "width_m": source["target_width_m"],
                "heading_rad": target_heading, "speed_mps": target_speed, "xy_m": target_xy.tolist(),
            },
        ],
        "collision_audit": collision,
    }


def run_pipeline(source_root: str | Path, output: str | Path, config: dict[str, Any], allow_test: bool = False) -> dict[str, Any]:
    validate_config(config)
    if config["source_split"] == "test" and not allow_test:
        raise ValueError("Project test split is locked; set allow_test only after the pipeline is frozen")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    export_root, meta_path, data_path, metadata, data = load_category(source_root, config["source_category"])
    selected, counts = select_source_events(metadata, data, config)
    records = []
    for source in selected:
        for variant in config["variants"]:
            record = simulate_collision(source, variant, config)
            records.append(record)
            _write_json(output / "scenarios" / f"{record['scenario_id']}.json", record)
    collision_time_errors = [
        record["collision_audit"]["first_collision_time_s"] - config["impact_time_s"]
        for record in records if record["collision_audit"]["collision"]
    ]
    summary = {
        "status": "SHRP2-conditioned kinematic collision pilot; not exact replay, closed-loop SUMO, D2RL, or a crash-rate estimate",
        "dataset": "SHRP2 NDS public bird's-eye trajectory reconstruction",
        "dataset_doi": config["dataset_doi"], "license": "CC BY 4.0",
        "source_category": config["source_category"], "source_split": config["source_split"],
        "test_split_used": bool(config["source_split"] == "test"),
        "source_file_sha256": {"event_meta.csv": _file_hash(meta_path), "event_data.h5": _file_hash(data_path)},
        "implementation_sha256_normalized_lf": _text_file_hash(Path(__file__)),
        "candidate_events_by_conflict": counts,
        "selected_source_events": [{key: source[key] for key in (
            "event_id", "target_id", "source_conflict", "simulation_type", "source_box_clearance_m",
            "source_sample_dt_s", "impact_conditioning_shift_m", "impact_conditioning_quality"
        )} for source in selected],
        "scenario_count": len(records),
        "verified_collision_scenarios": sum(record["collision_audit"]["collision"] for record in records),
        "initially_colliding_scenarios": sum(record["collision_audit"]["initial_collision"] for record in records),
        "first_collision_time_error_s": {
            "minimum": min(collision_time_errors) if collision_time_errors else None,
            "maximum": max(collision_time_errors) if collision_time_errors else None,
        },
        "conditioning_quality_counts": {
            quality: sum(record["impact_conditioning"]["quality"] == quality for record in records)
            for quality in ("high", "medium", "low")
        },
        "simulation_types": sorted({record["simulation_type"] for record in records}),
        "variant_names": [variant["name"] for variant in config["variants"]],
        "config": config,
        "limitations": [
            "The associated radar target is selected geometrically because the public file has no explicit primary target_id link.",
            "Impact contact is conditioned by shifting the selected target at impact; this is not exact accident replay.",
            "Actors follow constant-heading constant-speed kinematics and have not yet been replayed in SUMO.",
            "Trajectories after first contact are nonphysical and must not be used as a collision-dynamics model.",
            "The deterministic split is event-disjoint, not driver-disjoint, because no driver key is published here.",
            "No NDD or proposal action probabilities are inferred from SHRP2 trajectories.",
        ],
    }
    _write_json(output / "manifest.json", {
        "dataset_doi": config["dataset_doi"], "source_root": str(export_root),
        "records": [{"scenario_id": record["scenario_id"], "path": f"scenarios/{record['scenario_id']}.json"} for record in records],
    })
    _write_json(output / "summary.json", summary)
    return summary


def _collision_audit(times, ego_xy, target_xy, ego_heading, target_heading,
                     ego_length, ego_width, target_length, target_width, substeps):
    minimum, first = math.inf, None
    for index in range(len(times) - 1):
        for step in range(substeps + 1):
            ratio = step / substeps
            time = float(times[index] + ratio * (times[index + 1] - times[index]))
            ego = ego_xy[index] + ratio * (ego_xy[index + 1] - ego_xy[index])
            target = target_xy[index] + ratio * (target_xy[index + 1] - target_xy[index])
            clearance = signed_box_clearance(
                ego, ego_heading, ego_length, ego_width,
                target, target_heading, target_length, target_width,
            )
            minimum = min(minimum, clearance)
            if clearance <= 0 and first is None:
                first = time
    initial = signed_box_clearance(
        ego_xy[0], ego_heading, ego_length, ego_width,
        target_xy[0], target_heading, target_length, target_width,
    ) <= 0
    return {"collision": first is not None, "first_collision_time_s": first,
            "minimum_signed_clearance_m": float(minimum), "initial_collision": bool(initial),
            "method": f"oriented boxes with {substeps} linear substeps per sample interval"}


def _contact_distance(direction, heading_a, length_a, width_a, heading_b, length_b, width_b):
    low, high = 0.0, 2.0 * (length_a + width_a + length_b + width_b)
    for _ in range(60):
        middle = 0.5 * (low + high)
        clearance = signed_box_clearance(
            np.zeros(2), heading_a, length_a, width_a,
            np.asarray(direction) * middle, heading_b, length_b, width_b,
        )
        if clearance <= 0:
            low = middle
        else:
            high = middle
    return low


def _conditioned_impact_geometry(source, penetration_m):
    rotate = _rotation(-source["ego_heading_rad"])
    source_relative = rotate @ (
        np.asarray(source["target_center_m"]) - np.asarray(source["ego_center_m"])
    )
    target_heading = _wrap_angle(source["target_heading_rad"] - source["ego_heading_rad"])
    relative_velocity = (
        source["target_speed_mps"] * _heading(target_heading)
        - source["ego_speed_mps"] * np.array([1.0, 0.0])
    )
    if np.linalg.norm(relative_velocity) > 1e-9:
        direction = -relative_velocity / np.linalg.norm(relative_velocity)
    elif np.linalg.norm(source_relative) > 1e-9:
        direction = source_relative / np.linalg.norm(source_relative)
    else:
        direction = np.array([1.0, 0.0])
    distance = _contact_distance(
        direction, 0.0, source["ego_length_m"], source["ego_width_m"],
        target_heading, source["target_length_m"], source["target_width_m"],
    )
    target_impact = direction * max(0.0, distance - penetration_m)
    shift = float(np.linalg.norm(target_impact - source_relative))
    return target_impact, target_heading, shift


def _conditioning_quality(shift_m):
    if shift_m <= 1.0:
        return "high"
    if shift_m <= 3.0:
        return "medium"
    return "low"


def _point_segment_distance(point, start, end):
    segment = end - start
    scale = float(np.dot(segment, segment))
    if scale <= 1e-18:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, segment) / scale, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * segment)))


def _heading(angle):
    return np.array([math.cos(float(angle)), math.sin(float(angle))])


def _rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def _wrap_angle(angle):
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_file_hash(path):
    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/shrp2_collision_pilot.json")
    parser.add_argument("--allow_test", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run_pipeline(args.source_root, args.output, config, allow_test=args.allow_test)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
