"""Convert a measured SHRP2 reference into a documented initial-state seed."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def reference_to_seed(
    reference: dict[str, Any],
    duration_s: float = 6.0,
    sample_hz: int = 10,
) -> dict[str, Any]:
    """Create a kinematic initialization seed without claiming trajectory replay."""
    if duration_s <= 0 or sample_hz <= 0:
        raise ValueError("duration_s and sample_hz must be positive")
    source = reference.get("source") or {}
    actors = reference.get("actors") or {}
    if not source.get("event_id") or not source.get("associated_target_id"):
        raise ValueError("reference source lacks event_id or associated_target_id")
    if "CAV" not in actors or "BV_primary" not in actors:
        raise ValueError("reference must contain CAV and BV_primary actors")
    cav = actors["CAV"]
    target = actors["BV_primary"]
    for actor_id, actor in (("CAV", cav), ("BV_primary", target)):
        for key in ("xy_m", "reported_speed_mps", "reported_heading_rad"):
            if not actor.get(key):
                raise ValueError(f"{actor_id} reference lacks {key}")
    collision_audit = reference.get("collision_audit") or {}
    impact_time = collision_audit.get("first_sampled_contact_time_s")
    if (
        isinstance(impact_time, bool)
        or not isinstance(impact_time, (int, float))
        or not math.isfinite(impact_time)
        or impact_time <= 0
    ):
        raise ValueError(
            "reference collision_audit lacks a positive first_sampled_contact_time_s"
        )
    times = np.arange(0.0, duration_s + 0.5 / sample_hz, 1.0 / sample_hz)
    records = []
    for actor_id, actor in (("CAV", cav), ("BV_primary", target)):
        initial_xy = np.asarray(actor["xy_m"][0], dtype=float)
        speed = max(0.0, float(actor["reported_speed_mps"][0]))
        heading = float(actor["reported_heading_rad"][0])
        direction = np.array([math.cos(heading), math.sin(heading)])
        xy = initial_xy[None, :] + times[:, None] * speed * direction[None, :]
        row = {
            "id": actor_id,
            "role": "CAV" if actor_id == "CAV" else "BV",
            "length_m": float(actor["length_m"]),
            "width_m": float(actor["width_m"]),
            "heading_rad": heading,
            "speed_mps": speed,
            "xy_m": xy.tolist(),
        }
        if actor_id == "BV_primary":
            row["source_target_id"] = int(actor["source_target_id"])
        records.append(row)
    return {
        "schema_version": 1,
        "scenario_id": f"shrp2_{source['event_id']}_rear_end_reference_initial_state",
        "status": (
            "SHRP2 reference-conditioned initial-state seed; constant-speed extrapolation "
            "is used only for bridge initialization, not exact replay"
        ),
        "source": {
            "dataset": source.get("dataset"),
            "doi": source.get("doi"),
            "event_id": int(source["event_id"]),
            "source_split": source.get("source_split"),
            "source_conflict": "leading",
            "associated_target_id": int(source["associated_target_id"]),
            "association_rule": "selected by SHRP2 candidate trajectory quality audit",
            "reference_record_type": reference.get("record_type"),
        },
        "simulation_type": "rear_end",
        "variant": {
            "name": "reference_initial_state",
            "ego_speed_scale": 1.0,
            "target_speed_scale": 1.0,
        },
        "impact_conditioning": {
            "requested_impact_time_s": float(impact_time),
            "source": "collision_audit.first_sampled_contact_time_s",
            "measurement": "first 10 Hz resampled oriented-box contact; no between-sample interpolation",
            "quality": "audited",
            "quality_rule": "selected by source position/speed/heading consistency",
        },
        "time_s": times.tolist(),
        "actors": records,
        "bridge_policy": {
            "uses": "initial positions, speeds and headings only",
            "does_not_use": "pointwise reference coordinates after initialization",
            "next_step": "fit SUMO controller/fault parameters against the soft reference",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration_s", type=float, default=6.0)
    parser.add_argument("--sample_hz", type=int, default=10)
    parser.add_argument("--bridge_config")
    parser.add_argument("--bridge_output")
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    seed = reference_to_seed(reference, args.duration_s, args.sample_hz)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(seed, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if args.bridge_config or args.bridge_output:
        if not args.bridge_config or not args.bridge_output:
            parser.error("--bridge_config and --bridge_output must be provided together")
        from .shrp2_sumo_bridge import rear_end_template_from_seed

        bridge_config = json.loads(Path(args.bridge_config).read_text(encoding="utf-8"))
        template = rear_end_template_from_seed(seed, bridge_config)
        bridge_output = Path(args.bridge_output)
        bridge_output.parent.mkdir(parents=True, exist_ok=True)
        bridge_output.write_text(
            json.dumps(template, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    print(json.dumps(seed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
