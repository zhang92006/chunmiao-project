"""Map SHRP2 kinematic collision seeds to executable 2Lane SUMO templates.

Only same-direction rear-end seeds are mapped by this module.  Crossing and
opposing-turn seeds require road networks that the current project does not
ship, while low-quality seeds are intentionally excluded from primary studies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .templates import ScenarioTemplate


QUALITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Only SHRP2 SUMO bridge schema_version 1 is supported")
    qualities = config.get("allowed_quality")
    if not isinstance(qualities, list) or not qualities or any(
        quality not in QUALITY_ORDER for quality in qualities
    ):
        raise ValueError("allowed_quality must contain one or more known quality tiers")
    if "low" in qualities:
        raise ValueError("Low-quality SHRP2 seeds cannot enter the primary SUMO bridge")
    if config.get("mappable_types") != ["rear_end"]:
        raise ValueError("The current 2Lane bridge supports only mappable_types=['rear_end']")
    for name in ("duration_s", "cav_position_m", "context_gap_m", "context_speed_mps"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if config["duration_s"] <= 0 or config["context_gap_m"] <= 0:
        raise ValueError("duration_s and context_gap_m must be positive")


def bridge_seed_directory(
    seed_root: str | Path,
    output: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create executable templates for directly mappable SHRP2 seed records."""
    validate_config(config)
    seed_root = Path(seed_root)
    scenario_dir = seed_root / "scenarios"
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Missing SHRP2 seed directory: {scenario_dir}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")

    records = []
    for source_path in sorted(scenario_dir.glob("*.json")):
        record = json.loads(source_path.read_text(encoding="utf-8"))
        decision = _bridge_one(record, source_path, output, config)
        records.append(decision)

    summary = {
        "status": "SHRP2-to-SUMO bridge pilot; only created templates may be executed in the current 2Lane runtime",
        "seed_root": str(seed_root.resolve()),
        "source_summary": _load_source_summary(seed_root),
        "allowed_quality": config["allowed_quality"],
        "mappable_types": config["mappable_types"],
        "record_count": len(records),
        "template_count": sum(item["status"] == "template_created" for item in records),
        "blocked_count": sum(item["status"] == "blocked" for item in records),
        "excluded_count": sum(item["status"] == "excluded" for item in records),
        "records": records,
        "limitations": [
            "A created template preserves the SHRP2 seed's initial longitudinal gap and speeds after projection to 2Lane coordinates.",
            "It is a SUMO initialization candidate, not an exact crash replay and not a guaranteed collision.",
            "The current project has no validated intersection or opposing-direction network for crossing and opposing-turn seeds.",
            "Low-quality seeds are excluded from the primary bridge by configuration.",
        ],
    }
    _write_json(output / "bridge_summary.json", summary)
    return summary


def _bridge_one(
    record: dict[str, Any], source_path: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(record.get("scenario_id", source_path.stem))
    simulation_type = str(record.get("simulation_type", ""))
    quality = str(record.get("impact_conditioning", {}).get("quality", ""))
    decision = {
        "scenario_id": scenario_id,
        "source_path": str(source_path),
        "simulation_type": simulation_type,
        "quality": quality,
    }
    if quality not in config["allowed_quality"]:
        decision.update({
            "status": "excluded",
            "reason": f"quality={quality!r} is outside allowed_quality={config['allowed_quality']!r}",
        })
        return decision
    if simulation_type != "rear_end":
        decision.update({
            "status": "blocked",
            "reason": _topology_requirement(simulation_type),
        })
        return decision

    template = rear_end_template_from_seed(record, config)
    template_path = output / "templates" / f"{template['template_id']}.json"
    _write_json(template_path, template)
    ScenarioTemplate.from_dict(template)
    decision.update({
        "status": "template_created",
        "template_path": str(template_path),
        "map": "2Lane",
        "projection": template["bridge_metadata"],
    })
    return decision


def rear_end_template_from_seed(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Project a same-direction kinematic rear-end seed into 2Lane coordinates."""
    if record.get("simulation_type") != "rear_end":
        raise ValueError("rear_end_template_from_seed accepts only rear_end seeds")
    actors = record.get("actors", [])
    if len(actors) != 2 or actors[0].get("id") != "CAV" or actors[1].get("id") != "BV_primary":
        raise ValueError("SHRP2 seed must contain CAV and BV_primary actors")
    cav, primary = actors
    cav_xy, primary_xy = cav.get("xy_m", []), primary.get("xy_m", [])
    if not cav_xy or not primary_xy or len(cav_xy[0]) < 2 or len(primary_xy[0]) < 2:
        raise ValueError("SHRP2 seed lacks initial two-dimensional actor coordinates")
    source_gap_m = float(primary_xy[0][0]) - float(cav_xy[0][0])
    if source_gap_m <= 5.0:
        raise ValueError("Projected rear-end seed must place BV_primary safely ahead of CAV")

    cav_position = float(config["cav_position_m"])
    primary_position = cav_position + source_gap_m
    context_position = max(0.0, cav_position - float(config["context_gap_m"]))
    source = record.get("source", {})
    conditioning = record.get("impact_conditioning", {})
    template_id = f"sumo_{record['scenario_id']}"
    return {
        "template_id": template_id,
        "description": (
            "SHRP2 high-confidence rear-end initialization projected to the project's "
            "same-direction 2Lane map. SUMO collision occurrence must be validated "
            "after execution; this is not an exact accident replay."
        ),
        "map": "2Lane",
        "route": "route_0",
        "duration": float(config["duration_s"]),
        "tags": ["shrp2", "rear_end", "sumo_bridge", str(conditioning.get("quality", "unknown"))],
        "ego": {
            "id": "CAV", "role": "CAV", "route": "route_0", "lane_index": 1,
            "position": cav_position, "speed": float(cav["speed_mps"]), "controller": "IDM",
        },
        "actors": [
            {
                "id": "BV_primary", "role": "BV", "route": "route_0", "lane_index": 1,
                "position": primary_position, "speed": float(primary["speed_mps"]), "controller": "IDM",
            },
            {
                "id": "BV_context", "role": "BV", "route": "route_0", "lane_index": 0,
                "position": context_position, "speed": float(config["context_speed_mps"]), "controller": "IDM",
            },
        ],
        "events": [],
        "perturbations": [],
        "bridge_metadata": {
            "source_scenario_id": record["scenario_id"],
            "source_event_id": source.get("event_id"),
            "source_target_id": source.get("associated_target_id"),
            "source_split": source.get("source_split"),
            "source_quality": conditioning.get("quality"),
            "source_initial_longitudinal_gap_m": source_gap_m,
            "source_requested_impact_time_s": conditioning.get("requested_impact_time_s"),
            "mapping_rule": "align CAV to lane 1 at cav_position_m and preserve initial x-gap to BV_primary",
        },
    }


def _topology_requirement(simulation_type: str) -> str:
    requirements = {
        "crossing": "requires a validated intersection SUMO network and crossing route mapping",
        "opposing_turn": "requires a validated bidirectional intersection SUMO network and turn route mapping",
        "lateral_sideswipe": "requires a validated lane-change maneuver mapping; this pilot does not infer maneuvers from kinematic seeds",
    }
    return requirements.get(simulation_type, "unsupported SHRP2 simulation_type")


def _load_source_summary(seed_root: Path) -> dict[str, Any] | None:
    path = seed_root / "summary.json"
    if not path.is_file():
        return None
    source = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: source.get(key)
        for key in ("dataset_doi", "source_split", "test_split_used", "source_file_sha256")
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_root", required=True, help="SHRP2 collision pipeline output directory")
    parser.add_argument("--output", required=True, help="New or empty bridge output directory")
    parser.add_argument("--config", default="configs/shrp2_sumo_bridge_pilot.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(bridge_seed_directory(args.seed_root, args.output, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
