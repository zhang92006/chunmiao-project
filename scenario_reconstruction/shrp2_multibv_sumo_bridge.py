"""Map SHRP2 multi-BV source seeds to autonomous 2Lane SUMO templates.

Only same-direction two-lane initializations are projected. The bridge does
not replay measured actions and emits no forced events, so the resulting
templates can be handed to NADE/D2RL for autonomous interaction sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .templates import ScenarioTemplate


SUPPORTED_CONFLICTS = {"leading", "adjacent_lane", "merging", "none"}


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Only MultiBV SUMO bridge schema_version 1 is supported")
    if config.get("map") != "2Lane":
        raise ValueError("The current MultiBV bridge supports map='2Lane' only")
    for key in (
        "duration_s",
        "cav_position_m",
        "lane_offset_threshold_m",
        "minimum_gap_m",
        "maximum_gap_m",
        "minimum_initial_speed_mps",
        "maximum_initial_speed_mps",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be positive")
    if config["minimum_gap_m"] >= config["maximum_gap_m"]:
        raise ValueError("minimum_gap_m must be smaller than maximum_gap_m")
    if config["minimum_initial_speed_mps"] >= config["maximum_initial_speed_mps"]:
        raise ValueError(
            "minimum_initial_speed_mps must be smaller than maximum_initial_speed_mps"
        )
    for key in ("cav_lane_index", "default_bv_lane_index"):
        value = config.get(key)
        if value not in (0, 1):
            raise ValueError(f"{key} must be 0 or 1")
    if config.get("allowed_conflicts") is None or not set(config["allowed_conflicts"]) <= SUPPORTED_CONFLICTS:
        raise ValueError("allowed_conflicts contains an unsupported conflict type")


def _state_for_bridge(seed):
    states = seed.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("seed lacks states")
    frame = states[-1]
    if not isinstance(frame, list) or len(frame) < 3:
        raise ValueError("MultiBV seed must contain CAV and two BV states")
    if any(not isinstance(actor, list) or len(actor) != 4 for actor in frame):
        raise ValueError("MultiBV seed states must have four channels")
    return frame


def _lane_from_relative_y(relative_y, config):
    """Collapse source-frame lateral position to the nearest of two lanes."""
    if abs(float(relative_y)) <= float(config["lane_offset_threshold_m"]):
        return int(config["cav_lane_index"])
    return 1 - int(config["cav_lane_index"])


def multibv_template_from_seed(seed, config):
    """Create one autonomous K=2 template or reject unsafe projection."""
    validate_config(config)
    if seed.get("record_type") != "shrp2_measured_multibv_seed_v1":
        raise ValueError("unsupported seed record_type")
    source = seed.get("source") or {}
    conflict = str(source.get("conflict", "unknown"))
    if conflict not in config["allowed_conflicts"]:
        raise ValueError(f"conflict={conflict!r} requires another SUMO topology")
    frame = _state_for_bridge(seed)
    cav, primary, context = frame[:3]
    source_gap = float(primary[0]) - float(cav[0])
    if not config["minimum_gap_m"] <= source_gap <= config["maximum_gap_m"]:
        raise ValueError(f"primary_gap_out_of_range:{source_gap:.3f}")
    for label, actor in (("cav", cav), ("primary", primary), ("context", context)):
        speed = float(actor[2])
        if speed < 0:
            raise ValueError(f"negative_{label}_speed")
        if speed < float(config["minimum_initial_speed_mps"]):
            raise ValueError(
                f"{label}_speed_below_d2rl_domain:{speed:.3f}<"
                f"{float(config['minimum_initial_speed_mps']):.3f}"
            )
        if speed > float(config["maximum_initial_speed_mps"]):
            raise ValueError(
                f"{label}_speed_exceeds_d2rl_domain:{speed:.3f}>"
                f"{float(config['maximum_initial_speed_mps']):.3f}"
            )

    event_id = int(source["event_id"])
    split = str(source.get("split", "unknown"))
    category = str(source.get("category", "unknown"))
    template_id = f"shrp2_multibv_{category}_{event_id}_{split}".replace(" ", "_")
    cav_position = float(config["cav_position_m"])
    actors = []
    for index, (actor_state, actor_meta, role) in enumerate(
        zip(frame[1:3], seed.get("actors", [])[1:3], ("primary_risk_bv", "context_bv"))
    ):
        actor_id = "BV_primary" if index == 0 else "BV_context"
        gap = float(actor_state[0]) - float(cav[0])
        position = cav_position + gap
        if position < 0:
            raise ValueError(f"{actor_id}_position_out_of_map")
        actors.append({
            "id": actor_id,
            "role": "BV",
            "route": str(config.get("route", "route_0")),
            "lane_index": _lane_from_relative_y(actor_state[1] - float(cav[1]), config),
            "position": position,
            "speed": float(actor_state[2]),
            "controller": "IDM",
            "source_target_id": actor_meta.get("source_target_id"),
            "source_role": role,
        })
    projected = [("CAV", cav_position, int(config["cav_lane_index"]))]
    projected.extend(
        (actor["id"], float(actor["position"]), int(actor["lane_index"]))
        for actor in actors
    )
    for index, (left_id, left_position, left_lane) in enumerate(projected):
        for right_id, right_position, right_lane in projected[index + 1 :]:
            if left_lane != right_lane:
                continue
            same_lane_gap = abs(left_position - right_position)
            if same_lane_gap < float(config["minimum_gap_m"]):
                raise ValueError(
                    f"same_lane_gap_out_of_range:{left_id}:{right_id}:{same_lane_gap:.3f}"
                )
    return {
        "template_id": template_id,
        "description": (
            "SHRP2 multi-BV source-frame initialization projected to the project's "
            "2Lane map. No measured action is replayed; NADE/D2RL controls all interaction."
        ),
        "map": "2Lane",
        "route": str(config.get("route", "route_0")),
        "duration": float(config["duration_s"]),
        "tags": ["shrp2", "multibv", "autonomous_seed", split, conflict],
        "ego": {
            "id": "CAV", "role": "CAV", "route": str(config.get("route", "route_0")),
            "lane_index": int(config["cav_lane_index"]), "position": cav_position,
            "speed": max(0.0, float(cav[2])), "controller": "IDM",
        },
        "actors": actors,
        "events": [],
        "perturbations": [],
        "bridge_metadata": {
            "source_record_type": seed["record_type"],
            "source_event_id": event_id,
            "source_category": category,
            "source_split": split,
            "source_conflict": conflict,
            "source_context_target_ids": source.get("context_target_ids", []),
            "source_state_time_s": seed.get("time_s", [])[-1] if seed.get("time_s") else None,
            "mapping_rule": "align CAV to configured lane/position; preserve source longitudinal gaps and map lateral offset to two lanes",
            "lane_mapping_threshold_m": float(config["lane_offset_threshold_m"]),
            "minimum_initial_speed_mps": float(config["minimum_initial_speed_mps"]),
            "maximum_initial_speed_mps": float(config["maximum_initial_speed_mps"]),
            "drl_training_ready": False,
            "drl_next_step": "run autonomous NADE/D2RL rollout and require joint episode fields",
        },
    }


def bridge_seed_directory(seed_root, output, config):
    validate_config(config)
    seed_root, output = Path(seed_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty MultiBV bridge output directory")
    decisions = []
    for split in ("train", "validation", "test"):
        source_path = seed_root / split / "seeds.jsonl"
        if not source_path.is_file():
            continue
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            seed = json.loads(line)
            decision = {
                "event_id": seed.get("source", {}).get("event_id"),
                "split": split,
                "category": seed.get("source", {}).get("category"),
            }
            try:
                template = multibv_template_from_seed(seed, config)
                template_path = output / "templates" / split / f"{template['template_id']}.json"
                template_path.parent.mkdir(parents=True, exist_ok=True)
                template_path.write_text(
                    json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                ScenarioTemplate.from_dict(template)
                decision.update({"status": "template_created", "template_path": str(template_path)})
            except (KeyError, TypeError, ValueError) as exc:
                decision.update({"status": "blocked", "reason": str(exc)})
            decisions.append(decision)
    summary = {
        "schema_version": 1,
        "status": "SHRP2 multi-BV source seed to autonomous 2Lane SUMO bridge",
        "seed_root": str(seed_root.resolve()),
        "output": str(output.resolve()),
        "template_count": sum(item["status"] == "template_created" for item in decisions),
        "blocked_count": sum(item["status"] == "blocked" for item in decisions),
        "blocked_by_reason": dict(
            Counter(
                _reason_code(item.get("reason", ""))
                for item in decisions
                if item["status"] == "blocked"
            )
        ),
        "created_by_split": dict(Counter(item["split"] for item in decisions if item["status"] == "template_created")),
        "records": decisions,
        "drl_training_ready": False,
        "limitations": [
            "Only two-lane same-direction initializations are mapped; unsupported conflict types are blocked.",
            "Source lateral offsets are collapsed to two lane IDs using a threshold and require SUMO validation.",
            "Initial speeds outside the configured D2RL speed domain are blocked; source seeds remain unchanged.",
            "Templates contain no forced event; D2RL eligibility requires autonomous NADE joint logs after rollout.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "bridge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _reason_code(reason):
    """Collapse value-bearing rejection messages into stable audit categories."""
    return str(reason).split(":", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/shrp2_multibv_sumo_bridge.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = bridge_seed_directory(args.seed_root, args.output, config)
    print(json.dumps({
        key: result[key]
        for key in ("template_count", "blocked_count", "created_by_split", "blocked_by_reason")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
