from __future__ import annotations

import json
import importlib.util
import re
import sqlite3
from pathlib import Path
from typing import Any


IMPORTANT_TOPICS = {
    "/localization/kinematic_state": "ego_pose",
    "/vehicle/status/velocity_status": "ego_velocity",
    "/perception/object_recognition/objects": "perception_objects",
    "/planning/scenario_planning/trajectory": "planned_trajectory",
    "/control/command/control_cmd": "control_command",
    "/tf": "dynamic_tf",
    "/tf_static": "static_tf",
    "/clock": "simulation_clock",
}


def summarize_autoware_bag(metadata_path: str | Path, db3_path: str | Path) -> dict:
    metadata_path = Path(metadata_path)
    db3_path = Path(db3_path)
    metadata = _read_metadata(metadata_path)
    db_summary = _read_db3_summary(db3_path)
    yaml_summary = _metadata_summary(metadata)

    topics = _merge_topic_summaries(
        yaml_summary.get("topics", []),
        db_summary.get("topics", []),
    )
    duration_s = (
        yaml_summary.get("duration_s")
        or db_summary.get("duration_s")
        or _estimate_duration_from_topics(topics)
    )
    _add_bag_rates(topics, duration_s)

    return {
        "source": {
            "metadata_path": str(metadata_path),
            "db3_path": str(db3_path),
        },
        "duration_s": duration_s,
        "message_count": yaml_summary.get("message_count")
        or db_summary.get("message_count"),
        "start_time_ns": yaml_summary.get("start_time_ns"),
        "topics": topics,
        "available_signals": _available_signals(topics),
        "deserialization_libraries": _deserialization_libraries(),
        "current_simulation_limits": {
            "map": "2Lane",
            "lane_index_values": [0, 1],
            "supported_executed_event": "forced_bv_action",
            "notes": [
                "The current SUMO reconstruction path supports 2Lane templates.",
                "Without ROS2 message deserialization, this summary cannot decode CDR payloads into exact object positions.",
                "The LLM should generate conservative seed templates that can be validated and simulated.",
            ],
        },
        "template_generation_hints": _template_generation_hints(duration_s, topics),
    }


def write_summary(summary: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=4, ensure_ascii=False)
    return path


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        text = stream.read()
    try:
        import yaml

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = _parse_rosbag_metadata_text(text)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a mapping: {path}")
    return data


def _parse_rosbag_metadata_text(text: str) -> dict[str, Any]:
    duration_ns = _first_int_after(text, r"duration:\s*\n\s*nanoseconds:\s*(\d+)")
    start_time_ns = _first_int_after(
        text, r"starting_time:\s*\n\s*nanoseconds_since_epoch:\s*(\d+)"
    )
    message_count = _first_int_after(text, r"\n\s*message_count:\s*(\d+)")
    topics = []
    for block in text.split("- topic_metadata:")[1:]:
        name = _first_str_after(block, r"\n\s*name:\s*([^\n]+)")
        topic_type = _first_str_after(block, r"\n\s*type:\s*([^\n]+)")
        serialization_format = _first_str_after(
            block, r"\n\s*serialization_format:\s*([^\n]+)"
        )
        topic_count = _first_int_after(block, r"\n\s*message_count:\s*(\d+)")
        if name:
            topics.append(
                {
                    "topic_metadata": {
                        "name": name,
                        "type": topic_type,
                        "serialization_format": serialization_format,
                    },
                    "message_count": topic_count,
                }
            )
    return {
        "rosbag2_bagfile_information": {
            "duration": {"nanoseconds": duration_ns},
            "starting_time": {"nanoseconds_since_epoch": start_time_ns},
            "message_count": message_count,
            "topics_with_message_count": topics,
        }
    }


def _first_int_after(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _first_str_after(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip().strip('"')


def _metadata_summary(metadata: dict[str, Any]) -> dict:
    info = metadata.get("rosbag2_bagfile_information", {})
    duration_ns = (info.get("duration") or {}).get("nanoseconds")
    start_time_ns = (info.get("starting_time") or {}).get("nanoseconds_since_epoch")
    topics = []
    for item in info.get("topics_with_message_count", []):
        topic_metadata = item.get("topic_metadata", {})
        topics.append(
            {
                "name": topic_metadata.get("name"),
                "type": topic_metadata.get("type"),
                "serialization_format": topic_metadata.get("serialization_format"),
                "message_count": item.get("message_count"),
                "role": IMPORTANT_TOPICS.get(topic_metadata.get("name"), "other"),
            }
        )
    return {
        "duration_s": _ns_to_s(duration_ns),
        "start_time_ns": start_time_ns,
        "message_count": info.get("message_count"),
        "topics": topics,
    }


def _read_db3_summary(path: Path) -> dict:
    connection = sqlite3.connect(str(path))
    try:
        topic_rows = connection.execute(
            "select id, name, type, serialization_format from topics order by id"
        ).fetchall()
        count_rows = {
            topic_id: (count, first_ts, last_ts)
            for topic_id, count, first_ts, last_ts in connection.execute(
                "select topic_id, count(*), min(timestamp), max(timestamp) "
                "from messages group by topic_id"
            ).fetchall()
        }
        topics = []
        min_ts = None
        max_ts = None
        for topic_id, name, topic_type, serialization_format in topic_rows:
            count, first_ts, last_ts = count_rows.get(topic_id, (0, None, None))
            min_ts = first_ts if min_ts is None else min(min_ts, first_ts or min_ts)
            max_ts = last_ts if max_ts is None else max(max_ts, last_ts or max_ts)
            topics.append(
                {
                    "id": topic_id,
                    "name": name,
                    "type": topic_type,
                    "serialization_format": serialization_format,
                    "message_count": count,
                    "first_timestamp_ns": first_ts,
                    "last_timestamp_ns": last_ts,
                    "duration_s": _ns_to_s(
                        last_ts - first_ts
                        if first_ts is not None and last_ts is not None
                        else None
                    ),
                    "role": IMPORTANT_TOPICS.get(name, "other"),
                }
            )
        message_count = connection.execute("select count(*) from messages").fetchone()[0]
        duration_s = _ns_to_s(max_ts - min_ts if min_ts and max_ts else None)
        return {
            "duration_s": duration_s,
            "message_count": message_count,
            "topics": topics,
        }
    finally:
        connection.close()


def _merge_topic_summaries(yaml_topics: list[dict], db_topics: list[dict]) -> list[dict]:
    merged = {topic["name"]: dict(topic) for topic in yaml_topics if topic.get("name")}
    for topic in db_topics:
        name = topic.get("name")
        if not name:
            continue
        current = merged.setdefault(name, {})
        current.update({key: value for key, value in topic.items() if value is not None})
    for topic in merged.values():
        duration_s = topic.get("duration_s")
        message_count = topic.get("message_count")
        topic["rate_hz"] = (
            round(message_count / duration_s, 3)
            if duration_s and message_count
            else None
        )
    return sorted(merged.values(), key=lambda item: item.get("name", ""))


def _available_signals(topics: list[dict]) -> dict[str, bool]:
    roles = {topic.get("role") for topic in topics}
    return {
        "ego_pose": "ego_pose" in roles,
        "ego_velocity": "ego_velocity" in roles,
        "perception_objects": "perception_objects" in roles,
        "planned_trajectory": "planned_trajectory" in roles,
        "control_command": "control_command" in roles,
        "tf": "dynamic_tf" in roles or "static_tf" in roles,
    }


def _deserialization_libraries() -> dict[str, bool]:
    libraries = {}
    for name in ("rosbags", "rosbag2_py", "rclpy", "rosidl_runtime_py"):
        try:
            libraries[name] = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            libraries[name] = False
    return libraries


def _add_bag_rates(topics: list[dict], duration_s: float | None) -> None:
    if not duration_s:
        return
    for topic in topics:
        message_count = topic.get("message_count")
        topic["bag_rate_hz"] = (
            round(message_count / duration_s, 3) if message_count else None
        )


def _template_generation_hints(duration_s: float | None, topics: list[dict]) -> dict:
    has_objects = any(topic.get("role") == "perception_objects" for topic in topics)
    has_control = any(topic.get("role") == "control_command" for topic in topics)
    has_pose = any(topic.get("role") == "ego_pose" for topic in topics)
    topic_rates = {
        topic.get("role"): topic.get("bag_rate_hz")
        for topic in topics
        if topic.get("role") and topic.get("bag_rate_hz")
    }
    return {
        "recommended_template_family": "cut_in_or_close_following",
        "candidate_failure_types": [
            "forced_bv_action",
            "perception_dropout" if has_objects else None,
            "control_delay" if has_control else None,
        ],
        "target_duration_s": min(max(duration_s or 12.0, 6.0), 20.0),
        "topic_bag_rate_hz": topic_rates,
        "forced_bv_action_duration_estimate_s": _estimate_forced_action_duration(
            topic_rates
        ),
        "confidence_notes": [
            "Exact collision geometry requires decoded ROS2 messages.",
            "Topic availability supports ego motion and perceived object reasoning."
            if has_pose and has_objects
            else "Topic-level summary is available, but geometric reconstruction is approximate.",
        ],
    }


def _estimate_forced_action_duration(topic_rates: dict[str, float]) -> float:
    perception_rate = topic_rates.get("perception_objects")
    control_rate = topic_rates.get("control_command")
    candidate_rates = [
        rate for rate in (perception_rate, control_rate) if rate and rate > 0
    ]
    if not candidate_rates:
        return 0.8
    duration = 8.0 / min(candidate_rates)
    return round(max(0.5, min(2.0, duration)), 3)


def _estimate_duration_from_topics(topics: list[dict]) -> float | None:
    durations = [topic.get("duration_s") for topic in topics if topic.get("duration_s")]
    return max(durations) if durations else None


def _ns_to_s(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000.0
