from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


AUTOWARE_MSG_ROOT = Path("external/autoware_msgs")

DEFAULT_TOPICS = [
    "/localization/kinematic_state",
    "/vehicle/status/velocity_status",
    "/control/command/control_cmd",
    "/perception/object_recognition/objects",
    "/planning/scenario_planning/trajectory",
]


def build_autoware_typestore(
    msg_root: str | Path = AUTOWARE_MSG_ROOT,
    store: Stores = Stores.ROS2_FOXY,
):
    msg_root = Path(msg_root)
    if not msg_root.exists():
        raise FileNotFoundError(
            f"Autoware message definition directory not found: {msg_root}"
        )

    typestore = get_typestore(store)
    custom_types = {}
    for path in msg_root.rglob("*.msg"):
        package = path.parts[-3]
        typename = f"{package}/msg/{path.stem}"
        custom_types.update(get_types_from_msg(path.read_text(encoding="utf-8"), typename))
    typestore.register(custom_types)
    return typestore, custom_types


def decode_autoware_bag_summary(
    bag_dir: str | Path = ".",
    msg_root: str | Path = AUTOWARE_MSG_ROOT,
    topics: list[str] | None = None,
    max_samples: int = 3,
) -> dict[str, Any]:
    typestore, custom_types = build_autoware_typestore(msg_root)
    requested_topics = set(topics or DEFAULT_TOPICS)
    summary: dict[str, Any] = {
        "bag_dir": str(bag_dir),
        "msg_root": str(msg_root),
        "registered_custom_type_count": len(custom_types),
        "topics": {},
    }

    with Reader(Path(bag_dir)) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in requested_topics
        ]
        for connection in connections:
            summary["topics"][connection.topic] = {
                "msgtype": connection.msgtype,
                "connection_message_count": connection.msgcount,
                "decoded_count": 0,
                "decode_errors": [],
                "stats": _initial_topic_stats(connection.topic),
                "samples": [],
            }

        for connection, timestamp_ns, raw in reader.messages(connections=connections):
            topic_summary = summary["topics"][connection.topic]
            try:
                message = typestore.deserialize_cdr(raw, connection.msgtype)
            except Exception as exc:
                topic_summary["decode_errors"].append(
                    {
                        "timestamp_ns": timestamp_ns,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            topic_summary["decoded_count"] += 1
            _update_topic_stats(topic_summary["stats"], connection.topic, message)
            if len(topic_summary["samples"]) < max_samples:
                topic_summary["samples"].append(
                    _compact_message_sample(connection.topic, timestamp_ns, message)
                )

    return summary


def write_decoded_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=4, ensure_ascii=False)
    return path


def _initial_topic_stats(topic: str) -> dict[str, Any]:
    if topic == "/vehicle/status/velocity_status":
        return {
            "longitudinal_velocity_min": None,
            "longitudinal_velocity_max": None,
            "lateral_velocity_min": None,
            "lateral_velocity_max": None,
        }
    if topic == "/control/command/control_cmd":
        return {
            "target_velocity_min": None,
            "target_velocity_max": None,
            "target_acceleration_min": None,
            "target_acceleration_max": None,
            "steering_tire_angle_min": None,
            "steering_tire_angle_max": None,
        }
    if topic == "/perception/object_recognition/objects":
        return {
            "messages_with_objects": 0,
            "max_object_count": 0,
            "first_nonempty_sample": None,
        }
    if topic == "/planning/scenario_planning/trajectory":
        return {
            "max_point_count": 0,
            "first_trajectory_sample": None,
        }
    if topic == "/localization/kinematic_state":
        return {
            "first_pose": None,
            "last_pose": None,
            "linear_speed_min": None,
            "linear_speed_max": None,
        }
    return {}


def _update_topic_stats(stats: dict[str, Any], topic: str, message: Any) -> None:
    if topic == "/vehicle/status/velocity_status":
        _update_minmax(stats, "longitudinal_velocity", message.longitudinal_velocity)
        _update_minmax(stats, "lateral_velocity", message.lateral_velocity)
    elif topic == "/control/command/control_cmd":
        _update_minmax(stats, "target_velocity", message.longitudinal.velocity)
        _update_minmax(stats, "target_acceleration", message.longitudinal.acceleration)
        _update_minmax(stats, "steering_tire_angle", message.lateral.steering_tire_angle)
    elif topic == "/perception/object_recognition/objects":
        object_count = len(message.objects)
        stats["max_object_count"] = max(stats["max_object_count"], object_count)
        if object_count:
            stats["messages_with_objects"] += 1
            if stats["first_nonempty_sample"] is None:
                stats["first_nonempty_sample"] = _compact_predicted_objects(message)
    elif topic == "/planning/scenario_planning/trajectory":
        point_count = len(message.points)
        stats["max_point_count"] = max(stats["max_point_count"], point_count)
        if point_count and stats["first_trajectory_sample"] is None:
            stats["first_trajectory_sample"] = _compact_trajectory(message)
    elif topic == "/localization/kinematic_state":
        pose = _compact_pose(message.pose.pose)
        stats["first_pose"] = stats["first_pose"] or pose
        stats["last_pose"] = pose
        linear = message.twist.twist.linear
        speed = math.sqrt(linear.x * linear.x + linear.y * linear.y + linear.z * linear.z)
        _update_minmax(stats, "linear_speed", speed)


def _compact_message_sample(topic: str, timestamp_ns: int, message: Any) -> dict[str, Any]:
    sample = {"timestamp_ns": timestamp_ns}
    if topic == "/vehicle/status/velocity_status":
        sample.update(
            {
                "longitudinal_velocity": float(message.longitudinal_velocity),
                "lateral_velocity": float(message.lateral_velocity),
                "heading_rate": float(message.heading_rate),
            }
        )
    elif topic == "/control/command/control_cmd":
        sample.update(
            {
                "target_velocity": float(message.longitudinal.velocity),
                "target_acceleration": float(message.longitudinal.acceleration),
                "target_jerk": float(message.longitudinal.jerk),
                "steering_tire_angle": float(message.lateral.steering_tire_angle),
            }
        )
    elif topic == "/perception/object_recognition/objects":
        sample.update(_compact_predicted_objects(message))
    elif topic == "/planning/scenario_planning/trajectory":
        sample.update(_compact_trajectory(message))
    elif topic == "/localization/kinematic_state":
        sample.update(
            {
                "pose": _compact_pose(message.pose.pose),
                "linear_velocity": _compact_vector(message.twist.twist.linear),
            }
        )
    return sample


def _compact_predicted_objects(message: Any, limit: int = 3) -> dict[str, Any]:
    return {
        "object_count": len(message.objects),
        "objects": [_compact_predicted_object(obj) for obj in message.objects[:limit]],
    }


def _compact_predicted_object(obj: Any) -> dict[str, Any]:
    pose = obj.kinematics.initial_pose_with_covariance.pose
    twist = obj.kinematics.initial_twist_with_covariance.twist
    classification = obj.classification[0] if obj.classification else None
    return {
        "uuid": list(int(value) for value in obj.object_id.uuid.tolist()),
        "existence_probability": float(obj.existence_probability),
        "classification_label": int(classification.label) if classification else None,
        "classification_probability": (
            float(classification.probability) if classification else None
        ),
        "pose": _compact_pose(pose),
        "linear_velocity": _compact_vector(twist.linear),
        "shape_dimensions": _compact_vector(obj.shape.dimensions),
        "predicted_path_count": len(obj.kinematics.predicted_paths),
    }


def _compact_trajectory(message: Any, limit: int = 3) -> dict[str, Any]:
    points = message.points
    selected = list(points[:limit])
    if len(points) > limit:
        selected.append(points[-1])
    return {
        "point_count": len(points),
        "points": [_compact_trajectory_point(point) for point in selected],
    }


def _compact_trajectory_point(point: Any) -> dict[str, Any]:
    return {
        "pose": _compact_pose(point.pose),
        "longitudinal_velocity_mps": float(point.longitudinal_velocity_mps),
        "lateral_velocity_mps": float(point.lateral_velocity_mps),
        "acceleration_mps2": float(point.acceleration_mps2),
        "heading_rate_rps": float(point.heading_rate_rps),
        "front_wheel_angle_rad": float(point.front_wheel_angle_rad),
    }


def _compact_pose(pose: Any) -> dict[str, Any]:
    return {
        "position": _compact_point(pose.position),
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def _compact_point(point: Any) -> dict[str, float]:
    return {"x": float(point.x), "y": float(point.y), "z": float(point.z)}


def _compact_vector(vector: Any) -> dict[str, float]:
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}


def _update_minmax(stats: dict[str, Any], base_key: str, value: float) -> None:
    value = float(value)
    min_key = f"{base_key}_min"
    max_key = f"{base_key}_max"
    stats[min_key] = value if stats[min_key] is None else min(stats[min_key], value)
    stats[max_key] = value if stats[max_key] is None else max(stats[max_key], value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Autoware custom ROS2 CDR topics with rosbags."
    )
    parser.add_argument("--bag_dir", default=".", help="Directory containing metadata.yaml.")
    parser.add_argument(
        "--msg_root",
        default=str(AUTOWARE_MSG_ROOT),
        help="Directory containing cloned autoware_msgs packages.",
    )
    parser.add_argument(
        "--output",
        default="data_analysis/raw_data/AutowareDecoded/autoware_decoded_summary.json",
    )
    parser.add_argument("--max_samples", type=int, default=3)
    args = parser.parse_args()

    summary = decode_autoware_bag_summary(
        bag_dir=args.bag_dir,
        msg_root=args.msg_root,
        max_samples=args.max_samples,
    )
    output_path = write_decoded_summary(summary, args.output)
    print(f"Decoded summary written: {output_path}")
    for topic, topic_summary in summary["topics"].items():
        print(
            f"{topic}: decoded={topic_summary['decoded_count']} "
            f"errors={len(topic_summary['decode_errors'])}"
        )


if __name__ == "__main__":
    main()
