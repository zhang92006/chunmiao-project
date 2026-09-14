from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


VALID_FAILURE_TYPES = {
    "perception_dropout",
    "perception_delay",
    "perception_position_bias",
    "control_delay",
    "forced_bv_action",
    "calibration_cav_action",
}


@dataclass
class VehicleSpec:
    id: str
    role: str
    route: str
    lane_index: int
    position: float
    speed: float
    controller: str = "IDM"


@dataclass
class EventSpec:
    type: str
    actor: str
    start_time: float
    duration: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationSpec:
    field: str
    distribution: str
    low: float
    high: float


@dataclass
class ScenarioTemplate:
    template_id: str
    description: str
    map: str
    route: str
    duration: float
    ego: VehicleSpec
    actors: list[VehicleSpec]
    events: list[EventSpec]
    perturbations: list[PerturbationSpec]
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioTemplate":
        required = [
            "template_id",
            "description",
            "map",
            "route",
            "duration",
            "ego",
            "actors",
            "events",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required template fields: {', '.join(missing)}")

        template = cls(
            template_id=str(data["template_id"]),
            description=str(data["description"]),
            map=str(data["map"]),
            route=str(data["route"]),
            duration=float(data["duration"]),
            ego=_vehicle_from_dict(data["ego"], role="CAV"),
            actors=[
                _vehicle_from_dict(actor, role=str(actor.get("role", "BV")))
                for actor in data.get("actors", [])
            ],
            events=[_event_from_dict(event) for event in data.get("events", [])],
            perturbations=[
                _perturbation_from_dict(item)
                for item in data.get("perturbations", [])
            ],
            tags=[str(tag) for tag in data.get("tags", [])],
        )
        template.validate()
        return template

    def validate(self) -> None:
        if self.map != "2Lane":
            raise ValueError("Only map='2Lane' is supported by the current SUMO setup.")
        if self.duration <= 0:
            raise ValueError("duration must be positive.")

        vehicle_ids = {self.ego.id}
        for actor in self.actors:
            if actor.id in vehicle_ids:
                raise ValueError(f"Duplicate vehicle id: {actor.id}")
            vehicle_ids.add(actor.id)

        for vehicle in [self.ego, *self.actors]:
            if vehicle.speed < 0:
                raise ValueError(f"{vehicle.id}.speed must be non-negative.")
            if vehicle.position < 0:
                raise ValueError(f"{vehicle.id}.position must be non-negative.")
            if vehicle.lane_index not in (0, 1):
                raise ValueError(f"{vehicle.id}.lane_index must be 0 or 1 for 2Lane.")

        for event in self.events:
            if event.actor not in vehicle_ids:
                raise ValueError(f"Event actor does not exist: {event.actor}")
            if event.start_time < 0:
                raise ValueError(f"{event.type}.start_time must be non-negative.")
            if event.duration <= 0:
                raise ValueError(f"{event.type}.duration must be positive.")
            if event.start_time + event.duration > self.duration:
                raise ValueError(f"{event.type} extends beyond scenario duration.")
            if event.type not in VALID_FAILURE_TYPES:
                raise ValueError(f"Unsupported event type: {event.type}")
            if event.type == "calibration_cav_action":
                if event.actor != self.ego.id:
                    raise ValueError(
                        "calibration_cav_action may target only the ego CAV."
                    )
                if event.params.get("calibration_only") is not True:
                    raise ValueError(
                        "calibration_cav_action requires calibration_only=true."
                    )
                if event.params.get("not_for_d2rl_training") is not True:
                    raise ValueError(
                        "calibration_cav_action requires not_for_d2rl_training=true."
                    )
                if event.params.get("lateral", "central") != "central":
                    raise ValueError(
                        "calibration_cav_action supports longitudinal calibration only."
                    )

        for item in self.perturbations:
            if item.distribution != "uniform":
                raise ValueError("Only uniform perturbations are supported for now.")
            if item.low > item.high:
                raise ValueError(f"Invalid perturbation range for {item.field}.")


def load_template(path: str | Path) -> ScenarioTemplate:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = _safe_load_yaml(stream.read())
    if not isinstance(data, dict):
        raise ValueError("Template file must contain a YAML mapping.")
    return ScenarioTemplate.from_dict(data)


def _safe_load_yaml(text: str) -> Any:
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> Any:
    """Small fallback parser for the repository's scenario template YAML files."""
    raw_lines = text.splitlines()
    lines = []
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, stripped))
        index += 1
    value, next_index = _parse_yaml_block(lines, 0, 0)
    if next_index != len(lines):
        raise ValueError("Could not parse the complete YAML template.")
    return value


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int):
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(lines: list[tuple[int, str]], index: int, indent: int):
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, stripped = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation near: {stripped}")
        if stripped.startswith("- "):
            break
        key, value = _split_yaml_key_value(stripped)
        if value == ">":
            result[key], index = _parse_folded_scalar(lines, index + 1, indent)
        elif value == "":
            result[key], index = _parse_yaml_block(lines, index + 1, indent + 2)
        else:
            result[key] = _parse_yaml_scalar(value)
            index += 1
    return result, index


def _parse_yaml_list(lines: list[tuple[int, str]], index: int, indent: int):
    result: list[Any] = []
    while index < len(lines):
        current_indent, stripped = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or not stripped.startswith("- "):
            break
        item_text = stripped[2:].strip()
        if not item_text:
            item, index = _parse_yaml_block(lines, index + 1, indent + 2)
            result.append(item)
            continue
        if ":" in item_text:
            key, value = _split_yaml_key_value(item_text)
            item: dict[str, Any] = {}
            item[key] = (
                _parse_yaml_scalar(value)
                if value
                else _parse_yaml_block(lines, index + 1, indent + 2)[0]
            )
            index += 1
            while index < len(lines) and lines[index][0] == indent + 2:
                nested_key, nested_value = _split_yaml_key_value(lines[index][1])
                if nested_value == ">":
                    item[nested_key], index = _parse_folded_scalar(
                        lines, index + 1, indent + 2
                    )
                elif nested_value == "":
                    item[nested_key], index = _parse_yaml_block(
                        lines, index + 1, indent + 4
                    )
                else:
                    item[nested_key] = _parse_yaml_scalar(nested_value)
                    index += 1
            result.append(item)
        else:
            result.append(_parse_yaml_scalar(item_text))
            index += 1
    return result, index


def _parse_folded_scalar(
    lines: list[tuple[int, str]], index: int, parent_indent: int
):
    parts = []
    while index < len(lines) and lines[index][0] > parent_indent:
        parts.append(lines[index][1])
        index += 1
    return " ".join(parts), index


def _split_yaml_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Expected key/value line, got: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_yaml_scalar(value: str) -> Any:
    if value in ("[]", "{}"):
        return [] if value == "[]" else {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "~"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _vehicle_from_dict(data: dict[str, Any], role: str) -> VehicleSpec:
    return VehicleSpec(
        id=str(data["id"]),
        role=str(data.get("role", role)),
        route=str(data.get("route", "route_0")),
        lane_index=int(data["lane_index"]),
        position=float(data["position"]),
        speed=float(data["speed"]),
        controller=str(data.get("controller", "IDM")),
    )


def _event_from_dict(data: dict[str, Any]) -> EventSpec:
    return EventSpec(
        type=str(data["type"]),
        actor=str(data["actor"]),
        start_time=float(data["start_time"]),
        duration=float(data["duration"]),
        params=dict(data.get("params", {})),
    )


def _perturbation_from_dict(data: dict[str, Any]) -> PerturbationSpec:
    return PerturbationSpec(
        field=str(data["field"]),
        distribution=str(data.get("distribution", "uniform")),
        low=float(data["low"]),
        high=float(data["high"]),
    )
