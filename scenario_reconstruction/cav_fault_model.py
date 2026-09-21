"""Auditable CAV perception and control fault injection for scenario validation."""
from __future__ import annotations

import copy
from collections import deque
from typing import Any

from .templates import EventSpec


FAULT_EVENT_TYPES = {
    "perception_delay",
    "perception_dropout",
    "perception_position_bias",
    "control_delay",
}
OBSERVATION_SLOTS = (
    "Lead", "Foll", "LeftLead", "LeftFoll", "RightLead", "RightFoll",
)


class CAVFaultModel:
    """Transform CAV observations and decisions during declared fault windows."""

    def __init__(self, events: list[EventSpec]):
        self.events = [event for event in events if event.type in FAULT_EVENT_TYPES]
        max_delay = max((float(event.params.get("delay_s", 0.0)) for event in self.events), default=0.0)
        self._history_horizon_s = max_delay + 1.0
        self._observation_history: deque[tuple[float, dict[str, Any]]] = deque()
        self._control_history: deque[tuple[float, dict[str, Any]]] = deque()
        self.last_audit: dict[str, Any] = {"active_events": []}

    @property
    def enabled(self) -> bool:
        return bool(self.events)

    def transform_observation(self, current_time: float, observation: dict[str, Any]) -> dict[str, Any]:
        raw = copy.deepcopy(observation)
        self._observation_history.append((current_time, raw))
        self._trim_history(current_time)
        transformed = copy.deepcopy(raw)
        audit: dict[str, Any] = {"active_events": []}
        for event in self._active_events(current_time, "perception_delay"):
            delay_s = float(event.params["delay_s"])
            snapshot = self._latest_observation_at_or_before(current_time - delay_s)
            if snapshot is not None:
                self._replace_target_slots(transformed, snapshot, event.params)
                audit["active_events"].append({"type": event.type, "delay_s": delay_s, "target_vehicle": event.params.get("target_vehicle")})
        for event in self._active_events(current_time, "perception_dropout"):
            audit["active_events"].append({"type": event.type, "target_vehicle": event.params.get("target_vehicle"), "dropped_slots": self._drop_target_slots(transformed, event.params)})
        for event in self._active_events(current_time, "perception_position_bias"):
            offset_x_m = float(event.params["offset_x_m"])
            audit["active_events"].append({"type": event.type, "target_vehicle": event.params.get("target_vehicle"), "offset_x_m": offset_x_m, "biased_slots": self._bias_target_slots(transformed, event.params, offset_x_m)})
        self.last_audit = audit
        return transformed

    def delay_control(self, current_time: float, desired_action: dict[str, Any]) -> dict[str, Any]:
        action = copy.deepcopy(desired_action)
        for event in self._active_events(current_time, "control_delay"):
            delayed = self._latest_control_at_or_before(current_time - float(event.params["delay_s"]))
            if delayed is None:
                delayed = copy.deepcopy(event.params.get("initial_action", {"lateral": "central", "longitudinal": 0.0}))
            action = delayed
            self.last_audit.setdefault("active_events", []).append({"type": event.type, "delay_s": float(event.params["delay_s"]), "desired_action": copy.deepcopy(desired_action), "applied_action": copy.deepcopy(action)})
        self._control_history.append((current_time, copy.deepcopy(desired_action)))
        self._trim_history(current_time)
        return action

    def _active_events(self, current_time: float, event_type: str) -> list[EventSpec]:
        return [event for event in self.events if event.type == event_type and event.start_time <= current_time < event.start_time + event.duration]

    def _latest_observation_at_or_before(self, requested_time: float) -> dict[str, Any] | None:
        for time_s, snapshot in reversed(self._observation_history):
            if time_s <= requested_time + 1e-9:
                return snapshot
        return None

    def _latest_control_at_or_before(self, requested_time: float) -> dict[str, Any] | None:
        for time_s, action in reversed(self._control_history):
            if time_s <= requested_time + 1e-9:
                return copy.deepcopy(action)
        return None

    def _trim_history(self, current_time: float) -> None:
        cutoff = current_time - self._history_horizon_s
        while self._observation_history and self._observation_history[0][0] < cutoff:
            self._observation_history.popleft()
        while self._control_history and self._control_history[0][0] < cutoff:
            self._control_history.popleft()

    @staticmethod
    def _target_matches(value: dict[str, Any], params: dict[str, Any]) -> bool:
        return params.get("target_vehicle") is None or value.get("veh_id") == params.get("target_vehicle")

    def _replace_target_slots(self, destination: dict[str, Any], source: dict[str, Any], params: dict[str, Any]) -> None:
        for slot in OBSERVATION_SLOTS:
            source_value = source.get(slot)
            if source_value is None:
                if params.get("target_vehicle") is None:
                    destination[slot] = None
            elif self._target_matches(source_value, params):
                destination[slot] = copy.deepcopy(source_value)

    def _drop_target_slots(self, observation: dict[str, Any], params: dict[str, Any]) -> list[str]:
        changed = []
        for slot in OBSERVATION_SLOTS:
            value = observation.get(slot)
            if value is not None and self._target_matches(value, params):
                observation[slot] = None
                changed.append(slot)
        return changed

    def _bias_target_slots(self, observation: dict[str, Any], params: dict[str, Any], offset_x_m: float) -> list[str]:
        changed = []
        for slot in OBSERVATION_SLOTS:
            value = observation.get(slot)
            if value is None or not self._target_matches(value, params):
                continue
            if "position" in value:
                value["position"][0] += offset_x_m
            if "position3D" in value:
                value["position3D"][0] += offset_x_m
            if "distance" in value:
                value["distance"] += offset_x_m
            changed.append(slot)
        return changed
