from __future__ import annotations

from pathlib import Path

from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
from envs.nade import NADE
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.vehicle.vehicle import Vehicle

from .templates import EventSpec, ScenarioTemplate, VehicleSpec, load_template


class ScenarioNADE(NADE):
    """Template-initialized simulation, not an unbiased NDD probability sample."""

    def __init__(self, template: ScenarioTemplate | str | Path):
        self.scenario_template = load_template(template) if isinstance(template, (str, Path)) else template
        self.scenario_template.validate_runtime()
        self._applied_events: set[str] = set()
        # Template initialization itself has no derived p(initial)/q(initial).
        self.scenario_metadata = {
            "template_id": self.scenario_template.template_id,
            "duration": self.scenario_template.duration,
            "likelihood_valid": False,
            "likelihood_reason": "Template initialization and scripted interventions have no derived importance ratio",
            "weight_semantics": "upstream diagnostics only; not valid for training or probability estimation",
        }
        super().__init__(BVController=TreeSearchNADEBackgroundController, cav_model="IDM")

    def generate_traffic_flow(self, init_info=None):
        ego = self.scenario_template.ego
        self.generate_av(speed=ego.speed, id=ego.id, route=ego.route, position=ego.position,
                         av_lane_id=self._lane_id_from_index(ego.lane_index), controller_type=self.default_av_controller)
        for actor in self.scenario_template.actors:
            self._generate_actor(actor)

    def apply_template_events(self):
        current_time = self.simulator.get_time() - self.episode_info["start_time"]
        for index, event in enumerate(self.scenario_template.events):
            key = str(index)
            if not self._event_is_active(event, current_time):
                continue
            if bool(event.params.get("apply_once", True)) and key in self._applied_events:
                continue
            applied = self._apply_forced_bv_action(event)
            if applied:
                self._applied_events.add(key)
            # Do not manufacture ndd_step_info, epsilon, or weight_step_info.
            self.info_extractor.episode_log.setdefault("scenario_events", []).append({
                "event_index": index, "time": current_time, "type": event.type,
                "actor": event.actor, "applied": applied,
                "status": "action_sent" if applied else "missing_actor_or_illegal_action",
                "metrics": self._estimate_actor_metrics(event.actor),
            })

    def _step(self):
        control_info_list = super()._step()
        self.apply_template_events()
        return control_info_list

    def terminate_check(self):
        reason, stop, additional_info = self._terminate_check()
        now = self.simulator.get_time()
        elapsed = now - self.episode_info["start_time"]
        if not stop and elapsed + 1e-9 >= self.scenario_template.duration:
            reason, stop, additional_info = {5: "Template duration reached"}, True, {}
        if stop:
            self.episode_info["end_time"] = now
            self.episode_info["elapsed_time"] = elapsed
            self.info_extractor.get_terminate_info(stop, reason, additional_info)
        return stop, reason, additional_info

    def _generate_actor(self, actor: VehicleSpec):
        vehicle = Vehicle(id=actor.id, controller=Controller(), routeID=actor.route,
                          simulator=self.simulator, initial_speed=actor.speed,
                          initial_position=actor.position, initial_lane_id=self._lane_id_from_index(actor.lane_index))
        self.simulator._add_vehicle_to_sumo(vehicle, typeID="IDM")
        vehicle.install_controller(self.independent_controller_dict["BV"]())
        self.vehicle_list.add_vehicles([vehicle])
        self.info_extractor.add_initialization_info(actor.id, {
            "speed": actor.speed, "lane_id": vehicle.initial_lane_id,
            "route_id": actor.route, "position": actor.position, "controller": actor.controller,
        })

    def _apply_forced_bv_action(self, event: EventSpec):
        if event.actor not in self.vehicle_list:
            return False
        vehicle = self.vehicle_list[event.actor]
        action = {"lateral": str(event.params.get("lateral", "central")),
                  "longitudinal": float(event.params.get("longitudinal", 0.0))}
        if not vehicle.is_action_legal(action):
            return False
        vehicle.controller.action = action
        vehicle.act(action)
        return True

    def _estimate_actor_metrics(self, actor_id):
        cav, actor = self.vehicle_list.get("CAV"), self.vehicle_list.get(actor_id)
        if cav is None or actor is None:
            return None
        cav_info, actor_info = cav.observation.information["Ego"], actor.observation.information["Ego"]
        gap = actor_info["position"][0] - cav_info["position"][0] - 5.0
        closing = cav_info["velocity"] - actor_info["velocity"]
        same_lane = cav_info.get("lane_index") is not None and cav_info["lane_index"] == actor_info.get("lane_index")
        ttc = gap / closing if same_lane and closing > 0 and gap > 0 else None
        return {"longitudinal_gap_m": float(gap), "closing_speed_mps": float(closing),
                "same_lane": same_lane, "ttc_s": float(ttc) if ttc is not None else None,
                "risk_proxy": 1 / (1 + max(gap, 0)) if same_lane else 0,
                "metric_semantics": "heuristic; fixed 5 m vehicle length; not a likelihood"}

    def _lane_id_from_index(self, lane_index: int) -> str:
        lane_ids = sorted(lane.getID() for lane in self.simulator.get_available_lanes())
        if lane_index >= len(lane_ids):
            raise ValueError(f"lane_index={lane_index} is out of range for lanes {lane_ids}")
        return lane_ids[lane_index]

    @staticmethod
    def _event_is_active(event: EventSpec, current_time: float) -> bool:
        return event.start_time <= current_time < event.start_time + event.duration
