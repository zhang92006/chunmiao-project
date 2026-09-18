from __future__ import annotations

from pathlib import Path

from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
from envs.nade import NADE
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.vehicle.vehicle import Vehicle

from .multibv import select_multibv_training_actors
from .cav_fault_model import CAVFaultModel
from .fault_aware_vehicle import FaultAwareVehicle
from .templates import EventSpec, ScenarioTemplate, VehicleSpec, load_template


VALID_MULTIBV_PROPOSAL_MODES = {"naturalistic", "factorized", "joint_pair"}


class ScenarioNADE(NADE):
    """NADE environment initialized from a scenario reconstruction template."""

    def __init__(
        self,
        template: ScenarioTemplate | str | Path,
        multibv_proposal_mode: str = "joint_pair",
    ):
        self.scenario_template = (
            load_template(template)
            if isinstance(template, (str, Path))
            else template
        )
        self._logged_training_events: set[str] = set()
        if multibv_proposal_mode not in VALID_MULTIBV_PROPOSAL_MODES:
            valid = ", ".join(sorted(VALID_MULTIBV_PROPOSAL_MODES))
            raise ValueError(
                f"Unsupported multibv_proposal_mode={multibv_proposal_mode!r}; "
                f"expected one of {valid}"
            )
        self.multibv_proposal_mode = multibv_proposal_mode
        self.multi_bv_control_num = (
            2 if "multibv" in self.scenario_template.tags else 1
        )
        self.cav_fault_model = CAVFaultModel(self.scenario_template.events)
        super().__init__(
            BVController=TreeSearchNADEBackgroundController,
            cav_model="FaultAwareIDM" if self.cav_fault_model.enabled else "IDM",
        )
        self._record_scenario_metadata()

    def _record_scenario_metadata(self) -> None:
        """Keep the SHRP2 source identity through rollout and training export."""
        bridge = self.scenario_template.bridge_metadata
        source_keys = (
            "source_event_id",
            "source_category",
            "source_split",
            "source_conflict",
            "source_record_type",
        )
        self.info_extractor.episode_log["scenario_metadata"] = {
            "template_id": self.scenario_template.template_id,
            "tags": list(self.scenario_template.tags),
            "multibv_proposal_mode": self.multibv_proposal_mode,
            **{key: bridge[key] for key in source_keys if key in bridge},
        }

    def generate_traffic_flow(self, init_info=None):
        """Insert the CAV and key BVs from the template instead of NDD flow."""
        ego = self.scenario_template.ego
        self.generate_av(
            speed=ego.speed,
            id=ego.id,
            route=ego.route,
            position=ego.position,
            av_lane_id=self._lane_id_from_index(ego.lane_index),
            controller_type=self.default_av_controller,
        )

        for actor in self.scenario_template.actors:
            self._generate_actor(actor)

    def generate_av(
        self,
        speed=15.0,
        id="CAV",
        route="route_0",
        type_id="IDM",
        position=400.0,
        av_lane_id=None,
        controller_type=None,
    ):
        if not self.cav_fault_model.enabled:
            return super().generate_av(
                speed=speed,
                id=id,
                route=route,
                type_id=type_id,
                position=position,
                av_lane_id=av_lane_id,
                controller_type=controller_type or self.default_av_controller,
            )
        vehicle = FaultAwareVehicle(
            id=id,
            controller=Controller(),
            routeID=route,
            simulator=self.simulator,
            initial_speed=speed,
            initial_position=position,
            initial_lane_id=av_lane_id,
        )
        self.simulator._add_vehicle_to_sumo(vehicle, typeID=type_id)
        vehicle.install_controller((controller_type or self.default_av_controller)())
        vehicle.controller.fault_model = self.cav_fault_model
        self.vehicle_list.add_vehicles([vehicle])
        return id, {
            "speed": speed,
            "lane_id": av_lane_id,
            "route_id": route,
            "position": position,
            "fault_validation_only": True,
        }

    def apply_template_events(self):
        current_time = self.simulator.get_time()
        for event in self.scenario_template.events:
            if self._event_is_active(event, current_time):
                if event.type == "forced_bv_action":
                    event_key = self._event_key(event)
                    apply_once = bool(event.params.get("apply_once", True))
                    if not apply_once or event_key not in self._logged_training_events:
                        self._apply_forced_bv_action(event)
                    if event.params.get("calibration_only") is not True:
                        self._record_forced_training_step(event)
                elif event.type == "calibration_cav_action":
                    self._apply_calibration_cav_action(event)

    def _step(self):
        control_info_list = super()._step()
        self.apply_template_events()
        if self.cav_fault_model.enabled:
            self.info_extractor.episode_log.setdefault("cav_fault_step_info", {})[
                f"{self.simulator.get_time():.6f}"
            ] = self.cav_fault_model.last_audit
        return control_info_list

    def _terminate_check(self):
        """Preserve safety exits and also honor the template duration."""
        reason, stop, additional_info = super()._terminate_check()
        if stop or not _duration_reached(
            self.simulator.get_time(), self.scenario_template.duration
        ):
            return reason, stop, additional_info
        return (
            {5: "scenario duration reached"},
            True,
            {"scenario_duration": self.scenario_template.duration},
        )

    def _generate_actor(self, actor: VehicleSpec):
        vehicle = Vehicle(
            id=actor.id,
            controller=Controller(),
            routeID=actor.route,
            simulator=self.simulator,
            initial_speed=actor.speed,
            initial_position=actor.position,
            initial_lane_id=self._lane_id_from_index(actor.lane_index),
        )
        self.simulator._add_vehicle_to_sumo(vehicle, typeID="IDM")
        vehicle.install_controller(self.independent_controller_dict["BV"]())
        self.vehicle_list.add_vehicles([vehicle])
        self.info_extractor.add_initialization_info(
            actor.id,
            {
                "speed": actor.speed,
                "lane_id": vehicle.initial_lane_id,
                "route_id": actor.route,
                "position": actor.position,
                "controller": actor.controller,
            },
        )

    def _apply_forced_bv_action(self, event: EventSpec):
        if event.actor not in self.vehicle_list:
            return
        if event.params.get("calibration_only") is True:
            self._apply_calibration_longitudinal_action(event)
            return
        vehicle = self.vehicle_list[event.actor]
        action = {
            "lateral": str(event.params.get("lateral", "central")),
            "longitudinal": float(event.params.get("longitudinal", 0.0)),
        }
        if not vehicle.is_action_legal(action):
            return
        vehicle.controller.action = action
        vehicle.act(action)

    def _apply_calibration_cav_action(self, event: EventSpec):
        """Override the CAV response for an explicitly calibration-only window.

        This is an intervention used to test crash reachability.  It is not a
        perception-delay or control-delay implementation and is deliberately
        excluded from the D2RL training-event bookkeeping below.
        """
        if event.actor != self.scenario_template.ego.id:
            raise ValueError("calibration_cav_action may target only the ego CAV")
        if event.params.get("calibration_only") is not True:
            raise ValueError("calibration_cav_action requires calibration_only=true")
        if event.params.get("not_for_d2rl_training") is not True:
            raise ValueError(
                "calibration_cav_action requires not_for_d2rl_training=true"
            )
        self._apply_calibration_longitudinal_action(event)

    def _apply_calibration_longitudinal_action(self, event: EventSpec):
        """Apply low-speed calibration acceleration without the 20 m/s clamp.

        ``Vehicle.act`` uses the project's global high-speed action bounds and
        therefore clips SHRP2 low-speed actions to at least 20 m/s.  Calibration
        events use the simulator's acceleration primitive directly while still
        disabling SUMO's speed and lane-change safety checks explicitly.
        """
        if event.actor not in self.vehicle_list:
            return
        vehicle = self.vehicle_list[event.actor]
        self.simulator.set_vehicle_speedmode(vehicle.id, 0)
        self.simulator.set_vehicle_lanechangemode(vehicle.id, 0)
        current_lane_offset = self.simulator.get_vehicle_lateral_lane_position(
            vehicle.id
        )
        self.simulator.change_vehicle_sublane_dist(
            vehicle.id, -current_lane_offset, self.simulator.step_size
        )
        self.simulator.change_vehicle_speed(
            vehicle.id,
            float(event.params.get("longitudinal", 0.0)),
            vehicle.action_step_size,
        )

    def _record_forced_training_step(self, event: EventSpec):
        event_key = self._event_key(event)
        if event_key in self._logged_training_events:
            return
        self._logged_training_events.add(event_key)

        time_step = self._current_log_time()
        control_log = self.global_controller_instance_list[0].control_log
        metrics = self._estimate_event_metrics(event)
        multi_bv_num = int(event.params.get("multi_bv_num", 2))
        selection = select_multibv_training_actors(
            self,
            primary_actor_id=event.actor,
            agent_num=multi_bv_num,
        )
        if selection is None:
            return

        per_agent_metrics = [
            self._estimate_actor_metrics(actor_id) for actor_id in selection.ids
        ]
        per_agent_ndd = [
            float(metric["ndd_possi"]) for metric in per_agent_metrics
        ]
        if "ndd_possi" in event.params:
            per_agent_ndd[0] = float(event.params["ndd_possi"])
        elif control_log.get("ndd_possi") is not None:
            per_agent_ndd[0] = min(float(control_log["ndd_possi"]), per_agent_ndd[0])
        per_agent_weight = [
            float(min(0.09, max(1e-6, ndd_possi * 5000)))
            for ndd_possi in per_agent_ndd
        ]
        if "training_weight" in event.params:
            per_agent_weight[0] = float(event.params["training_weight"])
        joint_ndd = float(_product(per_agent_ndd))
        joint_weight = float(_product(per_agent_weight))
        criticality = float(control_log.get("criticality", 0.0))
        if criticality <= 0:
            criticality = float(event.params.get("criticality", metrics["criticality"]))
        epsilon = event.params.get("epsilon_placeholder", 0.5)
        if isinstance(epsilon, list):
            epsilon_values = [float(value) for value in epsilon[:multi_bv_num]]
        else:
            epsilon_values = [float(epsilon)] * multi_bv_num
        while len(epsilon_values) < multi_bv_num:
            epsilon_values.append(epsilon_values[-1])

        log = self.info_extractor.episode_log
        log["weight_step_info"][time_step] = {
            "joint": joint_weight,
            "per_agent": per_agent_weight,
        }
        log["drl_obs_step_info"][time_step] = {
            "joint": selection.joint_obs,
            "per_agent": selection.per_agent_obs,
        }
        log["criticality_step_info"][time_step] = criticality
        log["ndd_step_info"][time_step] = {
            "joint": joint_ndd,
            "per_agent": per_agent_ndd,
        }
        log["drl_epsilon_step_info"][time_step] = epsilon_values
        log["real_epsilon_step_info"][time_step] = epsilon_values
        log.setdefault("controlled_bv_ids_step_info", {})[time_step] = selection.ids
        log["current_weight"] = joint_weight
        log["weight_episode"] *= joint_weight
        log.setdefault("scenario_event_metrics", {})[time_step] = metrics
        log.setdefault("multi_bv_debug_step_info", {})[time_step] = {
            "requested_control_num": multi_bv_num,
            "selected_count": len(selection.ids),
            "underfilled": len(selection.ids) < multi_bv_num,
            "selected_bv_ids": selection.ids,
        }

    def _estimate_event_metrics(self, event: EventSpec) -> dict[str, float]:
        return self._estimate_actor_metrics(event.actor)

    def _estimate_actor_metrics(self, actor_id: str) -> dict[str, float]:
        cav = self.vehicle_list.get("CAV")
        actor = self.vehicle_list.get(actor_id)
        if cav is None or actor is None:
            return {
                "distance": 10000.0,
                "relative_speed": 0.0,
                "ttc": 10000.0,
                "ndd_possi": 1e-5,
                "criticality": 1.0,
            }

        cav_info = cav.observation.information["Ego"]
        actor_info = actor.observation.information["Ego"]
        distance = actor_info["position"][0] - cav_info["position"][0] - 5.0
        relative_speed = cav_info["velocity"] - actor_info["velocity"]
        if relative_speed > 0 and distance > 0:
            ttc = distance / relative_speed
        else:
            ttc = 10000.0

        # Rare-event proxy: closer, faster-closing cut-ins get smaller NDD probability.
        distance_term = max(0.0, min(distance, 40.0)) / 40.0
        ttc_term = max(0.0, min(ttc, 5.0)) / 5.0
        speed_term = max(0.0, min(abs(relative_speed), 20.0)) / 20.0
        ndd_possi = 1e-6 + 1.8e-5 * (0.45 * distance_term + 0.35 * ttc_term + 0.20 * (1.0 - speed_term))
        ndd_possi = float(max(1e-6, min(1.9e-5, ndd_possi)))
        criticality = float(max(0.01, min(1.0, 1.0 - min(ttc, 5.0) / 5.0)))
        return {
            "distance": float(distance),
            "relative_speed": float(relative_speed),
            "ttc": float(ttc),
            "ndd_possi": ndd_possi,
            "criticality": criticality,
        }

    def _lane_id_from_index(self, lane_index: int) -> str:
        lanes = self.simulator.get_available_lanes()
        lane_ids = sorted(lane.getID() for lane in lanes)
        if lane_index >= len(lane_ids):
            raise ValueError(
                f"lane_index={lane_index} is out of range for lanes {lane_ids}"
            )
        return lane_ids[lane_index]

    @staticmethod
    def _event_is_active(event: EventSpec, current_time: float) -> bool:
        return event.start_time <= current_time < event.start_time + event.duration

    @staticmethod
    def _event_key(event: EventSpec) -> str:
        return f"{event.type}:{event.actor}:{event.start_time}:{event.duration}"

    def _current_log_time(self) -> str:
        return f"forced_{self.simulator.get_time():.6f}"


def _product(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= float(value)
    return result


def _duration_reached(current_time: float, duration: float) -> bool:
    return float(current_time) >= float(duration)
