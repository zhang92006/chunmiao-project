from numpy import random
from numpy.core.numeric import full
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
import numpy as np
from copy import deepcopy
import collections
from itertools import combinations
import utils
from conf import conf
from controller.nddglobalcontroller import NDDBVGlobalController
from scenario_reconstruction.multibv import build_multibv_joint_obs
from scenario_reconstruction.joint_criticality import (
    joint_pair_proposal,
    pairwise_joint_criticality_details,
    sample_joint_action_pair,
)
from scenario_reconstruction.importance import probability_record

class NADEBVGlobalController(NDDBVGlobalController):
    controlled_bv_num = 4

    def __init__(self, env, veh_type="BV"):
        super().__init__(env, veh_type)
        self.joint_control_num = max(1, int(getattr(env, "multi_bv_control_num", 1)))
        self.drl_info = None
        self.drl_epsilon_value = -1
        self.real_epsilon_value = -1


    # @profile
    def step(self):
        """Control the selected bvs from the bvs candidates to realize the decided behavior
        """
        self.real_epsilon_value = -1
        self.drl_epsilon_value = -1
        self.control_log = {
            "criticality": 0,
            "discriminator_input": 0,
            "proposal_mode": self._proposal_mode(),
        }
        bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list = [], [], [], [], [], []
        vehicle_criticality_list = []
        self.reset_control_and_action_state()
        self.update_subscription(controller=TreeSearchNADEBackgroundController)

        # NDE decision
        for bv_id in self.controllable_veh_id_list:
            bv = self.env.vehicle_list[bv_id]
            bv.controller.step()

        # D2RL-trained intelligent testing environment (NADE) decision
        if self.apply_control_permission():
            if conf.experiment_config["mode"] == "NDE":
                for bv_id in self.controllable_veh_id_list:
                    bv = self.env.vehicle_list[bv_id]
                    if self.apply_control_permission():
                        bv.update()
            elif conf.experiment_config["mode"] != "NDE":
                bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, _ = self.select_controlled_bv_and_action()
                for bv_id in self.controllable_veh_id_list:
                    bv = self.env.vehicle_list[bv_id]
                    if bv in controlled_bvs_list:
                        nade_action = bv_action_idx_list[controlled_bvs_list.index(bv)]
                        if nade_action is not None:
                            self.control_log["ndd_possi"] = ndd_possi_list[controlled_bvs_list.index(bv)]
                            bv.controller.action = utils.action_id_to_action_command(
                                nade_action)
                            bv.controller.NADE_flag = True
                            bv.simulator.set_vehicle_color(bv.id, bv.color_blue)
                    if self.apply_control_permission():
                        bv.update() # apply bv.controller.action
            else:
                raise ValueError("conf experiment mode not recognized. should be NDE or D2RL")
        joint_proposal = self.control_log.get("joint_proposal_record")
        if joint_proposal is not None:
            self.control_log["weight_list_per_simulation"] = [
                float(joint_proposal["importance_weight"])
            ]
        else:
            self.control_log["weight_list_per_simulation"] = [
                val for val in weight_list if val is not None]
        if len(self.control_log["weight_list_per_simulation"]) == 0:
            self.control_log["weight_list_per_simulation"] = [1]
        if self.joint_control_num > 1:
            self._record_joint_training_context(
                controlled_bvs_list, weight_list, ndd_possi_list, vehicle_criticality_list
            )
        return vehicle_criticality_list

    def _record_joint_training_context(
        self, controlled_bvs_list, weight_list, ndd_possi_list, vehicle_criticality_list
    ):
        """Expose K selected BV observations and proposal terms to the extractor."""
        selected = [
            (index, bv)
            for index, bv in enumerate(controlled_bvs_list)
            if weight_list[index] is not None and ndd_possi_list[index] is not None
        ]
        if len(selected) != self.joint_control_num:
            return
        selected_indices = [index for index, _ in selected]
        selected_bvs = [bv for _, bv in selected]
        full_obs = getattr(self, "_joint_full_obs", None)
        if not full_obs:
            return
        selected_ids = [bv.id for bv in selected_bvs]
        episode_log = self.env.info_extractor.episode_log
        episode_weight = episode_log.get("weight_episode", 1.0)
        log_episode_weight = episode_log.get("log_importance_weight")
        self.control_log["joint_training"] = True
        self.control_log["joint_controlled_bv_ids"] = selected_ids
        joint_proposal = self.control_log.get("joint_proposal_record")
        self.control_log["weight_list_per_agent"] = [
            float(weight_list[index]) for index in selected_indices
        ]
        self.control_log["ndd_possi_list_per_agent"] = [
            float(ndd_possi_list[index]) for index in selected_indices
        ]
        self.control_log["drl_obs_joint"] = build_multibv_joint_obs(
            full_obs, selected_ids, episode_weight, log_episode_weight
        )
        self.control_log["drl_obs_per_agent"] = [
            build_multibv_joint_obs(
                full_obs, [bv_id], episode_weight, log_episode_weight
            )
            for bv_id in selected_ids
        ]
        self.control_log["discriminator_input"] = {
            "joint": self.control_log["drl_obs_joint"],
            "per_agent": self.control_log["drl_obs_per_agent"],
        }
        if joint_proposal is not None:
            self.control_log["weight_record"] = dict(joint_proposal["weight_record"])
            self.control_log["ndd_record"] = dict(joint_proposal["ndd_record"])
        else:
            fallback_proposal_type = (
                "naturalistic"
                if self._proposal_mode() == "naturalistic"
                else "factorized"
            )
            joint_weight = float(np.prod(self.control_log["weight_list_per_agent"]))
            joint_naturalistic = float(
                np.prod(self.control_log["ndd_possi_list_per_agent"])
            )
            joint_proposal_probability = (
                joint_naturalistic / joint_weight if joint_weight > 0.0 else None
            )
            self.control_log["weight_record"] = {
                "proposal_type": fallback_proposal_type,
                "joint": joint_weight,
                "per_agent": self.control_log["weight_list_per_agent"],
                "joint_naturalistic_probability": joint_naturalistic,
                "joint_proposal_probability": joint_proposal_probability,
            }
            self.control_log["ndd_record"] = {
                "proposal_type": fallback_proposal_type,
                "joint": joint_naturalistic,
                "per_agent": self.control_log["ndd_possi_list_per_agent"],
            }
        self._record_probability_terms()
        if self.drl_epsilon_value != -1:
            epsilon = self.drl_epsilon_value
            if isinstance(epsilon, (list, tuple, np.ndarray)):
                epsilon_values = [float(value) for value in list(epsilon)]
                if not epsilon_values:
                    epsilon_values = [0.0]
                epsilon_values = epsilon_values[: len(selected_bvs)]
                epsilon_values.extend(
                    [epsilon_values[-1]] * (len(selected_bvs) - len(epsilon_values))
                )
            else:
                epsilon_values = [float(epsilon) for _ in selected_bvs]
            if joint_proposal is not None:
                epsilon_values = [float(joint_proposal["epsilon"])] * len(selected_bvs)
            self.drl_epsilon_value = epsilon_values
            self.real_epsilon_value = list(epsilon_values)

    # @profile
    def select_controlled_bv_and_action(self):
        """Select the background vehicle controlled by D2RL-trained intelligent testing environment (NADE) and the corresponding action.

        Returns:
            list(float): List of action index for all studied background vehicles. 
            list(float): List of weight of each vehicle.
            float: Maximum criticality.
            list(float): List of behavior probability based on NDD.
            list(float): List of critical possibility.
            list(Vehicle): List of all studied vehicles.
        """
        num_controlled_critical_bvs = self.joint_control_num
        controlled_bvs_list = self.get_bv_candidates()
        CAV_obs = self.env.vehicle_list["CAV"].observation.information
        full_obs = self.get_full_obs_from_cav_obs_and_bv_list(CAV_obs, controlled_bvs_list)
        self._joint_full_obs = full_obs
        self.nade_candidates = controlled_bvs_list
        bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list = self.calculate_criticality_list(controlled_bvs_list, CAV_obs, full_obs)
        if self.joint_control_num > 1:
            bv_criticality_list, criticality_array_list = self._augment_pairwise_criticality(
                controlled_bvs_list, full_obs, criticality_array_list
            )
        whole_weight_list = []
        self.control_log["criticality"] = sum(bv_criticality_list)
        selected_bv_idx = sorted(
            range(len(bv_criticality_list)), key=lambda i: bv_criticality_list[i]
        )[-num_controlled_critical_bvs:]
        if self.joint_control_num > 1 and len(selected_bv_idx) == self.joint_control_num:
            selected_ids = [controlled_bvs_list[index].id for index in selected_bv_idx]
            discriminator_input = np.asarray(
                build_multibv_joint_obs(
                    full_obs,
                    selected_ids,
                    self.env.info_extractor.episode_log["weight_episode"],
                    self.env.info_extractor.episode_log.get("log_importance_weight"),
                ),
                dtype=np.float32,
            )
        else:
            discriminator_input = self.collect_discriminator_input_simplified(
                full_obs, controlled_bvs_list, bv_criticality_list
            )
        self.control_log["discriminator_input"] = discriminator_input.tolist()
        self.epsilon_value = -1
        underline_drl_action = self.get_underline_drl_action(discriminator_input, bv_criticality_list)
        proposal_mode = self._proposal_mode()
        epsilon_by_index, selected_epsilon_values = self._selected_epsilon_values(
            underline_drl_action, selected_bv_idx
        )
        if proposal_mode == "naturalistic":
            epsilon_by_index = {index: 1.0 for index in selected_bv_idx}
            selected_epsilon_values = [1.0 for _ in selected_bv_idx]
        if sum(bv_criticality_list) > 0:
            self.drl_epsilon_value = selected_epsilon_values
            self.real_epsilon_value = list(selected_epsilon_values)

        for i in range(len(controlled_bvs_list)):
            bv = controlled_bvs_list[i]
            bv_criticality = bv_criticality_list[i]
            bv_criticality_array = criticality_array_list[i]
            bv_pdf = bv.controller.get_NDD_possi()
            combined_bv_criticality_array = bv_criticality_array
            if i not in selected_bv_idx:
                bv_action_idx, weight, ndd_possi, critical_possi, single_weight_list = (
                    None, None, None, None, None
                )
            else:
                bv_action_idx, weight, ndd_possi, critical_possi, single_weight_list = bv.controller.Decompose_sample_action(
                    np.sum(combined_bv_criticality_array),
                    combined_bv_criticality_array,
                    bv_pdf,
                    epsilon_by_index[i],
                )
            if bv_action_idx is not None:
                bv_action_idx = bv_action_idx.item()
            bv_action_idx_list.append(bv_action_idx), weight_list.append(weight), ndd_possi_list.append(ndd_possi), IS_possi_list.append(critical_possi)
            if single_weight_list is not None:
                whole_weight_list.append(min(single_weight_list))
            else:
                whole_weight_list.append(None)

        joint_proposal = None
        if proposal_mode == "joint_pair":
            joint_proposal = self._sample_selected_joint_pair(
                selected_bv_idx, epsilon_by_index
            )
        if joint_proposal is not None:
            for index, action, marginal_weight, marginal_ndd, marginal_proposal in zip(
                joint_proposal["indices"],
                joint_proposal["action_pair"],
                joint_proposal["per_agent_marginal_weight"],
                joint_proposal["per_agent_ndd_probability"],
                joint_proposal["per_agent_proposal_probability"],
            ):
                bv_action_idx_list[index] = int(action)
                weight_list[index] = float(marginal_weight)
                ndd_possi_list[index] = float(marginal_ndd)
                IS_possi_list[index] = float(marginal_proposal)
                whole_weight_list[index] = float(joint_proposal["importance_weight"])
            self.control_log["joint_proposal_record"] = joint_proposal
                
        vehicle_criticality_list = deepcopy(bv_criticality_list)
        raw_weight_list = list(weight_list)
        raw_ndd_possi_list = list(ndd_possi_list)
        for i in range(len(controlled_bvs_list)):
            if i in selected_bv_idx:
                if whole_weight_list[i] and whole_weight_list[i]*self.env.info_extractor.episode_log["weight_episode"]*self.env.initial_weight < conf.weight_threshold:
                    bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], IS_possi_list[i] = None, None, None, None
            if i not in selected_bv_idx:
                bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], IS_possi_list[i] = None, None, None, None
        if self.joint_control_num > 1:
            self.control_log["multibv_selection_debug"] = {
                "candidate_ids": [bv.id for bv in controlled_bvs_list],
                "criticality": [
                    None if value is None else float(value)
                    for value in bv_criticality_list
                ],
                "raw_weight": [
                    None if value is None else float(value) for value in raw_weight_list
                ],
                "raw_ndd_possi": [
                    None if value is None else float(value) for value in raw_ndd_possi_list
                ],
                "selected_candidate_ids": [
                    controlled_bvs_list[index].id
                    for index in selected_bv_idx
                    if index < len(controlled_bvs_list)
                    and weight_list[index] is not None
                    and ndd_possi_list[index] is not None
                ],
                "sampled_action_ids": [
                    None if action_id is None else int(action_id)
                    for action_id in bv_action_idx_list
                ],
                "pair_criticality": self.control_log.get("joint_pair_criticality", []),
                "joint_proposal": self.control_log.get("joint_proposal_record"),
                "proposal_mode": proposal_mode,
            }
        if len(bv_criticality_list):
            max_vehicle_criticality = np.max(bv_criticality_list)
        else:
            max_vehicle_criticality = -np.inf

        return bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, discriminator_input

    def _proposal_mode(self):
        """Read the rollout proposal family without changing legacy defaults."""
        return getattr(self.env, "multibv_proposal_mode", "joint_pair")

    def _record_probability_terms(self):
        """Expose exact joint p/q terms to the episode information extractor."""
        weight_record = self.control_log.get("weight_record")
        ndd_record = self.control_log.get("ndd_record")
        if not isinstance(weight_record, dict) or not isinstance(ndd_record, dict):
            return
        naturalistic = weight_record.get(
            "joint_naturalistic_probability", ndd_record.get("joint")
        )
        proposal = weight_record.get("joint_proposal_probability")
        if proposal is None:
            joint_weight = float(weight_record.get("joint", 0.0))
            proposal = float(naturalistic) / joint_weight if joint_weight > 0.0 else None
        if naturalistic is None or proposal is None:
            return
        try:
            record = probability_record(
                str(weight_record.get("proposal_type", self._proposal_mode())),
                float(naturalistic),
                float(proposal),
            )
        except (TypeError, ValueError):
            return
        if not np.isclose(record["importance_weight"], float(weight_record["joint"])):
            raise ValueError("Recorded joint p/q disagrees with the sampled weight")
        self.control_log["probability_record"] = record

    def _augment_pairwise_criticality(
        self, controlled_bvs_list, full_obs, criticality_array_list
    ):
        """Add bounded BV-pair risk while preserving factorised NADE sampling."""
        enhanced_arrays = [np.asarray(values, dtype=float).copy() for values in criticality_array_list]
        pair_debug = []
        self._joint_pair_details = {}
        for first_index, second_index in combinations(range(len(controlled_bvs_list)), 2):
            first_bv, second_bv = controlled_bvs_list[first_index], controlled_bvs_list[second_index]
            first_array, second_array, debug, details = pairwise_joint_criticality_details(
                full_obs,
                first_bv.id,
                second_bv.id,
                first_bv.controller.get_NDD_possi(),
                second_bv.controller.get_NDD_possi(),
            )
            enhanced_arrays[first_index] += first_array
            enhanced_arrays[second_index] += second_array
            pair_debug.append(debug)
            self._joint_pair_details[(first_index, second_index)] = {
                "details": details,
                "ids": [first_bv.id, second_bv.id],
            }
        self.control_log["joint_pair_criticality"] = pair_debug
        enhanced_list = [float(np.sum(values)) for values in enhanced_arrays]
        return enhanced_list, enhanced_arrays

    def _sample_selected_joint_pair(self, selected_bv_idx, epsilon_by_index):
        """Sample an ordered BV action pair from a correlated IS proposal."""
        if self.joint_control_num != 2 or len(selected_bv_idx) != 2:
            return None
        key = tuple(sorted(int(index) for index in selected_bv_idx))
        pair = getattr(self, "_joint_pair_details", {}).get(key)
        if pair is None:
            return None
        epsilon = float(np.mean([epsilon_by_index[index] for index in key]))
        proposal = joint_pair_proposal(pair["details"], epsilon)
        if proposal is None:
            return None
        sampled = sample_joint_action_pair(proposal)
        first_action, second_action = sampled["action_pair"]
        naturalistic = proposal["naturalistic_pdf"]
        proposal_pdf = proposal["proposal_pdf"]
        per_agent_ndd = [
            float(np.sum(naturalistic[first_action, :])),
            float(np.sum(naturalistic[:, second_action])),
        ]
        per_agent_proposal = [
            float(np.sum(proposal_pdf[first_action, :])),
            float(np.sum(proposal_pdf[:, second_action])),
        ]
        marginal_weights = [
            natural / proposed
            for natural, proposed in zip(per_agent_ndd, per_agent_proposal)
        ]
        return {
            "proposal_type": "joint_pair",
            "selected_bv_ids": pair["ids"],
            "indices": list(key),
            "action_pair": [first_action, second_action],
            "epsilon": float(epsilon),
            "critical_mass": float(sampled["critical_mass"]),
            "naturalistic_probability": float(sampled["naturalistic_probability"]),
            "proposal_probability": float(sampled["proposal_probability"]),
            "importance_weight": float(sampled["importance_weight"]),
            "per_agent_ndd_probability": per_agent_ndd,
            "per_agent_proposal_probability": per_agent_proposal,
            "per_agent_marginal_weight": marginal_weights,
            "weight_record": {
                "proposal_type": "joint_pair",
                "joint": float(sampled["importance_weight"]),
                "per_agent": marginal_weights,
                "joint_naturalistic_probability": float(sampled["naturalistic_probability"]),
                "joint_proposal_probability": float(sampled["proposal_probability"]),
            },
            "ndd_record": {
                "proposal_type": "joint_pair",
                "joint": float(sampled["naturalistic_probability"]),
                "per_agent": per_agent_ndd,
            },
        }

    def _selected_epsilon_values(self, epsilon, selected_bv_idx):
        """Assign one epsilon to each selected BV in criticality rank order."""
        if isinstance(epsilon, (list, tuple, np.ndarray)):
            values = [float(value) for value in list(epsilon)]
        elif epsilon is None:
            values = [float(conf.epsilon_value)]
        else:
            values = [float(epsilon)]
        if not values:
            values = [float(conf.epsilon_value)]
        values = values[: len(selected_bv_idx)]
        values.extend([values[-1]] * (len(selected_bv_idx) - len(values)))
        return dict(zip(selected_bv_idx, values)), values

    def apply_control_permission(self):
        for vehicle in self.get_bv_candidates():
            if vehicle.controller.NADE_flag and utils.is_lane_change(vehicle.observation.information["Ego"]):
                return False
        return True
        
    # @profile
    def get_bv_candidates(self):
        """Find the Principal Other Vehicle (POV) candidates around the av

        Returns:
            list(Vehicle): List of background vehicles around the CAV.
        """
        av = self.env.vehicle_list["CAV"]
        av_pos = av.observation.local[av.id][66]
        av_context = av.observation.context
        av_context_bv_list = list(av_context.keys())
        bv_candidates = []
        bv_list = []
        # collect all bvs in a certain range of AV
        for bv_id in av_context_bv_list:
            bv_pos = av_context[bv_id][66]
            dist = utils.cal_euclidean_dist(av_pos, bv_pos)
            if dist <= conf.cav_obs_range:
                bv_list.append([bv_id, dist])
        # sort the bvs list by distance, and select the top controlled_bv_num nearest bvs
        bv_list.sort(key=lambda i: i[1])
        for i in range(len(bv_list)):
            if i < self.controlled_bv_num:
                bv_id = bv_list[i][0]
                bv = self.env.vehicle_list[bv_id]
                bv_candidates.append(bv)
        return bv_candidates

    @staticmethod
    # @profile
    def pre_load_predicted_obs_and_traj(full_obs):
        predicted_obs = {}
        trajectory_obs = {}
        action_list = ["left", "right", "still"]
        for veh_id in full_obs:
            for action in action_list:
                vehicle = full_obs[veh_id]
                if veh_id not in trajectory_obs:
                    trajectory_obs[veh_id] = {}
                if veh_id not in predicted_obs:
                    predicted_obs[veh_id] = {}
                predicted_obs[veh_id][action], trajectory_obs[veh_id][action] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(vehicle, action)
        return predicted_obs, trajectory_obs
    
    def collect_discriminator_input_simplified(self, full_obs, controlled_bvs_list, bv_criticality_list):
        """D2RL agent observation collection
        """
        CAV_global_position = list(full_obs["CAV"]["position"])
        CAV_speed = full_obs["CAV"]["velocity"]
        episode_log = self.env.info_extractor.episode_log
        log_weight = episode_log.get("log_importance_weight")
        if log_weight is not None and np.isfinite(float(log_weight)):
            tmp_weight = max(float(log_weight) / np.log(10.0), -300.0)
        else:
            tmp_weight = np.log10(max(float(episode_log["weight_episode"]), 1e-30))
        vehicle_info_list = []
        controlled_bv_num = 1
        total_bv_info_length = controlled_bv_num * 4
        # print(bv_criticality_list)
        if len(bv_criticality_list):
            selected_bv_index = np.argmax(np.array(bv_criticality_list))
            vehicle = controlled_bvs_list[selected_bv_index]
            veh_id = vehicle.id
            vehicle_single_obs = full_obs[veh_id]
            vehicle_local_position = list(vehicle_single_obs["position"])
            vehicle_relative_position = [vehicle_local_position[0]-CAV_global_position[0], vehicle_local_position[1]-CAV_global_position[1]]
            vehicle_relative_speed = vehicle_single_obs["velocity"] - CAV_speed
            predict_relative_position = vehicle_relative_position[0] + vehicle_relative_speed
            vehicle_info_list.extend(vehicle_relative_position +[vehicle_relative_speed] + [predict_relative_position])
        else:
            vehicle_info_list.extend([-20, -8, -10, -20]) # fill the state space with default values
        if len(vehicle_info_list) < total_bv_info_length:
            vehicle_info_list.extend([-1]*(total_bv_info_length - len(vehicle_info_list)))
        bv_criticality_flag = (sum(bv_criticality_list) > 0)
        if sum(bv_criticality_list) > 0:
            bv_criticality_value = np.log10(sum(bv_criticality_list))
        else:
            bv_criticality_value = 16
        if conf.simulation_config["map"] == "2LaneLong": # 2lane 4000m experiment
            CAV_position_lb, CAV_position_ub = [400, 40], [4400, 50]
        else: # 2lane 400m experiment
            CAV_position_lb, CAV_position_ub = [400, 40], [800, 50]
        CAV_velocity_lb, CAV_velocity_ub = 0, 20
        weight_lb = -30
        weight_ub = 0
        bv_criticality_flag_lb = 0
        bv_criticality_flag_ub = 1
        bv_criticality_value_lb = -16
        bv_criticality_value_ub = 0
        vehicle_info_lb, vehicle_info_ub = [-20, -8, -10, -20], [20, 8, 10, 20]
        lb_array = np.array(CAV_position_lb + [CAV_velocity_lb] + [weight_lb] + [bv_criticality_flag_lb] + [bv_criticality_value_lb] + vehicle_info_lb * controlled_bv_num)
        ub_array = np.array(CAV_position_ub + [CAV_velocity_ub] + [weight_ub] + [bv_criticality_flag_ub] + [bv_criticality_value_ub] + vehicle_info_ub * controlled_bv_num)
        total_obs_for_DRL_ori = np.array(CAV_global_position + [CAV_speed] + [tmp_weight] + [bv_criticality_flag] + [bv_criticality_value] + vehicle_info_list)
        total_obs_for_DRL = 2 * (total_obs_for_DRL_ori - lb_array)/(ub_array - lb_array) - 1 # normalize the observation
        total_obs_for_DRL = np.clip(total_obs_for_DRL, -5, 5) # clip the observation
        return np.float32(np.array(total_obs_for_DRL))

    def calculate_criticality_list(self, controlled_bvs_list, CAV_obs, full_obs):
        bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list = [], [], [], [], [], []        
        predicted_full_obs, predicted_traj_obs = NADEBVGlobalController.pre_load_predicted_obs_and_traj(full_obs)
        CAV_left_prob, CAV_still_prob, CAV_right_prob = NADEBVGlobalController._get_Surrogate_CAV_action_probability(cav_obs=self.env.vehicle_list["CAV"].observation.information)
        for bv in controlled_bvs_list:
            bv_criticality, criticality_array = bv.controller.Decompose_decision(
                CAV_obs, SM_LC_prob=[CAV_left_prob, CAV_still_prob, CAV_right_prob], full_obs=full_obs, predicted_full_obs=predicted_full_obs, predicted_traj_obs=predicted_traj_obs)
            bv_criticality_list.append(bv_criticality), criticality_array_list.append(criticality_array)
        return bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list


    def get_underline_drl_action(self, discriminator_input, bv_criticality_list):
        underline_drl_action = None # 1 - adversarial maneuver probability
        if sum(bv_criticality_list) > 0:
            # critical time step
            if conf.simulation_config["epsilon_setting"] == "drl": # using D2RL agent to output the adversarial maneuver probability
                if conf.discriminator_agent is None:
                    conf.discriminator_agent = conf.load_discriminator_agent()
                underline_drl_action = conf.discriminator_agent.compute_action(discriminator_input)
                if sum(bv_criticality_list) > 0:
                    print(underline_drl_action, self.env.info_extractor.episode_log["weight_episode"])
                values = np.asarray(underline_drl_action, dtype=float).reshape(-1)
                if len(values) == 1:
                    underline_drl_action = float(np.clip(values[0], 0.0, 1.0))
                else:
                    underline_drl_action = np.clip(values, 0.0, 1.0).tolist()
            elif conf.simulation_config["epsilon_setting"] == "fixed": # ! need to be corrected
                underline_drl_action = conf.epsilon_value
                underline_drl_action = conf.epsilon_value
        return underline_drl_action

    @staticmethod
    # @profile
    def _get_Surrogate_CAV_action_probability(cav_obs):
        """Predict the action probability of the CAV based on surrogate model"""
        CAV_left_prob, CAV_right_prob = 0, 0
        CAV_still_prob = conf.epsilon_still_prob
        left_gain, right_gain = 0, 0
        left_LC_safety_flag, right_LC_safety_flag = False, False
        lane_index_list = [-1, 1]  # -1: right turn; 1: left turn
        for lane_index in lane_index_list:
            LC_safety_flag, gain = NADEBVGlobalController._Mobil_surraget_model(
                cav_obs=cav_obs, lane_index=lane_index)
            if gain is not None:
                if lane_index == -1:
                    right_gain = np.clip(gain, 0., None)
                    right_LC_safety_flag = LC_safety_flag
                elif lane_index == 1:
                    left_gain = np.clip(gain, 0., None)
                    left_LC_safety_flag = LC_safety_flag
        assert(left_gain >= 0 and right_gain >= 0)

        if not cav_obs["Ego"]["could_drive_adjacent_lane_left"]:
            left_LC_safety_flag = 0
            left_gain = 0
        elif not cav_obs["Ego"]["could_drive_adjacent_lane_right"] == 0:
            right_LC_safety_flag = 0
            right_gain = 0

        CAV_left_prob += conf.epsilon_lane_change_prob*left_LC_safety_flag
        CAV_right_prob += conf.epsilon_lane_change_prob*right_LC_safety_flag

        max_remaining_LC_prob = 1-conf.epsilon_still_prob-CAV_left_prob-CAV_right_prob

        total_gain = left_gain+right_gain
        obtained_LC_prob_for_sharing = np.clip(utils.remap(total_gain, [0, conf.SM_MOBIL_max_gain_threshold], [
                                               0, max_remaining_LC_prob]), 0, max_remaining_LC_prob)
        CAV_still_prob += (max_remaining_LC_prob -
                           obtained_LC_prob_for_sharing)

        if total_gain > 0:
            CAV_left_prob += obtained_LC_prob_for_sharing * \
                (left_gain/(left_gain + right_gain))
            CAV_right_prob += obtained_LC_prob_for_sharing * \
                (right_gain/(left_gain + right_gain))

        assert(0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001)
        return CAV_left_prob, CAV_still_prob, CAV_right_prob

    @staticmethod
    # @profile
    def _Mobil_surraget_model(cav_obs, lane_index):
        """Apply the Mobil surrogate model for CAV Lane change to calculate the gain for this lane change maneuver. If it does not have safety issue, then return True, gain; otherwise False, None.

        Args:
            lane_index (integer): Candidate lane for the change.

        Returns:
            (bool, float): The first output stands for safety flag (whether ADS will crash immediately after doing LC), the second output is gain (Now could even smaller than 0).
        """
        gain = None
        cav_info = cav_obs['Ego']

        if lane_index == -1:  # right turn
            new_preceding = cav_obs["RightLead"]
            new_following = cav_obs["RightFoll"]
        if lane_index == 1:  # left turn
            new_preceding = cav_obs["LeftLead"]
            new_following = cav_obs["LeftFoll"]

        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
        if new_following:
            r_new_following = new_following["distance"]
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=cav_info)

        old_preceding = cav_obs["Lead"]
        old_following = cav_obs["Foll"]
        self_pred_a = utils.acceleration(
            ego_vehicle=cav_info, front_vehicle=new_preceding)

        if new_following_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return True, 0

        # calculate acceleration advantage
        self_a = utils.acceleration(
            ego_vehicle=cav_info, front_vehicle=old_preceding)
        old_following_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=cav_info)
        old_following_pred_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + conf.Surrogate_POLITENESS * \
            (new_following_pred_a - new_following_a +
             old_following_pred_a - old_following_a)
        return True, gain

    # @profile
    def get_full_obs_from_cav_obs_and_bv_list(self, CAV_obs, bv_list):
        # This observation will be a dict containing CAV and all BV candidates
        full_obs = collections.OrderedDict()
        full_obs["CAV"] = CAV_obs["Ego"]
        vehicle_id_list = [vehicle.id for vehicle in bv_list]
        cav_surrounding = self._process_cav_context(vehicle_id_list)
        av_pos = CAV_obs["Ego"]["position"]
        for vehicle in bv_list:
            vehicle_id = vehicle.observation.information["Ego"]["veh_id"]
            full_obs[vehicle_id] = vehicle.observation.information["Ego"]
            bv_pos = vehicle.observation.information["Ego"]["position"]
        return full_obs

    # @profile
    def _process_cav_context(self, vehicle_id_list):
        """fetch information of all bvs from the cav context information
        """
        cav = self.env.vehicle_list["CAV"]
        cav_pos = cav.observation.local["CAV"][66]
        cav_context = cav.observation.context
        cav_surrounding = {}
        cav_surrounding["CAV"] = {
            "range": 0,
            "lane_width": self.env.simulator.get_vehicle_lane_width("CAV"),
            "lateral_offset": cav.observation.local["CAV"][184],
            "lateral_speed": cav.observation.local["CAV"][50],
            "position": cav_pos,
            "prev_action": cav.observation.information["Ego"]["prev_action"],
            "relative_lane_index": 0,
            "speed": cav.observation.local["CAV"][64]
        }
        total_vehicle_id_list = list(
            set(vehicle_id_list) | set(cav_context.keys()))
        for veh_id in total_vehicle_id_list:
            bv_pos = cav_context[veh_id][66]
            distance = self.env.simulator.get_vehicles_dist_road("CAV", veh_id)

            if distance > conf.cav_obs_range+5:
                distance_alter = self.env.simulator.get_vehicles_dist_road(
                    veh_id, "CAV")
                if distance_alter > conf.cav_obs_range+5:
                    continue
                else:
                    distance = -distance_alter
                    relative_lane_index = - \
                        self.env.simulator.get_vehicles_relative_lane_index(
                            veh_id, "CAV")
            else:
                relative_lane_index = self.env.simulator.get_vehicles_relative_lane_index(
                    "CAV", veh_id)
            cav_surrounding[veh_id] = {
                "range": distance,
                "lane_width": self.env.simulator.get_vehicle_lane_width(veh_id),
                "lateral_offset": cav_context[veh_id][184],
                "lateral_speed": cav_context[veh_id][50],
                "position": bv_pos,
                "prev_action": self.env.vehicle_list[veh_id].observation.information["Ego"]["prev_action"],
                "relative_lane_index": relative_lane_index,
                "speed": cav_context[veh_id][64]
            }
        return cav_surrounding
