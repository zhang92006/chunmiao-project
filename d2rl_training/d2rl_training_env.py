try:
	from gym import spaces, core
except ModuleNotFoundError:
	class _Box:
		def __init__(self, low, high, shape):
			self.low = low
			self.high = high
			self.shape = shape

		def sample(self):
			return np.random.uniform(self.low, self.high, self.shape).astype(np.float32)

	class _Env:
		@property
		def unwrapped(self):
			return self

	class _Spaces:
		Box = _Box

	class _Core:
		Env = _Env

	spaces = _Spaces()
	core = _Core()
import os, glob
import random
import json
import numpy as np
import logging


class D2RLTrainingEnv(core.Env):
	def __init__(self, yaml_conf):
		data_folders = [yaml_conf["root_folder"] + folder for folder in yaml_conf["data_folders"]]
		data_folder_weights = yaml_conf["data_folder_weights"]
		self.yaml_conf = yaml_conf
		self.multi_bv_training = bool(yaml_conf.get("multi_bv_training", False))
		self.multi_bv_num = int(yaml_conf.get("multi_bv_num", 2))
		if self.multi_bv_training and self.multi_bv_num < 1:
			raise ValueError("multi_bv_num must be positive when multi_bv_training=true")
		self.action_dim = self.multi_bv_num if self.multi_bv_training else 1
		self.observation_dim = 6 + 4 * self.multi_bv_num if self.multi_bv_training else 10
		self.action_space = spaces.Box(low=0.001, high=0.999, shape=(self.action_dim, ))
		self.observation_space = spaces.Box(low=-5, high=5, shape=(self.observation_dim, ))
		
		self.constant, self.weight_reward, self.exposure, self.positive_weight_reward=0,0,0,0 # some customized metric logging
		self.total_episode, self.total_steps = 0, 0
		if isinstance(data_folders, list):
			data_folder = random.choices(data_folders, weights=data_folder_weights)[0]
		else:
			data_folder = data_folders
		self.crash_data_path_list, self.safe_data_path_list, self.crash_data_weight_list, self.crash_target_weight_list = self.get_path_list(data_folder)
		self.all_data_path_list = self.crash_data_path_list + self.safe_data_path_list
		self.episode_data_path = ""
		self.episode_data = None
		
		self.unwrapped.trials = 100
		self.unwrapped.reward_threshold = 1.5
	
	def get_path_list(self, data_folder):
		crash_target_weight_list = None
		if os.path.exists(data_folder + "/crash_weight_dict.json"):
			with open(data_folder + "/crash_weight_dict.json") as data_file:
				crash_weight_dict = json.load(data_file)
				self.crash_weight_dict = crash_weight_dict
				crash_data_path_list = list(crash_weight_dict.keys())
				crash_data_weight_list = [crash_weight_dict[path][0] for path in crash_data_path_list]
				log_weight_path = os.path.join(data_folder, "crash_log_weight_dict.json")
				if os.path.exists(log_weight_path):
					crash_data_weight_list = self._stable_sampling_weights(
						crash_data_path_list,
						log_weight_path,
					)
		else:
			raise ValueError("No weight information!")
		tested_but_safe_path = os.path.join(data_folder, "tested_and_safe")
		if os.path.exists(data_folder + "/safe_weight_dict.json"):
			with open(data_folder + "/safe_weight_dict.json") as data_file:
				safe_weight_dict = json.load(data_file)
				safe_data_path_list = list(safe_weight_dict.keys())
		elif os.path.isdir(tested_but_safe_path):
			safe_data_path_list = glob.glob(tested_but_safe_path+"/*.json")
		else:
			safe_data_path_list = []
		logging.info(f'{len(crash_data_path_list)} Crash Events, {len(safe_data_path_list)} Safe Events')
		return crash_data_path_list, safe_data_path_list, crash_data_weight_list, crash_target_weight_list

	@staticmethod
	def _stable_sampling_weights(crash_paths, log_weight_path):
		"""Normalize finite log weights without changing their relative ratios.

		The scale is intentionally removed because ``random.choices`` only needs
		relative non-negative weights. It lets the training sampler consume valid
		rare-event episodes whose raw ``p/q`` underflowed to zero.
		"""
		with open(log_weight_path) as data_file:
			payload = json.load(data_file)
		log_weights = payload.get("log_weights", payload)
		values = np.asarray([float(log_weights[path]) for path in crash_paths], dtype=float)
		if not np.isfinite(values).all():
			raise ValueError("crash_log_weight_dict.json contains a non-finite log weight")
		stable = np.exp(values - np.max(values))
		if not np.isfinite(stable).all() or not np.any(stable > 0):
			raise ValueError("Could not derive positive stable crash sampling weights")
		return stable.tolist()
	
	def reset(self, episode_data_path=None):
		self.constant, self.weight_reward, self.exposure, self.positive_weight_reward=0,0,0,0
		self.total_episode = 0
		self.total_steps = 0
		self.episode_data_path = ""
		self.episode_data = None
		return self._reset(episode_data_path)

	def filter_episode_data(self, episode_data):
		invalid_timestep_list = []
		for timestep in episode_data["weight_step_info"]:
			weight = self._joint_value(episode_data["weight_step_info"][timestep])
			if weight < 1.001 and weight > 0.9:
				invalid_timestep_list.append(timestep)
				logging.debug(f"popping out {episode_data['weight_step_info']}")
		for invalid_time_step in invalid_timestep_list:
			episode_data["weight_step_info"].pop(invalid_time_step, None)
			episode_data["drl_epsilon_step_info"].pop(invalid_time_step, None)
			episode_data["real_epsilon_step_info"].pop(invalid_time_step, None)
			episode_data["criticality_step_info"].pop(invalid_time_step, None)
			episode_data["ndd_step_info"].pop(invalid_time_step, None)
			episode_data["drl_obs_step_info"].pop(invalid_time_step, None)
		# logging.debug(str(episode_data))
		return episode_data

	def sample_data_this_episode(self):
		if self.crash_data_weight_list:
			episode_data_path = random.choices(self.crash_data_path_list, weights=self.crash_data_weight_list)[0]
		else:
			raise ValueError("No weight information!")
		return episode_data_path
	
	def _reset(self, episode_data_path=None):
		self.total_episode += 1
		if not episode_data_path:
			self.episode_data_path = self.sample_data_this_episode()
		else:
			self.episode_data_path = episode_data_path
		with open(self.episode_data_path) as data_file:
			self.episode_data = self.filter_episode_data(json.load(data_file))
		if self.episode_data is not None:
			all_obs = self.episode_data["drl_obs_step_info"]
			time_step_list = list(all_obs.keys())
			if len(time_step_list):
				init_obs = np.float32(self._training_observation(all_obs[time_step_list[0]]))
				return init_obs
			else:
				return self._reset()
		else:
			return self._reset()
	
	def step(self, action):
		action = self._normalize_action(action)
		obs = self._get_observation()
		done, _ = self._get_done()
		time_step_list = list(self.episode_data["drl_obs_step_info"].keys())
		criticality_this_step = self.episode_data["criticality_step_info"][time_step_list[self.total_steps]]
		self.episode_data["drl_epsilon_step_info"][time_step_list[self.total_steps]] = action
		reward = self._get_reward()
		info = self._get_info()
		self.total_steps += 1
		return obs, reward, done, info

	def _get_info(self):
		return {}
	
	def close(self):
		return
	
	def _get_observation(self):
		all_obs = self.episode_data["drl_obs_step_info"]
		time_step_list = list(all_obs.keys())
		try:
			obs = np.float32(self._training_observation(all_obs[time_step_list[self.total_steps]]))
		except:
			print(self.total_steps, time_step_list)
			obs = np.float32(self._training_observation(all_obs[time_step_list[-1]]))
		return obs

	@staticmethod
	def _joint_value(record):
		"""Return a scalar from either legacy or MultiBV step records."""
		if isinstance(record, dict):
			return float(record["joint"])
		return float(record)

	@staticmethod
	def _primary_observation(record):
		"""Project MultiBV observations to the legacy 10-D primary-agent view."""
		if isinstance(record, dict):
			per_agent = record.get("per_agent", [])
			if not per_agent:
				raise ValueError("MultiBV observation has no per_agent entries")
			return per_agent[0]
		return record

	def _training_observation(self, record):
		"""Return the configured legacy or centralized MultiBV observation.

		The legacy path deliberately keeps the original 10-D first-agent view.
		With ``multi_bv_training=true``, the learner instead receives the logged
		joint CAV-plus-K-BV observation and must provide one epsilon per BV.
		"""
		if not self.multi_bv_training:
			return self._primary_observation(record)
		if not isinstance(record, dict):
			raise ValueError("MultiBV training requires joint observation records")
		joint = record.get("joint")
		if not isinstance(joint, list) or len(joint) != self.observation_dim:
			raise ValueError(
				f"Expected {self.observation_dim}-D joint observation for "
				f"K={self.multi_bv_num}"
			)
		return joint

	def _normalize_action(self, action):
		"""Validate a policy output and retain all K epsilon values in joint mode."""
		values = np.asarray(action, dtype=np.float32).reshape(-1)
		if len(values) != self.action_dim:
			raise ValueError(
				f"Expected {self.action_dim}-D action, got {len(values)} values"
			)
		if not np.isfinite(values).all() or (values < 0.001).any() or (values > 0.999).any():
			raise ValueError("D2RL epsilon actions must be finite and lie in [0.001, 0.999]")
		return values.tolist() if self.multi_bv_training else float(values[0])

	def get_multiple_adv_action_num(self, weight_info):
		adv_action_num = 0
		for timestep in weight_info:
			if self._joint_value(weight_info[timestep]) < 0.99:
				adv_action_num += 1
		return adv_action_num

	def _get_reward(self): # ! Aim to remove the magnitude of the environment
		stop, reason = self._get_done()
		if not stop:
			return 0
		else:			
			drl_epsilon_weight = self._get_drl_epsilon_weight(self.episode_data["weight_step_info"], self.episode_data["drl_epsilon_step_info"], self.episode_data["ndd_step_info"], self.episode_data["criticality_step_info"])
			if 1 in reason:
				print(self.episode_data["drl_epsilon_step_info"])
				adv_action_num = self.get_multiple_adv_action_num(self.episode_data["weight_step_info"])
				if adv_action_num > 1:
					return 0 # if multiple adversarial action is detected, this episode will be of no use
				clip_reward_threshold = self.yaml_conf["clip_reward_threshold"]
				q_amplifier_reward = clip_reward_threshold - drl_epsilon_weight * 500 * clip_reward_threshold # drl epsilon weight reward
				if q_amplifier_reward < -clip_reward_threshold:
					q_amplifier_reward = -clip_reward_threshold
				print("final_reward:", q_amplifier_reward)
				return q_amplifier_reward
			else:
				return 0

	def _get_drl_epsilon_weight(self, weight_info, epsilon_info, ndd_info, criticality_info=None):
		total_q_amplifier = 1
		for timestep in epsilon_info:
			if timestep in weight_info:
				if self.multi_bv_training:
					total_q_amplifier *= self._joint_epsilon_weight(
						weight_info[timestep],
						epsilon_info[timestep],
						ndd_info.get(timestep) if ndd_info is not None else None,
					)
					continue
				weight = self._joint_value(weight_info[timestep])
				epsilon = epsilon_info[timestep]
				if isinstance(epsilon, list):
					epsilon = epsilon[0]
				if weight > 1:
					total_q_amplifier = total_q_amplifier * (1/epsilon)
				elif weight < 0.999:
					if timestep not in ndd_info:
						ndd_tmp = criticality_info[timestep]
					else:
						ndd_tmp = self._joint_value(ndd_info[timestep])
					total_q_amplifier = total_q_amplifier * (1/(1- epsilon)) * ndd_tmp
		return total_q_amplifier	

	@staticmethod
	def _joint_epsilon_weight(weight_record, epsilon_record, ndd_record):
		"""Return the correct IS term for factorised or correlated BV proposals."""
		if not isinstance(weight_record, dict) or not isinstance(ndd_record, dict):
			raise ValueError("MultiBV importance weighting requires joint step records")
		if weight_record.get("proposal_type") == "joint_pair":
			naturalistic = float(weight_record["joint_naturalistic_probability"])
			proposal = float(weight_record["joint_proposal_probability"])
			if naturalistic < 0 or proposal <= 0:
				raise ValueError("Joint-pair proposal probabilities must be non-negative/positive")
			result = naturalistic / proposal
			if not np.isclose(result, float(weight_record["joint"])):
				raise ValueError("Joint-pair weight must equal naturalistic/proposal")
			return result
		weights = weight_record.get("per_agent")
		ndd_values = ndd_record.get("per_agent")
		epsilons = np.asarray(epsilon_record, dtype=float).reshape(-1)
		if not isinstance(weights, list) or not isinstance(ndd_values, list):
			raise ValueError("MultiBV importance weighting requires per_agent values")
		if not (len(weights) == len(ndd_values) == len(epsilons)):
			raise ValueError("MultiBV weight, epsilon, and NDD lengths must match")
		result = 1.0
		for weight, epsilon, ndd in zip(weights, epsilons, ndd_values):
			if not np.isfinite(epsilon) or epsilon <= 0 or epsilon >= 1:
				raise ValueError("MultiBV epsilon values must lie strictly between zero and one")
			if float(weight) > 1:
				result *= 1 / float(epsilon)
			elif float(weight) < 0.999:
				result *= float(ndd) / (1 - float(epsilon))
		return result

	def _get_done(self):
		stop = False
		reason = None
		if self.total_steps == len(self.episode_data["drl_obs_step_info"].keys())-1:
			stop = True
			if self.episode_data["collision_result"]:
				reason = {1: "CAV and BV collision"}
			else:
				reason = {4: "CAV safely exist"}
		return stop, reason

if __name__ == "__main__":
    env = D2RLTrainingEnv()
    
    for i in range(100):
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
