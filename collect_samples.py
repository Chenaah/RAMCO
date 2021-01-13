import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from tqdm import tqdm
from helper_funcs import frames_to_mp4
from policy_random import Policy_Random

from gym.spaces import Box
from cpprb import ReplayBuffer
from tf2rl.experiments.utils import save_path
from tf2rl.envs.utils import is_discrete, get_act_dim
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer
from tf2rl.misc.discount_cumsum import discount_cumsum
import logging
import wandb

from tf2rl.algos.ppo import PPO
from tf2rl.algos.sac import SAC
from tf2rl.algos.td3 import TD3


class CollectSamples(object):

	def __init__(self, env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories):
		self.env = env
		self.policy = policy
		self.visualize_at_all = visualize_rollouts
		self.which_agent = which_agent

		self.low = self.env.observation_space.low
		self.high = self.env.observation_space.high
		self.shape = self.env.observation_space.shape

		self.use_low = self.low + (self.high-self.low)/3.0  ##invalid value encountered in substract
		self.use_high = self.high - (self.high-self.low)/3.0

		self.dt_steps = dt_steps
		self.dt_from_xml = dt_from_xml

		self.follow_trajectories = follow_trajectories
		
	def collect_samples(self, num_rollouts, steps_per_rollout):
		observations_list = []
		actions_list = []
		starting_states_list=[]
		rewards_list = []
		visualization_frequency = 10
		pbar = tqdm(total=num_rollouts, ascii=True)
		for rollout_number in range(num_rollouts):
			org_observation = self.env.reset()
			min_len=0

			observation = org_observation
			min_len = 2
			if (steps_per_rollout==23333):
				steps_per_rollout = 100
				min_len = 50
			starting_state = np.copy(observation)
			observations = np.zeros((1,2))
			trails_counter = 0
			while not (np.shape(observations)[0] >= min_len):
				observations, actions, reward_for_rollout = self.perform_rollout(observation, steps_per_rollout, 
																		rollout_number, visualization_frequency)
				trails_counter += 1
				if (trails_counter > 100):
					if (min_len == 5):
						min_len -= 1
					else:
						min_len -= 5
					trails_counter = 0

			rewards_list.append(reward_for_rollout)
			observations= np.array(observations)
			actions= np.array(actions)
			observations_list.append(observations)
			actions_list.append(actions)
			starting_states_list.append(starting_state)
			pbar.update(1)
		pbar.close()

		#return list of length = num rollouts
		#each entry of that list contains one rollout
		#each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
		print('observations_list, actions_list, starting_states_list, rewards_list')  #(60, xxx, 29) (60, xxx, 8) (60, xxx) (60,)

		# print(np.shape(observations_list), np.shape(actions_list), np.shape(starting_states_list),np.shape(rewards_list))
		# print(observations_list[0][0])
		# print(starting_states_list[0])
		return observations_list, actions_list, starting_states_list, rewards_list

	def perform_rollout(self, observation, steps_per_rollout, rollout_number, visualization_frequency):
		observations = []
		actions = []
		visualize = False
		reward_for_rollout = 0
		self.env.reset()
		if((rollout_number%visualization_frequency)==0):
			#print("currently performing rollout #", rollout_number)
			if(self.visualize_at_all):
				all_states=[]
				print ("---- visualizing a rollout ----")
				visualize=True

		for step_num in range(steps_per_rollout):
			action, _ = self.policy.get_action(observation)
			#print("[MESSAGE FROM COLLECT_SAMPLES]  The action just took: \n", action)

			observations.append(observation)
			actions.append(action)

			org_next_observation, reward, terminal, env_info = self.env.step(action)
			try:
				cost = env_info['cost']
			except (TypeError, KeyError, IndexError):
				cost = 0

			next_observation = org_next_observation
			reward_for_rollout += reward

			observation = np.copy(next_observation)
			
			if terminal:
				#print(f"Had to stop rollout after {step_num} steps because terminal state was reached.")
				break

			if(visualize):
				if(self.which_agent==0):
					curr_state = self.env.render()
					all_states.append(np.expand_dims(curr_state, axis=0))
				else:
					self.env.render()
					time.sleep(self.dt_steps*self.dt_from_xml)

		if(visualize and (self.which_agent==0)):
			all_states= np.concatenate(all_states, axis=0)
			plt.plot(all_states[:,0], all_states[:,1], 'r')
			plt.show()
		return observations, actions, reward_for_rollout

class CollectSamples_random(object):

	def __init__(self, env, max_steps, _output_dir=None, _episode_max_steps=100, gpu=0, wandb_on=False):

		self.env = env
		self._policy = Policy_Random(env)
		self.visualize_at_all = False

		self.max_steps = max_steps

		self.wandb_on = wandb_on


		self.low = self.env.observation_space.low
		self.high = self.env.observation_space.high
		self.shape = self.env.observation_space.shape

		self.use_low = self.low + (self.high-self.low)/3.0  ##invalid value encountered in substract
		self.use_high = self.high - (self.high-self.low)/3.0

		self.global_total_steps = 0
		self.global_num_episodes = 0
		self.global_cost = 0

		
	def collect_samples(self):
		observations_list = []
		actions_list = []
		starting_states_list=[]
		rewards_list = []
		visualization_frequency = 10
		pbar = tqdm(total=self.max_steps, ascii=True)
		step_counter = 0
		while(step_counter < self.max_steps):
			org_observation = self.env.reset()
			min_len=0

			observation = org_observation

			starting_state = np.copy(observation)

			observations, actions, reward_for_rollout, episode_cost = self.perform_rollout(observation)
			self.global_num_episodes += 1
			self.global_cost += episode_cost
			step_counter += len(actions)
			cost_rate = self.global_cost/self.global_total_steps

			if self.wandb_on:
				wandb.log({'Training_Return': reward_for_rollout, 'Training_Cost': episode_cost, 'Cost_Rate': cost_rate}, step=self.global_num_episodes)
				# print("DEBUG! LOGGING ...")

			tf.summary.scalar(name="Common/Training_Return", data=reward_for_rollout, step=self.global_total_steps)
			tf.summary.scalar(name="Common/Training_Cost", data=episode_cost, step=self.global_total_steps)
			tf.summary.scalar(name="Common/Cost_Rate", data=cost_rate, step=self.global_total_steps)

			

			if not (step_counter>self.max_steps): # otherwise ignore this rollout

				rewards_list.append(reward_for_rollout)
				observations= np.array(observations)
				actions= np.array(actions)
				observations_list.append(observations)
				actions_list.append(actions)
				starting_states_list.append(starting_state)

				pbar.update(len(actions))
			
		pbar.close()

		#return list of length = num rollouts
		#each entry of that list contains one rollout
		#each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
		#print('observations_list, actions_list, starting_states_list, rewards_list')  #(60, xxx, 29) (60, xxx, 8) (60, xxx) (60,)

		# print(np.shape(observations_list), np.shape(actions_list), np.shape(starting_states_list),np.shape(rewards_list))
		# print(observations_list[0][0])
		# print(starting_states_list[0])
		return observations_list, actions_list, starting_states_list, rewards_list

	def perform_rollout(self, observation, steps_per_rollout=1000, rollout_number=1, visualization_frequency=1):
		observations = []
		actions = []
		visualize = False
		reward_for_rollout = 0
		episode_cost = 0
		self.env.reset()
		if((rollout_number%visualization_frequency)==0):
			#print("currently performing rollout #", rollout_number)
			if(self.visualize_at_all):
				all_states=[]
				print ("---- visualizing a rollout ----")
				visualize=True

		for step_num in range(steps_per_rollout):
			action, _ = self._policy.get_action(observation)
			#print("[MESSAGE FROM COLLECT_SAMPLES]  The action just took: \n", action)

			observations.append(observation)
			actions.append(action)

			org_next_observation, reward, terminal, env_info = self.env.step(action)
			try:
				cost = env_info['cost']
			except (TypeError, KeyError, IndexError):
				cost = 0
			
			self.global_total_steps += 1

			next_observation = org_next_observation
			reward_for_rollout += reward
			episode_cost += cost

			observation = np.copy(next_observation)
			
			if terminal:
				#print(f"Had to stop rollout after {step_num} steps because terminal state was reached.")
				break

			if(visualize):
				if(self.which_agent==0):
					curr_state = self.env.render()
					all_states.append(np.expand_dims(curr_state, axis=0))
				else:
					self.env.render()
					time.sleep(self.dt_steps*self.dt_from_xml)

		if(visualize and (self.which_agent==0)):
			all_states= np.concatenate(all_states, axis=0)
			plt.plot(all_states[:,0], all_states[:,1], 'r')
			plt.show()
		return observations, actions, reward_for_rollout, episode_cost

class CollectSamples_SAC(CollectSamples_random):
	
	def __init__(self, env, max_steps, _output_dir, _episode_max_steps=100, gpu=0, wandb_on=False, *args, **kwargs):
		super().__init__(env, max_steps, _output_dir, _episode_max_steps, gpu, wandb_on, *args, **kwargs)

		#self._set_from_args(args)
		# experiment settings

		self._max_steps = max_steps
		self._episode_max_steps = _episode_max_steps
		self._n_experiments = 1
		self._show_progress = False
		self._save_model_interval = int(400)
		self._save_summary_interval = int(1e3)
		self._normalize_obs = False
		self._logdir = _output_dir
		self._model_dir = _output_dir + '/../checkpoints/policy'
		# replay buffer
		self._use_prioritized_rb = False
		self._use_nstep_rb = False
		self._n_step = 4
		# test settings
		self._evaluate = False
		self._test_interval = int(1e4)  #TODO
		self._show_test_progress = False
		self._test_episodes = 3
		self._save_test_path = False
		self._save_test_movie = False
		self._show_test_images = False
		self.wandb_on = wandb_on
		self.evaluation_interval = 2#20 #TODO


		self._policy = SAC(
						state_shape=env.observation_space.shape,
						action_dim=env.action_space.high.size,
						gpu=gpu,
						max_action=env.action_space.high[0],
						batch_size=100,
						n_warmup=1000)
		self._env = env
		self._test_env = copy.copy(self._env)
		if self._normalize_obs:
			assert isinstance(env.observation_space, Box)
			self._obs_normalizer = EmpiricalNormalizer(
				shape=env.observation_space.shape)

		# prepare log directory
		self._output_dir = self._logdir
		self.logger = initialize_logger(
			logging_level=logging.getLevelName('INFO'),
			output_dir=self._output_dir)

		if self._evaluate:
			assert self._model_dir is not None
		self._set_check_point(self._model_dir)

		self.replay_buffer = get_replay_buffer(
			self._policy, self._env, self._use_prioritized_rb,
			self._use_nstep_rb, self._n_step)

		# prepare TensorBoard output
		# self.writer = tf.summary.create_file_writer(self._output_dir)
		# self.writer.set_as_default()

	def collect_samples(self):#observations_list, actions_list, starting_states_list, rewards_list (60, xxx, 29) (60, xxx, 8) (60, xxx) (60,)
		observations_list = []
		actions_list = []
		starting_states_list=[]
		rewards_list = []
		
		
		tf.summary.experimental.set_step(self.global_total_steps)
		episode_steps = 0
		episode_return = 0
		episode_cost = 0
		episode_start_time = time.perf_counter()
		n_episode = 0

		# self.replay_buffer = get_replay_buffer(
		# 	self._policy, self._env, self._use_prioritized_rb,
		# 	self._use_nstep_rb, self._n_step)

		obs = self._env.reset()
		observations = []
		actions = []
		starting_states_list.append(obs)

		while self.global_total_steps < self._max_steps:
			if self.global_total_steps < self._policy.n_warmup:
				action = self._env.action_space.sample()
			else:
				action = self._policy.get_action(obs)

			next_obs, reward, done, env_info = self._env.step(action)
			try:
				cost = env_info['cost']
			except (TypeError, KeyError, IndexError):
				cost = 0


			observations.append(obs)
			actions.append(action)
			rewards_list.append(reward)

			if self._show_progress:
				self._env.render()
			episode_steps += 1
			episode_return += reward
			episode_cost += cost
			self.global_total_steps += 1
			tf.summary.experimental.set_step(self.global_total_steps)

			done_flag = done
			if hasattr(self._env, "_max_episode_steps") and \
					episode_steps == self._env._max_episode_steps:
				done_flag = False
			self.replay_buffer.add(obs=obs, act=action,
							  next_obs=next_obs, rew=reward, done=done_flag)
			obs = next_obs

			if done or episode_steps == self._episode_max_steps:
				obs = self._env.reset()

				n_episode += 1
				self.global_num_episodes += 1
				self.global_cost += episode_cost
				cost_rate = self.global_cost/self.global_total_steps

				fps = episode_steps / (time.perf_counter() - episode_start_time)
				self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 7.4f} Cost: {4: 5.4f} FPS: {5:5.2f}".format(
					n_episode, self.global_total_steps, episode_steps, episode_return, episode_cost, fps))
				tf.summary.scalar(name="Common/Training_Return", data=episode_return, step=self.global_total_steps)
				tf.summary.scalar(name="Common/Training_Cost", data=episode_cost, step=self.global_total_steps)
				tf.summary.scalar(name="Common/Cost_Rate", data=cost_rate, step=self.global_total_steps)
				tf.summary.scalar(name="MFAgent/FPS", data=fps, step=self.global_total_steps)
				
				if self.wandb_on:
					wandb.log({'Training_Return': episode_return, 'Training_Cost': episode_cost, 'Cost_Rate': cost_rate}, step=self.global_num_episodes)

				
				rewards_list.append(episode_return)
				observations_list.append(np.array(observations))
				actions_list.append(np.array(actions))
				starting_states_list.append(obs)

				#print('=============DEBUG:actions_list',np.shape(actions_list),'=============')
				#print(actions_list)

				observations = []
				actions = []
				episode_steps = 0
				episode_return = 0
				episode_cost = 0
				episode_start_time = time.perf_counter()

			if self.global_total_steps < self._policy.n_warmup:
				continue

			if self.global_total_steps % self._policy.update_interval == 0:
				samples = self.replay_buffer.sample(self._policy.batch_size)
				with tf.summary.record_if(self.global_total_steps % self._save_summary_interval == 0):
					self._policy.train(
						samples["obs"], samples["act"], samples["next_obs"],
						samples["rew"], np.array(samples["done"], dtype=np.float32),
						None if not self._use_prioritized_rb else samples["weights"])
				if self._use_prioritized_rb:
					td_error = self._policy.compute_td_error(
						samples["obs"], samples["act"], samples["next_obs"],
						samples["rew"], np.array(samples["done"], dtype=np.float32))
					self.replay_buffer.update_priorities(
						samples["indexes"], np.abs(td_error) + 1e-6)

			if self.global_total_steps % self._test_interval == 0:
				# print('Evaluating the MFAgent ...')
				time1 = time.time()
				self.evaluate_policy()
			if False:
				avg_test_return = self.evaluate_policy(self.global_total_steps)
				self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
					total_steps, avg_test_return, self._test_episodes))
				tf.summary.scalar(
					name="Common/average_test_return", data=avg_test_return)
				tf.summary.scalar(name="Common/fps", data=fps)
				self.writer.flush()

			if self.global_total_steps % self._save_model_interval == 0:
				self.checkpoint_manager.save()

		tf.summary.flush()

			#return list of length = num rollouts
			#each entry of that list contains one rollout
			#each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
		return observations_list, actions_list, starting_states_list, rewards_list

	def train(self):
		samples = self.replay_buffer.sample(self._policy.batch_size)
		with tf.summary.record_if(self.global_total_steps % self._save_summary_interval == 0):
			self._policy.train(
				samples["obs"], samples["act"], samples["next_obs"],
				samples["rew"], np.array(samples["done"], dtype=np.float32),
				None if not self._use_prioritized_rb else samples["weights"])
		if self._use_prioritized_rb:
			td_error = self._policy.compute_td_error(
				samples["obs"], samples["act"], samples["next_obs"],
				samples["rew"], np.array(samples["done"], dtype=np.float32))
			self.replay_buffer.update_priorities(
				samples["indexes"], np.abs(td_error) + 1e-6)

	def _set_check_point(self, model_dir):
		# Save and restore model
		self._checkpoint = tf.train.Checkpoint(policy=self._policy)
		self.checkpoint_manager = tf.train.CheckpointManager(
			self._checkpoint, directory=model_dir, max_to_keep=5)

		# if model_dir is not None:
		# 	assert os.path.isdir(model_dir)
		# 	self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
		# 	self._checkpoint.restore(self._latest_path_ckpt)
		# 	self.logger.info("Restored {}".format(self._latest_path_ckpt))

	def evaluate_policy(self, total_steps=None):
		tf.summary.experimental.set_step(self.global_total_steps)
		if self._normalize_obs:
			self._test_env.normalizer.set_params(
				*self._env.normalizer.get_params())
		avg_test_return = 0.
		avg_test_cost = 0.
		if self._save_test_path:
			replay_buffer = get_replay_buffer(
				self._policy, self._test_env, size=self._episode_max_steps)
		for i in range(self._test_episodes):
			timer0 = time.time()
			episode_return = 0.
			episode_cost = 0.
			frames = []
			obs = self._test_env.reset()
			# print(f'LOOP{i}-PERIOD1: ', time.time() - timer0)
			timer0 = time.time()
			for _ in range(self._episode_max_steps):
				action = self._policy.get_action(obs, test=True)
				next_obs, reward, done, env_info = self._test_env.step(action)
				try:
					cost = env_info['cost']
				except (TypeError, KeyError, IndexError):
					cost = 0
				if self._save_test_path:
					replay_buffer.add(obs=obs, act=action,
									  next_obs=next_obs, rew=reward, done=done)

				if self._save_test_movie:
					frames.append(self._test_env.render(mode='rgb_array'))
				elif self._show_test_progress:
					self._test_env.render()
				episode_return += reward
				episode_cost += cost
				obs = next_obs
				if done:
					break
			# print(f'LOOP{i}-PERIOD2: ', time.time() - timer0)
			prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}_cost_{2:010.4f}".format(
				self.global_total_steps, i, episode_return, episode_cost)
			if self._save_test_path:
				save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
						  os.path.join(self._output_dir, prefix + ".pkl"))
				replay_buffer.clear()
			if self._save_test_movie:
				# frames_to_gif(frames, prefix, self._output_dir)
				frames_to_mp4(frames, prefix, self._output_dir)
			avg_test_return += episode_return
			avg_test_cost += episode_cost
		if self._show_test_images:
			images = tf.cast(
				tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
				tf.uint8)
			tf.summary.image('train/input_img', images,)

		avg_test_return = avg_test_return / self._test_episodes
		if self.wandb_on:
			wandb.log({'Evaluation_Return': avg_test_return, 'Evaluation_Cost': avg_test_cost}, step=self.global_num_episodes)

		tf.summary.scalar(name="MFAgent/Evaluation_Return", data=avg_test_return, step=self.global_total_steps)
		tf.summary.scalar(name="MFAgent/Evaluation_Cost", data=avg_test_cost, step=self.global_total_steps)


		return avg_test_return, avg_test_cost

	def load_checkpoints(self, global_total_steps, global_num_episodes, global_cost):
		self.global_total_steps = global_total_steps
		self.global_num_episodes = global_num_episodes
		self.global_cost = global_cost

		self._latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
		self._checkpoint.restore(self._latest_path_ckpt)
		self.logger.info("Restored {}".format(self._latest_path_ckpt))

		tf.summary.experimental.set_step(self.global_total_steps)


class CollectSamples_PPO(CollectSamples_SAC):
	

	def __init__(self, env, max_steps, _output_dir, _episode_max_steps=100, gpu=0, wandb_on=False, *args, **kwargs):
		super(CollectSamples_PPO, self).__init__(env, max_steps, _output_dir, _episode_max_steps, gpu, wandb_on, *args, **kwargs)

		# experiment settings
		self._max_steps = max_steps
		self._episode_max_steps = _episode_max_steps
		self._n_experiments = 1
		self._show_progress = False
		self._save_model_interval = int(1e4)
		self._save_summary_interval = int(1e3)
		self._normalize_obs = False
		self._logdir = _output_dir
		# self._model_dir = None
		# replay buffer
		self._use_prioritized_rb = False
		self._use_nstep_rb = False
		self._n_step = 4
		# test settings
		self._evaluate = False
		# self._test_interval = int(1e4)
		self._show_test_progress = False
		# self._test_episodes = 5
		self._save_test_path = False
		self._save_test_movie = False
		self._show_test_images = False
		# PPO settings
		self._enable_gae = False
		self._normalize_adv = False
		self._horizon = 2048



		self._output_dir = _output_dir

		self._policy = PPO(
						state_shape=env.observation_space.shape,
						action_dim=get_act_dim(env.action_space),
						is_discrete=is_discrete(env.action_space),
						max_action=None if is_discrete(
							env.action_space) else env.action_space.high[0],
						batch_size=64,
						actor_units=(64, 64),
						critic_units=(64, 64),
						n_epoch=10,
						lr_actor=3e-4,
						lr_critic=3e-4,
						hidden_activation_actor="tanh",
						hidden_activation_critic="tanh",
						discount=0.99,
						lam=0.95,
						entropy_coef=0.,
						horizon=self._horizon,
						normalize_adv=self._normalize_adv,
						enable_gae=self._enable_gae,
						gpu=gpu)

		# Prepare buffer
		self.replay_buffer = get_replay_buffer(
			self._policy, self._env)
		kwargs_local_buf = get_default_rb_dict(
			size=self._policy.horizon, env=self._env)
		kwargs_local_buf["env_dict"]["logp"] = {}
		kwargs_local_buf["env_dict"]["val"] = {}
		if is_discrete(self._env.action_space):
			kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
		self.local_buffer = ReplayBuffer(**kwargs_local_buf)


	def collect_samples(self):

		observations_list = []
		actions_list = []
		starting_states_list=[]
		rewards_list = []


		# # Prepare buffer
		# self.replay_buffer = get_replay_buffer(
		# 	self._policy, self._env)
		# kwargs_local_buf = get_default_rb_dict(
		# 	size=self._policy.horizon, env=self._env)
		# kwargs_local_buf["env_dict"]["logp"] = {}
		# kwargs_local_buf["env_dict"]["val"] = {}
		# if is_discrete(self._env.action_space):
		# 	kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
		# self.local_buffer = ReplayBuffer(**kwargs_local_buf)

		episode_steps = 0
		episode_return = 0
		episode_cost = 0
		episode_start_time = time.time()
		total_steps = np.array(0, dtype=np.int32)
		# self.global_total_steps = total_steps
		n_episode = 0
		obs = self._env.reset()
		observations = []
		actions = []
		starting_states_list.append(obs)

		tf.summary.experimental.set_step(self.global_total_steps)
		while self.global_total_steps < self._max_steps:
			# Collect samples
			for _ in range(self._policy.horizon):
				if self._normalize_obs:
					obs = self._obs_normalizer(obs, update=False)
				act, logp, val = self._policy.get_action_and_val(obs)
				next_obs, reward, done, env_info = self._env.step(act)
				try:
					cost = env_info['cost']
				except (TypeError, KeyError, IndexError):
					cost = 0

				observations.append(obs)
				actions.append(act)

				if self._show_progress:
					self._env.render()

				episode_steps += 1
				self.global_total_steps += 1
				# self.global_total_steps = total_steps
				episode_return += reward
				episode_cost += cost

				done_flag = done
				if hasattr(self._env, "_max_episode_steps") and \
						episode_steps == self._env._max_episode_steps:
					done_flag = False
				self.local_buffer.add(
					obs=obs, act=act, next_obs=next_obs,
					rew=reward, done=done_flag, logp=logp, val=val)
				obs = next_obs

				if done or episode_steps == self._episode_max_steps:
					tf.summary.experimental.set_step(self.global_total_steps)
					self.finish_horizon()
					obs = self._env.reset()
					n_episode += 1
					self.global_num_episodes += 1
					self.global_cost += episode_cost
					cost_rate = self.global_cost/self.global_total_steps
					fps = episode_steps / (time.time() - episode_start_time)
					self.logger.info(
						"Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 7.4f} Cost: {4: 5.4f} FPS: {5:5.2f}".format(
							n_episode, int(self.global_total_steps), episode_steps, episode_return, episode_cost, fps))
					# tf.summary.scalar(name="Common/training_return", data=episode_return)
					
					tf.summary.scalar(name="Common/Training_Return", data=episode_return, step=self.global_total_steps)
					tf.summary.scalar(name="Common/Training_Cost", data=episode_cost, step=self.global_total_steps)
					tf.summary.scalar(name="Common/Cost_Rate", data=cost_rate, step=self.global_total_steps)
					tf.summary.scalar(name="MFAgent/FPS", data=fps, step=self.global_total_steps)

					if self.wandb_on:
						wandb.log({'Training_Return': episode_return, 'Training_Cost': episode_cost, 'Cost_Rate': cost_rate}, step=self.global_num_episodes)

					starting_states_list.append(obs)
					rewards_list.append(episode_return)
					observations_list.append(np.array(observations))
					actions_list.append(np.array(actions))

					
					#print('=============DEBUG:actions_list',np.shape(actions_list),'=============')
					#print(actions_list)

					observations = []
					actions = []
					episode_steps = 0
					episode_return = 0
					episode_start_time = time.time()

				if self.global_total_steps % self._test_interval == 0:
					# print('Evaluating the MFAgent ...')
					self.evaluate_policy()
				#if total_steps % self._test_interval == 0:
				if False:
					avg_test_return = self.evaluate_policy(self.global_total_steps)
					self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
						self.global_total_steps, avg_test_return, self._test_episodes))
					tf.summary.scalar(
						name="Common/average_test_return", data=avg_test_return)
					self.writer.flush()

				if self.global_total_steps % self._save_model_interval == 0:
					self.checkpoint_manager.save()

			self.finish_horizon(last_val=val)

			tf.summary.experimental.set_step(self.global_total_steps)

			# Train actor critic
			if self._policy.normalize_adv:
				samples = self.replay_buffer._encode_sample(np.arange(self._policy.horizon))
				mean_adv = np.mean(samples["adv"])
				std_adv = np.std(samples["adv"])
				# Update normalizer
				if self._normalize_obs:
					self._obs_normalizer.experience(samples["obs"])
			with tf.summary.record_if(self.global_total_steps % self._save_summary_interval == 0):
				for _ in range(self._policy.n_epoch):
					samples = self.replay_buffer._encode_sample(
						np.random.permutation(self._policy.horizon))
					if self._normalize_obs:
						samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
					if self._policy.normalize_adv:
						adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
					else:
						adv = samples["adv"]
					for idx in range(int(self._policy.horizon / self._policy.batch_size)):
						target = slice(idx * self._policy.batch_size,
									   (idx + 1) * self._policy.batch_size)
						self._policy.train(
							states=samples["obs"][target],
							actions=samples["act"][target],
							advantages=adv[target],
							logp_olds=samples["logp"][target],
							returns=samples["ret"][target])

		tf.summary.flush()

		#print('====================DEGUG:observations_list',np.shape(observations_list),'====================')
		#print(observations_list)
		#print('====================DEGUG:actions_list',np.shape(actions_list),'====================')

		return observations_list, actions_list, starting_states_list, rewards_list

	def finish_horizon(self, last_val=0):
		samples = self.local_buffer._encode_sample(
			np.arange(self.local_buffer.get_stored_size()))
		rews = np.append(samples["rew"], last_val)
		vals = np.append(samples["val"], last_val)

		# GAE-Lambda advantage calculation
		deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
		if self._policy.enable_gae:
			advs = discount_cumsum(
				deltas, self._policy.discount * self._policy.lam)
		else:
			advs = deltas

		# Rewards-to-go, to be targets for the value function
		rets = discount_cumsum(rews, self._policy.discount)[:-1]
		self.replay_buffer.add(
			obs=samples["obs"], act=samples["act"], done=samples["done"],
			ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
		self.local_buffer.clear()

	def evaluate_policy(self, total_steps=None):
		avg_test_return = 0.
		avg_test_cost = 0.
		if self._save_test_path:
			replay_buffer = get_replay_buffer(
				self._policy, self._test_env, size=self._episode_max_steps)
		for i in range(self._test_episodes):
			episode_return = 0.
			episode_cost = 0.
			frames = []
			obs = self._test_env.reset()
			for _ in range(self._episode_max_steps):
				if self._normalize_obs:
					obs = self._obs_normalizer(obs, update=False)
				act, _ = self._policy.get_action(obs, test=True)
				act = act if not hasattr(self._env.action_space, "high") else \
					np.clip(act, self._env.action_space.low, self._env.action_space.high)
				next_obs, reward, done, env_info = self._test_env.step(act)
				try:
					cost = env_info['cost']
				except (TypeError, KeyError, IndexError):
					cost = 0
				if self._save_test_path:
					replay_buffer.add(
						obs=obs, act=act, next_obs=next_obs,
						rew=reward, done=done)

				if self._save_test_movie:
					frames.append(self._test_env.render(mode='rgb_array'))
				elif self._show_test_progress:
					self._test_env.render()
				episode_return += reward
				episode_cost += cost
				obs = next_obs
				if done:
					break
			prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}_cost_{2:010.4f}".format(
				self.global_total_steps, i, episode_return, episode_cost)
			if self._save_test_path:
				save_path(replay_buffer.sample(self._episode_max_steps),
						  os.path.join(self._output_dir, prefix + ".pkl"))
				replay_buffer.clear()
			if self._save_test_movie:
				# frames_to_gif(frames, prefix, self._output_dir)
				frames_to_mp4(frames, prefix, self._output_dir)
			avg_test_return += episode_return
			avg_test_cost += episode_cost
		if self._show_test_images:
			images = tf.cast(
				tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
				tf.uint8)
			tf.summary.image('train/input_img', images, )

		avg_test_return = avg_test_return / self._test_episodes
		avg_test_cost = avg_test_cost / self._test_episodes
		if self.wandb_on:
			wandb.log({'Evaluation_Return': avg_test_return, 'Evaluation_Cost': avg_test_cost}, step=self.global_num_episodes)
		tf.summary.scalar(name="MFAgent/Evaluation_Return", data=avg_test_return, step=self.global_total_steps)
		tf.summary.scalar(name="MFAgent/Evaluation_Cost", data=avg_test_cost, step=self.global_total_steps)
		
		return avg_test_return, avg_test_cost

	def train(self):
		if self._policy.normalize_adv:
			samples = self.replay_buffer._encode_sample(np.arange(self._policy.horizon))
			mean_adv = np.mean(samples["adv"])
			std_adv = np.std(samples["adv"])
			# Update normalizer
			if self._normalize_obs:
				self._obs_normalizer.experience(samples["obs"])
		with tf.summary.record_if(self.global_total_steps % self._save_summary_interval == 0):
			for _ in range(self._policy.n_epoch):
				samples = self.replay_buffer._encode_sample(
					np.random.permutation(self._policy.horizon))
				if self._normalize_obs:
					samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
				if self._policy.normalize_adv:
					adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
				else:
					adv = samples["adv"]
				for idx in range(int(self._policy.horizon / self._policy.batch_size)):
					target = slice(idx * self._policy.batch_size,
								   (idx + 1) * self._policy.batch_size)
					self._policy.train(
						states=samples["obs"][target],
						actions=samples["act"][target],
						advantages=adv[target],
						logp_olds=samples["logp"][target],
						returns=samples["ret"][target])

class CollectSamples_TD3(CollectSamples_SAC):
	
	def __init__(self, env, max_steps, _output_dir, _episode_max_steps=100, gpu=0, wandb_on=False, *args, **kwargs):
		super(CollectSamples_TD3, self).__init__(env, max_steps, _output_dir, _episode_max_steps, gpu, wandb_on, *args, **kwargs)

		self._policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=gpu,
        max_action=env.action_space.high[0],
        batch_size=100,
        n_warmup=1000)
        # trainer = Trainer(policy, env, args, test_env=test_env)

    