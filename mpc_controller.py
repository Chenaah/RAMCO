import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import copy
from six.moves import cPickle
from data_manipulation import from_observation_to_usablestate
from reward_functions import RewardFunctions
from helper_funcs import get_state, get_reward, stack_tensor_dict_list, frames_to_mp4
from gym.wrappers import Monitor
import os
import mpc_core
from tqdm import tqdm

import wandb

class MPCController:

	def __init__(self, env_inp, dyn_model, horizon, which_agent, steps_per_episode, dt_steps, num_control_samples, 
				mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index, z_index, yaw_index, 
				joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index, dt, 
				MF_agent, cmaes_args, cma_seed, alpha, boosting, wandb_on, mpc_optimiser, save_movies, dyn_actor):

		self.dt = dt
		#init vars
		self.env = env_inp #copy.deepcopy(env_inp)
		self.N = num_control_samples
		self.which_agent = which_agent
		self.MF_agent = MF_agent
		
		self.horizon = horizon
		self.dyn_model = dyn_model
		self.steps_per_episode = steps_per_episode 
		self.mean_x = mean_x
		self.mean_y = mean_y
		self.mean_z = mean_z
		self.std_x = std_x
		self.std_y = std_y
		self.std_z = std_z
		self.x_index = x_index
		self.y_index = y_index
		self.z_index = z_index 
		self.yaw_index = yaw_index
		self.joint1_index = joint1_index
		self.joint2_index = joint2_index
		self.frontleg_index = frontleg_index 
		self.frontshin_index = frontshin_index 
		self.frontfoot_index = frontfoot_index 
		self.xvel_index = xvel_index 
		self.orientation_index = orientation_index
		self.actions_ag = actions_ag
		self.print_minimal = print_minimal
		self.cmaes_args = cmaes_args
		self.cma_seed = cma_seed
		self.alpha = alpha
		self.boosting = boosting
		self.wandb = wandb_on
		self.dyn_actor = dyn_actor
		self.mpc_optimiser = mpc_optimiser
		self.total_controlled_steps = 0
		self.total_controlled_episodes = 0
		self.save_movies = save_movies #TODO
		self.recording_interval = 100*200 #STEPS
		self.evaluation_interval = 50 #EPISODES
		if which_agent==101 or which_agent==102 or which_agent==103:
			self.save_movies = False
		self.recording = False
		self.evaluation = False
		self.reward_functions = RewardFunctions(self.which_agent, self.x_index, self.y_index, self.z_index, self.yaw_index, 
												self.joint1_index, self.joint2_index, self.frontleg_index, self.frontshin_index, 
												self.frontfoot_index, self.xvel_index, self.orientation_index)       

		self.action_chooser = [mpc_core.Randomshooting(self.env, self.N, self.horizon, self.which_agent, self.dyn_model, self.dt),
							  mpc_core.Drunk(self.env),
							  mpc_core.CrossEntropyMethod(self.env, self.N, self.horizon, self.which_agent, self.dyn_model, self.dt),
							  mpc_core.CMAES(self.env, self.horizon, self.which_agent, self.dyn_model, self.dt, self.MF_agent._policy, self.cmaes_args, self.cma_seed, self.alpha),
							  mpc_core.MultiGaussian(self.env, self.horizon, self.which_agent, self.dyn_model, self.dt, self.MF_agent, 
							  	self.MF_agent._policy, self.cmaes_args, self.cma_seed, self.alpha, self.boosting, self.dyn_actor),
							  mpc_core.SimplifiedCMAES(self.env, self.horizon, self.which_agent, self.dyn_model, self.dt, self.MF_agent, 
							  	self.MF_agent._policy, self.cmaes_args, self.cma_seed, self.alpha, self.boosting, self.dyn_actor)][3+mpc_optimiser]


	def perform_rollout(self, starting_fullenvstate, starting_observation, starting_observation_NNinput, desired_states, follow_trajectories, 
						horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, noise_actions, noise_amount, info, action_chooser_info=None):
		
		#lists for saving info
		traj_taken=[] #list of states that go into NN
		actions_taken=[]
		observations = [] #list of observations (direct output of the env)
		rewards = []
		agent_infos = []
		env_infos = []
		frames = []

		#init vars
		stop_taking_steps = False
		episode_return = 0
		episode_cost = 0
		step=0
		curr_line_segment = 0
		self.horiz_penalty_factor = horiz_penalty_factor
		self.forward_encouragement_factor = forward_encouragement_factor
		self.heading_penalty_factor = heading_penalty_factor

		#extend the list of desired states so you don't run out
		temp = np.tile(np.expand_dims(desired_states[-1], axis=0), (10,1))
		self.desired_states = np.concatenate((desired_states, temp))

		#reset env to the given full env state
		if(info["record"] == True):
			vdir = info["save_dir"] +'/summary/tapes/' + info["scope"] + '_' + str(info["rollout_num"])
			if not os.path.isdir(vdir):
				os.makedirs(vdir)
			self.env = Monitor(self.env, vdir, force=True)
		vdir = info["save_dir"] +'/summary/tapes'
		if not os.path.isdir(vdir):
			os.makedirs(vdir)
	

		#============ RESET THE ENVIRONMENT ============#

		self.env.reset()
		if (starting_fullenvstate != 0):
			self.env.sim.set_state(starting_fullenvstate)

		#################################################

		#current observation
		obs = np.copy(starting_observation)
		#current observation in the right format for NN
		curr_state = np.copy(starting_observation_NNinput)
		traj_taken.append(curr_state)

		#select task or reward func
		reward_func = self.reward_functions.get_reward_func(follow_trajectories, self.desired_states, horiz_penalty_factor, 
															forward_encouragement_factor, heading_penalty_factor)

		#take steps according to the chosen task/reward function

		####################

		if (action_chooser_info == None):
			action_chooser_info = {"action_chooser": 2, "K": int(self.N*0.3)}


		self.action_chooser.first_action = True

		####################
		pbar = tqdm(total=self.steps_per_episode, ascii=True)

		while(stop_taking_steps==False):

			#get optimal action
			best_action, best_sim_number, best_sequence, logp_val = self.action_chooser.get_action(curr_state, curr_line_segment, reward_func, 
										save_path=info["save_dir"]+"/summary/MPC_log/"+info["scope"] + '_' +str(info["rollout_num"])+"/step"+str(step)+"/")  #curr_state -> obs
			moved_to_next=0


			if self.total_controlled_steps % self.recording_interval == 0 and self.save_movies:
				self.recording = True
			
			if self.recording:
				frame = self.env.render(mode='rgb_array')
				frames.append(frame)

			#advance which line segment we are on
			if(follow_trajectories):
				if(moved_to_next[best_sim_number]==1):
					curr_line_segment+=1
					print("MOVED ON TO LINE SEGMENT ", curr_line_segment)

			#noise the action
			action_to_take= np.copy(best_action)

			#whether to execute noisy or clean actions
			if(self.actions_ag=='nn'):
				noise_actions=True
			if(self.actions_ag=='nc'):
				noise_actions=True
			if(self.actions_ag=='cc'):
				noise_actions=False

			clean_action = np.copy(action_to_take)
			if(noise_actions):
				noise = noise_amount * npr.normal(size=action_to_take.shape)#
				action_to_take = action_to_take + noise
				action_to_take=np.clip(action_to_take, -1,1)

			#execute the action
			org_next_state, rew, done, env_info = self.env.step(action_to_take)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			try:
				cost = env_info['cost']
			except (TypeError, KeyError, IndexError):
				cost = 0
			next_state = org_next_state #get_state(self.env)

			if self.MF_agent._policy.policy_name == 'SAC':

				self.MF_agent.replay_buffer.add(obs=curr_state, act=action_to_take,
								  next_obs=next_state, rew=rew, done=done)
				self.MF_agent.train()

			elif self.MF_agent._policy.policy_name == 'PPO':
				if not (self.which_agent == 100 and self.MF_agent.global_num_episodes > 250):

					self.MF_agent.local_buffer.add(
					obs=curr_state, act=action_to_take, next_obs=next_state,
					rew=rew, done=done, logp=logp_val[0], val=logp_val[1])

			elif self.MF_agent._policy.policy_name == 'random':
				pass

			pbar.update(1)
			step+=1
			self.total_controlled_steps += 1
			self.MF_agent.global_total_steps += 1

			tf.summary.scalar(name="MPC/Aggiter{}Roll{}/Reward".format(info["aggIter"], info["rollout_num"]), data=rew, step=step)
			tf.summary.scalar(name="MPC/Aggiter{}Roll{}/Cost".format(info["aggIter"], info["rollout_num"]), data=cost, step=step)

			#check if done
			if(done):
				stop_taking_steps=True
			else:
				#save things
				observations.append(obs)
				rewards.append(rew)
				env_infos.append(env_info)
				episode_return += rew
				episode_cost += cost

				#whether to save clean or noisy actions
				if(self.actions_ag=='nn'):
					actions_taken.append(np.array([action_to_take]))
				if(self.actions_ag=='nc'):
					actions_taken.append(np.array([clean_action]))
				if(self.actions_ag=='cc'):
					actions_taken.append(np.array([clean_action]))

				#this is the observation returned by taking a step in the env
				obs=np.copy(next_state)

				#get the next state (usable by NN)
				just_one=True
				next_state = from_observation_to_usablestate(next_state, self.which_agent, just_one)
				curr_state=np.copy(next_state)
				traj_taken.append(curr_state)

				
				#print('STEP: ',step)

				#bookkeeping
				if(not(self.print_minimal)):
					if(step%1==0):
						#print("done step ", step, ", score: ", episode_return)
						pass


				#when to stop
				if(follow_trajectories):
					if((step>=self.steps_per_episode) or (curr_line_segment>5)):
						stop_taking_steps = True
				else:
					if(step>=self.steps_per_episode):
						#print('STEP: ',step, 'STOP!')
						stop_taking_steps = True

		pbar.close()

		if(not(self.print_minimal)):
			print("DONE TAKING ", step, " STEPS.")
			print("Score: ", episode_return, "   Cost: ", episode_cost)

			
		mydict = None
		self.MF_agent.global_num_episodes += 1
		self.total_controlled_episodes +=1 
		self.MF_agent.global_cost += episode_cost

		if self.MF_agent._policy.policy_name == 'PPO':
			if not (self.which_agent == 100 and self.MF_agent.global_num_episodes > 250):
				self.MF_agent.finish_horizon()
				self.MF_agent.train()

		if self.recording:
			print('Saving movies ...')
			prefix = "S{0:07d}R{1:08.2f}C{1:08.2f}".format(self.MF_agent.global_total_steps, episode_return, episode_cost)
			frames_to_mp4(frames, prefix, vdir)
			frames = []
			self.recording = False

		# Evaluation
		# print('[DEBUG]  TESTING THE EVALUATION MODULE')
		# self.MF_agent.evaluate_policy()
		if self.total_controlled_episodes % self.evaluation_interval == 0 and self.MF_agent._policy.policy_name != 'random':
			print('Evaluating the MFAgent ...')
			self.MF_agent.evaluate_policy()
			

		cost_rate = self.MF_agent.global_cost/self.MF_agent.global_total_steps
		if self.wandb:
			print('Uploading training results ...')
			wandb.log({'Training_Return': episode_return, 'Training_Cost': episode_cost, 'Cost_Rate': cost_rate})
		tf.summary.scalar(name="Common/Training_Return", data=episode_return, step=self.MF_agent.global_total_steps)
		tf.summary.scalar(name="Common/Training_Cost", data=episode_cost, step=self.MF_agent.global_total_steps)
		tf.summary.scalar(name="Common/Cost_Rate", data=cost_rate, step=self.MF_agent.global_total_steps)


		if(step != 0):

			mydict = dict(
			observations=np.array(observations),
			actions=np.array(actions_taken),
			rewards=np.array(rewards),
			agent_infos=agent_infos,
			env_infos=None)

		return traj_taken, actions_taken, episode_return, mydict # mydict: for saving rollouts!

	def get_action(self, curr_nn_state, curr_line_segment, reward_func):
		#randomly sample N candidate action sequences
		all_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0]))

		#forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
		many_in_parallel = True
		resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(all_samples), many_in_parallel, self.env, self.which_agent)
		resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]

		#init vars to evaluate the trajectories
		scores=np.zeros((self.N,))
		done_forever=np.zeros((self.N,))
		move_to_next=np.zeros((self.N,))
		curr_seg = np.tile(curr_line_segment,(self.N,))
		curr_seg = curr_seg.astype(int)
		prev_forward = np.zeros((self.N,))
		moved_to_next = np.zeros((self.N,))
		prev_pt = resulting_states[0]

		all_actions = np.swapaxes(all_samples,0,1)

		#accumulate reward over each timestep
		for pt_number in range(resulting_states.shape[0]-1):

			#array of "the point"... for each sim
			pt = resulting_states[pt_number] # N x state

			actionns = all_actions[pt_number]

			#how far is the point from the desired trajectory
			#how far along the desired traj have you moved since the last point
			min_perp_dist, curr_forward, curr_seg, moved_to_next = self.calculate_geometric_trajfollow_quantities(pt, curr_seg, moved_to_next)

			#update reward score
			#scores, done_forever = reward_func(pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, curr_seg, 
			#                                    moved_to_next, done_forever, all_samples, pt_number)
			scores = get_reward(pt, prev_pt, actionns, scores, self.dt)

			#update vars
			prev_forward = np.copy(curr_forward)
			prev_pt = np.copy(pt)

		#pick best action sequence
		best_score = np.min(np.negative(scores))
		best_sim_number = np.argmin(np.negative(scores) )
		best_sequence = all_samples[best_sim_number]
		best_action = np.copy(best_sequence[0])

		

		return best_action, best_sim_number, best_sequence, moved_to_next

	def calculate_geometric_trajfollow_quantities(self, pt, curr_seg, moved_to_next):

		#arrays of line segment points... for each sim
		curr_start = self.desired_states[curr_seg]
		curr_end = self.desired_states[curr_seg+1]
		next_start = self.desired_states[curr_seg+1]
		next_end = self.desired_states[curr_seg+2]

		#initialize
		min_perp_dist = np.ones((self.N, ))*5000

		####################################### closest distance from point to current line segment

		#vars
		a = pt[:,self.x_index]- curr_start[:,0]
		b = pt[:,self.y_index]- curr_start[:,1]
		c = curr_end[:,0]- curr_start[:,0]
		d = curr_end[:,1]- curr_start[:,1]

		#project point onto line segment
		which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

		#point on line segment that's closest to the pt
		closest_pt_x = np.copy(which_line_section)
		closest_pt_y = np.copy(which_line_section)
		closest_pt_x[which_line_section<0] = curr_start[:,0][which_line_section<0]
		closest_pt_y[which_line_section<0] = curr_start[:,1][which_line_section<0]
		closest_pt_x[which_line_section>1] = curr_end[:,0][which_line_section>1]
		closest_pt_y[which_line_section>1] = curr_end[:,1][which_line_section>1]
		closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,0] + 
							np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
		closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,1] + 
							np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

		#min dist from pt to that closest point (ie closes dist from pt to line segment)
		min_perp_dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + 
								(pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

		####################################### "forward-ness" of the pt... for each sim
		curr_forward = which_line_section

		###################################### closest distance from point to next line segment

		#vars
		a = pt[:,self.x_index]- next_start[:,0]
		b = pt[:,self.y_index]- next_start[:,1]
		c = next_end[:,0]- next_start[:,0]
		d = next_end[:,1]- next_start[:,1]

		#project point onto line segment
		which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), 
										(np.multiply(c,c) + np.multiply(d,d)))

		#point on line segment that's closest to the pt
		closest_pt_x = np.copy(which_line_section)
		closest_pt_y = np.copy(which_line_section)
		closest_pt_x[which_line_section<0] = next_start[:,0][which_line_section<0]
		closest_pt_y[which_line_section<0] = next_start[:,1][which_line_section<0]
		closest_pt_x[which_line_section>1] = next_end[:,0][which_line_section>1]
		closest_pt_y[which_line_section>1] = next_end[:,1][which_line_section>1]
		closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,0] + 
							np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
		closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,1] + 
							np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

		#min dist from pt to that closest point (ie closes dist from pt to line segment)
		dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + 
						(pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

		############################################ 

		#pick which line segment it's closest to, and update vars accordingly
		curr_seg[dist<=min_perp_dist] += 1
		moved_to_next[dist<=min_perp_dist] = 1
		curr_forward[dist<=min_perp_dist] = which_line_section[dist<=min_perp_dist]
		min_perp_dist = np.min([min_perp_dist, dist], axis=0)

		return min_perp_dist, curr_forward, curr_seg, moved_to_next


	



