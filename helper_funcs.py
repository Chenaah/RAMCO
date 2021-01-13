import os
import copy
import logging
import time
import tensorflow as tf
import numpy as np
from normalized_env import normalize
import gym
from gym import wrappers
# from collect_samples import CollectSamples_SAC
from scipy.stats import truncnorm
from skvideo.io import FFmpegWriter
import collections

def get_truncated_normal(mean=0, sd=0.5, low=-1, up=1, size=1):
	return truncnorm((low-mean)/sd,(up-mean)/sd,loc=mean,scale=sd).rvs(size)

def add_noise(data_inp, noiseToSignal):
	data= copy.deepcopy(data_inp)
	mean_data = np.mean(data, axis = 0)
	std_of_noise = mean_data*noiseToSignal
	for j in range(mean_data.shape[0]):
		if(std_of_noise[j]>0):
			data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
	return data

def perform_rollouts(policy, num_rollouts, steps_per_rollout, visualize_rollouts, CollectSamples, 
					env, which_agent, dt_steps, dt_from_xml, follow_trajectories):
	#collect training data by performing rollouts
	print("Beginning to do ", num_rollouts, " rollouts.")
	c = CollectSamples(env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories)
	states, controls, starting_states, rewards_list = c.collect_samples(num_rollouts, steps_per_rollout)

	print("Performed ", len(states), " rollouts, each with around ", len(states[0]), " steps.")
	return states, controls, starting_states, rewards_list # states: list, [Mrollout, Nsteps, Xdim]
														   # starting_states: list, [Mrollout, Xdim]
def perform_rollouts_SAC(max_num_steps, env):

	print("Beginning to do ", max_num_steps, " steps of rollouts.")

	c = CollectSamples_SAC(env, max_num_steps)
	states, controls, starting_states, rewards_list = c.collect_samples()

	print("Performed ", len(states), " rollouts, each with around ", len(states[0]), " steps.")

	return states, controls, starting_states, rewards_list

def cartpole_cost_numpy(state, pole_length=0.6, cost_sigma=0.25):
	cart_x = state[0]
	theta = state[2]
	pole_x = pole_length*np.sin(theta)
	pole_y = pole_length*np.cos(theta)
	tip_position = np.array([cart_x + pole_x, pole_y])

	target = np.array([0.0, pole_length])
	sq_distance = np.sum((tip_position - target)**2)
	cost = 1 - np.exp(-0.5*sq_distance/cost_sigma**2)
	return cost

def create_env(which_agent):
	# SwimmerEnv = gym.make('Swimmer-v2')
	# HalfCheetahEnv = gym.make('HalfCheetah-v2')
	# HopperEnv = gym.make('Hopper-v2')
	# Walker2dEnv = gym.make('Walker2d-v2')
	# AntEnv = gym.make('Ant-v2')
	# AntEnv2 = gym.make("AntBulletEnv-v0")

	# setup environment
	# RAMCO Environments: AntX 2->100, Eidose1 4->101, Eidose2 40->102, Eidose3 400->103;
	# Additional Environments: InvertedDoublePendulum 0->110, CartPoleSwingUp 5->111, HalfCheetahEnv 6->112, PusherEnv 7->113, Reacher 11->114
	# Backup Environments: AntX-v0 1->120, AntBulletEnv-v0 3->121
	# Toy Environments: Toy 110->130
	# GRAMCO Environments: ScratchItchPR2X 8->200, DressingPR2X 201

	if(which_agent==100):
		env = gym.make('AntX-v1')
		dt_from_xml = 0.01*20
	
	elif(which_agent==101):
		import eidos_env_ii
		env = eidos_env_ii.Eidos(s_dim=10, a_dim=10,
								load_dynamics_model="Eidos_models/dynamics_s10a10.h5", 
								load_reward_model="Eidos_models/reward_s10a10.h5")
		dt_from_xml = 2333
	elif(which_agent==102):
		import eidos_env_ii
		env = eidos_env_ii.Eidos(s_dim=100, a_dim=10,
								load_dynamics_model="Eidos_models/dynamics_s100a10.h5", 
								load_reward_model="Eidos_models/reward_s100a10.h5")
		dt_from_xml = 2333
	elif(which_agent==103):
		import eidos_env_ii
		env = eidos_env_ii.Eidos(s_dim=1000, a_dim=10,
								load_dynamics_model="Eidos_models/dynamics_s1000a10.h5", 
								load_reward_model="Eidos_models/reward_s1000a10.h5")
		dt_from_xml = 2333

	elif(which_agent==110):
		env = gym.make('InvertedDoublePendulum-v2')
	
	elif(which_agent==111):
		from cartpole_swingup import CartPoleSwingUp
		env = CartPoleSwingUp()
		dt_from_xml = 0.01

	elif(which_agent==112):
		from env.half_cheetah import HalfCheetahEnv
		env = HalfCheetahEnv()
		dt_from_xml = 0.01*5

	elif(which_agent==113):
		from env.pusher import PusherEnv
		env = PusherEnv()
		dt_from_xml = 0.01*4

	elif(which_agent==114):
		env = gym.make("Reacher-v2") 
		dt_from_xml = 0.01*2

	elif(which_agent==120):
		env = gym.make('AntX-v0')
		dt_from_xml = 0.01*20
	
	elif(which_agent==121):
		import pybullet
		import pybullet_envs
		pybullet.connect(pybullet.DIRECT)
		env = gym.make("AntBulletEnv-v0") 

	elif(which_agent==130):
		env = gym.make("Toy-v0") 
		dt_from_xml = 0.01*2

	elif(which_agent==200):
		import assistive_gym
		env = gym.make('ScratchItchPR2X-v0')
		dt_from_xml = 0.02*5

	elif(which_agent==201):
		import assistive_gym
		env = gym.make('DressingPR2X-v0')
		dt_from_xml = 0.01*10

	elif(which_agent==202):
		import assistive_gym
		env = gym.make('BedBathingPR2X-v0')
		dt_from_xml = 0.02*5
	#get dt value from env
	# if(which_agent==2):
	# 	env.reset()
	# 	dt_from_xml = env.robot.scene.dt 
	# elif(which_agent==4):
	# 	dt_from_xml = 2333
	# else:
	# 	dt_from_xml = env.model.opt.timestep
	print("\n\n the dt is: ", dt_from_xml, "\n\n")

	#set vars
	tf.random.set_seed(2)
	gym.logger.setLevel(logging.WARNING)
	dimO = env.observation_space.shape
	dimA = env.action_space.shape
	print ('--------------------------------- \nState space dimension: ', dimO)
	print ('Action space dimension: ', dimA, "\n -----------------------------------")

	return env, dt_from_xml


def only_reward(st, prev_st, a, dt, which_agent, goal=None, batch=False):

	if(which_agent==100):
		if not batch:
			xposafter = st[0]
			xposbefore = prev_st[0]
			forward_reward = (xposafter - xposbefore)/dt
			ctrl_cost = .5 * np.square(a).sum()
			contact_cost = 0  #####TEMP
			survive_reward = 1.0

		else:
			xposafter = st[:,0]
			xposbefore = prev_st[:, 0]
			forward_reward = (xposafter - xposbefore)/dt
			ctrl_cost = .5 * np.sum(np.square(a), axis=1)
			contact_cost = 0  #####TEMP
			survive_reward = 1.0


		r = forward_reward - ctrl_cost - contact_cost + survive_reward

	elif(which_agent == 101):
		reward_model = tf.keras.models.load_model('Eidos_models/reward_model.h5')
		s_a = np.atleast_2d(np.concatenate((st,a)))
		r_array = reward_model.predict(s_a)
		r = r_array.item()

	elif(which_agent==110):
		x, y = st[0:2]
		v1, v2 = st[8:10]
		dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
		vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
		alive_bonus = 10
		r = alive_bonus - dist_penalty - vel_penalty
		if(y <= 1):
			r = 0
		
	elif(which_agent == 111):
		r = -cartpole_cost_numpy(prev_st)

	elif(which_agent == 112):
		reward_ctrl = -0.1 * np.square(a).sum()
		reward_run = st[0] - 0.0 * np.square(st[2])
		r = reward_run + reward_ctrl

	elif(which_agent == 113):
		obj_pos = prev_st[17:20],
		vec_1 = obj_pos - prev_st[14:17]
		vec_2 = obj_pos - goal
		reward_near = -np.sum(np.abs(vec_1))
		reward_dist = -np.sum(np.abs(vec_2))
		reward_ctrl = -np.square(a).sum()
		r = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

	elif(which_agent==114):
		reward_dist = - np.linalg.norm(prev_st[-1])
		reward_ctrl = - np.square(a).sum()
		r = reward_dist + reward_ctrl

	elif(which_agent==120):
		xposafter = st[0]
		xposbefore = prev_st[0]
		forward_reward = (xposafter - xposbefore)/dt
		ctrl_cost = .5 * np.square(a).sum()
		contact_cost = 0  #####TEMP
		survive_reward = 1.0
		r = forward_reward - ctrl_cost - contact_cost + survive_reward
		if not (st[2] >= 0.2 and st[2] <= 1.0):
			r = 0  # Die

	elif(which_agent==130):
		reward_dist = - np.linalg.norm(prev_st[-1])
		reward_ctrl = - np.square(a).sum()
		r = reward_dist + reward_ctrl

	elif(which_agent == 200):
		if not batch:
			reward_distance = -np.linalg.norm(st[7:10])
			reward_action = -np.sum(np.square(a))
			tool_force_at_target = st[-3]
			reward_force_scratch = st[-2]
			preferences_score = st[-1]
		else:
			reward_distance = -np.linalg.norm(st[:,7:10])
			reward_action = -np.sum(np.square(a), axis=1)
			tool_force_at_target = st[:,-3]
			reward_force_scratch = st[:,-2]
			preferences_score = st[:,-1]
		r = 1*reward_distance + 0.01*reward_action + 0.01*tool_force_at_target + 2.0*reward_force_scratch + preferences_score

	elif(which_agent == 201):
		if not batch:
			reward_dressing = st[-2]
			reward_action = -np.sum(np.square(a))
			preferences_score = st[-1]
		else:
			reward_dressing = st[:,-2]
			reward_action = -np.sum(np.square(a), axis=1)
			preferences_score = st[:,-1]
		r = 1*reward_dressing + 0.01*reward_action + preferences_score

	elif(which_agent == 202):
		if not batch:
			reward_distance = st[-3]
			reward_action = -np.sum(np.square(a))
			reward_new_contact_points = st[-2]
			preferences_score = st[-1]
		else:
			reward_distance = st[:,-3]
			reward_action = -np.sum(np.square(a),axis=1)
			reward_new_contact_points = st[:,-2]
			preferences_score = st[:,-1]
		r = 1.0*reward_distance + 0.01*reward_action + 5.0*reward_new_contact_points + preferences_score

	else:
		raise NameError('NO ENVIRONMENT MATCHED')

	return r



def visualize_rendering(starting_state, list_of_actions, env_inp, dt_steps, dt_from_xml, which_agent):
	env=copy.deepcopy(env_inp)

	if(which_agent==5):
		env.reset()
	else:
		env.reset(starting_state)

	for action in list_of_actions:

		if(action.shape[0]==1):
			env.step(action[0], collectingInitialData=False)
		else:
			env.step(action, collectingInitialData=False)

		if(which_agent==5):
			junk=1
		else:
			env.render()
			time.sleep(dt_steps*dt_from_xml)

	print("Done rendering.")
	return

def log_to_file(strt, note=None, save_path=None):
	date = time.strftime("%Y-%m-%d", time.localtime())
	now = time.strftime("%H:%M:%S", time.localtime())
	there = False
	if note==None:
		name = f'{save_path}log_{date}.txt'
	else:
		note = ''
		name = f'{save_path}log_{note}.txt'
	if os.path.isfile(name):
		there = True
	with open(name,'a') as f:
		if there:
			f.write('\n')
		f.write(f"[{now}] {strt}")
		f.write('\n')


def get_state(env):

	#return env.state_now

	# envname = env.unwrapped.spec.id
	# if(envname == "Ant-v2"):
	# 	state = np.concatenate([
	# 			env.sim.data.qpos.flat,
	# 			env.sim.data.qvel.flat,
	# 			np.clip(env.sim.data.cfrc_ext, -1, 1).flat,
	# 		])[0:29]
	# elif(envname == "AntBulletEnv-v0"):
	# 	state = np.concatenate([env.robot.body_xyz[:2], env.robot.calc_state()])
	# elif(envname == "InvertedDoublePendulum-v2"):
	# 	x, _, y = env.sim.data.site_xpos[0]
	# 	state =  np.concatenate([
	# 			[x],[y],
	# 			env.sim.data.qpos[:1],  # cart x pos
	# 			np.sin(env.sim.data.qpos[1:]),  # link angles
	# 			np.cos(env.sim.data.qpos[1:]),
	# 			np.clip(env.sim.data.qvel, -10, 10),
	# 			np.clip(env.sim.data.qfrc_constraint, -10, 10)
	# 			]).ravel()
		

	# else:
	# raise NameError

	return state


def get_reward(pt, prev_pt, actions, scores, dt, discount_factor=0.95, which_agent=0, model=None, etc=None):
	scores, pt, prev_pt, actions = np.array(scores), np.array(pt), np.array(prev_pt), np.array(actions)
	if (model==None):
		# if False:
		# 	for i in range(len(pt)):  # for each sample
		# 		st = pt[i]
		# 		prev_st = prev_pt[i]
		# 		a = actions[i]

		# 		r = only_reward(st, prev_st, a, dt, which_agent, goal=etc)
				
		# 		scores_copied[i] += r*(discount_factor)  ###TODO: a discount
		#scores_copied = np.negative(scores_copied)
		#logg = f"======== one of a sample ========\nforward_reward = {forward_reward}  ctrl_cost = {ctrl_cost}  contact_cost = {contact_cost}  survive_reward = {survive_reward}"
		#log_to_file(logg)
		# scores = scores + np.array([only_reward(st, prev_st, a, dt, which_agent)*(discount_factor) for st,prev_st,a in zip(pt,prev_pt,actions)])
		scores = scores + only_reward(pt, prev_pt, actions, dt, which_agent, batch=True)*(discount_factor)
	else:
		s_a = np.concatenate((pt,actions), axis=1)
		r_batch = model.predict(s_a)
		r_batch = r_batch.flatten()

		scores = scores + r_batch*(discount_factor)

	return scores




def stack_tensor_dict_list(tensor_dict_list):
	
	"""Stack a list of dictionaries of {tensors or dictionary of tensors}.

	Args:
		tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
			dictionary of tensors}.

	Return:
		dict: a dictionary of {stacked tensors or dictionary of
			stacked tensors}

	"""
	keys = list(tensor_dict_list[0].keys())
	ret = dict()
	for k in keys:
		example = tensor_dict_list[0][k]
		dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
		if isinstance(example, dict):
			v = stack_tensor_dict_list(dict_list)
		else:
			v = np.array(dict_list)
		ret[k] = v
	return ret


# def frames_to_mp4(frames, prefix, save_dir, interval=None, fps=30):

# 	file = "{}/{}.mp4".format(save_dir, prefix)

# 	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
# 	writer = cv2.VideoWriter(file, fourcc, fps, (1920, 1080))

# 	for frame in frames:
# 		writer.write(cv2.cvtColor(np.array(frame)[:,:,:3], cv2.COLOR_RGB2BGR)) # Write out frame to video

# 	writer.release()

def frames_to_mp4(frames, prefix, save_dir, interval=None, fps=30):
	file = "{}/{}.mp4".format(save_dir, prefix)
	writer = FFmpegWriter(file)
	for frame in frames:
		writer.writeFrame(frame)
	writer.close()

def flatten(d, parent_key='', sep='/'):
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, collections.MutableMapping):
			items.extend(flatten(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)