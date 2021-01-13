import os
import numpy as np
import numpy.random as npr
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import copy
import sys
from six.moves import cPickle
import yaml
import argparse
import json
import shutil
import glob
from policy_random import Policy_Random
from trajectories import make_trajectory
from trajectories import get_trajfollow_params
from data_manipulation import *
from helper_funcs import *
from dynamics_model import Dyn_Model, BNN, ForwardActor
from mpc_controller import MPCController
from collect_samples import CollectSamples_SAC, CollectSamples_PPO, CollectSamples_random, CollectSamples_TD3
import wandb
import psutil
import ray

def main():


	#################################################
	############ commandline arguments ##############
	#################################################

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='antx')
	parser.add_argument('--seed', type=int, default=233)
	# date_and_time = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
	run_num_i = 0
	while os.path.exists("LOG/RUN_{}{:03d}".format(int(time.strftime("%Y%m%d", time.localtime())), run_num_i)):
		run_num_i += 1
	parser.add_argument('--run_num', type=int, default=int("{}{:03d}".format(int(time.strftime("%Y%m%d", time.localtime())), run_num_i)))
	parser.add_argument('--num_warmup', type=int, default=20000)
	parser.add_argument('--warmup_type', type=int, default=1, help='[0] Random, [1] SAC, [2] PPO, [3] TD3')
	parser.add_argument('--mpc_optimiser', type=int, default=1, help='[0] CMAES, [1] MultiGaussian, [2] SimplifiedCMAES')
	parser.add_argument('--alpha', type=float, default=0.05, help='Alpha of CVaR')
	parser.add_argument('--boosting', action="store_true", default=False, help='Default: False')

	parser.add_argument('--use_existing_training_data', action="store_true", dest='use_existing_training_data', default=False)
	parser.add_argument('--use_existing_dynamics_model', action="store_true", dest='use_existing_dynamics_model', default=False)
	parser.add_argument('--num_controlled_rollouts', type=int, default=60)
	parser.add_argument('--might_render', action="store_true", dest='might_render', default=False)
	parser.add_argument('--visualize_MPC_rollout', action="store_true", dest='visualize_MPC_rollout', default=False)
	parser.add_argument('--perform_forwardsim_for_vis', action="store_true", dest='perform_forwardsim_for_vis', default=False)
	parser.add_argument('--print_minimal', action="store_true", dest='print_minimal', default=False)
	parser.add_argument('--gpu', type=int, default=-1)
	parser.add_argument('--wandb', action="store_true", default=False)
	parser.add_argument('--save_movies', action="store_true", default=False)
	parser.add_argument('--continuous', action="store_true", default=False)
	parser.add_argument('--new_config', action="store_true", default=False)
	parser.add_argument('--desired_traj_type', type=str, default='straight') #TODO

	args = parser.parse_args()

	if args.gpu < 0:
		tf.config.experimental.set_visible_devices([], 'GPU')
	else:
		physical_devices = tf.config.list_physical_devices('GPU')
		tf.config.set_visible_devices(physical_devices[args.gpu], 'GPU')
		tf.config.experimental.set_memory_growth(physical_devices[args.gpu], True)
		# tf.config.experimental.set_virtual_device_configuration(physical_devices[args.gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])



	########################################
	######### params from yaml file ########
	########################################

	#load in parameters from specified file
	save_dir = 'LOG/RUN_'+ str(args.run_num)

	if not args.continuous or args.new_config:
		yaml_path = os.path.abspath('config/'+args.config+'.yaml')
		assert(os.path.exists(yaml_path))
		with open(yaml_path, 'r') as f:
			params = yaml.load(f)

		#from args
		print_minimal= args.print_minimal
		num_warmup = args.num_warmup
		warmup_type = args.warmup_type
		alpha = args.alpha
		boosting = args.boosting
		wandb_on = args.wandb
		mpc_optimiser = args.mpc_optimiser
		save_movies = args.save_movies
		counters_dict = {}

	else:
		with open(save_dir+'/checkpoints'+'/params.pkl', 'rb') as f:
			params = pickle.load(f)
		if os.path.isfile(save_dir+'/checkpoints'+'/counters.pkl'):
			with open(save_dir+'/checkpoints'+'/counters.pkl', 'rb') as f:
				counters_dict = pickle.load(f)
		else:
			counters_dict = {}


		print_minimal= params['parse']['print_minimal']
		num_warmup = params['parse']['num_warmup']
		warmup_type = params['parse']['warmup_type']
		alpha = params['parse']['alpha']
		boosting = params['parse']['boosting']
		wandb_on = params['parse']['wandb']
		mpc_optimiser = params['parse']['mpc_optimiser']
		save_movies = params['parse']['save_movies']

		yaml_path = os.path.abspath('config/'+args.config+'.yaml')
		if os.path.isfile(yaml_path):
			with open(yaml_path, 'r') as f:
				new_params = yaml.load(f)  
				params['controller']['distributed'] = new_params['controller']['distributed'] # Parameters would not affect results cound be overwritten.
		
	#save params from specified file
	which_agent = params['which_agent']
	follow_trajectories = params['follow_trajectories']
	#data collection
	use_threading = params['data_collection']['use_threading']
	num_rollouts_train = params['data_collection']['num_rollouts_train']
	num_rollouts_val = params['data_collection']['num_rollouts_val']
	#dynamics model
	num_fc_layers = params['dyn_model']['num_fc_layers']
	depth_fc_layers = params['dyn_model']['depth_fc_layers']
	batchsize = params['dyn_model']['batchsize']
	lr = params['dyn_model']['lr']
	nEpoch = params['dyn_model']['nEpoch']
	fraction_use_new = params['dyn_model']['fraction_use_new']
	#controller
	horizon = params['controller']['horizon']
	num_control_samples = params['controller']['num_control_samples']
	cmaes_N_factor = params['controller']['N_factor']
	cmaes_maxiter = params['controller']['maxiter']
	cmaes_num_MCscore = params['controller']['num_MCscore']
	cmaes_forward_particles = params['controller']['forward_particles']
	cmaes_distributed = params['controller']['distributed']
	#aggregation
	num_aggregation_iters = params['aggregation']['num_aggregation_iters']
	num_trajectories_for_aggregation = params['aggregation']['num_trajectories_for_aggregation']
	rollouts_forTraining = params['aggregation']['rollouts_forTraining']
	#noise
	make_aggregated_dataset_noisy = params['noise']['make_aggregated_dataset_noisy']
	make_training_dataset_noisy = params['noise']['make_training_dataset_noisy']
	noise_actions_during_MPC_rollouts = params['noise']['noise_actions_during_MPC_rollouts']
	#steps
	dt_steps = params['steps']['dt_steps']
	steps_per_episode = params['steps']['steps_per_episode']
	steps_per_rollout_train = params['steps']['steps_per_rollout_train']
	steps_per_rollout_val = params['steps']['steps_per_rollout_val']
	#saving
	min_rew_for_saving = params['saving']['min_rew_for_saving']
	#generic
	visualize_True = params['generic']['visualize_True']
	visualize_False = params['generic']['visualize_False']
		




	####TODO
	record = False
	cem_topK = 300


	########################################
	### make directories for saving data ###
	########################################

	
	if not os.path.exists(save_dir):
		if args.continuous:
			raise RuntimeError('Logging directory not found.')
		os.makedirs(save_dir)
		print('\n================  CREATING  '+ save_dir + '  ================\n')
	if not os.path.exists(save_dir+'/losses'):
		os.makedirs(save_dir+'/losses')
	if not os.path.exists(save_dir+'/config'):
		os.makedirs(save_dir+'/config')
	if not os.path.exists(save_dir+'/checkpoints'):
		if args.continuous:
			raise RuntimeError('Checkpoints directory not found.')
		os.makedirs(save_dir+'/checkpoints')
		os.makedirs(save_dir+'/checkpoints/policy')
		os.makedirs(save_dir+'/checkpoints/dynamics')
	if not os.path.exists(save_dir+'/models'):
		os.makedirs(save_dir+'/models')
	if not os.path.exists(save_dir+'/saved_forwardsim'):
		os.makedirs(save_dir+'/saved_forwardsim')
	if not os.path.exists(save_dir+'/saved_trajfollow'):
		os.makedirs(save_dir+'/saved_trajfollow')
	if not os.path.exists(save_dir+'/training_data'):
		os.makedirs(save_dir+'/training_data')
	if not os.path.exists(save_dir+'/summary'):
		os.makedirs(save_dir+'/summary')
	if not os.path.exists(save_dir+'/summary/plots'):
		os.makedirs(save_dir+'/summary/plots')
	if not os.path.exists(save_dir+'/src'):
		os.makedirs(save_dir+'/src')
	if not os.path.exists(save_dir+'/summary/MPC_rewards'):
		os.makedirs(save_dir+'/summary/MPC_rewards')

	filenames = glob.glob('*.py')  # put copy of all python files in log_dir
	for filename in filenames:     # for reference
		shutil.copy(filename, save_dir+'/src')


	########################################
	############## set vars ################
	########################################

	#set seeds
	npr.seed(args.seed)
	#tf.set_random_seed(args.seed)
	tf.random.set_seed(args.seed)
	cma_seed = args.seed
	
	#data collection, either with or without multi-threading
	if(use_threading):
		from collect_samples_threaded import CollectSamples
	else:
		from collect_samples import CollectSamples

	#more vars
	x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(which_agent)
	tf_datatype = tf.float64
	noiseToSignal = 0.01

	# n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated
	actions_ag='nc'
	
	#################################################
	######## save param values to a file ############
	#################################################

	# param_dict={}
	# param_dict['which_agent']= which_agent
	# param_dict['use_existing_training_data']= str(args.use_existing_training_data)
	# param_dict['desired_traj_type']= args.desired_traj_type
	# param_dict['visualize_MPC_rollout']= str(args.visualize_MPC_rollout)
	# param_dict['num_controlled_rollouts']= args.num_controlled_rollouts
	# param_dict['seed']= args.seed
	# param_dict['follow_trajectories']= str(follow_trajectories)
	# param_dict['use_threading']= str(use_threading)
	# param_dict['num_rollouts_train']= num_rollouts_train
	# param_dict['num_fc_layers']= num_fc_layers
	# param_dict['depth_fc_layers']= depth_fc_layers
	# param_dict['batchsize']= batchsize
	# param_dict['lr']= lr
	# param_dict['nEpoch']= nEpoch
	# param_dict['fraction_use_new']= fraction_use_new
	# param_dict['horizon']= horizon
	# param_dict['num_control_samples']= num_control_samples
	# param_dict['N_factor'] = cmaes_N_factor
	# param_dict['maxiter'] = cmaes_maxiter
	# param_dict['num_MCscore'] = cmaes_num_MCscore
	# param_dict['forward_particles'] = cmaes_forward_particles
	# param_dict['distributed'] = cmaes_distributed
	# param_dict['alpha'] = alpha

	# param_dict['num_aggregation_iters']= num_aggregation_iters
	# param_dict['num_trajectories_for_aggregation']= num_trajectories_for_aggregation
	# param_dict['rollouts_forTraining']= rollouts_forTraining
	# param_dict['make_aggregated_dataset_noisy']= str(make_aggregated_dataset_noisy)
	# param_dict['make_training_dataset_noisy']= str(make_training_dataset_noisy)
	# param_dict['noise_actions_during_MPC_rollouts']= str(noise_actions_during_MPC_rollouts)
	# param_dict['dt_steps']= dt_steps
	# param_dict['steps_per_episode']= steps_per_episode
	# param_dict['steps_per_rollout_train']= steps_per_rollout_train
	# param_dict['steps_per_rollout_val']= steps_per_rollout_val
	# param_dict['min_rew_for_saving']= min_rew_for_saving
	# param_dict['x_index']= x_index
	# param_dict['y_index']= y_index
	# param_dict['tf_datatype']= str(tf_datatype)
	# param_dict['noiseToSignal']= noiseToSignal
	param_to_save = params
	if not args.continuous:
		param_to_save['parse'] = vars(args)

		with open(save_dir+'/checkpoints'+'/params.pkl', 'wb') as f:
			pickle.dump(param_to_save, f, pickle.HIGHEST_PROTOCOL)
		with open(save_dir+'/config'+'/params.txt', 'w') as f:
			f.write(json.dumps(param_to_save))

	param_dict = flatten(param_to_save)


	# LOGGING SETTING
	typeL = ['R','S','P', 'T'][warmup_type]
	typeL1 = ['CM', 'M','C'][mpc_optimiser]
	if wandb_on:
		if which_agent==101:
			wandb.init(config=param_dict, project="Eidos", name=f'GRAMCO[{typeL}{typeL1}] on EidosE1')
		elif which_agent==102:
			wandb.init(config=param_dict, project="Eidos", name=f'GRAMCO[{typeL}{typeL1}] on EidosE2')
		elif which_agent==103:
			wandb.init(config=param_dict, project="Eidos", name=f'GRAMCO[{typeL}{typeL1}] on EidosE3')
		elif which_agent==200:
			wandb.init(config=param_dict, project="Assistive Gym", name=f'GRAMCO[{typeL}{typeL1}] on ScratchItchPR2X')#+args.config.capitalize())
		elif which_agent==201:
			wandb.init(config=param_dict, project="Assistive Gym", name=f'GRAMCO[{typeL}{typeL1}] on DressingPR2X')
		elif which_agent==202:
			wandb.init(config=param_dict, project="Assistive Gym", name=f'GRAMCO[{typeL}{typeL1}] on BedBathingPR2X')

		elif which_agent==100:
			wandb.init(config=param_dict, project="AntX", name=f'GRAMCO[{typeL}{typeL1}] on AntX')
		


	writer = tf.summary.create_file_writer(logdir=save_dir+'/summary/',max_queue=1)
	writer.set_as_default()
	#################################################
	### initialize the experiment
	#################################################

	if(not(print_minimal)):
		print("\n#####################################")
		print("Initializing environment")
		print("#####################################\n")

	#create env
	env, dt_from_xml= create_env(which_agent)

	#create random policy for data collection
	random_policy = Policy_Random(env)

	#################################################
	### set GPU options for TF
	#################################################
	

	# gpu_device = args.gpu
	# gpu_frac = 0.3
	# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
	'''
	config = tf.ConfigProto(gpu_options=gpu_options,
							log_device_placement=False,
							allow_soft_placement=True,
							inter_op_parallelism_threads=1,
							intra_op_parallelism_threads=1)
	'''

	# config = tf.ConfigProto()

	# with tf.Session(config=config) as sess:
	if(True):

		#################################################
		### deal with data
		#################################################

		# if(args.use_existing_training_data):
		# 	if(not(print_minimal)):
		# 		print("\n#####################################")
		# 		print("Retrieving training data & policy from saved files")
		# 		print("#####################################\n")
			
		# 	states_data= np.load(save_dir + '/training_data/states_data.npy') # input1: state
		# 	actions_data= np.load(save_dir + '/training_data/actions_data.npy') # input2: control
		# 	delta_states_data= np.load(save_dir + '/training_data/delta_states_data.npy') # output: nextstate-state
		# 	states_val= np.load(save_dir + '/training_data/states_val.npy', allow_pickle=True)
		# 	controls_val= np.load(save_dir + '/training_data/controls_val.npy', allow_pickle=True)
		# 	forwardsim_x_true= np.load(save_dir + '/training_data/forwardsim_x_true.npy', allow_pickle=True)
		# 	forwardsim_y= np.load(save_dir + '/training_data/forwardsim_y.npy', allow_pickle=True)

		WU_output_dir = save_dir+'/MF_agent'
		MF_core = [CollectSamples_random(env, num_warmup, WU_output_dir, _episode_max_steps=steps_per_rollout_train, gpu=args.gpu, wandb_on=wandb_on),
				   CollectSamples_SAC(env, num_warmup, WU_output_dir, _episode_max_steps=steps_per_rollout_train, gpu=args.gpu, wandb_on=wandb_on),
				   CollectSamples_PPO(env, num_warmup, WU_output_dir, _episode_max_steps=steps_per_rollout_train, gpu=args.gpu, wandb_on=wandb_on),
				   CollectSamples_TD3(env, num_warmup, WU_output_dir, _episode_max_steps=steps_per_rollout_train, gpu=args.gpu, wandb_on=wandb_on)][warmup_type]

		if args.continuous:

			if(not(print_minimal)):
				print("\n#####################################")
				print("Retrieving training data & policy from saved files")
				print("#####################################\n")

			states_data= np.load(save_dir + '/training_data/states_data.npy') # input1: state
			actions_data= np.load(save_dir + '/training_data/actions_data.npy') # input2: control
			delta_states_data= np.load(save_dir + '/training_data/delta_states_data.npy') # output: nextstate-state

			if not os.path.isfile(save_dir + '/training_data/states_val.npy'):
				print('No validation data found, perform rollouts to collect validation data.')
				states_val, controls_val, _, _ = perform_rollouts(random_policy, num_rollouts_val, steps_per_rollout_val, visualize_False, 
															CollectSamples, env, which_agent, dt_steps, dt_from_xml, follow_trajectories)
			else:
				states_val = np.load(save_dir + '/training_data/states_val.npy', allow_pickle=True)
				controls_val = np.load(save_dir + '/training_data/controls_val.npy', allow_pickle=True)

			if not counters_dict == {}:
				MF_core.load_checkpoints(counters_dict['global_total_steps'], counters_dict['global_num_episodes'], counters_dict['global_cost']) 
			else:
				MF_core.load_checkpoints(0,0,0)

		else:

			if(not(print_minimal)):
				print("\n#####################################")
				print("Performing rollouts to collect training data")
				print("#####################################\n")

			
			states, controls, _, _ = MF_core.collect_samples()
			#print('STATE SHAPE A: ',np.shape(states))


			if(not(print_minimal)):
				print("\n#####################################")
				print("Performing rollouts to collect validation data")
				print("#####################################\n")

			start_validation_rollouts = time.time()
			states_val, controls_val, _, _ = perform_rollouts(random_policy, num_rollouts_val, steps_per_rollout_val, visualize_False, 
															CollectSamples, env, which_agent, dt_steps, dt_from_xml, follow_trajectories)
			states_val = np.array(states_val)
			# if(not(print_minimal)):
			# 	print("\n#####################################")
			# 	print("Convert from env observations to NN 'states' ")
			# 	print("#####################################\n")

			# #training
			# states = from_observation_to_usablestate(states, which_agent, False)
			# #print('STATE SHAPE B: ',np.shape(states))
			# #validation
			# states_val = from_observation_to_usablestate(states_val, which_agent, False)
			# states_val = np.array(states_val)

			if(not(print_minimal)):
				print("\n#####################################")
				print("Data formatting: create inputs and labels for NN ")
				print("#####################################\n")

			states_data , actions_data = generate_training_data_inputs(states, controls)
			delta_states_data = generate_training_data_outputs(states, which_agent)

			if(not(print_minimal)):
				print("\n#####################################")
				print("Add noise")
				print("#####################################\n")

			#add a little dynamics noise (next state is not perfectly accurate, given correct state and action)
			if(make_training_dataset_noisy):
				states_data = add_noise(states_data, noiseToSignal)
				delta_states_data = add_noise(delta_states_data, noiseToSignal)

			if(args.perform_forwardsim_for_vis):

				if(not(print_minimal)):
					print("\n#####################################")
					print("Perform rollout & save for forward sim")
					print("#####################################\n")

				states_forwardsim_orig, controls_forwardsim, _,_ = perform_rollouts(random_policy, 1, 100,           # states_forwardsim_orig:[Mrollout, Nsteps, Xdim]
																				visualize_False, CollectSamples, 
																				env, which_agent, dt_steps, 
																				dt_from_xml, follow_trajectories)
				states_forwardsim = np.copy(from_observation_to_usablestate(states_forwardsim_orig, which_agent, False))
				forwardsim_x_true, forwardsim_y = generate_training_data_inputs(states_forwardsim, controls_forwardsim)
				np.save(save_dir + '/training_data/forwardsim_x_true.npy', forwardsim_x_true)
				np.save(save_dir + '/training_data/forwardsim_y.npy', forwardsim_y)
			
			if(not(print_minimal)):
				print("\n#####################################")
				print("Saving data")
				print("#####################################\n")

			np.save(save_dir + '/training_data/states_data.npy', states_data)
			np.save(save_dir + '/training_data/actions_data.npy', actions_data)
			np.save(save_dir + '/training_data/delta_states_data.npy', delta_states_data)
			np.save(save_dir + '/training_data/states_val.npy', states_val)
			np.save(save_dir + '/training_data/controls_val.npy', controls_val)
			

		if(not(print_minimal)):
			print("Done getting data.")
			print("states_data dim: ", states_data.shape)

		#################################################
		### init vars
		#################################################


		training_loss_list=[]
		forwardsim_score_list=[]
		old_loss_list=[]
		new_loss_list=[]
		errors_1_per_agg=[]
		errors_5_per_agg=[]
		errors_10_per_agg=[]
		errors_50_per_agg=[]
		errors_100_per_agg=[]
		list_avg_rew=[]
		list_num_datapoints=[]
		avg_loss_per_epoch_list=[]
		counter_agg_iters_real = 0
		if not args.continuous or counters_dict == {}:
			counter_agg_iters=0
			counters_dict['counter_agg_iters'] = counter_agg_iters
			states_data_new = np.zeros((0,states_data.shape[1]))
			actions_data_new = np.zeros((0,actions_data.shape[1]))
			delta_states_data_new = np.zeros((0,delta_states_data.shape[1]))
		else:
			counter_agg_iters = counters_dict['counter_agg_iters']# if not counters_dict == {} else 0
			states_data_new= np.load(save_dir + '/training_data/states_data_new.npy')
			actions_data_new= np.load(save_dir + '/training_data/actions_data_new.npy')
			delta_states_data_new= np.load(save_dir + '/training_data/delta_states_data_new_.npy')

		
		
		#################################################
		### preprocess the old training dataset
		#################################################

		if(not(print_minimal)):
			print("\n#####################################")
			print("Preprocessing 'old' training data")
			print("#####################################\n")

		#every component (i.e. x position) should become mean 0, std 1
		mean_x = np.mean(states_data, axis = 0)
		states_data = states_data - mean_x
		std_x = np.std(states_data, axis = 0)
		states_data = np.nan_to_num(states_data/std_x)

		mean_y = np.mean(actions_data, axis = 0) 
		actions_data = actions_data - mean_y
		std_y = np.std(actions_data, axis = 0)
		actions_data = np.nan_to_num(actions_data/std_y)

		mean_z = np.mean(delta_states_data, axis = 0) 
		delta_states_data = delta_states_data - mean_z
		std_z = np.std(delta_states_data, axis = 0)
		delta_states_data = np.nan_to_num(delta_states_data/std_z)

		## concatenate state and action, to be used for training dynamics
		print("Shape of states_data: ", np.shape(states_data), "Shape of actions_data: ",np.shape(actions_data))
		inputs = np.concatenate((states_data, actions_data), axis=1)
		outputs = np.copy(delta_states_data)

		#doing a render here somehow allows it to not produce an error later
		might_render = (record and which_agent==1)
		if(args.visualize_MPC_rollout or args.might_render):
			might_render=True
		if(might_render):
			new_env, _ = create_env(which_agent)
			new_env.render()

		# Defind initial state
		fixed_first_state = False
		if(fixed_first_state):
			env.reset()
			starting_fullenvstate = env.sim.get_state()
		else:
			starting_fullenvstate = 0

		##############################################
		########## THE AGGREGATION LOOP ##############
		##############################################

		#dimensions
		assert inputs.shape[0] == outputs.shape[0]
		inputSize = inputs.shape[1]
		outputSize = outputs.shape[1]
	
		#initialize dynamics model
		dyn_cp_dir = save_dir + '/checkpoints/dynamics'
		sess = 0
		dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, which_agent, x_index, y_index, num_fc_layers,
							depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal, dyn_cp_dir)

		
		if cmaes_distributed:
			# ray.init()
			ray.init(num_cpus=min(psutil.cpu_count(logical=False), cmaes_num_MCscore))
			dyn_actor = ForwardActor.remote(depth_fc_layers, inputSize, outputSize, dyn_model.model_prob, dyn_model.model_lam, std_z, mean_z, std_y, mean_y, std_x, mean_x)  # n_hidden, n_in, n_out, prob, model_lam

		else:
			dyn_actor = None

		cmaes_args = {
		"N_factor": cmaes_N_factor,
		"maxiter": cmaes_maxiter,
		"num_MCscore": cmaes_num_MCscore,
		"forward_particles": cmaes_forward_particles,
		"distributed": cmaes_distributed
		}
		#create mpc controller
		mpc_controller = MPCController(env, dyn_model, horizon, which_agent, steps_per_episode, dt_steps, num_control_samples, 
										mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index, 
										z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, 
										frontfoot_index, xvel_index, orientation_index, dt_from_xml, MF_core, cmaes_args, cma_seed, alpha, boosting, wandb_on, mpc_optimiser, save_movies, dyn_actor)


		while(counter_agg_iters < num_aggregation_iters):

			#make saver
			# if(counter_agg_iters==0):
			# 	saver = tf.train.Saver(max_to_keep=0)

			print("\n#####################################")
			print("AGGREGATION ITERATION ", counter_agg_iters)
			print("#####################################\n")

			#save the aggregated dataset used to train during this agg iteration
			# save_dataset = False
			# if save_dataset:
			# 	np.save(save_dir + '/training_data/states_data_new_iter'+ str(counter_agg_iters) + '.npy', states_data_new)
			# 	np.save(save_dir + '/training_data/actions_data_new_iter'+ str(counter_agg_iters) + '.npy', actions_data_new)
			# 	np.save(save_dir + '/training_data/delta_states_data_new_iter'+ str(counter_agg_iters) + '.npy', delta_states_data_new)

			starting_big_loop = time.time()

			if(not(print_minimal)):
				print("\n#####################################")
				print("Preprocessing 'new' training data  (!!! START BIG LOOP)")
				print("#####################################\n")

			states_data_new_preprocessed = np.nan_to_num((states_data_new - mean_x)/std_x)
			actions_data_new_preprocessed = np.nan_to_num((actions_data_new - mean_y)/std_y)
			delta_states_data_new_preprocessed = np.nan_to_num((delta_states_data_new - mean_z)/std_z)

			std_x[std_x==0] = 0.001
			std_y[std_y==0] = 0.001
			std_z[std_z==0] = 0.001

			## concatenate state and action, to be used for training dynamics
			inputs_new = np.concatenate((states_data_new_preprocessed, actions_data_new_preprocessed), axis=1)
			outputs_new = np.copy(delta_states_data_new_preprocessed)
			#outputs_new = np.copy(delta_states_data_new)

			if(not(print_minimal)):
				print("\n#####################################")
				print("Training the dynamics model")
				print("#####################################\n")

			#train model or restore model
			# if(args.use_existing_dynamics_model):
			# 	restore_path = save_dir+ '/models/DynamicsModel_final.h5'
			# 	# saver.restore(sess, restore_path)
			# 	_ = dyn_model.bnn_network(inputs[0:2,:]) # Initilisation
			# 	dyn_model.bnn_network.load_weights(restore_path)


			# 	print("Model restored from ", restore_path)
			# 	training_loss=0
			# 	avg_loss_per_epoch=0
			# 	old_loss=0
			# 	new_loss=0
			if args.continuous and counter_agg_iters_real==0:
				dyn_model.load_checkpoints()
				training_loss, old_loss, new_loss, avg_loss_per_epoch = 0, 0, 0, 0
			else:
				training_loss, old_loss, new_loss, avg_loss_per_epoch = dyn_model.train(inputs, outputs, inputs_new, outputs_new, 
																						nEpoch, save_dir, fraction_use_new, num_aggregation_iters=num_aggregation_iters)
			if cmaes_distributed:
				dyn_actor.load_weights.remote(dyn_model.bnn_network.get_weights())


			#how good is model on training data
			training_loss_list.append(training_loss)  ### may be wrong
			avg_loss_per_epoch_list.append(avg_loss_per_epoch)
			np.save(save_dir + '/summary/All_dynamics_training_loss_forALLaggiters.npy', np.array(avg_loss_per_epoch_list)) 

			#how good is model on old dataset
			old_loss_list.append(old_loss)
			#how good is model on new dataset
			new_loss_list.append(new_loss)

			print("\nTraining loss: ", training_loss)

			#####################################
			## Saving model
			#####################################

			# save_path = saver.save(sess, save_dir+ '/models/DynamicsModel_aggIter' +str(counter_agg_iters)+ '.ckpt') #model_aggIter -> DynamicsModel__aggIter
			# save_path = saver.save(sess, save_dir+ '/models/DynamicsModel_final.ckpt') #finalModel -> DynamicsModel_final
			# dyn_model.bnn_network.save_weights(save_dir+ '/models/DynamicsModel_aggIter' +str(counter_agg_iters)+ '.h5')

			dyn_model.bnn_network.save_weights(save_dir+ '/models/DynamicsModel.h5')
			
			if(not(print_minimal)):
				print("Model saved at ", save_dir+ '/models/DynamicsModel.h5')

			#####################################
			## calculate multi-step validation metrics
			#####################################

			if(not(print_minimal)):
				print("\n#####################################")
				print("Calculating Validation Metrics")
				print("#####################################\n")

			#####################################
			## init vars for multi-step validation metrics
			#####################################

			validation_inputs_states = []
			labels_1step = []
			labels_5step = []
			labels_10step = []
			labels_50step = []
			labels_100step = []
			controls_100step=[]

			#####################################
			## make the arrays to pass into forward sim
			#####################################

			#print("============ States for validation: ", np.shape(states_val))

			for i in range(num_rollouts_val):

				#print(f"The iteration {i} of {num_rollouts_val} validation ...")
				

				length_curr_rollout = states_val[i].shape[0]

				
				#print(f"The length of current rollout is {states_val[i].shape} ...")

				if(length_curr_rollout>100):

					#########################
					#### STATE INPUTS TO NN
					#########################

					## take all except the last 100 pts from each rollout
					validation_inputs_states.append(states_val[i][0:length_curr_rollout-100])

					#########################
					#### CONTROL INPUTS TO NN
					#########################

					#100 step controls
					list_100 = []
					for j in range(100):
						list_100.append(controls_val[i][0+j:length_curr_rollout-100+j])
						##for states 0:x, first apply acs 0:x, then apply acs 1:x+1, then apply acs 2:x+2, etc...
					list_100=np.array(list_100) #100xstepsx2
					list_100= np.swapaxes(list_100,0,1) #stepsx100x2
					controls_100step.append(list_100)

					#########################
					#### STATE LABELS- compare these to the outputs of NN (forward sim)
					#########################
					labels_1step.append(states_val[i][0+1:length_curr_rollout-100+1])
					labels_5step.append(states_val[i][0+5:length_curr_rollout-100+5])
					labels_10step.append(states_val[i][0+10:length_curr_rollout-100+10])
					labels_50step.append(states_val[i][0+50:length_curr_rollout-100+50])
					labels_100step.append(states_val[i][0+100:length_curr_rollout-100+100])

			if (validation_inputs_states == []):
				#raise NameError('The data for validation may be not enough...')
				print('[VALIDATION SKIPPED] The data for validation may not be enough... \n')
			else:
				validation_inputs_states = np.concatenate(validation_inputs_states)
				controls_100step = np.concatenate(controls_100step)
				labels_1step = np.concatenate(labels_1step)
				labels_5step = np.concatenate(labels_5step)
				labels_10step = np.concatenate(labels_10step)
				labels_50step = np.concatenate(labels_50step)
				labels_100step = np.concatenate(labels_100step)

				# print('[TEMP] =========> INPUT SATES [1] : ', np.mean(validation_inputs_states))

				#####################################
				## pass into forward sim, to make predictions
				#####################################

				many_in_parallel = True
				predicted_100step = dyn_model.do_forward_sim(validation_inputs_states, controls_100step, 
															many_in_parallel, env, which_agent, num_particles=5)
				
				#####################################
				## Calculate validation metrics (mse loss between predicted and true)
				#####################################

				array_meanx = np.tile(np.expand_dims(mean_x, axis=0),(labels_1step.shape[0],1))
				array_stdx = np.tile(np.expand_dims(std_x, axis=0),(labels_1step.shape[0],1))

				error_1step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[1]-array_meanx,array_stdx)) 
									-np.nan_to_num(np.divide(labels_1step-array_meanx,array_stdx))))
				error_5step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[5]-array_meanx,array_stdx))
									-np.nan_to_num(np.divide(labels_5step-array_meanx,array_stdx))))
				error_10step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[10]-array_meanx,array_stdx))
										-np.nan_to_num(np.divide(labels_10step-array_meanx,array_stdx))))
				error_50step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[50]-array_meanx,array_stdx)) 
										-np.nan_to_num(np.divide(labels_50step-array_meanx,array_stdx))))
				error_100step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[100]-array_meanx,array_stdx)) 
										-np.nan_to_num(np.divide(labels_100step-array_meanx,array_stdx))))
				print("Multistep error values: ", error_1step, error_5step, error_10step, error_50step, error_100step,"\n")

				errors_1_per_agg.append(error_1step)
				errors_5_per_agg.append(error_5step)
				errors_10_per_agg.append(error_10step)
				errors_50_per_agg.append(error_50step)
				errors_100_per_agg.append(error_100step)

			#####################################
			## Perform 1 forward simulation, for visualization purposes (compare predicted traj vs true traj)
			#####################################

			if(args.perform_forwardsim_for_vis):
				if(not(print_minimal)):
					print("\n#####################################")
					print("Performing a forward sim of the learned model. using pre-saved dataset. just for visualization")
					print("#####################################\n")
			
				#for a given set of controls,
				#compare sim traj vs. learned model's traj 
				#(dont expect this to be good cuz error accum)
				many_in_parallel = False
				forwardsim_x_pred = dyn_model.do_forward_sim(forwardsim_x_true, forwardsim_y, many_in_parallel, env, which_agent)    
				forwardsim_x_pred = np.array(forwardsim_x_pred)

				# save results of forward sim
				np.save(save_dir + '/saved_forwardsim/forwardsim_states_true_'+str(counter_agg_iters)+'.npy', forwardsim_x_true)
				np.save(save_dir + '/saved_forwardsim/forwardsim_states_pred_'+str(counter_agg_iters)+'.npy', forwardsim_x_pred)

				forwardsim_x_pred_particles = dyn_model.predict_for_visualisation(forwardsim_x_true, forwardsim_y, many_in_parallel, env, which_agent, 
																				  num_particles=5, repeat=100)

				np.save(save_dir + '/saved_forwardsim/forwardsim_states_pred_particles_'+str(counter_agg_iters)+'.npy', np.array(forwardsim_x_pred_particles))




			#####################################
			######## EXECUTE CONTROLLER #########
			#####################################

			if(not(print_minimal)):
				print("##############################################")
				print("#### Execute the controller to follow desired trajectories")
				print("##############################################\n")

			###################################################################
			### Try to follow trajectory... collect rollouts
			###################################################################

			#init vars
			list_rewards=[]
			starting_states=[]
			selected_multiple_u = []
			resulting_multiple_x = []

			#get parameters for trajectory following
			horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, desired_snake_headingInit = get_trajfollow_params(which_agent, args.desired_traj_type)
			if(follow_trajectories==False):
				desired_snake_headingInit=0

			for rollout_num in range(num_trajectories_for_aggregation):

				if(not(print_minimal)):
					print("\nPerforming MPC rollout #", rollout_num)

				#reset env and set the desired traj 

				starting_observation = env.reset()
				if (fixed_first_state):
					env.sim.set_state(starting_fullenvstate)
				# starting_observation = get_state(env)
				starting_state = None

				#desired trajectory to follow
				starting_observation_NNinput = from_observation_to_usablestate(starting_observation, which_agent, True)
				desired_x = make_trajectory(args.desired_traj_type, starting_observation_NNinput, x_index, y_index, which_agent)
				#desired_x = None

				#perform 1 MPC rollout
				#depending on follow_trajectories, either move forward or follow desired_traj_type
				if(noise_actions_during_MPC_rollouts):
					curr_noise_amount = 0.005
				else:
					curr_noise_amount=0

				mpc_info = {"save_dir": save_dir, "scope": "aggIter"+str(counter_agg_iters), "rollout_num": rollout_num, "record": record, "aggIter": counter_agg_iters}
				action_chooser_info = {"action_chooser": 2, "K": cem_topK}
				resulting_x, selected_u, ep_rew, outputdict = mpc_controller.perform_rollout(starting_fullenvstate, starting_observation, 
																		starting_observation_NNinput, desired_x, 
																		follow_trajectories, horiz_penalty_factor, 
																		forward_encouragement_factor, heading_penalty_factor, 
																		noise_actions_during_MPC_rollouts, curr_noise_amount, 
																		mpc_info, action_chooser_info)

				#save info from MPC rollout
				list_rewards.append(ep_rew)
				selected_multiple_u.append(selected_u)
				resulting_multiple_x.append(resulting_x)
				starting_states.append(starting_state)

				if False:
					np.save(save_dir + '/summary/MPC_rewards/MPC_rewards_for_'+ str(len(list_rewards)) +'rollouts_Iter' + str(counter_agg_iters) + "_"
						+ str(rollout_num) + '.npy', outputdict["rewards"])

			if(args.visualize_MPC_rollout):
				input("\n\nPAUSE BEFORE VISUALIZATION... Press Enter to continue...")
				for vis_index in range(num_trajectories_for_aggregation):
					visualize_rendering(None, selected_multiple_u[vis_index], env, dt_steps, dt_from_xml, which_agent)

			#bookkeeping
			avg_rew = np.mean(np.array(list_rewards))
			std_rew = np.std(np.array(list_rewards))
			print("############# Avg reward for ", num_trajectories_for_aggregation, " MPC rollouts: ", avg_rew)
			print("############# Std reward for ", num_trajectories_for_aggregation, " MPC rollouts: ", std_rew)
			print("############# Rewards for the ", num_trajectories_for_aggregation, " MPC rollouts: ", list_rewards)

			if False:
				np.save(save_dir + '/summary/MPC_scores_for_'+ str(len(list_rewards)) +'rollouts_Iter' + str(counter_agg_iters) +'.npy', np.array(list_rewards))

			#save pts_used_so_far + performance achieved by those points
			list_num_datapoints.append(states_data.shape[0]+states_data_new.shape[0])
			list_avg_rew.append(avg_rew)

			# print("\n############# TRAINING THE MODEL-FREE AGENT")
			# timer0 = time.time()
			# MF_core.train()
			# print('############# DONE, COST ',time.time()-timer0, ' s')


			##############################
			### Aggregate data
			##############################

			full_states_list = []
			full_controls_list = []
			if(counter_agg_iters<(num_aggregation_iters-1)):

				##############################
				### aggregate some rollouts into training set
				##############################

				x_array = np.array(resulting_multiple_x)[0:(rollouts_forTraining+1)]
				#print('OOOOOOOOOO  ', np.shape(x_array), np.mean(x_array))
				if(True): #which_agent==6 or which_agent==1
					u_array = np.array(selected_multiple_u)[0:(rollouts_forTraining+1)]
				else:
					u_array = np.squeeze(np.array(selected_multiple_u), axis=2)[0:(rollouts_forTraining+1)]
					#print('AAAAAAAA  ', np.shape(u_array), np.mean(u_array))

				for i in range(rollouts_forTraining):
					
					if(True):#which_agent==6 or which_agent==1
						x= np.array(x_array[i])
						u= np.squeeze(u_array[i], axis=1)
					else:
						x= x_array[i] #[N+1, NN_inp]
						u= u_array[i] #[N, actionSize]
					
					newstates_data= np.copy(x[0:-1, :])
					newactions_data= np.copy(u)
					newdelta_states_data= np.copy(x[1:, :]-x[0:-1, :])
					
					# make this new data a bit noisy before adding it into the dataset
					if(make_aggregated_dataset_noisy):
						newstates_data = add_noise(newstates_data, noiseToSignal)
						newdelta_states_data = add_noise(newdelta_states_data, noiseToSignal)

					# the actual aggregation
					states_data_new = np.concatenate((states_data_new, newstates_data))
					actions_data_new = np.concatenate((actions_data_new, newactions_data))
					delta_states_data_new = np.concatenate((delta_states_data_new, newdelta_states_data))


				np.save(save_dir + '/training_data/states_data_new' + '.npy', states_data_new)
				np.save(save_dir + '/training_data/actions_data_new' + '.npy', actions_data_new)
				np.save(save_dir + '/training_data/delta_states_data_new_' + '.npy', delta_states_data_new)
				##############################
				### aggregate the rest of the rollouts into validation set
				##############################

				x_array = np.array(resulting_multiple_x)[rollouts_forTraining:len(resulting_multiple_x)] 
				# ^ dim: [rollouts_forValidation x stepsPerEpisode+1 x stateSize]
				if(True):#which_agent==6 or which_agent==1
					u_array = np.array(selected_multiple_u)[rollouts_forTraining:len(resulting_multiple_x)] 
				else:
					u_array = np.squeeze(np.array(selected_multiple_u), axis=2)[rollouts_forTraining:len(resulting_multiple_x)] 
					# rollouts_forValidation x stepsPerEpisode x acSize 

				full_states_list = []
				full_controls_list = []
				for i in range(states_val.shape[0]):
					full_states_list.append(states_val[i])
					full_controls_list.append(controls_val[i])
				for i in range(x_array.shape[0]):
					x = np.array(x_array[i])
					full_states_list.append(x[0:-1,:])
					full_controls_list.append(np.squeeze(u_array[i]))
				states_val = np.array(full_states_list)
				controls_val = np.array(full_controls_list)

				np.save(save_dir + '/training_data/states_val.npy', states_val)
				np.save(save_dir + '/training_data/controls_val.npy', controls_val)

			#save trajectory following stuff (aka trajectory taken) for plotting
			# if False:
			# 	np.save(save_dir + '/saved_trajfollow/startingstate_iter' + str(counter_agg_iters) +'.npy', starting_state)
			# 	np.save(save_dir + '/saved_trajfollow/control_iter' + str(counter_agg_iters) +'.npy', selected_u)
			# 	np.save(save_dir + '/saved_trajfollow/true_iter' + str(counter_agg_iters) +'.npy', desired_x)
			# 	np.save(save_dir + '/saved_trajfollow/pred_iter' + str(counter_agg_iters) +'.npy', np.array(resulting_multiple_x))

			#bookkeeping
			if(not(print_minimal)):
				print("\n\nDONE WITH BIG LOOP ITERATION ", counter_agg_iters ,"\n\n")
				print("training dataset size: ", states_data.shape[0] + states_data_new.shape[0])
				if(len(full_states_list)>0):
					print("validation dataset size: ", np.concatenate(full_states_list).shape[0])
				print("Time taken: {:0.2f} s\n\n".format(time.time()-starting_big_loop))
			counter_agg_iters += 1
			counter_agg_iters_real += 1

			counters_dict['global_total_steps'] = MF_core.global_total_steps
			counters_dict['global_num_episodes'] = MF_core.global_num_episodes
			counters_dict['global_cost'] = MF_core.global_cost
			counters_dict['counter_agg_iters'] = counter_agg_iters

			with open(save_dir+'/checkpoints'+'/counters.pkl', 'wb') as f:
				pickle.dump(counters_dict, f, pickle.HIGHEST_PROTOCOL)

			#save things after every agg iteration
			np.save(save_dir + '/losses/errors_1_per_agg.npy', errors_1_per_agg)
			np.save(save_dir + '/losses/errors_5_per_agg.npy', errors_5_per_agg)
			np.save(save_dir + '/losses/errors_10_per_agg.npy', errors_10_per_agg)
			np.save(save_dir + '/losses/errors_50_per_agg.npy', errors_50_per_agg)
			np.save(save_dir + '/losses/errors_100_per_agg.npy', errors_100_per_agg)
			
			np.save(save_dir + '/losses/list_training_loss.npy', training_loss_list) 
			np.save(save_dir + '/losses/dynamics_training_old_loss.npy', old_loss_list)
			np.save(save_dir + '/losses/dynamics_training_new_loss.npy', new_loss_list)

			if False:
				np.save(save_dir + '/summary/Best_dynamics_training_loss_for'+ str(num_aggregation_iters) + 'aggiters.npy''.npy', np.array(training_loss_list)) 
				np.save(save_dir + '/summary/Avg_rollout_rewards_for_' + str(num_aggregation_iters) + 'aggiters.npy', np.array(list_avg_rew))
				np.save(save_dir + '/summary/All_dynamics_training_loss_for' + str(num_aggregation_iters) + 'aggiters.npy', np.array(avg_loss_per_epoch_list)) 


		##############################
		### perform a bunch of MPC rollouts to save for later mbmf TRPO usage
		##############################

		# all_rollouts_to_save = []
		# if(args.num_controlled_rollouts>0):
		# 	print("##############################################")
		# 	print("#### Performing MPC rollouts")
		# 	print("##############################################\n")

		# 	#init vars
		# 	list_rewards=[]
		# 	starting_states=[]
		# 	num_saved = 0
		# 	rollout_num = 0
		# 	while(num_saved < args.num_controlled_rollouts):
		# 		if(not(print_minimal)):
		# 			print("\nSo far, saved ", num_saved, " rollouts")
		# 			print("Currently, on rollout #", rollout_num)

		# 		#reset env before performing rollout

		# 		starting_observation = env.reset()
		# 		if (fixed_first_state):
		# 			env.sim.set_state(starting_fullenvstate)
				
		# 		starting_state = np.copy(starting_observation)
		# 		starting_observation_NNinput = from_observation_to_usablestate(starting_observation, which_agent, True)

		# 		#perform 1 MPC rollout
		# 		startrollout = time.time()
		# 		curr_noise_amount=0
		# 		if (rollout_num % 10 == 0):
		# 			mpc_info = {"save_dir": save_dir, "scope": "rollout_for_mbmf", "rollout_num": rollout_num, "record": record}
		# 		else:
		# 			mpc_info = {"save_dir": save_dir, "scope": "rollout_for_mbmf", "rollout_num": rollout_num, "record": False}
		# 		_, _, ep_rew, rollout_saved = mpc_controller.perform_rollout(starting_fullenvstate, starting_observation, 
		# 															starting_observation_NNinput, desired_x, 
		# 															follow_trajectories, horiz_penalty_factor, 
		# 															forward_encouragement_factor, heading_penalty_factor, 
		# 															noise_actions_during_MPC_rollouts, curr_noise_amount, mpc_info)

		# 		if(not(print_minimal)):
		# 			print("Time taken for a single rollout: {:0.2f} s\n\n".format(time.time()-startrollout))

		# 		#save rollouts
		# 		rollout_num += 1
		# 		if(ep_rew>min_rew_for_saving):
		# 			list_rewards.append(ep_rew)
		# 			all_rollouts_to_save.append(rollout_saved)
		# 			starting_states.append(starting_state)
		# 			num_saved += 1


			

		# #bookkeeping
		# if(len(list_rewards)>0):

		# 	#get avg rew
		# 	avg_rew = np.mean(np.array(list_rewards))
		# 	print("############# Avg over all selected runs: ", avg_rew)
		# 	print("############# Rewards of all selected runs: ", list_rewards)

		# 	#save the rollouts for later MBMF usage
		# 	pathname_savedMPCrollouts = save_dir + '/savedRollouts_avg'+ str(int(avg_rew)) +'.save'
		# 	pathname2_savedMPCrollouts = save_dir + '/savedRollouts.save'
		# 	f = open(pathname_savedMPCrollouts, 'wb')
		# 	cPickle.dump(all_rollouts_to_save, f, protocol=cPickle.HIGHEST_PROTOCOL)
		# 	f.close()
		# 	f = open(pathname2_savedMPCrollouts, 'wb')
		# 	cPickle.dump(all_rollouts_to_save, f, protocol=cPickle.HIGHEST_PROTOCOL)
		# 	f.close()

		# 	#save the starting states of these rollouts, in case want to visualize them later
		# 	f = open(save_dir + '/savedRollouts_startingStates.save', 'wb')
		# 	cPickle.dump(starting_states, f, protocol=cPickle.HIGHEST_PROTOCOL)
		# 	f.close()
   
		# 	print("Saved MPC rollouts.")

		# np.save(save_dir + '/datapoints_MB.npy', list_num_datapoints)
		# np.save(save_dir + '/performance_MB.npy', list_avg_rew)

		# np.save(save_dir + '/summary/MPC_scores_for_'+ str(len(list_rewards)) +'rollouts_MBMF_used.npy', np.array(list_rewards))

		print("ALL DONE.")

		return

if __name__ == '__main__':
	main()
