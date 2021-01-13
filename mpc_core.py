import numpy as np
import numpy.random as npr
from helper_funcs import get_reward, log_to_file, get_truncated_normal
import cma
from cma import restricted_gaussian_sampler as rgs
import nevergrad as ng
from tqdm import tqdm
import ray
import psutil
from scipy.stats import norm, truncnorm
from dynamics_model import ForwardActor
import time

class Drunk():
	def __init__(self, env):
		self.env = env

	def get_action(self, curr_nn_state, curr_line_segment, reward_func):

		best_action = self.env.action_space.sample()

		return best_action, 0, 0, 0


class Randomshooting():
	def __init__(self, env, N, horizon, which_agent, dyn_model, dt):
		self.env = env
		self.N = N
		self.horizon = horizon
		self.which_agent = which_agent
		self.dyn_model = dyn_model
		self.dt = dt


	def get_action(self, curr_nn_state, curr_line_segment=None, reward_func=get_reward, **kwang):
	#randomly sample N candidate action sequences
		all_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0]))

		#forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
		many_in_parallel = True                           #  N * Sdim
		#log_to_file(str([curr_nn_state,0]), "input_to_predict_in_mpc")
		resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(all_samples), many_in_parallel, self.env, self.which_agent, num_particles=10)
		resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]
		a_random_state = resulting_states[3, 500, :]
		#log_to_file(str(a_random_state))

		#init vars to evaluate the trajectories
		scores=np.zeros((self.N,))
		done_forever=np.zeros((self.N,))
		move_to_next=np.zeros((self.N,))
		#curr_seg = np.tile(curr_line_segment,(self.N,))
		#curr_seg = curr_seg.astype(int)
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
			#min_perp_dist, curr_forward, curr_seg, moved_to_next = self.calculate_geometric_trajfollow_quantities(pt, curr_seg, moved_to_next)


			#update reward score
			#scores, done_forever = reward_func(pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, curr_seg, 
			#                                    moved_to_next, done_forever, all_samples, pt_number)
			scores = get_reward(pt, prev_pt, actionns, scores, self.dt)
			#print(scores)

			#update vars
			#prev_forward = np.copy(curr_forward)
			prev_pt = np.copy(pt)

		#pick best action sequence
		#print(scores)
		best_score = np.max(scores)
		best_sim_number = np.argmax(scores) 
		best_sequence = all_samples[best_sim_number]
		best_action = np.copy(best_sequence[0])

		

		return best_action, best_sim_number, best_sequence, moved_to_next


class CrossEntropyMethod():
	def __init__(self, env, N, horizon, which_agent, dyn_model, dt, k=512, update_iter=10):

		self.env = env
		self.N = N
		self.k = k
		self.update_iter = update_iter
		self.horizon = horizon
		self.which_agent = which_agent
		self.dyn_model = dyn_model
		self.dt = dt
		self.init_coef = 0.8
		self.a_dim = env.action_space.shape[0]
		self.mu = np.zeros([horizon, self.a_dim])
		self.sigma = np.ones([horizon, self.a_dim]) * self.init_coef
		self.min = env.action_space.low
		self.max = env.action_space.high
		self.epsilon = 0.5
		
	def get_action(self, curr_nn_state, curr_line_segment=None, reward_func=get_reward, run_num=None, save_path=None):

		
		#print(np.shape(all_samples))
		self.m = int(self.N*(1-self.epsilon)) # the number of normal random samples

		pbar = tqdm(total=self.update_iter, ascii=True)
		for cem_iter in range(self.update_iter):


			all_samples = np.zeros([self.horizon, self.a_dim, self.N])

			for step in range(self.horizon):
				#print(f"----for {step} in {range(self.horizon)}")
				for dim in range(self.a_dim):
					#print(f"--------------for {dim} in {range(self.a_dim)}")
					#print(np.shape(all_samples[step, dim, :]))
					

					all_samples[step, dim, 0:self.m] = np.random.normal(loc=self.mu[step, dim], scale=self.sigma[step, dim], size=self.m)
					all_samples[step, dim, 0:self.m] = np.clip(all_samples[step, dim, 0:self.m], self.min[dim], self.max[dim])

					all_samples[step, dim, self.m:self.N] = npr.uniform(self.min[dim], self.max[dim], size=self.N-self.m)

			#print("RANDOM SAMPLES: ", np.mean(all_samples[:, :, self.m:self.N]))
			#print("GASSIAN SAMPLES: ", np.mean(all_samples[:, :, 0:self.m]))


			#print("I AM NOW OUT OF THIS LOOP...")
			#print(f"Iteration {cem_iter} of the CEM:  ")
			#print(f"Actions: {np.mean(all_samples)},  mu: {np.mean(self.mu)},  sigma: {np.mean(self.sigma)}")

			num_MCscore = 10


			all_samples = np.swapaxes(all_samples,1,2)
			all_samples = np.swapaxes(all_samples,0,1)  #[N, horizon, dims]

			resulting_states_list = []
			for _ in range(num_MCscore):
				resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(all_samples), True, self.env, self.which_agent)
				resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]
				resulting_states_list.append(resulting_states)
			resulting_states = np.mean(resulting_states_list, axis=0)

			#init vars to evaluate the trajectories
			scores=np.zeros((self.N,))
			prev_pt = resulting_states[0]

			all_actions = np.swapaxes(all_samples,0,1)  #[horizon, N, dims]

			for pt_number in range(resulting_states.shape[0]-1):

				pt = resulting_states[pt_number] # N x state

				actionns = all_actions[pt_number]
				scores = get_reward(pt, prev_pt, actionns, scores, self.dt)
				prev_pt = np.copy(pt)


			max_index = np.argsort(scores)[::-1][:self.k]
			selected_actions = all_samples[max_index,:,:]  # [k, horizon, dims]

			self.mu = selected_actions.mean(axis=0)
			#self.sigma = selected_actions.std(axis=0)
			#self.sigma = all_samples.std(axis=0)
			self.sigma = selected_actions.std(axis=0)
			
			if(np.mean(self.sigma) < 0.5):
				self.sigma = np.ones([self.horizon, self.a_dim]) * self.epsilon

			pbar.update(1)

		pbar.close()

		best_score = np.max(scores)
		best_sim_number = np.argmax(scores) 
		best_sequence = all_samples[best_sim_number]


		if(np.random.random() > self.epsilon):
			best_action = np.copy(best_sequence[0])
		else:
			best_action = np.random.uniform(self.min[0], self.max[0], size=self.a_dim)
			#print("RANDOM!")

		if (self.epsilon > 0.01):
			self.epsilon *= 0.9999
		else:
			self.epsilon = 0.01

		#print(f"ACTION:  {np.mean(best_action)}    SIGMA:  {np.mean(self.sigma)}    EPSILON:  {self.epsilon}    PREDICT:  {best_score}")
		step_info = f"ACTION:  {np.mean(best_action)}    SIGMA:  {np.mean(self.sigma)}    EPSILON:  {self.epsilon}    PREDICT:  {best_score}"
		log_to_file(step_info, "step_info", save_path=self.save_path)

		return best_action, best_sim_number, best_sequence, 0




class CMAES():
	def __init__(self, env, horizon, which_agent, dyn_model, dt, reference_policy, args, cma_seed=233, alpha=0.01, dyn_actor=None):

		self.env = env
		self.horizon = horizon
		self.which_agent = which_agent
		self.dyn_model = dyn_model
		self.dt = dt
		self.init_coef = 0.4
		self.a_dim = env.action_space.shape[0]
		self.s_dim = env.observation_space.shape[0]
		self.min = env.action_space.low
		self.max = env.action_space.high
		self.N = (4+int(3*np.log(horizon*self.a_dim)))*args["N_factor"] #100

		self.alpha = alpha
		#self.maxiter = 100 + 150 * ((horizon*self.a_dim)+3)**2 // self.N**0.5
		self.maxiter = args["maxiter"] #200
		
		self.num_MCscore = args["num_MCscore"] #10
		self.forward_particles = args["forward_particles"] #5 --> 1
		self.initialiser_forward_particles = 5
		self.distributed = args["distributed"] #False
		if(self.distributed):
			# ray.shutdown()
			# ray.init(memory=2000 * 1024 * 1024, num_cpus=psutil.cpu_count(logical=False))
			# ray.init(num_cpus=min(psutil.cpu_count(logical=False), self.num_MCscore))
			self.dyn_actor = dyn_actor


		self.reference_policy = reference_policy
		self.warmup_alg = self.reference_policy.policy_name

		self.reward_model = None
		self.cma_seed = cma_seed

		self.logp_val = 0
		self.first_action = True

		if(which_agent==130):
			self.maxiter = 2
		if (which_agent==101):
			import tensorflow as tf
			self.reward_model = tf.keras.models.load_model('Eidos_models/reward_s10a10.h5', compile=False)
		elif (which_agent==102):
			import tensorflow as tf
			self.reward_model = tf.keras.models.load_model('Eidos_models/reward_s100a10.h5', compile=False)
		elif (which_agent==103):
			import tensorflow as tf
			self.reward_model = tf.keras.models.load_model('Eidos_models/reward_s1000a10.h5', compile=False)
		
	def get_reference_actions(self, obs):

		actions = []

		for _ in range(self.horizon):
			
			if (self.warmup_alg == 'PPO') :
				#action,_ = self.reference_policy.get_action(obs, test=True) ## TEMP: SAC ONLY RETURN ACTION
				action, logp, val = self.reference_policy.get_action_and_val(obs, test=True)

				self.logp_val = [logp, val]
				action = np.clip(action, self.min, self.max)
			elif (self.warmup_alg == 'SAC' or self.warmup_alg == 'TD3'): 
				action = self.reference_policy.get_action(obs, test=True)
				action = np.clip(action, self.min, self.max)
			elif (self.warmup_alg == 'random'):
				action = [0]*self.a_dim
			if(False):
				print('======DEBUG:action from PPO',np.shape(actions))
				print(action)
			actions.append(action)
			resulting_state = self.dyn_model.do_forward_sim([obs], [action], False, self.env, self.which_agent, num_particles=self.initialiser_forward_particles)
			obs = resulting_state[0]

		

		return actions


	def get_action(self, curr_nn_state, curr_line_segment=None, reward_func=get_reward, save_path = "."):

		if (self.warmup_alg == 'random'):
			initial_mean = [0]*(self.a_dim*self.horizon)
		else:
			reference_actions = self.get_reference_actions(curr_nn_state)
			initial_mean = np.array(reference_actions).flatten()
			
		# print('!!!!!!!!!',np.shape(reference_actions))
		# print('!!!!!!!!!',np.mean(reference_actions))

		self.es = cma.CMAEvolutionStrategy(initial_mean, self.init_coef, 
											rgs.GaussVDSampler.extend_cma_options({'bounds': [self.min[0], self.max[0]],
																				 'maxiter': self.maxiter,
																				 'popsize': self.N,
																				 'verbose': -1,
																				 'verb_filenameprefix': save_path,
																				 'seed': self.cma_seed}))
		popsize = self.es.popsize

		dis_len = self.maxiter
		pbar = tqdm(total=dis_len, ascii=True)
		counter = 0
		while not self.es.stop():
			all_samples = self.es.ask()
			action_samples = np.array(all_samples).reshape([self.N, self.horizon, self.a_dim])

			
			resulting_states_list = []
			if not (self.distributed):
				for _ in range(self.num_MCscore):
					resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(action_samples), True, self.env, self.which_agent, num_particles=self.forward_particles)
					resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]
					resulting_states_list.append(resulting_states)
					#print("resulting_states: ", np.mean(resulting_states))
			else:
				states_stored = ray.put([curr_nn_state,0])
				actions_stored = ray.put(action_samples)
				curr_weights = self.dyn_model.bnn_network.get_weights()


				predictors = [self.dyn_model.remote_forward_sim.remote(
					curr_weights, states_stored, actions_stored, 
					True, self.env, self.which_agent, self.forward_particles,
					self.dyn_model.std_z, self.dyn_model.mean_z, self.dyn_model.std_y, self.dyn_model.mean_y, self.dyn_model.std_x, self.dyn_model.mean_x
					) for _ in range(self.num_MCscore)]

				resulting_states_list = ray.get(predictors)
				# print(np.shape(resulting_states_list))
				# for resulting_states in resulting_states_list:
				# 	print(np.shape(resulting_states))
			# resulting_states = np.mean(resulting_states_list, axis=0)
			scores_sampled = [self.score(action_samples, np.array(resulting_states), discount=0.95) for resulting_states in resulting_states_list]


			self.es.tell(all_samples, self.cost(scores_sampled, reduce_='CVaR'))

			#self.es.disp()
			#idx = self.es.countiter
			self.es.logger.add()

			if(counter >= dis_len):
				pbar.total = counter +1
			pbar.update(1)
			counter += 1
			#log_to_file(step_info, "step_info", save_path=self.save_path)
		pbar.close()

		best_sequence = np.array(self.es.result.xbest).reshape([self.horizon, self.a_dim])
		best_action = best_sequence[0]

		return best_action, 0, best_sequence, 0

	def cost(self, scores_sampled, reduce_='ave'):
		if (reduce_ == 'ave'):
			cost = np.negative(np.mean(scores_sampled, axis=0))
		elif (reduce_ == 'CVaR'):
			ALPHA = self.alpha
			sig = np.std(scores_sampled, axis=0)
			mu = np.mean(scores_sampled, axis=0)
			cost = ALPHA**-1 * norm.pdf(norm.ppf(ALPHA))*sig - mu

		return cost

	def score(self, action_samples, resulting_states, discount=0.9):

		all_actions = np.swapaxes(action_samples,0,1)

		scores=np.zeros((self.N,))
		prev_pt = resulting_states[0]
		try:
			etc = self.env.ac_goal_pos
		except AttributeError:
			etc = None

		for pt_number in range(resulting_states.shape[0]-1):

			pt = resulting_states[pt_number] # N x state

			actionns = all_actions[pt_number]

			scores = get_reward(pt, prev_pt, actionns, scores, self.dt, discount_factor=discount**pt_number, 
								which_agent=self.which_agent, model=self.reward_model, etc=etc)

			prev_pt = np.copy(pt)

		return scores


class MultiGaussian(CMAES):
	def __init__(self, env, horizon, which_agent, dyn_model, dt, MF_agent, reference_policy, args, cma_seed=233, alpha=0.01, boosting=False, dyn_actor=None):
		super(MultiGaussian, self).__init__(env, horizon, which_agent, dyn_model, dt, reference_policy, args, cma_seed, alpha, dyn_actor)

		self.N = 3000
		# self.distributed = False
		self.Hammersley = False
		#self.forward_particles = 1

		self.print_timer = False

		self.boosting = boosting
		self.MF_agent = MF_agent

	def get_action(self, curr_nn_state, curr_line_segment=None, reward_func=get_reward, save_path = "."):
		timer1 = time.time()

		if (self.warmup_alg == 'random'):
			initial_mean = [0]*(self.a_dim*self.horizon)
		else:
			reference_actions = self.get_reference_actions(curr_nn_state)
			initial_mean = np.array(reference_actions).flatten()
			
		timer0 = time.time()
		# if self.Hammersley:
		# 	instrum = ng.p.Instrumentation(ng.p.Array(shape=(self.a_dim*self.horizon,)).set_bounds(self.min[0], self.max[0])) 
		# 	optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint(parametrization=instrum,budget=self.N)
		# 	optimizer.suggest(initial_mean)
		# 	if self.print_timer:
		# 		print('\nTIME COSTED BY CREATING OPTIMISERS: ', time.time()-timer0 ,' s')
		# 	timer0 = time.time()
		# 	all_samples = [optimizer.ask().value[0][0] for _ in range(self.N)]
		# 	action_samples = np.array(all_samples).reshape([self.N, self.horizon, self.a_dim])

		all_samples = get_truncated_normal(mean=np.array(initial_mean), sd=0.4*np.ones_like(initial_mean),low=self.min[0], up=self.max[0], size=(self.N, self.a_dim*self.horizon))
		action_samples = np.array(all_samples).reshape([self.N, self.horizon, self.a_dim])
		if self.print_timer:
			print('\nTIME COSTED BY SAMPLING CANDIDATES: ', time.time()-timer0 ,' s')
		timer0 = time.time()
		if not (self.distributed):
			resulting_states_list = []
			for _ in range(self.num_MCscore):
				resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(action_samples), True, self.env, self.which_agent, num_particles=self.forward_particles)
				resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]
				resulting_states_list.append(resulting_states)
		else:
			states_stored = ray.put([curr_nn_state,0])
			actions_stored = ray.put(action_samples)

			predictors = [self.dyn_actor.remote_forward_sim.remote(
				None, states_stored, actions_stored, 
				True, None, None, self.forward_particles,
				None, None, None, None, None, None
				) for _ in range(self.num_MCscore)]
			# predictors = [self.dyn_model.remote_forward_sim.remote(
			# 	curr_weights, states_stored, actions_stored, 
			# 	True, self.env, self.which_agent, self.forward_particles,
			# 	self.dyn_model.std_z, self.dyn_model.mean_z, self.dyn_model.std_y, self.dyn_model.mean_y, self.dyn_model.std_x, self.dyn_model.mean_x
			# 	) for _ in range(self.num_MCscore)]

			resulting_states_list = ray.get(predictors)
		if self.print_timer:
			print('\nTIME COSTED BY PREDICTING TRAJECTORIES: ', time.time()-timer0 ,' s')

		timer0 = time.time()
		scores_sampled = [self.score(action_samples, np.array(resulting_states), discount=0.95) for resulting_states in resulting_states_list]
		
		if self.print_timer:
			print('\nTIME COSTED BY CALCUATING REWARDS: ', time.time()-timer0 ,' s')

		if self.boosting:
			boost_obs =  np.tile(curr_nn_state, (self.N,1))
			boost_actions = action_samples[:,0,:]
			boost_next_obs = np.mean(resulting_states_list,axis=0)[1]
			boost_rews = get_reward(boost_next_obs, boost_obs, boost_actions, [0]*self.N, self.dt, discount_factor=1, 
								which_agent=self.which_agent, model=self.reward_model)

			for action_to_take, next_state, rew in zip(boost_actions, boost_next_obs, boost_rews):
				self.MF_agent.replay_buffer.add(obs=curr_nn_state, act=action_to_take, next_obs=next_state, rew=rew, done=False)
				self.MF_agent.train()




		timer0 = time.time()
		cost = self.cost(scores_sampled, reduce_='CVaR')
		if self.print_timer:
			print('\nTIME COSTED BY CALCUATING COSTS: ', time.time()-timer0 ,' s')

		best_sequence = action_samples[np.argmin(cost)]
		best_action = best_sequence[0]
		if self.print_timer:
			print('\n==> TIME COSTED BY SEARCHING ACTIONS: ', time.time()-timer1 ,' s')

		self.first_action = False

		return best_action, 0, best_sequence, self.logp_val


class SimplifiedCMAES(CMAES):
	def __init__(self, env, horizon, which_agent, dyn_model, dt, MF_agent, reference_policy, args, cma_seed=233, alpha=0.01, boosting = True, dyn_actor=None):
		super(SimplifiedCMAES, self).__init__(env, horizon, which_agent, dyn_model, dt, reference_policy, args, cma_seed, alpha, dyn_actor)

		self.N = args["N_factor"]#500#3000
		self.max_iters = self.maxiter#3#5

		#self.distributed = False
		self.Hammersley = False
		#self.forward_particles = 1

		self.print_timer = False

		self.boosting = boosting
		self.MF_agent = MF_agent

		self.dO, self.dU = self.env.observation_space.shape[0], self.env.action_space.shape[0]
		self.ac_ub, self.ac_lb = self.env.action_space.high, self.env.action_space.low

		self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.horizon])

		self.optimiser = CAMES_optimiser(
            sol_dim=self.horizon * self.dU,
            max_iters=self.max_iters, popsize=self.N, num_elites=int(self.N/10), alpha=0.1,
            lower_bound=np.tile(self.ac_lb, [self.horizon]),
            upper_bound=np.tile(self.ac_ub, [self.horizon]),
            cost_function=self._compile_cost,
            init_var=self.init_var
        )

	def get_action(self, curr_nn_state, curr_line_segment=None, reward_func=get_reward, save_path = "."):
		timer0 = time.time()
		if (self.warmup_alg == 'random'):
			initial_mean = np.array([0]*(self.a_dim*self.horizon))
		else:
			reference_actions = self.get_reference_actions(curr_nn_state)
			initial_mean = np.array(reference_actions).flatten()
		if self.print_timer:
			print('\nTIME COSTED BY GETTING REFERENCE ACTIONS: ', time.time()-timer0 ,' s') # 0.31s
		timer0 = time.time()
		best_sequence = self.optimiser.obtain_solution(state=curr_nn_state, init_mean=initial_mean)
		best_sequence = np.array(best_sequence).reshape([self.horizon, self.a_dim])

		best_action = best_sequence[0]
		if self.print_timer:
			print('\n==> TIME COSTED BY SEARCHING ACTIONS: ', time.time()-timer0 ,' s')

		return best_action, 0, best_sequence, self.logp_val


	def _compile_cost(self, state, action):

		timer1 = time.time()
		curr_nn_state = state
		action_samples = np.array(action).reshape([self.N, self.horizon, self.a_dim])

		if not (self.distributed):
			resulting_states_list = []
			for _ in range(self.num_MCscore):
				resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(action_samples), True, self.env, self.which_agent, num_particles=self.forward_particles)
				resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]
				resulting_states_list.append(resulting_states)
		else:
			states_stored = ray.put([curr_nn_state,0])
			actions_stored = ray.put(action_samples)

			predictors = [self.dyn_actor.remote_forward_sim.remote(
				None, states_stored, actions_stored, 
				True, None, None, self.forward_particles,
				None, None, None, None, None, None
				) for _ in range(self.num_MCscore)]

			resulting_states_list = ray.get(predictors)
		if self.print_timer:
			print('\nTIME COSTED BY PREDICTING TRAJECTORIES: ', time.time()-timer1 ,' s')  #1.08s * 5

		timer1 = time.time()
		scores_sampled = [self.score(action_samples, np.array(resulting_states), discount=0.95) for resulting_states in resulting_states_list]
		
		if self.print_timer:
			print('\nTIME COSTED BY CALCUATING REWARDS: ', time.time()-timer1 ,' s')  #1.22s * 5

		timer1 = time.time()
		cost = self.cost(scores_sampled, reduce_='CVaR')

		if self.print_timer:
			print('\nTIME COSTED BY CALCUATING COSTS: ', time.time()-timer1 ,' s')  #0.0005 * 5

		return cost



class CAMES_optimiser():

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25, init_var=0):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """

        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        # self.sol_dim = sol_dim
        # self.max_iters = max_iters
        # self.popsize = 2
        # self.num_elites = 1

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.init_var = init_var

        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, state, init_mean, init_var=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, self.init_var, 0
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)

            costs = self.cost_function(state, samples)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean
