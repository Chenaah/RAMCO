import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym

def add_noise_to_signal(signal, SNR):

	if not (SNR==None):
		noise=np.random.randn(*signal.shape)
		noise=noise-np.mean(noise)
		signal_power=(1/signal.shape[0])*sum(signal*signal)
		noise_variance=signal_power/np.power(10,(SNR/10))
		noise=(np.sqrt(noise_variance)/np.std(noise))*noise
		noisy_signal=signal+noise

	else:
		noisy_signal = signal

	return noisy_signal


class Eidos(gym.Env):
	def __init__(self, s_dim=100, a_dim=10, o_dim=None, hidden_units=512, seed=233, 
				 s_SNR=None, o_SNR=60, r_SNR=None, terminal_error=-1, max_steps=100, reward_scale=1,
				 load_dynamics_model='default', load_reward_model='default', 
				 save_dynamics_model='default', save_reward_model="default",
				 reward_type=1):
		super(Eidos, self).__init__()

		self.save_dynamics_model = save_dynamics_model
		self.save_reward_model = save_reward_model
		self.load_dynamics_model = load_dynamics_model
		self.load_reward_model = load_reward_model

		self.hidden_units = hidden_units

		if o_dim==None:
			o_dim = s_dim
		elif o_dim > s_dim:
			raise ValueError("Observation space should be smaller than state space.")


		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(a_dim,), dtype=np.float32)
		self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(o_dim,), dtype=np.float32)

		tf.random.set_seed(seed)
		self.s_SNR = s_SNR
		self.o_SNR = o_SNR
		self.r_SNR = r_SNR
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.o_dim = o_dim
		self.terminal_error=terminal_error
		self.max_steps=max_steps
		self.reward_scale = reward_scale

		self.observation_mask = np.random.choice(s_dim, size=o_dim, replace=False)
		self.terminal_state = np.random.uniform(low=-1, high=1, size=s_dim)

		if isinstance(self.load_dynamics_model, str):
			if(self.load_dynamics_model == 'default'):
				if os.path.exists(f"Eidos_models/dynamics_s{s_dim}a{a_dim}.h5"):
					self.load_dynamics_model = f"Eidos_models/dynamics_s{s_dim}a{a_dim}.h5"
					self.dynamics_model = tf.keras.models.load_model(self.load_dynamics_model, compile=False)
					self.save_dynamics_model = False
				else:
					self.dynamics_model = self._build_dynamics()
			else:
				self.dynamics_model = tf.keras.models.load_model(self.load_dynamics_model, compile=False)
				self.save_dynamics_model = False
		else:
			self.dynamics_model = self._build_dynamics()

		if isinstance(self.load_reward_model, str):
			if(self.load_reward_model == 'default'):
				if os.path.exists(f"Eidos_models/reward_s{s_dim}a{a_dim}.h5"):
					self.load_reward_model = f"Eidos_models/reward_s{s_dim}a{a_dim}.h5"
					self.reward_model = tf.keras.models.load_model(self.load_reward_model, compile=False)
					self.save_reward_model = False
				else:
					self.reward_model = self._build_reward()
			else:
				self.reward_model = tf.keras.models.load_model(self.load_reward_model, compile=False)
				self.save_reward_model = False
		else:
			self.reward_model = self._build_reward()

		self.state_now = np.zeros((s_dim,))
		self.observation_now = np.zeros((o_dim,))

		self.step_counter = 0
		self.distance = None

		if isinstance(self.save_dynamics_model, str):
			
			if self.save_dynamics_model=='default':
				self.save_dynamics_model = f"Eidos_models/dynamics_s{s_dim}a{a_dim}.h5"
			model_path = os.path.dirname(self.save_dynamics_model)
			if not os.path.exists(model_path):
				os.makedirs(model_path)
			self.dynamics_model.save(self.save_dynamics_model)
		if isinstance(self.save_reward_model, str):
			if self.save_reward_model=='default':
				self.save_reward_model = f"Eidos_models/reward_s{s_dim}a{a_dim}.h5"
			model_path = os.path.dirname(self.save_reward_model)
			if not os.path.exists(model_path):
				os.makedirs(model_path)
			self.reward_model.save(self.save_reward_model)

		self.reward_type = reward_type



	def step(self, action):

		self.step_counter += 1
		state_before = self.state_now
		s_next = self._next_state(self.state_now, action)
		self.state_now = s_next
		observation = self._state2observation(s_next)
		self.observation_now = observation
		if (self.reward_type==0):
			reward = self._reward(self.state_now, action)
		elif (self.reward_type==1):
			reward = self._reward(state_before, action)
		done = self._done(s_next)
		info = self.distance

		return observation, reward, done, info

	def reset(self):
		self.step_counter = 0
		self.state_now = np.clip(np.random.normal(0, 1, self.s_dim), -1, 1)
		self.observation_now = self._state2observation(self.state_now)
		self.distance = (np.square(self.state_now, self.terminal_state)).mean()

		return self.observation_now


	def render(self):
		pass

	def _done(self, s):
		done = False
		self.distance = (np.square(s, self.terminal_state)).mean()

		if(self.distance < self.terminal_error):
			done = True
		elif(self.step_counter >= self.max_steps):
			done = True

		return done


	def _reward(self, s, a):
		s_a = np.concatenate((s,a))
		s_a = np.atleast_2d(s_a)
		reward = self.reward_model.predict(s_a)
		
		reward = reward[0][0]
		r_noisy = add_noise_to_signal(reward, self.r_SNR)

		return (r_noisy)*self.reward_scale


	def _state2observation(self, s):

		o = s[self.observation_mask]
		o_noisy = add_noise_to_signal(o, self.o_SNR)
		return o_noisy

	def _next_state(self, s, a):
		s_a = np.concatenate((s,a))
		s_a = np.atleast_2d(s_a)
		s_next = self.dynamics_model.predict(s_a)
		s_next = s_next[0]
		s_noisy = add_noise_to_signal(s_next, self.s_SNR)
		return s_noisy

	def _build_dynamics(self):
		dynamics_model = tf.keras.Sequential()
		dynamics_model.add(layers.Dense(self.hidden_units, activation='tanh', input_shape=(self.s_dim+self.a_dim,)))
		dynamics_model.add(layers.Dense(self.hidden_units, activation='tanh'))
		dynamics_model.add(layers.Dense(self.s_dim))
		return dynamics_model

	def _build_reward(self):
		reward_model = tf.keras.Sequential()
		reward_model.add(layers.Dense(self.hidden_units, activation='tanh', input_shape=(self.s_dim+self.a_dim,)))
		reward_model.add(layers.Dense(1, activation='tanh'))
		return reward_model

	

def test():

	env = Eidos(s_dim=2, a_dim=2,max_steps=10)
	env.reset()
	done = False
	score = 0
	while not done:
		s,r,done,info = env.step(env.action_space.sample())
		score += r
		print(score)
