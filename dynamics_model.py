
import numpy as np
import numpy.random as npr
import tensorflow as tf
# from tensorflow.contrib.distributions import Bernoulli
import tensorflow_probability as tfp
import time
import math
# import psutil
import ray
from helper_funcs import log_to_file
from tqdm import tqdm

class VariationalDense__old:
	"""Variational Dense Layer Class"""
	def __init__(self, n_in, n_out, model_prob, model_lam, tf_datatype):
		self.model_prob = model_prob
		self.model_lam = model_lam
		self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf_datatype)
		self.model_M = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01, dtype=tf_datatype), dtype=tf_datatype)
		self.model_m = tf.Variable(tf.zeros([n_out], dtype=tf_datatype),dtype=tf_datatype)
		self.model_W = tf.matmul(
			tf.diag(self.model_bern.sample((n_in, ))), self.model_M
		)

	def __call__(self, X, activation=None):
		if (activation == None):
			activation = tf.identity
		output = activation(tf.matmul(X, self.model_W) + self.model_m)
		if self.model_M.shape[1] == 1:
			output = tf.squeeze(output)
		return output

	@property
	def regularization(self):
		return self.model_lam * (
			self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
			tf.reduce_sum(tf.square(self.model_m))
		)

class VariationalDense(tf.keras.layers.Layer):
	def __init__(self, n_out, prob, lam, tf_datatype, *args, **kwargs):
		self.autocast = False
		super(VariationalDense, self).__init__(*args, **kwargs)

		self.n_out = n_out
		self.prob = prob
		self.lam = lam
		self.tf_datatype = tf_datatype



	def build(self, input_shape):
		self.n_in = int(input_shape[-1])
		self.M = self.add_weight('M', 
								 shape=[self.n_in, self.n_out],
								 dtype=self.tf_datatype,
								 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
								 trainable=True)
		self.m = self.add_weight('m',
								 shape=[self.n_out],
								 dtype=self.tf_datatype,
								 initializer=tf.zeros_initializer(),
								 trainable=True
								 )
		self.bern = tfp.distributions.Bernoulli(probs=self.prob, dtype=self.tf_datatype)

	@tf.function
	def call(self, input, activation=tf.identity):
		input = tf.dtypes.cast(input, dtype=self.tf_datatype)
		self.W = tf.linalg.matmul(
				 tf.linalg.diag(self.bern.sample((self.n_in, ))), self.M
				 )
		output = tf.linalg.matmul(input, self.W) + self.m
		output = activation(output)

		return output

	@property
	def regularization(self):
		return self.lam * (
			self.prob * tf.reduce_sum(tf.square(self.M)) +
			tf.reduce_sum(tf.square(self.m))
		)

class BNN(tf.keras.Model):

	def __init__(self, n_hidden=500, n_out=3, prob=0.9, lam=1e-2, tf_datatype=tf.float64,name='BNN', **kwargs):
		self.autocast = False
		super(BNN, self).__init__(name=name, **kwargs)
		self.n_hidden = n_hidden
		self.prob = prob
		self.lam = lam

		self.L_1 = VariationalDense(n_hidden, prob, lam, tf_datatype, dtype='float64')
		self.A_1 = tf.keras.layers.ReLU(dtype=tf_datatype)
		self.L_2 = VariationalDense(n_hidden, prob, lam, tf_datatype, dtype='float64')
		self.A_2 = tf.keras.layers.ReLU(dtype=tf_datatype)
		self.L_3 = VariationalDense(n_out, prob, lam, tf_datatype, dtype='float64')

	def call(self, input):
		x = self.L_1(input, activation=tf.nn.relu)
		#x = self.A_1(x)
		x = self.L_2(x, activation=tf.nn.relu)
		#x = self.A_2(x)
		output = self.L_3(x)

		self.add_loss(self.L_1.regularization+self.L_2.regularization+self.L_3.regularization)

		return output


class Dyn_Model:

	def __init__(self, inputSize, outputSize, sess, learning_rate, batchsize, which_agent, x_index, y_index, 
				num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal, model_dir):

		#init vars
		self.sess = sess
		self.batchsize = batchsize
		self.which_agent = which_agent
		self.x_index = x_index
		self.y_index = y_index
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.mean_x = mean_x
		self.mean_y = mean_y
		self.mean_z = mean_z
		self.std_x = std_x
		self.std_y = std_y
		self.std_z = std_z
		self.print_minimal = print_minimal
		self.depth_fc_layers = depth_fc_layers

		self.model_prob = 0.9
		self.model_lam = 1e-2
		self._model_dir = model_dir

		#placeholders
		# self.x_ = tf.placeholder(tf_datatype, shape=[None, self.inputSize], name='x') #inputs
		# self.z_ = tf.placeholder(tf_datatype, shape=[None, self.outputSize], name='z') #labels

		#forward pass
		# self.curr_nn_output = self.feedforward_network(self.x_, self.inputSize, self.outputSize, 
		# 										num_fc_layers, depth_fc_layers, tf_datatype)

		self.bnn_network = BNN(n_hidden=self.depth_fc_layers, n_out=self.outputSize, prob=self.model_prob,
							lam=self.model_lam, tf_datatype=tf_datatype, dtype='float64')
		#loss
		# self.sse_ = tf.reduce_sum(tf.square(self.z_ - self.curr_nn_output))

		# self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))

		# self.model_loss = (
		# 	# Negative log-likelihood.
		# 	self.sse_ +
		# 	# Regularization.
		# 	self.model_L_1.regularization +
		# 	self.model_L_2.regularization +
		# 	self.model_L_3.regularization
		# 	) / self.batchsize


		# self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.model_loss)
		self.loss_metric = tf.keras.metrics.Mean()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

		self._checkpoint = tf.train.Checkpoint(dynamics=self.bnn_network)
		self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint, directory=self._model_dir, max_to_keep=5)

	def feedforward_network_old(self, inputState, inputSize, outputSize, junk, n_hidden, tf_datatype):

		model_prob = 0.9
		model_lam = 1e-2 #1e-2

		self.model_L_1 = VariationalDense(inputSize, n_hidden, model_prob, model_lam, tf_datatype)
		self.model_L_2 = VariationalDense(n_hidden, n_hidden, model_prob, model_lam, tf_datatype)
		self.model_L_3 = VariationalDense(n_hidden, outputSize, model_prob, model_lam, tf_datatype)

		model_out_1 = self.model_L_1(inputState, tf.nn.relu)
		model_out_2 = self.model_L_2(model_out_1, tf.nn.relu)
		output = self.model_L_3(model_out_2)

		return output

	def loss(self, X, Y):
		sse = tf.reduce_sum(tf.square(X - Y))
		loss = (sse + self.bnn_network.losses)/self.batchsize
		return loss

	def train(self, dataX, dataZ, dataX_new, dataZ_new, nEpoch, save_dir, fraction_use_new, num_aggregation_iters=None):

		#init vars
		start = time.time()
		training_loss_list = []
		range_of_indeces = np.arange(dataX.shape[0])
		nData_old = dataX.shape[0]
		num_new_pts = dataX_new.shape[0]

		#how much of new data to use per batch
		if(num_new_pts<(self.batchsize*fraction_use_new)):
			batchsize_new_pts = num_new_pts #use all of the new ones
		else:
			batchsize_new_pts = int(self.batchsize*fraction_use_new)

		#how much of old data to use per batch
		batchsize_old_pts = int(self.batchsize- batchsize_new_pts)

		print(f"------------------- I would be trained for {nEpoch} epoch -------------------")

		avg_loss_per_epoch = []

		#training loop
		for i in range(nEpoch):
			
			#reset to 0
			avg_loss=0
			num_batches=0

			#randomly order indeces (equivalent to shuffling dataX and dataZ)
			old_indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)
			#train from both old and new dataset
			if(batchsize_old_pts>0): 
				#print(" [A] I AM TRAINING....................")

				#get through the full old dataset
				for batch in range(int(math.floor(nData_old / batchsize_old_pts))):

					#randomly sample points from new dataset
					if(num_new_pts==0):
						dataX_new_batch = dataX_new
						dataZ_new_batch = dataZ_new
					else:
						new_indeces = npr.randint(0,dataX_new.shape[0], (batchsize_new_pts,))
						dataX_new_batch = dataX_new[new_indeces, :]
						dataZ_new_batch = dataZ_new[new_indeces, :]

					#walk through the randomly reordered "old data"
					dataX_old_batch = dataX[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
					dataZ_old_batch = dataZ[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
					
					#combine the old and new data
					dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
					dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))


					#one iteration of feedforward training
					with tf.GradientTape() as tape:
						output = self.bnn_network(dataX_batch)
						# print("  REAL DATA: ",  np.mean(dataZ_batch))
						# print("  -------->  ", np.mean(output))
						sse = tf.reduce_sum(tf.square(dataZ_batch - output))
						mse = tf.reduce_mean(tf.square(dataZ_batch - output))
						#loss = (sse + self.bnn_network.losses)/self.batchsize

						loss = sse/self.batchsize
					grads = tape.gradient(loss, self.bnn_network.trainable_weights)
					self.optimizer.apply_gradients(zip(grads, self.bnn_network.trainable_weights))

					self.loss_metric(mse)

					# _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
					# 											feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
					training_loss_list.append(loss)
					avg_loss+= mse.numpy()
					num_batches+=1

			#train completely from new set
			else: 
				print(" [B] I AM TRAINING....................")
				for batch in range(int(math.floor(num_new_pts / batchsize_new_pts))):

					#walk through the shuffled new data
					dataX_batch = dataX_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]
					dataZ_batch = dataZ_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]

					#one iteration of feedforward training
					with tf.GradientTape() as tape:
						output = self.bnn_network(dataX_batch)
						sse = tf.reduce_sum(tf.square(dataZ_batch - output))
						mse = sse/self.batchsize
						loss = (sse + self.bnn_network.losses)/self.batchsize
					grads = tape.gradient(loss, self.bnn_network.trainable_weights)
					self.optimizer.apply_gradients(zip(grads, self.bnn.trainable_weights))

					self.loss_metric(mse)

					# _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
					# 											feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

					training_loss_list.append(loss)
					avg_loss+= mse.numpy()
					num_batches+=1

				#shuffle new dataset after an epoch (if training only on it)
				p = npr.permutation(dataX_new.shape[0])
				dataX_new = dataX_new[p]
				dataZ_new = dataZ_new[p]

			#save losses after an epoch
			if False:
				np.save(save_dir + '/training_losses.npy', training_loss_list)

			avg_loss_curr = avg_loss/num_batches
			if(not(self.print_minimal)):
				if(((i+1)%10)==0):
					print("\n=== Epoch {} ===".format(i+1))
					print ("loss: ", avg_loss_curr)
			tf.summary.scalar(name=f"Dynamics/Aggiter{num_aggregation_iters}/Loss", data=avg_loss_curr, step=i+1)

			avg_loss_per_epoch.append(avg_loss_curr)
		
		if(not(self.print_minimal)):
			print ("Training set size: ", (nData_old + dataX_new.shape[0]))
			print("Training duration: {:0.2f} s".format(time.time()-start))

		#get loss of curr model on old dataset
		avg_old_loss=0
		iters_in_batch=0
		for batch in range(int(math.floor(nData_old / self.batchsize))):
			# Batch the training data
			dataX_batch = dataX[batch*self.batchsize:(batch+1)*self.batchsize, :]
			dataZ_batch = dataZ[batch*self.batchsize:(batch+1)*self.batchsize, :]
			#one iteration of feedforward training
			# loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
			output = self.bnn_network(dataX_batch)

			loss = self.loss(dataZ_batch, output)
			avg_old_loss+= loss
			iters_in_batch+=1
		old_loss =  avg_old_loss/iters_in_batch

		#get loss of curr model on new dataset
		avg_new_loss=0
		iters_in_batch=0
		for batch in range(int(math.floor(dataX_new.shape[0] / self.batchsize))):
			# Batch the training data
			dataX_batch = dataX_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
			dataZ_batch = dataZ_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
			#one iteration of feedforward training
			# loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
			output = self.bnn_network(dataX_batch)

			loss = self.loss(dataZ_batch, output)
			avg_new_loss+= loss
			iters_in_batch+=1
		if(iters_in_batch==0):
			new_loss=0
		else:
			new_loss =  avg_new_loss/iters_in_batch

		self.checkpoint_manager.save()

		#done
		return (avg_loss/num_batches), old_loss, new_loss, avg_loss_per_epoch

	def run_validation(self, inputs, outputs):

		#init vars
		nData = inputs.shape[0]
		avg_loss=0
		iters_in_batch=0

		for batch in range(int(math.floor(nData / self.batchsize))):
			# Batch the training data
			dataX_batch = inputs[batch*self.batchsize:(batch+1)*self.batchsize, :]
			dataZ_batch = outputs[batch*self.batchsize:(batch+1)*self.batchsize, :]

			#one iteration of feedforward training
			# z_predictions, loss = self.sess.run([self.curr_nn_output, self.mse_], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
			z_predictions = self.bnn_network(dataX_batch)
			loss = self.loss(dataZ_batch, z_predictions)

			avg_loss+= loss
			iters_in_batch+=1

		#avg loss + all predictions
		print ("Validation set size: ", nData)
		print ("Validation set's total loss: ", avg_loss/iters_in_batch)

		return (avg_loss/iters_in_batch)

	#multistep prediction using the learned dynamics model at each step
	def do_forward_sim(self, forwardsim_x_true, forwardsim_y, many_in_parallel, env_inp, which_agent, num_particles=20):

		#init vars
		state_list = []
		std_list = []

		if(many_in_parallel):
			#init vars
			N= forwardsim_y.shape[0]
			horizon = forwardsim_y.shape[1]
			array_stdz = np.tile(np.expand_dims(self.std_z, axis=0),(N,1))
			array_meanz = np.tile(np.expand_dims(self.mean_z, axis=0),(N,1))
			array_stdy = np.tile(np.expand_dims(self.std_y, axis=0),(N,1))
			array_meany = np.tile(np.expand_dims(self.mean_y, axis=0),(N,1))
			array_stdx = np.tile(np.expand_dims(self.std_x, axis=0),(N,1))
			array_meanx = np.tile(np.expand_dims(self.mean_x, axis=0),(N,1))

			self.array_stdz = array_stdz
			self.array_meanz = array_meanz
			# print('[TEMP] =========> INPUT SATES [2] : ', np.mean(forwardsim_x_true))

			#log_to_file(f"array_stdx \n {array_stdx} \narray_stdy\n {array_stdy}")

			if(len(forwardsim_x_true)==2):
				#N starting states, one for each of the simultaneous sims
				curr_states=np.tile(forwardsim_x_true[0], (N,1))
			else:
				curr_states=np.copy(forwardsim_x_true)
			# print('[TEMP] =========> INPUT SATES [3] : ', np.mean(curr_states))

			#advance all N sims, one timestep at a time
			for timestep in range(horizon):

				#keep track of states for all N sims
				state_list.append(np.copy(curr_states))
				# print('[TEMP] =========> INPUT SATES [4] : ', np.mean(curr_states))

				#log_to_file("ORIGINAL: " + str(np.shape(curr_states)))

				#make [N x (state,action)] array to pass into NN
				states_preprocessed = np.nan_to_num(np.divide((curr_states-array_meanx), array_stdx))
				#log_to_file(str(np.shape(states_preprocessed)))
				actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
				#print('[TEMP] state: ',  np.mean(curr_states),'  state_pre: ', np.mean(states_preprocessed),'  action_pre: ', np.mean(actions_preprocessed))
				inputs_list= np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

				particles = []

				parallel_particles = False

				if not parallel_particles:

					for i in range(num_particles):
						#run the N sims all at once
						# model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list}) 
						model_output = self.bnn_network(inputs_list)
						model_output = model_output.numpy()
						# print('[TEMP] =========> OUTPUT SATES [1] : ', np.mean(model_output))
						# print('[TEMP] OUTPUT: ', np.mean(inputs_list), '  ====>  ' , np.mean(model_output) )
						state_differences = np.multiply(model_output,array_stdz)+array_meanz
						# print('[TEMP] =========> OUTPUT SATES [2] : ', np.mean(state_differences))
						# print('[TEMP] ... model_output[-0-] : ', np.mean(model_output), ',  ',np.shape(model_output))
						# print('[TEMP] ... array_stdz : ', np.mean(array_stdz), ',  ',np.shape(array_stdz))
						# print('[TEMP] ... array_meanz : ', np.mean(array_meanz), ',  ',np.shape(array_meanz))
						particles.append(state_differences)
						#print("OUTPUT:  ", np.mean(model_output))
				else:

					num_cpus = psutil.cpu_count(logical=False)
					print("Let me see how many CPUs you have......", num_cpus)
					ray.init(num_cpus=num_cpus)

					inputs_list_id = ray.put(inputs_list)
					sess_id = ray.put(self.sess)
					curr_nn_output_id = ray.put(self.curr_nn_output)
					x__id = ray.put(self.x_)
					array_stdz_id = ray.put(array_stdz)
					array_meanz_id = ray.put(array_meanz)


					for _ in range(num_particles):
						
						igotit = ray.get(get_output_parallel.remote(inputs_list_id, sess_id, curr_nn_output_id,
								 x__id, array_stdz_id, array_meanz_id))
						particles.append(igotit)

				# print('[TEMP] =========> OUTPUT SATES [3] : ', np.mean(particles))


				particles_mean = np.mean(particles, axis=0)
				particles_std = np.std(particles, axis=0)
				#print("[TEMP] ", np.shape(particles_mean), np.shape(particles_std))

				sampled_state_differences = np.random.normal(loc=particles_mean, scale=particles_std)
				# print('[TEMP] =========> OUTPUT SATES [4] : ', np.mean(sampled_state_differences))
				#update the state info
				curr_states = curr_states + sampled_state_differences
				# print('[TEMP] =========> OUTPUT SATES [5] : ', np.mean(curr_states))
				std_list.append(particles_std)

			#return a list of length = horizon+1... each one has N entries, where each entry is (13,)
			state_list.append(np.copy(curr_states))
		else:
			curr_state = np.copy(forwardsim_x_true[0]) #curr state is of dim NN input

			for curr_control in forwardsim_y:
				if(False):
					print('====================DEGUG:curr_controlA', np.shape(curr_control),'====================')
					print(curr_control)

				state_list.append(np.copy(curr_state))
				curr_control = np.expand_dims(curr_control, axis=0)

				if(False):
					print('====================DEGUG:curr_controlB', np.shape(curr_control),'====================')
					print(curr_control)

				#subtract mean and divide by standard deviation
				curr_state_preprocessed = curr_state - self.mean_x
				curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/self.std_x)
				curr_control_preprocessed = curr_control - self.mean_y
				curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/self.std_y)
				inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

				if(False):
					print('====================DEGUG:inputs_preprocessed', np.shape(inputs_preprocessed),'====================')
					print(inputs_preprocessed)
					print('====================================================================================')

				particles = []

				for i in range(num_particles):

					#run through NN to get prediction
					# model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed}) 
					model_output = self.bnn_network(inputs_preprocessed)
					model_output = model_output.numpy()
					#multiply by std and add mean back in
					state_differences= (model_output[0]*self.std_z)+self.mean_z

					particles.append(state_differences)

				particles_mean = np.mean(particles, axis=0)
				particles_std = np.std(particles, axis=0)

				sampled_state_differences = np.random.normal(loc=particles_mean, scale=particles_std)

				#update the state info
				next_state = curr_state + sampled_state_differences

				#copy the state info
				curr_state= np.copy(next_state)
				std_list.append(particles_std)

			state_list.append(np.copy(curr_state))
			  
		return state_list






	def predict_for_visualisation(self, forwardsim_x_true, forwardsim_y, junk, env_inp, which_agent, num_particles=5, repeat=200):

		state_list_list = []

		pbar = tqdm(total=repeat, ascii=True)
		for particle in range(repeat):
			state_list = self.do_forward_sim(forwardsim_x_true, forwardsim_y, False, env_inp, which_agent, num_particles)
			state_list_list.append(state_list)
			pbar.update(1)
		pbar.close()
		  
		return state_list_list

	def load_checkpoints(self):
		self._latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
		self._checkpoint.restore(self._latest_path_ckpt)
		_ = self.bnn_network(np.zeros( (1, self.inputSize))) # initialisation
		print("Restored {}".format(self._latest_path_ckpt))

	# @ray.remote
	# def remote_forward_sim(weights, forwardsim_x_true, forwardsim_y, many_in_parallel, env_inp, which_agent=1, num_particles=20,
	# 					   std_z=0, mean_z=0, std_y=0, mean_y=0, std_x=0, mean_x=0):

	# 	# Build another network
		
	# 	#quit()
	# 	n_out_TEMP = np.shape(weights[-1])[0]
	# 	n_hidden =  np.shape(weights[-2])[0]
	# 	prob_TEMP = 0.9
	# 	model_lam_TEMP = 1e-2
	# 	new_bnn_network = BNN(n_hidden=n_hidden, n_out=n_out_TEMP, prob=prob_TEMP,
	# 						lam=model_lam_TEMP, dtype='float64')
	# 	_ = new_bnn_network(np.zeros( (1, np.shape(weights[0])[0]) )) # initialisation
	# 	new_bnn_network.set_weights(weights)
	# 	#init vars
	# 	state_list = []
	# 	std_list = []

	# 	if(many_in_parallel):
	# 		#init vars
	# 		N= forwardsim_y.shape[0]
	# 		horizon = forwardsim_y.shape[1]
	# 		array_stdz = np.tile(np.expand_dims(std_z, axis=0),(N,1))
	# 		array_meanz = np.tile(np.expand_dims(mean_z, axis=0),(N,1))
	# 		array_stdy = np.tile(np.expand_dims(std_y, axis=0),(N,1))
	# 		array_meany = np.tile(np.expand_dims(mean_y, axis=0),(N,1))
	# 		array_stdx = np.tile(np.expand_dims(std_x, axis=0),(N,1))
	# 		array_meanx = np.tile(np.expand_dims(mean_x, axis=0),(N,1))

	# 		# print('[TEMP] =========> INPUT SATES [2] : ', np.mean(forwardsim_x_true))

	# 		#log_to_file(f"array_stdx \n {array_stdx} \narray_stdy\n {array_stdy}")



	# 		if(len(forwardsim_x_true)==2):
	# 			#N starting states, one for each of the simultaneous sims
	# 			curr_states=np.tile(forwardsim_x_true[0], (N,1))
	# 		else:
	# 			curr_states=np.copy(forwardsim_x_true)
	# 		# print('[TEMP] =========> INPUT SATES [3] : ', np.mean(curr_states))

	# 		#advance all N sims, one timestep at a time
	# 		for timestep in range(horizon):

	# 			#keep track of states for all N sims
	# 			state_list.append(np.copy(curr_states))
	# 			# print('[TEMP] =========> INPUT SATES [4] : ', np.mean(curr_states))

	# 			#log_to_file("ORIGINAL: " + str(np.shape(curr_states)))

	# 			#make [N x (state,action)] array to pass into NN
	# 			states_preprocessed = np.nan_to_num(np.divide((curr_states-array_meanx), array_stdx))
	# 			#log_to_file(str(np.shape(states_preprocessed)))
	# 			actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
	# 			#print('[TEMP] state: ',  np.mean(curr_states),'  state_pre: ', np.mean(states_preprocessed),'  action_pre: ', np.mean(actions_preprocessed))
	# 			inputs_list= np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

	# 			particles = []

	# 			parallel_particles = False

	# 			if True:

	# 				for i in range(num_particles):
	# 					#run the N sims all at once
	# 					# model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list}) 
	# 					model_output = new_bnn_network(inputs_list)
	# 					model_output = model_output.numpy()
	# 					# print('[TEMP] =========> OUTPUT SATES [1] : ', np.mean(model_output))
	# 					# print('[TEMP] OUTPUT: ', np.mean(inputs_list), '  ====>  ' , np.mean(model_output) )
	# 					state_differences = np.multiply(model_output,array_stdz)+array_meanz
	# 					# print('[TEMP] =========> OUTPUT SATES [2] : ', np.mean(state_differences))
	# 					# print('[TEMP] ... model_output[-0-] : ', np.mean(model_output), ',  ',np.shape(model_output))
	# 					# print('[TEMP] ... array_stdz : ', np.mean(array_stdz), ',  ',np.shape(array_stdz))
	# 					# print('[TEMP] ... array_meanz : ', np.mean(array_meanz), ',  ',np.shape(array_meanz))
	# 					particles.append(state_differences)
	# 					#print("OUTPUT:  ", np.mean(model_output))

	# 			# print('[TEMP] =========> OUTPUT SATES [3] : ', np.mean(particles))


	# 			particles_mean = np.mean(particles, axis=0)
	# 			particles_std = np.std(particles, axis=0)
	# 			#print("[TEMP] ", np.shape(particles_mean), np.shape(particles_std))

	# 			sampled_state_differences = np.random.normal(loc=particles_mean, scale=particles_std)
	# 			# print('[TEMP] =========> OUTPUT SATES [4] : ', np.mean(sampled_state_differences))
	# 			#update the state info
	# 			curr_states = curr_states + sampled_state_differences
	# 			# print('[TEMP] =========> OUTPUT SATES [5] : ', np.mean(curr_states))
	# 			std_list.append(particles_std)

	# 		#return a list of length = horizon+1... each one has N entries, where each entry is (13,)
	# 		state_list.append(np.copy(curr_states))
	# 	else:
	# 		curr_state = np.copy(forwardsim_x_true[0]) #curr state is of dim NN input

	# 		for curr_control in forwardsim_y:

	# 			state_list.append(np.copy(curr_state))
	# 			curr_control = np.expand_dims(curr_control, axis=0)

	# 			#subtract mean and divide by standard deviation
	# 			curr_state_preprocessed = curr_state - mean_x
	# 			curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/std_x)
	# 			curr_control_preprocessed = curr_control - mean_y
	# 			curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/std_y)
	# 			inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

	# 			particles = []

	# 			for i in range(num_particles):

	# 				#run through NN to get prediction
	# 				# model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed}) 
	# 				model_output = new_bnn_network(inputs_preprocessed)
	# 				model_output = model_output.numpy()
	# 				#multiply by std and add mean back in
	# 				state_differences= (model_output[0]*std_z)+mean_z

	# 				particles.append(state_differences)

	# 			particles_mean = np.mean(particles, axis=0)
	# 			particles_std = np.std(particles, axis=0)

	# 			sampled_state_differences = np.random.normal(loc=particles_mean, scale=particles_std)

	# 			#update the state info
	# 			next_state = curr_state + sampled_state_differences

	# 			#copy the state info
	# 			curr_state= np.copy(next_state)
	# 			std_list.append(particles_std)

	# 		state_list.append(np.copy(curr_state))
			  
	# 	return state_list


@ray.remote
class ForwardActor(object):
	def __init__(self, n_hidden, n_in, n_out, prob, model_lam, std_z=0, mean_z=0, std_y=0, mean_y=0, std_x=0, mean_x=0):
		self.new_bnn_network = BNN(n_hidden=n_hidden, n_out=n_out, prob=prob, lam=model_lam, dtype='float64')
		_ = self.new_bnn_network(np.zeros( (1, n_in) )) # initialisation
		self.std_z, self.mean_z, self.std_y, self.mean_y, self.std_x, self.mean_x = std_z, mean_z, std_y, mean_y, std_x, mean_x

	def test(self,weight):
		print('???????????????????????????????????????????????????????????????????????????/')

	def load_weights(self, weights):
		# Set weights
		self.new_bnn_network.set_weights(weights)
		


	def remote_forward_sim(self, junk, forwardsim_x_true, forwardsim_y, many_in_parallel, env_inp, which_agent=1, num_particles=20,
						   std_z=0, mean_z=0, std_y=0, mean_y=0, std_x=0, mean_x=0):

		
		# n_out_TEMP = np.shape(weights[-1])[0]
		# n_hidden =  np.shape(weights[-2])[0]
		# prob_TEMP = 0.9
		# model_lam_TEMP = 1e-2
		# new_bnn_network = net
		std_z, mean_z, std_y, mean_y, std_x, mean_x = self.std_z, self.mean_z, self.std_y, self.mean_y, self.std_x, self.mean_x
		
		state_list = []
		std_list = []

		if(many_in_parallel):
			#init vars
			N= forwardsim_y.shape[0]
			horizon = forwardsim_y.shape[1]
			array_stdz = np.tile(np.expand_dims(std_z, axis=0),(N,1))
			array_meanz = np.tile(np.expand_dims(mean_z, axis=0),(N,1))
			array_stdy = np.tile(np.expand_dims(std_y, axis=0),(N,1))
			array_meany = np.tile(np.expand_dims(mean_y, axis=0),(N,1))
			array_stdx = np.tile(np.expand_dims(std_x, axis=0),(N,1))
			array_meanx = np.tile(np.expand_dims(mean_x, axis=0),(N,1))

			# print('[TEMP] =========> INPUT SATES [2] : ', np.mean(forwardsim_x_true))

			#log_to_file(f"array_stdx \n {array_stdx} \narray_stdy\n {array_stdy}")



			if(len(forwardsim_x_true)==2):
				#N starting states, one for each of the simultaneous sims
				curr_states=np.tile(forwardsim_x_true[0], (N,1))
			else:
				curr_states=np.copy(forwardsim_x_true)
			# print('[TEMP] =========> INPUT SATES [3] : ', np.mean(curr_states))

			#advance all N sims, one timestep at a time
			for timestep in range(horizon):

				#keep track of states for all N sims
				state_list.append(np.copy(curr_states))
				# print('[TEMP] =========> INPUT SATES [4] : ', np.mean(curr_states))

				#log_to_file("ORIGINAL: " + str(np.shape(curr_states)))

				#make [N x (state,action)] array to pass into NN
				states_preprocessed = np.nan_to_num(np.divide((curr_states-array_meanx), array_stdx))
				#log_to_file(str(np.shape(states_preprocessed)))
				actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
				#print('[TEMP] state: ',  np.mean(curr_states),'  state_pre: ', np.mean(states_preprocessed),'  action_pre: ', np.mean(actions_preprocessed))
				inputs_list= np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

				particles = []

				parallel_particles = False

				if True:

					for i in range(num_particles):
						#run the N sims all at once
						# model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list}) 
						model_output = self.new_bnn_network(inputs_list)
						model_output = model_output.numpy()
						# print('[TEMP] =========> OUTPUT SATES [1] : ', np.mean(model_output))
						# print('[TEMP] OUTPUT: ', np.mean(inputs_list), '  ====>  ' , np.mean(model_output) )
						state_differences = np.multiply(model_output,array_stdz)+array_meanz
						# print('[TEMP] =========> OUTPUT SATES [2] : ', np.mean(state_differences))
						# print('[TEMP] ... model_output[-0-] : ', np.mean(model_output), ',  ',np.shape(model_output))
						# print('[TEMP] ... array_stdz : ', np.mean(array_stdz), ',  ',np.shape(array_stdz))
						# print('[TEMP] ... array_meanz : ', np.mean(array_meanz), ',  ',np.shape(array_meanz))
						particles.append(state_differences)
						#print("OUTPUT:  ", np.mean(model_output))

				# print('[TEMP] =========> OUTPUT SATES [3] : ', np.mean(particles))


				particles_mean = np.mean(particles, axis=0)
				particles_std = np.std(particles, axis=0)
				#print("[TEMP] ", np.shape(particles_mean), np.shape(particles_std))

				sampled_state_differences = np.random.normal(loc=particles_mean, scale=particles_std)
				# print('[TEMP] =========> OUTPUT SATES [4] : ', np.mean(sampled_state_differences))
				#update the state info
				curr_states = curr_states + sampled_state_differences
				# print('[TEMP] =========> OUTPUT SATES [5] : ', np.mean(curr_states))
				std_list.append(particles_std)

			#return a list of length = horizon+1... each one has N entries, where each entry is (13,)
			state_list.append(np.copy(curr_states))
		else:
			curr_state = np.copy(forwardsim_x_true[0]) #curr state is of dim NN input

			for curr_control in forwardsim_y:

				state_list.append(np.copy(curr_state))
				curr_control = np.expand_dims(curr_control, axis=0)

				#subtract mean and divide by standard deviation
				curr_state_preprocessed = curr_state - mean_x
				curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/std_x)
				curr_control_preprocessed = curr_control - mean_y
				curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/std_y)
				inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

				particles = []

				for i in range(num_particles):

					#run through NN to get prediction
					# model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed}) 
					model_output = self.new_bnn_network(inputs_preprocessed)
					model_output = model_output.numpy()
					#multiply by std and add mean back in
					state_differences= (model_output[0]*std_z)+mean_z

					particles.append(state_differences)

				particles_mean = np.mean(particles, axis=0)
				particles_std = np.std(particles, axis=0)

				sampled_state_differences = np.random.normal(loc=particles_mean, scale=particles_std)

				#update the state info
				next_state = curr_state + sampled_state_differences

				#copy the state info
				curr_state= np.copy(next_state)
				std_list.append(particles_std)

			state_list.append(np.copy(curr_state))
			  
		return state_list


def get_output_parallel(inputs_list, sess, curr_nn_output, x_, array_stdz, array_meanz):


	with sess.graph.as_default():

		model_output = sess.run([curr_nn_output], feed_dict={x_: inputs_list}) 
		state_differences = np.multiply(model_output[0],array_stdz)+array_meanz
		#print("...", np.mean(state_differences))
		state_differences = np.array(state_differences)

	#print("...", np.mean(state_differences))
	return state_differences