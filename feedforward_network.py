import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform

class VariationalDense:
	"""Variational Dense Layer Class"""
	def __init__(self, n_in, n_out, model_prob, model_lam, tf_datatype):
		self.model_prob = model_prob
		self.model_lam = model_lam
		self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf_datatype)
		self.model_M = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01))
		self.model_m = tf.Variable(tf.zeros([n_out]))
		self.model_W = tf.matmul(
			tf.diag(self.model_bern.sample((n_in, ))), self.model_M
		)

	def __call__(self, X, activation=tf.identity):
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


