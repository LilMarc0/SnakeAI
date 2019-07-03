from tflearn import fully_connected, lstm, \
	batch_normalization, dropout, input_data, conv_1d, max_pool_1d, initializations, activation, mean_square
import tensorflow as tf
import numpy as np
from collections import deque
import random
import pickle
import os

class ReplayBuffer(object):
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def save(self, path):
        pickle.dump(self.buffer, open(path, "wb"))

    def load(self, path):
        self.buffer = pickle.load(open(path, "rb"))
        print('\033[92m' + 'Buffer found with {} data points \033[0m'.format(len(self.buffer)))

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch


class Actor:
	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
		self.saver = tf.train.Saver ()
		self.sess = sess
		self.s_dim = tf.placeholder (tf.int8, state_dim)
		self.obs_dim = state_dim
		self.a_dim = action_dim

		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.batch_size = batch_size
		self.inputs, self.out, self.scaled_out = self.create_actor_network (self.obs_dim, self.a_dim)

		self.network_params = tf.trainable_variables()

		self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(self.obs_dim, self.a_dim)

		self.target_network_params = tf.trainable_variables()[len(self.network_params):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign (tf.multiply (self.network_params[i], self.tau) + \
												   tf.multiply (self.target_network_params[i], 1. - self.tau))
			 for i in range (len (self.target_network_params))]
		# This gradient will be provided by the critic network
		self.action_gradient = tf.placeholder (tf.float32, [None, action_dim])

		# Combine the gradients, dividing by the batch size to
		# account for the fact that the gradients are summed over the
		# batch by tf.gradients
		self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
		self.actor_gradients = list(map(lambda x: tf.div (x, self.batch_size), self.unnormalized_actor_gradients))

		# Optimization Op
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

		self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

	def create_actor_network(self, state_dim, action_dim):
		inputs = input_data(shape=state_dim)
		net = conv_1d (inputs, 128, 2, 2)
		net = max_pool_1d(net, 2, 2, 'same')
		net = batch_normalization (net)

		net = conv_1d (net, 256, 2, 2)
		net = max_pool_1d (net, 2, 2, 'same')
		net = batch_normalization (net)

		shape = net.get_shape ().as_list ()
		net = fully_connected (net, 1024, activation='relu', regularizer='L2')
		net = dropout (net, 0.8)
		net = fully_connected (net, 1024, activation='relu', regularizer='L2')

		# Final layer weights are init to Uniform[-3e-3, 3e-3]
		w_init = initializations.uniform (minval=-0.003, maxval=0.003)
		out = fully_connected (net, action_dim, activation='softmax', weights_init=w_init)
		# Scale output to -action_bound to action_bound
		scaled_out = tf.multiply (out, self.action_bound)
		return inputs, out, scaled_out

	def train(self, inputs, a_gradient):
		self.sess.run (self.optimize, feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def predict(self, inputs):
		return self.sess.run (self.scaled_out, feed_dict={
			self.inputs: inputs
		})

	def predict_target(self, inputs):
		return self.sess.run (self.target_scaled_out, feed_dict={
			self.target_inputs: inputs
		})

	def update_target_network(self):
		self.sess.run (self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars

	def save(self):
		print('Saving actor...')
		save_path = self.saver.save (self.sess, "./actor/actor.ckpt")

	def load(self):
		if os.path.isdir('./actor'):
			if os.path.isfile("./actor/actor.ckpt.index"):
				print ('Restoring actor...')
				res_path = self.saver.restore (self.sess, "./actor/actor.ckpt")
				print ('Actor restored...')
		else:
			print('Created actor directory')
			os.mkdir("./actor")




class Critic:
	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
		self.saver = tf.train.Saver ()
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau
		self.gamma = gamma

		self.inputs, self.action, self.out = self.create_critic_network (self.s_dim, self.a_dim)

		self.network_params = tf.trainable_variables ()[num_actor_vars:]

		self.target_inputs, self.target_action, self.target_out = self.create_critic_network (self.s_dim, self.a_dim)

		self.target_network_params = tf.trainable_variables ()[(len (self.network_params) + num_actor_vars):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign (tf.multiply (self.network_params[i], self.tau) + \
												   tf.multiply (self.target_network_params[i], 1. - self.tau))
			 for i in range (len (self.target_network_params))]
		# Network target (y_i)
		# Obtained from the target networks
		self.predicted_q_value = tf.placeholder (tf.float32, [None, 1])

		# Define loss and optimization Op
		self.loss = mean_square (self.predicted_q_value, self.out)
		self.optimize = tf.train.AdamOptimizer (self.learning_rate).minimize (self.loss)

		# Get the gradient of the net w.r.t. the action
		self.action_grads = tf.gradients (self.out, self.action)

	def create_critic_network(self, state_dim, action_dim):
		inputs = input_data(shape=state_dim)
		action = input_data(shape=[None, action_dim])

		net = conv_1d (inputs, 128, 2, 2)
		net = max_pool_1d (net, 2, 2, 'same')
		net = batch_normalization (net)

		net = conv_1d (net, 256, 2, 2)
		net = max_pool_1d (net, 2, 2, 'same')
		net = batch_normalization (net)

		print(net.get_shape().as_list())

		net = fully_connected (net, 1024, activation='relu')
		net = dropout(net, 0.8)
		net = fully_connected (net, 1024, activation='relu')

		# Add the action tensor in the 2nd hidden layer
		# Use two temp layers to get the corresponding weights and biases
		t1 = fully_connected (net, 2048)
		t2 = fully_connected (action, 2048)

		net = activation (
			tf.matmul (net, t1.W) + tf.matmul (action, t2.W) + t2.b, activation='relu')

		# linear layer connected to 1 output representing Q(s,a)
		# Weights are init to Uniform[-3e-3, 3e-3]
		w_init = initializations.uniform (minval=-0.003, maxval=0.003)
		out = fully_connected (net, 1, weights_init=w_init)
		return inputs, action, out

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: inputs,
			self.action: action,
			self.predicted_q_value: predicted_q_value
		})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.action: action
		})

	def predict_target(self, inputs, action):
		return self.sess.run (self.target_out, feed_dict={
			self.target_inputs: inputs,
			self.target_action: action
		})

	def action_gradients(self, inputs, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.inputs: inputs,
			self.action: actions
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def save(self):
		print ('Saving critic...')
		save_path = self.saver.save (self.sess, "./critic/critic.ckpt")

	def load(self):
		if os.path.isdir("./critic"):
			if os.path.isfile("./critic/critic.ckpt.index"):
				print ('Restoring critic...')
				self.saver.restore (self.sess, "./critic/critic.ckpt")
				print ('Critic restored...')
		else:
			print('Created critic directory')
			os.mkdir("./critic")
	class OrnsteinUhlenbeckActionNoise:
		def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
			self.theta = theta
			self.mu = mu
			self.sigma = sigma
			self.dt = dt
			self.x0 = x0
			self.reset ()

		def __call__(self):
			x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
				self.sigma * np.sqrt (self.dt) * np.random.normal (size=self.mu.shape)
			self.x_prev = x
			return x

		def reset(self):
			self.x_prev = self.x0 if self.x0 is not None else np.zeros_like (self.mu)

		def __repr__(self):
			return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format (self.mu, self.sigma)

class OrnsteinUhlenbeckActionNoise:
	def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
				self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)