from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt


class DQNagent:
	def __init__(self, nb_actions, state_size, mem_size, batch_size, optimizer=tf.keras.optimizers.Adam,
	             learningrate=0.001, gamma=0.8, eps=1.0, eps_min=0.05, tau=4):
		self.nb_actions = nb_actions
		self.state_size = state_size

		self.tau = tau

		self.state_buffer = deque(maxlen=tau)
		self.next_state_buffer = deque(maxlen=tau)

		for i in range(self.tau):
			self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))
			self.next_state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))

		self.mem_size = mem_size
		self.batch_size = batch_size

		self.learningRate = learningrate
		self.gamma = gamma

		self.current_eps = eps
		self.min_eps = eps_min

		self.memory = deque(maxlen=mem_size)

		# tf.keras.optimizersAdam(learning_rate=1e-4, epsilon=1e-6)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-6)

		self.model_policy = self.create_model_cnn()
		self.model_target = self.create_model_cnn()
		self.update_weights()

		self.agent_loss = []

	def create_model_cnn(self):
		a = Input(shape=(self.state_size[0], self.state_size[1], self.tau))

		c = Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), input_shape=(self.state_size[0], self.state_size[1], self.tau), activation='relu')(a)
		c = Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')(c)
		c = Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')(c)

		f = Flatten()(c)
		d0 = Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")(f)
		# d1 = Dense(128, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")(f)
		# d2 = Dense(64, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")(d1)
		d3 = Dense(self.nb_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="linear")(d0)

		model = Model(inputs=a, outputs=d3)
		model.compile(loss="mse", optimizer=self.optimizer)
		return model

	def create_model_no_cnn(self):
		pass

	def update_weights(self):
		self.model_target.set_weights(self.model_policy.get_weights())

	# def update_epsilonv2(self, counter, factor=0.05):
	#
	# 	replay_start_size = 10000
	# 	if counter < replay_start_size:
	# 		eps = self.current_eps
	#
	#
	# 	self.current_eps = max(self.min_eps, self.current_eps*decay)

	def update_epsilon(self, decay=0.9999):
		self.current_eps = max(self.min_eps, self.current_eps * decay)

	def display_model(self):
		self.model_policy.summary()

	def act(self, state):
		self.state_buffer.append(state)
		if np.random.rand() < self.current_eps:
			return np.random.randint(self.nb_actions)

		ss_state = np.stack(self.state_buffer, axis=2)
		r_state = np.expand_dims(ss_state, axis=0)

		actions_t = self.model_policy(r_state, training=False)
		actions = actions_t.numpy()
		return np.argmax(actions[0])

	def add_to_memory(self, state, action, reward, n_state, terminal):
		self.memory.append((state, action, reward, n_state, terminal))

	def get_minibatch(self):
		minibatch = np.array(random.sample(self.memory, self.batch_size))
		s = np.array([np.stack(minibatch[i][0], axis=2) for i in range(len(minibatch))])
		a = np.array(minibatch[:, 1])
		r = np.array(minibatch[:, 2])
		n_s = np.array([np.stack(minibatch[i][3], axis=2) for i in range(len(minibatch))])
		t = np.array(minibatch[:, 4])

		return s, a, r, n_s, t


	def optimize(self):
		# s, a, r, n_s, t
		# minibatch = random.sample(self.memory, self.batch_size)
		# s, a, r, n_s, t = [], [], [], [], []
		# for i in range(len(minibatch)):
		# 	s.append(minibatch[i][0])
		# 	a.append(minibatch[i][1])
		# 	r.append(minibatch[i][2])
		# 	n_s.append(minibatch[i][3])
		# 	t.append(minibatch[i][4])
		# s = np.array(s)
		# a = np.array(a)
		# r = np.array(r)
		# n_s = np.array(n_s)
		# t = np.array(t)
		s, a, r, n_s, t = self.get_minibatch()

		#compute target

		# exp_actions = a_tensor.numpy()


		# gradient descent
		with tf.GradientTape() as tape:
			a_tensor = self.model_target(n_s)
			a_max_next = tf.math.reduce_max(a_tensor, axis=1)
			target_q = r + self.gamma * a_max_next * (1 - t)

			one_hot_actions = tf.one_hot(a, self.nb_actions, 1.0, 0.0)

			q_values = self.model_policy(s)

			Q = tf.reduce_sum(q_values * one_hot_actions, axis=1)
			loss = tf.keras.losses.MSE(tf.stop_gradient(target_q), Q)

		model_grad = tape.gradient(loss, self.model_policy.trainable_variables)
		clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in model_grad]
		self.model_policy.optimizer.apply_gradients(zip(clipped_gradients, self.model_policy.trainable_variables))

		self.agent_loss.append(loss)
		return loss.numpy()

	def save(self, some_string=''):
		if some_string:
			self.model_policy.save("model/"+some_string+"/model_policy.h5")
		else:
			self.model_policy.save("model/model_policy.h5")

	def load(self):
		pass

	def plot_loss(self):
		plt.plot(self.agent_loss)
		plt.title('Evolution of the loss of the Agent')
		plt.xlabel('Learning step')
		plt.ylabel('Loss value')
		plt.show()
