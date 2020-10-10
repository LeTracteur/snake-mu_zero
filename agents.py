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
	             learningrate=0.001, gamma=0.9, eps=1.0, eps_min=0.01):
		self.nb_actions = nb_actions
		self.state_size = state_size

		self.mem_size = mem_size
		self.batch_size = batch_size

		self.learningRate = learningrate
		self.gamma = gamma

		self.current_eps = eps
		self.min_eps = eps_min

		self.memory = deque(maxlen=mem_size)

		self.optimizer = optimizer(learning_rate=self.learningRate)

		self.model_policy = self.create_model_cnn()
		self.model_target = self.create_model_cnn()
		self.update_weights()

		self.agent_loss = []

	def create_model_cnn(self):
		a = Input(shape=(self.state_size[0],self.state_size[1], 4))

		c = Conv2D(filters=32, kernel_size=3, padding="same", input_shape=(self.state_size[0], self.state_size[1], 4), activation='relu')(a)
		# m = MaxPooling2D(pool_size=(2, 2))(c)

		c = Conv2D(filters=32, kernel_size=3, padding="same",activation='relu')(c)
		# m = MaxPooling2D(pool_size=(2, 2))(c)

		c = Conv2D(filters=32, kernel_size=3, padding="same", activation='relu')(c)
		c = Conv2D(filters=32, kernel_size=3, padding="same", activation='relu')(c)

		f = Flatten()(c)
		d0 = Dense(64, activation="relu")(f)
		d1 = Dense(32, activation="relu")(d0)
		d2 = Dense(16, activation="relu")(d1)
		d3 = Dense(self.nb_actions, activation="linear")(d2)

		model = Model(inputs=a, outputs=d3)
		model.compile(loss='mse', optimizer=self.optimizer)
		return model

	def update_weights(self):
		self.model_target.set_weights(self.model_policy.get_weights())

	def update_epsilon(self, decay=0.999):
		self.current_eps = max(self.min_eps, self.current_eps*decay)

	def display_model(self):
		self.model_policy.summary()

	def act(self, state):
		if np.random.rand() < self.current_eps:
			return np.random.randint(self.nb_actions)

		r_state = np.expand_dims(state, axis=0)
		actions_t = self.model_policy(r_state, training=False)
		actions = actions_t.numpy()
		return np.argmax(actions[0])

	def add_to_memory(self, state, action, reward, n_state, terminal):
		self.memory.append((state, action, reward, n_state, terminal))

	def optimize(self):
		# s, a, r, n_s, t
		minibatch = random.sample(self.memory, self.batch_size)
		s, a, r, n_s, t = [], [], [], [], []
		for i in range(len(minibatch)):
			s.append(minibatch[i][0])
			a.append(minibatch[i][1])
			r.append(minibatch[i][2])
			n_s.append(minibatch[i][3])
			t.append(minibatch[i][4])
		s = np.array(s)
		a = np.array(a)
		r = np.array(r)
		n_s = np.array(n_s)
		t = np.array(t)

		#compute target
		a_tensor = self.model_target(n_s)
		exp_actions = a_tensor.numpy()
		target_q = r + self.gamma * np.amax(exp_actions) * (1 - t)

		# gradient descent
		with tf.GradientTape() as tape:
			one_hot_actions = tf.one_hot(a, self.nb_actions)
			q_values = self.model_policy(s)
			Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
			loss = tf.keras.losses.MSE(target_q, Q)

		model_grad = tape.gradient(loss, self.model_policy.trainable_variables)
		self.model_policy.optimizer.apply_gradients(zip(model_grad, self.model_policy.trainable_variables))

		self.agent_loss.append(loss)

	def save(self):
		self.model_policy.save("model/model_policy.h5")

	def load(self):
		pass

	def plot_loss(self):
		plt.plot(self.agent_loss)
		plt.title('Evolution of the loss of the Agent')
		plt.xlabel('Learning step')
		plt.ylabel('Loss value')
		plt.show()
