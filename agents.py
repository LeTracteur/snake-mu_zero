from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt


class ExperienceReplay:
    def __init__(self, state_size, max_size=10000, batch_size=64, alpha=0.7, beta=0.7):
        self.batch_size = batch_size
        self.max_size = max_size
        self.state_size = state_size
        self.alpha = alpha
        self.beta = beta
        
        self.buffer_counter = 0
        
        self.states =  np.empty((self.max_size, self.state_size[0], self.state_size[1], 4), dtype=np.uint8)
        self.next_states = np.empty((self.max_size, self.state_size[0], self.state_size[1], 4), dtype=np.uint8)
        self.rewards = np.empty(self.max_size, dtype=np.float32)
        self.actions = np.empty(self.max_size, dtype=np.int32)
        self.terminals = np.empty(self.max_size, dtype=np.bool)
        self.loss = np.zeros(self.max_size, dtype=np.float32)

    def add_to_memory(self, state, action, reward, n_state, terminal, loss=1.0):
        
        if self.buffer_counter > self.max_size:
            pos = np.argmin(self.loss)
        else:
            pos = self.buffer_counter
            self.buffer_counter += 1
        self.states[pos] = np.stack(state, axis=2)
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.next_states[pos] = np.stack(n_state, axis=2)
        self.terminals[pos] = terminal
        self.loss[pos] = loss

    def update_weights_per(self, loss):
        """
        This function updates the priorization weights and sampling
        probabilities.
        Input:
           loss: A loss vector for the whole dataset
        """
        Err = np.sqrt(loss) + 1e-9
        V = np.power(Err, self.alpha)
        P = V / np.sum(V)
        sample_weights = np.power(len(P) * P, -self.beta)
        return P, sample_weights

    def get_minibatch_per(self):
        p, weights = self.update_weights_per(self.loss)
        minibatch_idx = np.random.choice(self.max_size, self.batch_size, p=p)
        # print(minibatch_idx)
        selected_w = weights[minibatch_idx]
        return self.states[minibatch_idx], self.actions[minibatch_idx], self.rewards[minibatch_idx], self.next_states[minibatch_idx], self.terminals[minibatch_idx], selected_w, minibatch_idx

    def update_batch_loss(self, idx, loss):
        self.loss[idx] = loss.numpy()

class DQNagent:
    def __init__(self, nb_actions, state_size, mem_size, batch_size, optimizer=tf.keras.optimizers.Adam,
                 learningrate=0.001, gamma=0.9, eps=1.0, eps_min=0.01, tau=4, alpha=0.7, beta=0.5):
        self.nb_actions = nb_actions
        self.state_size = state_size

        self.tau = tau
        self.alpha = alpha
        self.beta = beta

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
        self.max_eps = eps
        self.min_eps = eps_min

        self.memory = ExperienceReplay(state_size, mem_size, batch_size, alpha, beta)
        self.MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # tf.keras.optimizersAdam(learning_rate=1e-4, epsilon=1e-6)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-6)

        # self.model_policy = self.create_model_cnn()
        # self.model_target = self.create_model_cnn()
        self.model_policy = self.create_model_2()
        self.model_target = self.create_model_2()

        self.update_weights()

        self.agent_loss = []

    def create_model_cnn(self):
        a = Input(shape=(self.state_size[0], self.state_size[1], self.tau))
        b = tf.subtract(tf.divide(a,4.0),0.5)
        c = Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), input_shape=(self.state_size[0], self.state_size[1], self.tau), activation='relu')(b)
        # c = Conv2D(filters=32, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')(c)
        c = Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')(c)
        c = Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')(c)

        f = Flatten()(c)
        d0 = Dense(128, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")(f)
        # d1 = Dense(128, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")(f)
        # d2 = Dense(64, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")(d1)
        d3 = Dense(self.nb_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="linear")(d0)

        model = Model(inputs=a, outputs=d3)
        model.compile(loss="mse", optimizer=self.optimizer)
        return model

    def create_model_2(self):
        a = Input(shape=(self.state_size[0], self.state_size[1], self.tau))

        b = tf.subtract(tf.divide(a, 4.0), 0.5)

        c = Conv2D(filters=16, kernel_size=3, strides=1, padding='SAME', input_shape=(self.state_size[0], self.state_size[1], self.tau), activation='selu')(b)
        c = Conv2D(filters=32, kernel_size=3, strides=1, padding='SAME', activation='selu')(c)

        f = Flatten()(c)
        d0 = Dense(256, activation="selu")(f)
        d3 = Dense(self.nb_actions, activation="linear")(d0)

        model = Model(inputs=a, outputs=d3)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=0.001))
        return model

    def update_weights(self):
        self.model_target.set_weights(self.model_policy.get_weights())

    def update_epsilon(self, decay=0.9999):
        self.current_eps = max(self.min_eps, self.current_eps * decay)

    def update_epsilonv2(self, max_episodes, exp_phase_lenght):
        self.current_eps = max(self.min_eps, self.current_eps - (self.max_eps - self.min_eps)/(max_episodes * exp_phase_lenght))

    def display_model(self):
        self.model_policy.summary()

    def act(self, state, snake_list=None):
        self.state_buffer.append(state)
        if np.random.rand() < self.current_eps:
            # head = snake_list[-1]
            # before = snake_list[-2]
            # x_res = head[0] - before[0]
            # y_res = head[1] - before[1]
            # if y_res < 0:
            # 	return np.random.choice([0, 1, 3])
            # elif y_res >0:
            # 	return np.random.choice([2, 1, 3])
            # elif x_res > 0:
            # 	return np.random.choice([0, 1, 2])
            # else:
            # 	return np.random.choice([0, 2, 3])
            return np.random.randint(self.nb_actions)

        ss_state = np.stack(self.state_buffer, axis=2)
        r_state = np.expand_dims(ss_state, axis=0)

        actions_t = self.model_policy(r_state, training=False)
        actions = actions_t.numpy()
        return np.argmax(actions[0])

    def add_to_memory(self, state, action, reward, n_state, terminal):
        self.memory.add_to_memory(state, action, reward, n_state, terminal, 1.0)

    def get_minibatch(self):
        return self.memory.get_minibatch_per()

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
        s, a, r, n_s, t, _, _ = self.get_minibatch()

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

    def optimize_per(self):
        s, a, r, n_s, t, w, samples_pos = self.get_minibatch()

        # gradient descent
        with tf.GradientTape() as tape:
            a_tensor = self.model_target(n_s)
            a_max_next = tf.math.reduce_max(a_tensor, axis=1)
            target_q = r + self.gamma * a_max_next * (1 - t)

            one_hot_actions = tf.one_hot(a, self.nb_actions, 1.0, 0.0)

            q_values = self.model_policy(s)

            Q = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = self.MSE(tf.stop_gradient(target_q), Q)
            self.memory.update_batch_loss(samples_pos, loss)
            loss = tf.reduce_mean(loss*w)
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
