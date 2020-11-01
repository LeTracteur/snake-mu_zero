import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import Model
import numpy as np

@tf.custom_gradient
def scale_gradient(x, scale):
    def grad(dy):
        return scale * dy
    return tf.identity(x), grad


def scale_grad(tensor, scale):
    return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor


class FullyConnectedNetwork(Model):
    def __init__(self, layer_sizes, output_size, activation=tf.nn.leaky_relu):
        super().__init__()
        self._act = activation
        self._size_list = layer_sizes
        self._output_size = output_size
        self.dense = [tfkl.Dense(i, activation=self._act) for i in self._size_list]
        self.final = tfkl.Dense(output_size, activation=self._act)

    def call(self, x):
        out = self.dense[0](x)
        for i in range(1, len(self.dense)):
            out = self.dense[i](out)
            out = self.final(out)
        return out

class ResidualBlock(Model):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.conv1 = tfkl.Conv2D(filters, (3,3), strides=(stride,stride), padding='same', use_bias=False)
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(filters, (3,3), strides=(stride,stride), padding='same', use_bias=False)
        self.bn2 = tfkl.BatchNormalization()

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = tf.nn.relu(out)
        return out

class DownSample(Model):
    def __init__(self, depth):
        super().__init__()
        self.conv1 = tfkl.Conv2D(depth // 2, (3,3), strides=(2,2), padding='same', use_bias=False)
        self.resblocks1 = [ResidualBlock(depth // 2) for _ in range(2)]
        self.conv2 = tfkl.Conv2D(depth, (3,3), strides=(2,2), padding='same', use_bias=False)
        self.resblocks2 = [ResidualBlock(depth) for _ in range(3)]
        self.pooling1 = tfkl.AveragePooling2D((3,3), strides=(2,2), padding='same')
        self.resblocks3 = [ResidualBlock(depth) for _ in range(3)]
        self.pooling2 = tfkl.AveragePooling2D((3,3), strides=(2,2), padding='same')

    def call(self, x):
        out = self.conv1(x)
        for block in self.resblocks1:
            out = block(out)
        out = self.conv2(out)
        for block in self.resblocks2:
            out = block(out)
        out = self.pooling1(out)
        for block in self.resblocks3:
            out = block(out)
        out = self.pooling2(out)
        return out

class RepresentationNetwork(Model):
    def __init__(self, settings):
        super().__init__()
        self.use_ds = settings.use_downsampling
        if self.use_ds:
            self.ds = DownSample(settings.depth)
        else:
            self.conv = tfkl.Conv2D(settings.depth, (3,3), strides=(1,1), padding='same', use_bias=False)
            self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(settings.depth) for _ in range(settings.blocks)]

    def call(self, x):
        if self.use_ds:
            out = self.ds(x)
        else:
            out = self.conv(x)
            out = self.bn(out)
            out = tf.nn.relu(out)

        for block in self.resblocks:
            out = block(out)
        return out

class DynamicsNetwork(Model):
    def __init__(self, settings):
        super().__init__()
        self.conv = tfkl.Conv2D(settings.depth, (3,3), strides=(1,1), padding='same', use_bias=False)
        self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(settings.depth) for _ in range(settings.blocks)]
        self.conv1x1 = tfkl.Conv2D(settings.reduced_depth, (1,1))
        self.flat = tfkl.Flatten()
        self.fc = FullyConnectedNetwork(settings.reward_layers, settings.support_size*2+1, activation=None)

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = tf.nn.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        out = self.conv1x1(out)
        out = self.flat(out)
        reward = self.fc(out)
        return state, reward

class PredictionNetwork(Model):
    def __init__(self, settings):
        super().__init__()
        self.resblocks =  [ResidualBlock(settings.depth) for _ in range(settings.blocks)]

        self.conv1x1 = tfkl.Conv2D(settings.reduced_depth, (1,1))
        self.flat = tfkl.Flatten()
        self.fc_value = FullyConnectedNetwork(settings.value_layers, settings.support_size*2+1, activation=None)
        self.fc_policy = FullyConnectedNetwork(settings.policy_layers, settings.action_space, activation=None)

    def call(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.conv1x1(out)
        out = self.flat(out)
        value = self.fc_value(out)
        policy = self.fc_policy(out)
        return policy, value

class MuZero(Model):
    def __init__(self, settings):
        super().__init__()
        self.sts = settings
        self.training_step = 0

    def build(self):
        self.representation_network = RepresentationNetwork(self.sts)
        self.dynamics_network = DynamicsNetwork(self.sts)
        self.prediction_network = PredictionNetwork(self.sts)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.sts.learning_rate,
                                                  beta_1=self.sts.adam_beta_1,
                                                  beta_2=self.sts.adam_beta_2,
                                                  epsilon=float(self.sts.adam_epsilon))
        # self.optimizer = tf.keras.optimizers.Adam()
        self.value_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.reward_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.policy_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def train(self, data):
        # obs, actions, target_value, target_reward, target_policy = data

        with tf.GradientTape() as model_tape:
            value, reward, policy_logits, hidden_state = self.initial_inference(data['observation_batch'])
            predictions = [[value, reward, policy_logits]]
            for i in range(1, data['actions'].shape[-1]):
                value, reward, policy_logits, hidden_state = self.recurrent_inference(hidden_state, data['actions'][:, i])
                # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                hidden_state = scale_grad(hidden_state, 0.5)#egister_hook(lambda grad: grad * 0.5)
                predictions.append([value, reward, policy_logits])

            # Compute losses
            value, reward, policy_logits = predictions[0]
            value_loss, _, policy_loss = self.loss_function(value, reward, policy_logits, data['target_values'][:, 0], data['target_rewards'][:, 0], data['target_policies'][:, 0])
            reward_loss = 0.

            for i, prediction in enumerate(predictions[1:], 1):
                current_value_loss, current_reward_loss, current_policy_loss = self.loss_function(prediction[0], prediction[1], prediction[2], data['target_values'][:, i], data['target_rewards'][:, i], data['target_policies'][:, i])
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                value_loss += scale_grad(current_value_loss, 1./i)*data['mask_policy'][:,i]#register_hook(lambda grad: grad / gradient_scale_batch[:, i])
                reward_loss += scale_grad(current_reward_loss, 1./i)*data['mask_policy'][:,i]#.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
                policy_loss += scale_grad(current_policy_loss, 1./i)*data['mask_policy'][:,i]#.register_hook(lambda grad: grad / gradient_scale_batch[:, i])

                loss = value_loss * self.sts.value_loss_weight + reward_loss + policy_loss

            loss = tf.reduce_mean(loss)

        # Optimize
        grads = model_tape.gradient(loss, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_trainable_variables()))
        self.training_step += 1
        return tf.reduce_mean(value_loss), tf.reduce_mean(reward_loss), tf.reduce_mean(policy_loss), loss

    def save_weights(self):
        self.representation_network.save_weights(self.sts.model_path)
        self.dynamics_network.save_weights(self.sts.model_path)
        self.prediction_network.save_weights(self.sts.model_path)

    def load_weights(self):
        self.representation_network.load_weights(self.sts.model_path)
        self.dynamics_network.load_weights(self.sts.model_path)
        self.prediction_network.load_weights(self.sts.model_path)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, obs):
        state = self.representation_network(obs)
        # [0,1] scaling
        min_state = tf.reduce_min(state, axis=(-2,-1), keepdims=True)
        max_state = tf.reduce_max(state, axis=(-2,-1), keepdims=True)
        scale_state = (max_state - min_state) + 1e-5
        state_norm = (state - min_state) / scale_state
        return state_norm

    def dynamics(self, state, action):
        # Stack encoded_state with action as plane
        # print(action)
        action_one_hot = tf.ones((1, state.shape[1], state.shape[2], state.shape[0]), dtype=tf.dtypes.float32)
        action_one_hot = action_one_hot * action / self.sts.action_space
        action_one_hot = tf.transpose(action_one_hot, [3, 1, 2, 0])
        x = tf.concat((state, action_one_hot), -1)
        nxt_state, reward = self.dynamics_network(x)

        # [0,1] scaling
        min_nxt_state = tf.reduce_min(nxt_state, axis=(-2,-1), keepdims=True)
        max_nxt_state = tf.reduce_max(nxt_state, axis=(-2,-1), keepdims=True)
        scale_nxt_state = max_nxt_state - min_nxt_state + 1e-5
        nxt_state_normalized = (nxt_state - min_nxt_state) / scale_nxt_state
        return nxt_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = tf.one_hot(tf.ones(len(observation),dtype=tf.int32)*self.sts.support_size, self.sts.support_size*2+1)
        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, state, action):
        nxt_state, reward = self.dynamics(state, action)
        policy_logits, value = self.prediction(nxt_state)
        return value, reward, policy_logits, nxt_state

    def loss_function(self, pred_value, pred_reward, pred_policy_logits, true_value, true_reward, true_policy):
        value_loss = self.value_loss(scalar_to_support(true_value, self.sts.support_size), pred_value)
        reward_loss = self.reward_loss(scalar_to_support(true_reward, self.sts.support_size), pred_reward)
        policy_loss = self.policy_loss(true_policy, pred_policy_logits)
        return value_loss, reward_loss, policy_loss

    # def cb_get_trainable_variables(self):
    def get_trainable_variables(self):
        networks = (self.representation_network, self.dynamics_network, self.prediction_network)
        return [variables for variables_list in map(lambda n:n.trainable_variables, networks) for variables in variables_list]
        # return get_trainable_variables


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    """
    # Decode to a scalar
    prob = tf.nn.softmax(logits, axis=1)
    support = tf.range(-support_size, support_size + 1, 1, dtype=tf.float32)
    x = tf.reduce_sum(tf.multiply(prob,support), axis=-1, keepdims=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = tf.sign(x) * (((tf.sqrt(1 + 4 * 0.001 * (tf.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))** 2 - 1)
    return x

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + 0.001 * x
    x = tf.clip_by_value(x, -support_size, support_size)
    floor = tf.floor(x)
    prob = x - floor
    logits1 = tf.transpose(tf.one_hot(tf.cast(floor, dtype=tf.int32)+support_size, 2*support_size+1))*(1-prob)
    logits2 = tf.transpose(tf.one_hot(tf.cast(floor, dtype=tf.int32)+1+support_size, 2*support_size+1))*(prob)
    logits = tf.transpose(logits1+logits2)
    return logits
