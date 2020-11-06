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
    def __init__(self, layer_sizes, activation=tf.nn.leaky_relu):
        super().__init__()
        self._act = tf.nn.leaky_relu
        self._size_list = layer_sizes
        self.dense = [tfkl.Dense(i, activation=self._act) for i in self._size_list]

    def call(self, x):
        out = self.dense[0](x)
        for i in range(1, len(self.dense)):
            out = self.dense[i](out)
        return out

class ResidualBlock(Model):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.conv1 = tfkl.Conv2D(filters, (3,3), strides=(stride,stride), padding='same', use_bias=False, activation=tf.nn.leaky_relu)
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(filters, (3,3), strides=(stride,stride), padding='same', use_bias=False, activation=tf.nn.leaky_relu)
        self.bn2 = tfkl.BatchNormalization()

    def call(self, x, training):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out += x
        out = tf.nn.relu(out)
        return out

class DownSample(Model):
    def __init__(self, depth):
        super().__init__()
        self.conv1 = tfkl.Conv2D(depth // 2, (3,3), strides=(2,2), padding='same', use_bias=False, activation=tf.nn.leaky_relu)
        self.resblocks1 = [ResidualBlock(depth // 2) for _ in range(2)]
        self.conv2 = tfkl.Conv2D(depth, (3,3), strides=(2,2), padding='same', use_bias=False, activation=tf.nn.leaky_relu)
        self.resblocks2 = [ResidualBlock(depth) for _ in range(3)]
        self.pooling1 = tfkl.AveragePooling2D((3,3), strides=(2,2), padding='same')
        self.resblocks3 = [ResidualBlock(depth) for _ in range(3)]
        self.pooling2 = tfkl.AveragePooling2D((3,3), strides=(2,2), padding='same')

    def call(self, x, training):
        out = self.conv1(x)
        for block in self.resblocks1:
            out = block(out, training)
        out = self.conv2(out)
        for block in self.resblocks2:
            out = block(out, training)
        out = self.pooling1(out)
        for block in self.resblocks3:
            out = block(out, training)
        out = self.pooling2(out)
        return out

class RepresentationNetwork(Model):
    def __init__(self, settings):
        super().__init__()
        self.use_ds = settings.use_downsampling
        if self.use_ds:
            self.ds = DownSample(settings.depth)
        else:
            self.conv = tfkl.Conv2D(settings.depth, (3,3), strides=(1,1), padding='same', use_bias=False, activation=tf.nn.leaky_relu)
            self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(settings.depth) for _ in range(settings.blocks)]

    def call(self, x, training):
        if self.use_ds:
            out = self.ds(x, training)
        else:
            out = self.conv(x)
            out = self.bn(out, training=training)
            out = tf.nn.relu(out)

        for block in self.resblocks:
            out = block(out)
        return out

class DynamicsNetwork(Model):
    def __init__(self, settings):
        super().__init__()
        self.conv = tfkl.Conv2D(settings.depth, (3,3), strides=(1,1), padding='same', use_bias=False, activation=tf.nn.leaky_relu)
        self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(settings.depth) for _ in range(settings.blocks)]
        self.conv1x1 = tfkl.Conv2D(settings.reduced_depth, (1,1), activation=tf.nn.leaky_relu)
        self.flat = tfkl.Flatten()
        #self.fc = FullyConnectedNetwork(settings.reward_layers, settings.support_size*2+1, activation=None)
        self.fc = FullyConnectedNetwork(settings.reward_layers)
        self.reward = tfkl.Dense(1)

    def call(self, x, training):
        out = self.conv(x)
        out = self.bn(out, training=training)
        out = tf.nn.relu(out)
        for block in self.resblocks:
            out = block(out,training)
        state = out
        out = self.conv1x1(out)
        out = self.flat(out)
        out = self.fc(out)
        reward = self.reward(out)
        return state, reward

class PredictionNetwork(Model):
    def __init__(self, settings):
        super().__init__()
        self.c1 = tfkl.Conv2D(settings.reduced_depth, (3,3), padding='same', activation = tf.nn.leaky_relu)
        self.c2 = tfkl.Conv2D(settings.reduced_depth, (3,3), padding='same', activation = tf.nn.leaky_relu)
        self.flat = tfkl.Flatten()
        self.fc = FullyConnectedNetwork(settings.prediction_layers)
        self.value = tfkl.Dense(settings.support_size*2+1)
        self.policy = tfkl.Dense(settings.action_space)

    def call(self, x, training):
        out = self.c1(x)
        out = self.c2(out)
        out = self.flat(out)
        value = self.value(out)
        policy = self.policy(out)
        return tf.nn.softmax(policy), value

class MuZero:
    def __init__(self, settings):
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
        self.value_loss =  tf.keras.losses.CategoricalCrossentropy()#reduction=tf.keras.losses.Reduction.NONE)
        self.reward_loss = tf.keras.losses.MeanSquaredError()#reduction=tf.keras.losses.Reduction.NONE)
        self.policy_loss =  tf.keras.losses.CategoricalCrossentropy()#reduction=tf.keras.losses.Reduction.NONE)

    def train(self, data):
        obs, targets_init, targets_time, actions_time, mask_time, dynamic_mask_time = data
        with tf.GradientTape() as model_tape:
            # Acquire state
            value, reward, policy_logits, hidden_state = self.initial_inference(np.array(obs), training=True)
            target_value, target_reward, target_policy = zip(*targets_init)
            # Creates masks to handle non-exisiting policies (end of episode)
            target_policy = target_policy
            mask_policy = list(map(lambda l: bool(l), target_policy))
            target_policy = list(filter(lambda l: bool(l), target_policy))
            policy_logits = tf.boolean_mask(policy_logits, mask_policy)

            # Compute initial losses
            value_loss, _, policy_loss = self.loss_function(value, reward, policy_logits, np.array(target_value,dtype=np.float32),
                                                                                          np.array(target_reward,dtype=np.float32),
                                                                                          np.array(target_policy,dtype=np.float32))
            reward_loss = 0.
            # Compute losses
            k = 1
            for actions, targets, mask, dynamic_mask in zip(actions_time, targets_time,
                                                            mask_time, dynamic_mask_time):
                target_value, target_reward, target_policy = zip(*targets)
                # Mask hidden state
                hidden_state = tf.boolean_mask(hidden_state, dynamic_mask)
                # Apply Dynamics
                value, reward, policy_logits, hidden_state = self.recurrent_inference(hidden_state, np.array(actions,dtype=np.float32), training=True)
                # Mask targets
                target_value = tf.boolean_mask(target_value, mask)
                target_reward = tf.boolean_mask(target_reward, mask)
                # Mask policy
                target_policy = [policy for policy, b in zip(target_policy, mask) if b]
                mask_policy = list(map(lambda l: bool(l), target_policy))
                target_policy = tf.convert_to_tensor([policy for policy in target_policy if policy])
                target_policy = tf.reshape(target_policy,[-1,self.sts.action_space])
                policy_logits = tf.boolean_mask(policy_logits, mask_policy)
                # Compute loss and scale gradient
                current_value_loss, current_reward_loss, current_policy_loss = self.loss_function(value, reward, policy_logits,
                                                                                                  target_value,
                                                                                                  target_reward,
                                                                                                  target_policy)
                value_loss += scale_grad(current_value_loss, 1./k)
                reward_loss += scale_grad(current_reward_loss, 1./k)
                policy_loss += scale_grad(current_policy_loss, 1./k)
                loss = value_loss * self.sts.value_loss_weight + reward_loss + policy_loss
                hidden_state = scale_grad(hidden_state, 0.5)
                k += 1 

            loss = tf.reduce_mean(loss)
        
        # Optimize
        grads = model_tape.gradient(loss, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_trainable_variables()))
        self.training_step += 1
        return value_loss/k, reward_loss/(k-1), policy_loss/k, loss/k

    def save_weights(self):
        self.representation_network.save_weights(self.sts.model_path+'/representation/weights')
        self.dynamics_network.save_weights(self.sts.model_path+'/dynamic/weights')
        self.prediction_network.save_weights(self.sts.model_path+'/predictions/weights')

    def load_weights(self):
        self.representation_network.load_weights(self.sts.model_path+'/representation/weights')
        self.dynamics_network.load_weights(self.sts.model_path+'/dynamic/weights')
        self.prediction_network.load_weights(self.sts.model_path+'/predictions/weights')

    def prediction(self, encoded_state, training=False):
        policy, value = self.prediction_network(encoded_state, training)
        return policy, value

    def representation(self, obs, training=False):
        state = self.representation_network(obs, training)
        # [0,1] scaling
        min_state = tf.reduce_min(state, axis=(-2,-1), keepdims=True)
        max_state = tf.reduce_max(state, axis=(-2,-1), keepdims=True)
        scale_state = (max_state - min_state) + 1e-5
        state_norm = (state - min_state) / scale_state
        return state_norm

    def dynamics(self, state, action, training=False):
        # Stack encoded_state with action as plane
        action_one_hot = tf.ones((1, state.shape[1], state.shape[2], state.shape[0]), dtype=tf.dtypes.float32)
        action_one_hot = action_one_hot * action / self.sts.action_space
        action_one_hot = tf.transpose(action_one_hot, [3, 1, 2, 0])
        x = tf.concat((state, action_one_hot), -1)
        nxt_state, reward = self.dynamics_network(x, training)
        # [0,1] scaling
        min_nxt_state = tf.reduce_min(nxt_state, axis=(-2,-1), keepdims=True)
        max_nxt_state = tf.reduce_max(nxt_state, axis=(-2,-1), keepdims=True)
        scale_nxt_state = max_nxt_state - min_nxt_state + 1e-5
        nxt_state_normalized = (nxt_state - min_nxt_state) / scale_nxt_state
        return nxt_state_normalized, reward

    def initial_inference(self, observation, training=False):
        encoded_state = self.representation(observation, training=training)
        policy_logits, value = self.prediction(encoded_state)
        reward = tf.ones(len(observation),dtype=tf.int32)
        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, state, action, training=False):
        nxt_state, reward = self.dynamics(state, action,training=training)
        policy_logits, value = self.prediction(nxt_state,training=training)
        return value, reward, policy_logits, nxt_state

    def loss_function(self, pred_value, pred_reward, pred_policy_logits, true_value, true_reward, true_policy):
        #value_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_value, labels=scalar_to_support(true_value, self.sts.support_size)))
        value_loss =  tf.reduce_mean(-scalar_to_support(true_value, self.sts.support_size)*tf.nn.log_softmax(pred_value))
        reward_loss = tf.reduce_mean(tf.square(tf.subtract(true_reward, pred_reward)))
        policy_loss = tf.reduce_mean(- true_policy*tf.math.log(pred_policy_logits))
        #print(value_loss)
        #print(pred_policy_logits)
        #print(true_policy)
        return value_loss, reward_loss, policy_loss

    def get_trainable_variables(self):
        networks = (self.representation_network, self.dynamics_network, self.prediction_network)
        return [variables for variables_list in map(lambda n:n.trainable_variables, networks) for variables in variables_list]


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
