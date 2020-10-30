import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import Model

class FullyConnectedNetwork(tfkl.Layer):
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

class ResidualBlock(tfkl.Layer):
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

class DownSample(tfkl.Layer):
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
            self.conv = tfkl.Conv2D(settings.depth, (3,3), strides=(2,2), padding='same', use_bias=False)
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
        self.conv = tfkl.Conv2D(settings.depth, (3,3), strides=(2,2), padding='same', use_bias=False)
        self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(settings.depth) for _ in range(settings.blocks)]
        self.conv1x1 = tfkl.Conv2D(settings.reduced_depth, (1,1))
        self.flat = tfkl.Flatten()
        self.fc = FullyConnectedNetwork(settings.reward_layers, settings.support_size, activation=None)

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

        self.conv1x1 = tfkl.Conv2D(settings.reduced_depth, kernel_size=(1,1))
        self.flat = tfkl.Flatten()
        self.fc_value = FullyConnectedNetwork(settings.value_layers, setting.support_size, activation=None)
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

        self.representation_network = RepresentationNetwork(self.settings)
        self.dynamics_network = DynamicsNetwork(self.settings)
        self.prediction_network = PredictionNetwork(self.settings)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, obs):
        st = self.representation_network(obs)
        # [0,1] scaling
        min_state = tf.reduce_min(state, axis=(-2,-1), keepdims=True)
        max_state = tf.reuce_max(state, axis=(-2,-1), keepdims=True)
        scale_state = (max_state - min_state) + 1e-5
        state_norm = (state - min_state) / scale_state
        return st_norm

    def dynamics(self, state, action):
        # Stack encoded_state with action as plane
        action_one_hot = tf.ones((state.shape[0], state.shape[1], state.shape[2], 1), dtype=tf.dtypes.float32)
        action_one_hot = action * action_one_hot / self.action_space_size
        x = tf.cat((state, action_one_hot), dim=-1)
        nxt_state, reward = self.dynamics_network(x)

        # [0,1] scaling
        min_nxt_st = tf.reduce_min(nxt_state, axis=(-2,-1), keepdims=True)
        max_nxt_st = tf.reuce_max(nxt_state, axis=(-2,-1), keepdims=True)
        scale_nxt_state = max_nxt_state - min_nxt_state + 1e-5
        nxt_state_normalized = (nxt_state - min_nxt_state) / scale_nxt_state
        return nxt_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = tf.onehot(tf.ones(len(observation),dtype=tf.int32)*self.full_support_size//2, support_size)
        return (value, reward, policy_logits, encoded_state)

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    """
    # Decode to a scalar
    prob = tf.nn.softmax(logits, dim=1)
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
    floor2 = tf.floor(x)
    prob = x - floor
    logits1 = tf.transpose(tf.one_hot(floor+support_size, 2*support_size+1))*(1-prob)
    logits2 = tf.transpose(tf.one_hot(floor+1+support_size, 2*support_size+1))*(prob)
    logits = tf.transpose(logits1+logits2)
    return logits
