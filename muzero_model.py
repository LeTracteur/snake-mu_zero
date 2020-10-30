import tf
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

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, x):
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
    def __init__(self, stacked_obs, blocks, depth, ds):
        super().__init__()
        self.use_ds = ds
        if self.use_ds:
            self.ds = DownSample(depth)
        else:
            self.conv = tfkl.Conv2D(depth, (3,3), strides=(2,2), padding='same', use_bias=False)
            self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(depth) for _ in range(blocks)]

    def __call__(self, x):
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
    def __init__(self, blocks, depth, reduced_depth, fc_reward_layers, full_sup_size):
        super().__init__()
        self.conv = tfkl.Conv2D(depth, (3,3), strides=(2,2), padding='same', use_bias=False)
        self.bn = tfkl.BatchNormalization()
        self.resblocks = [ResidualBlock(depth) for _ in range(blocks)]
        self.conv1x1 = tfkl.Conv2D(reduced_depth, (1,1))
        self.flat = tfkl.Flatten()
        self.fc = FullyConnectedNetwork(fc_reward_layers, full_sup_size, activation=None)

    def __call__(self, x):
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
    def __init__(self, act_space, blocks, depth, reduced_depth, fc_value_layers, fc_policy_layers, full_sup_size):
        super().__init__()
        self.resblocks =  [ResidualBlock(depth) for _ in range(blocks)]

        self.conv1x1 = tfkl.Conv2D(reduced_depth, kernel_size=(1,1))
        self.flat = tfkl.Flatten()
        self.fc_value = FullyConnectedNetwork(fc_value_layers, full_sup_size, activation=None)
        self.fc_policy = FullyConnectedNetwork(fc_policy_layers, act_space, activation=None)

    def __call__(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.conv1x1(out)
        out = self.flat(out)
        value = self.fc_value(out)
        policy = self.fc_policy(out)
        return policy, value

class MuZero(Model):
    def __init__(self, ):
        super().__init__()
        self.action_space_size = act_dim
        self.full_support_size = 2 * support_size + 1
        #if ds:
        #    block_output_size = reduced_depth * (obs_shape[1] // 16) * (obs_shape[2] // 16)
        #else: 
        #    block_output_size = reduced_depth * obs_shape[1] * obs_shape[2]

        self.representation_network = RepresentationNetwork(obs_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            ds,
        )

        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels,
            reduced_channels,
            fc_reward_layers,
            self.full_support_size,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            depth,
            reduced_depth,
            fc_value_layers,
            fc_policy_layers,
            self.full_support_size,
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, obs):
        st = self.representation_network(obs)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_st = tf.reduce_min(tf.reshape(st, [-1, st.shape[1], st.shape[2] * st.shape[3]]), axis=2, keepdim=True)
        max_st = tf.reuce_max(tf.reshape(st, [-1, st.shape[1], st.shape[2] * st.shape[3]]), axis=2, keepdim=True)
        scale_st = max_st - min_st
        scale_st[scale_st < 1e-5] += 1e-5
        st_norm = (st - min_st) / scale_st
        return st_norm

    def dynamics(self, state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = tf.ones((state.shape[0], 1, state.shape[2], state.shape[3], dtype=tf.dtypes.float32)
        action_one_hot = action[:, :, None, None] * action_one_hot / self.action_space_size
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        nxt_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_nxt_state = (nxt_state.view(-1, nxt_state.shape[1], nxt_state.shape[2] * nxt_state.shape[3]).min(2, keepdim=True)[0].unsqueeze(-1))
        max_nxt_state = (nxt_state.view(-1, nxt_state.shape[1], nxt_state.shape[2] * nxt_state.shape[3]).max(2, keepdim=True)[0].unsqueeze(-1))
        scale_nxt_state = max_nxt_state - min_nxt_state
        scale_nxt_state[scale_nxt_state < 1e-5] += 1e-5
        nxt_state_normalized = (nxt_state - min_nxt_state) / scale_nxt_state
        return nxt_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = (tf.zeros(1, self.full_support_size).scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0).repeat(len(observation), 1))
        return (value, reward, policy_logits, encoded_state)

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = tf.nn.softmax(logits, dim=1)
    support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape)#.float().to(device=probabilities.device))
    x = tf.reduce_sum(support * probabilities, axis=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = tf.sign(x) * (((tf.sqrt(1 + 4 * 0.001 * (tf.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))** 2 - 1)
    return x

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + 0.001 * x
    x = tf.clip_by_value(x, -support_size, support_size)
    floor = tf.floor(x)
    floor2 = tf.floor(x)
    prob = x - floor
    logits1 = tf.transpose(tf.one_hot(floor, 2*support_size+1))*(1-prob)
    logits2 = tf.transpose(tf.one_hot(floor+1, 2*support_size+1))*(prob)
    logits = tf.transpose(logits1+logits2)
    return logits
