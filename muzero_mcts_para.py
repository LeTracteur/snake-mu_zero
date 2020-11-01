import numpy as np
from muzero_model import support_to_scalar, scalar_to_support
import tensorflow as tf
import math


# def my_max(a_lis, b_list):
#     res = []
#     for i in range(len(a_list)):
#         if elem > number:
#             res.append(elem)
#         else:
#             res.append(number)
#     return res


class MCTS:
    def __init__(self, settings):
        self.sts = settings

    def run(self, model, observation, legal_actions, to_play, add_exploration_noise):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        roots = [Node(0) for _ in range(observation.shape[0])]
        #observation = (torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device))
        root_pred_value, reward, policy_logits, hidden_state = model.initial_inference(observation)
        root_pred_value = support_to_scalar(root_pred_value, self.sts.support_size)
        reward = support_to_scalar(reward, self.sts.support_size)
        for i,root in enumerate(roots):
            root.expand(legal_actions, to_play, tf.expand_dims(reward[i], 0), tf.expand_dims(policy_logits[i], 0), tf.expand_dims(hidden_state[i], 0))
            if add_exploration_noise:
                root.add_exploration_noise(
                    dirichlet_alpha=self.sts.root_dirichlet_alpha,
                    exploration_fraction=self.sts.root_exploration_fraction,
                )

        list_min_max_stats = [MinMaxStats() for _ in range(observation.shape[0])]
        # min_max_stats =

        max_tree_depth = [0]*observation.shape[0]
        for _ in range(self.sts.num_simulations):
            node_list = []
            search_path_list = []
            action_list = []
            tree_depth = []
            for i,root in enumerate(roots):
                virtual_to_play = to_play
                node = root
                search_path = [node]
                current_tree_depth = 0
                while node.expanded():
                    current_tree_depth += 1
                    action, node = self.select_child(node, list_min_max_stats[i])
                    search_path.append(node)

                    # Players play turn by turn
                    if virtual_to_play + 1 < len(self.sts.players):
                        virtual_to_play = self.sts.players[virtual_to_play + 1]
                    else:
                        virtual_to_play = self.sts.players[0]
                node_list.append(node)
                search_path_list.append(search_path)
                action_list.append(action)
                tree_depth.append(current_tree_depth)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            hidden_state_list = [search_p[-2].hidden_state for search_p in search_path_list]
            # print(np.array(hidden_state_list).shape)
            value, reward, policy_logits, hidden_state = model.recurrent_inference(np.squeeze(np.array(hidden_state_list)), np.array(action))

            value = support_to_scalar(value, self.sts.support_size)
            reward = support_to_scalar(reward, self.sts.support_size)
            for i,node in enumerate(node_list):
                node.expand(self.sts.action_pos, virtual_to_play, tf.expand_dims(reward[i], 0), tf.expand_dims(policy_logits[i], 0), tf.expand_dims(hidden_state[i], 0))
            for i, search_path in enumerate(search_path_list):
                # print(search_path)
                self.backpropagate(search_path, tf.expand_dims(value[i], 0), virtual_to_play, list_min_max_stats[i])
            print(max_tree_depth)
            print(tree_depth)
            max_tree_depth = list(map(max, zip(max_tree_depth, tree_depth)))

        return roots, max_tree_depth

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        #scores = [self.ucb_score(node, child, min_max_stats) for action, child in node.children.items()]
        #exp = np.exp(scores - np.max(scores))
        #prob = exp / exp.sum()
        #action = np.random.choice([action for action, child in node.children.items()], p = prob)
        max_ucb = max([self.ucb_score(node, child, min_max_stats) for action, child in node.children.items()])
        action = np.random.choice([action for action, child in node.children.items() if self.ucb_score(node, child, min_max_stats) == max_ucb])
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (math.log((parent.visit_count + self.sts.pb_c_base + 1) / self.sts.pb_c_base) + self.sts.pb_c_init)
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(child.reward + self.sts.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.sts.discount * node.value())

            value = node.reward + self.sts.discount * value

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy = {}
        for a in actions:
            try:
                policy[a] = 1 / sum(tf.exp(policy_logits[0] - policy_logits[0][a]))
            except OverflowError:
                print("Warning: prior has been approximated")
                policy[a] = 0.0
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def select_action(node, temperature):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function
    in the config.
    """
    visit_counts = np.array([child.visit_count for child in node.children.values()])
    actions = [action for action in node.children.keys()]
    if temperature == 0:
        action = actions[np.argmax(visit_counts)]
    elif temperature == float("inf"):
        action = np.random.choice(actions)
    else:
        # See paper appendix Data Generation
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
        action = np.random.choice(actions, p=visit_count_distribution)

    return action

def select_temperature(episode):
    if episode < 10000:
        temperature = float("inf")
    elif episode < 20000:
        temperature = 1.0
    elif episode < 50000:
        temperature = 0.5
    else:
        temperature = 0.25
    return temperature
