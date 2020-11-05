import numpy as np
from muzero_model import support_to_scalar, scalar_to_support
import tensorflow as tf
import math


#class MCTS:
#    def __init__(self, settings):
#        self.sts = settings

def run_mcts(settings, root, action, model, to_play, training):
    """
    At the root of the search tree we use the representation function to obtain a
    hidden state given the current observation.
    We then run a Monte Carlo Tree Search using only action sequences and the model
    learned by the network.
    """
    #root = Node(0)
    #observation = (torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device))
    #root_pred_value, reward, policy_logits, hidden_state = model.initial_inference(np.expand_dims(observation, 0))
    #root_pred_value = support_to_scalar(root_pred_value, self.sts.support_size)
    #reward = support_to_scalar(reward, self.sts.support_size)
    #reward = reward.numpy().item()
    #root.expand(legal_actions, to_play, reward, policy_logits, hidden_state)
    #if add_exploration_noise:
    #    root.add_exploration_noise(
    #        dirichlet_alpha=self.sts.root_dirichlet_alpha,
    #        exploration_fraction=self.sts.root_exploration_fraction,
    #    )

    min_max_stats = MinMaxStats()
    for _ in range(settings.num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(settings, node, min_max_stats)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next hidden
        # state given an action and the previous hidden state
        parent = search_path[-2]
        value, reward, policy_logits, hidden_state = model.recurrent_inference(parent.hidden_state, action, training=training)
        # Cast
        value = support_to_scalar(value, settings.support_size)
        reward = reward.numpy().item()
        # expand
        expand_node(node, settings.action_pos, True, reward, policy_logits, hidden_state)
        backpropagate(search_path, value, True, settings.discount, min_max_stats)

    return root

def backpropagate(search_path, value, to_play, discount, min_max_stats):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value

def select_child(settings, node, min_max_stats):
    """
    Select the child with the highest UCB score.
    """
    #scores = [self.ucb_score(node, child, min_max_stats) for action, child in node.children.items()]
    #exp = np.exp(scores - np.max(scores))
    #prob = exp / exp.sum()
    #action = np.random.choice([action for action, child in node.children.items()], p = prob)
    #max_ucb = max([self.ucb_score(node, child, min_max_stats) for action, child in node.children.items()])
    #action = np.random.choice([action for action, child in node.children.items() if self.ucb_score(node, child, min_max_stats) == max_ucb])
    _, action, child = max((ucb_score(settings, node, child, min_max_stats), action, child) for action, child in node.children.items())
    return action, node.children[action]
    
def ucb_score(settings, parent, child, min_max_stats):
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.
    """
    pb_c = (math.log((parent.visit_count + settings.pb_c_base + 1) / settings.pb_c_base) + settings.pb_c_init)
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
 
    prior_score = pb_c * child.prior
 
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.value())
    else:
        value_score = 0
 
    return prior_score + value_score

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

def expand_node(node, actions, to_play, reward, policy_logits, hidden_state):
    """
    We expand a node using the value, reward and policy prediction obtained from the
    neural network.
    """
    node.to_play = to_play
    node.reward = reward
    node.hidden_state = hidden_state
    policy = {}
    policy = {a: math.exp(policy_logits[0,a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p/policy_sum)
    return node

def add_exploration_noise(node, settings):
    """
    At the start of each search, we add dirichlet noise to the prior of the root to
    encourage the search to explore new actions.
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([settings.dirichlet_alpha] * len(actions))
    frac = settings.exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

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

'''
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
'''

def select_action(node, num_episodes, training = True):
    """
    After running simulations inside in MCTS, we select an action based on the root's children visit counts.
    During training we use a softmax sample for exploration.
    During evaluation we select the most visited child.
    """
    visit_counts = [child.visit_count for child in node.children.values()]
    actions = [action for action in node.children.keys()]
    if training:
        temp = get_temperature(num_episodes)
        action = softmax_sample(visit_counts, actions, temp)
    else:
        action, _ = max(node.children.items(), key=lambda item: item[1].visit_count)
    return action

def get_temperature(episode):
    if episode < 2000:
        temperature = 1.0
    elif episode < 4000:
        temperature = 0.5
    elif episode < 6000:
        temperature = 0.25
    else:
        temperature = 0.125
    return temperature

def softmax_sample(visit_counts, actions, temp):
    counts_exp = np.exp(visit_counts) * (1 / temp)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]

def simulate_and_select(settings, model, observation, num_episodes, action, training=True):
    root = Node(0)
    root_value, reward, policy_logits, hidden_state = model.initial_inference(np.expand_dims(observation, 0), training=training)
    root_value = support_to_scalar(root_value, settings.support_size)
    reward = reward.numpy().item()
    root = expand_node(root, settings.action_pos, True, reward, policy_logits, hidden_state)
    if training:
        add_exploration_noise(root, settings)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the networks.
    root = run_mcts(settings, root, action, model, True, training=training)
    action = select_action(root, num_episodes, training=training)
    return action, root
