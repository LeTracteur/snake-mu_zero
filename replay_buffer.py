import numpy as np
import muzero_model


class ReplayBuffer:
    def __init__(self, settings):
        self.buffer_size = int(eval(settings.buffer_size))
        self.batch_size = settings.batch_size
        self.num_unroll_steps = settings.unroll_steps
        self.td_steps = settings.td_steps
        self.allow_reanalize = settings.reanalize
        self.num_stacked_obs = settings.stacked_frame
        self.support_size = settings.support_size

        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def get_batch(self, model):
        batch = {"observation_batch": [],
                 "target_values": [],
                 "target_rewards": [],
                 "target_policies": [],
                 "target_actions": []}

        games = [self.get_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.get_pos_in_g(g)) for g in games]

        for (g, i) in game_pos:
            t_val, t_reward, t_policies, t_actions = self.make_target(g, i, model)
            batch["observation_batch"].append(g.get_stacked_observations(i, self.num_stacked_obs))
            batch["target_values"].append(t_val)
            batch["target_rewards"].append(t_reward)
            batch["target_policies"].append(t_policies)
            batch["target_actions"].append(t_actions)
        return batch
        # return [(g.make_image(i), g.history[i:i + num_unroll_steps], g.make_target(i, num_unroll_steps, td_steps, g.to_play())) for (g, i) in game_pos]

    def get_game(self):
        g_id = np.random.choice(len(self.buffer))
        return self.buffer[g_id]

    def get_pos_in_g(self, game):
        pos = np.random.choice(len(game.actions_history))
        return pos

    def make_target(self, game, start_index, model):
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for idx in range(start_index, start_index + self.num_unroll_steps + 1):

            value = self.compute_value(game, idx, model)

            if idx < len(game.root_values):
                target_values.append(value)
                target_rewards.append(game.rewards_history[idx])
                target_policies.append(game.child_visits[idx])
                actions.append(game.actions_history[idx])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append([])
                actions.append(np.random.choice(game.actions_history))

        return target_values, target_rewards, target_policies, actions

    def compute_value(self, game, idx, model):
        bootstrap_index = idx + self.td_steps
        if bootstrap_index < len(game.root_values):
            if self.allow_reanalize:
                observation = game.get_stacked_observations(bootstrap_index, self.num_stacked_obs)
                last_step_val = muzero_model.support_to_scalar(model.initial_inference(observation)[0], self.support_size).item()
                value = last_step_val * game.discount ** self.td_steps
            else:
                value = game.root_values[bootstrap_index] * game.discount ** self.td_steps
        else:
            value = 0

        for i, reward in enumerate(game.rewards_history[idx:bootstrap_index]):
            value += reward * game.discount ** i
        return value