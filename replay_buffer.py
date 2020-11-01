import numpy as np
import muzero_model
import os
import pickle
import glob
import shutil

class ReplayBuffer:
    def __init__(self, settings):
        self.buffer_size = int(eval(settings.buffer_size))
        self.batch_size = settings.batch_size
        self.num_unroll_steps = settings.unroll_steps
        self.td_steps = settings.td_steps
        self.allow_reanalize = settings.reanalize
        self.num_stacked_obs = settings.stacked_frame
        self.support_size = settings.support_size
        self.action_space = settings.action_space

        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def load_games_from_folder(self, load_from_seen=False):
        if not os.path.exists('games'):
            print('No folder were to find game')
            exit(1)
        else:
            if not os.path.exists('games/seen'):
                os.mkdir("games/seen")
            if load_from_seen:
                game_to_load = glob.glob("games/seen/*.game")
                for g in game_to_load:
                    with open(g, 'rb') as f:
                        game = pickle.load(f)
                    self.save_game(game)

            game_to_load = glob.glob("games/*.game")
            if game_to_load:
                for g in game_to_load:
                    with open(g, 'rb') as f:
                        game = pickle.load(f)
                    self.save_game(game)
                    name = g.split('/')[-1]
                    shutil.move(g, "games/seen/"+name)

    def get_batch(self, model):
        batch = {"observation_batch": [],
                 "target_values": [],
                 "target_rewards": [],
                 "target_policies": [],
                 "actions": [],
                 "mask_policy": []}

        games = [self.get_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.get_pos_in_g(g)) for g in games]

        for (g, i) in game_pos:
            t_val, t_reward, t_policies, t_actions, mask_policy = self.make_target(g, i, model)
            batch["observation_batch"].append(g.get_stacked_observations(i, self.num_stacked_obs))
            batch["target_values"].append(t_val)
            batch["target_rewards"].append(t_reward)
            batch["target_policies"].append(t_policies)
            batch["actions"].append(t_actions)
            batch["mask_policy"].append(mask_policy)
        for key in batch.keys():
            batch[key] = np.array(batch[key], dtype=np.float32)
        return batch

    def get_game(self):
        g_id = np.random.choice(len(self.buffer))
        return self.buffer[g_id]

    def get_pos_in_g(self, game):
        pos = np.random.choice(len(game.actions_history))
        return pos

    def make_target(self, game, start_index, model):
        target_values, target_rewards, target_policies, actions, mask_policy = [], [], [], [], []
        for idx in range(start_index, start_index + self.num_unroll_steps + 1):

            value = self.compute_value(game, idx, model)

            if idx < len(game.root_values):
                target_values.append(value)
                target_rewards.append(game.rewards_history[idx])
                target_policies.append(game.child_visits[idx])
                actions.append(game.actions_history[idx])
                mask_policy.append(True)
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append([0.0 for _ in range(self.action_space)])
                actions.append(np.random.choice(game.actions_history))
                mask_policy.append(False)

        return target_values, target_rewards, target_policies, actions, mask_policy

    def compute_value(self, game, idx, model):
        bootstrap_index = idx + self.td_steps
        if bootstrap_index < len(game.root_values):
            if self.allow_reanalize:
                observation = game.get_stacked_observations(bootstrap_index, self.num_stacked_obs)
                observation = np.expand_dims(observation,0)
                last_step_val = muzero_model.support_to_scalar(model.initial_inference(observation)[0], self.support_size).numpy().item()
                value = last_step_val * game.discount ** self.td_steps
            else:
                value = game.root_values[bootstrap_index] * game.discount ** self.td_steps
        else:
            value = 0

        for i, reward in enumerate(game.rewards_history[idx:bootstrap_index]):
            value += reward * game.discount ** i
        return value

    def last_n_games_stat(self, n):
        last_n = self.buffer[-n:]
        steps = sum([len(last_n[i].rewards_history) for i in range(n)]) / n
        reward = sum([sum(last_n[i].rewards_history) for i in range(n)]) / n
        obs_histo = last_n[-1].observation_history
        return steps, reward, obs_histo
