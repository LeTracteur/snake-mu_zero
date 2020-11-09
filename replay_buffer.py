import numpy as np
import muzero_model
import os
import pickle
import glob
import shutil
from itertools import zip_longest


class ReplayBuffer:
    def __init__(self, settings, with_per):
        self.buffer_size = int(eval(settings.buffer_size))
        self.batch_size = settings.batch_size
        self.num_unroll_steps = settings.unroll_steps
        self.td_steps = settings.td_steps
        self.allow_reanalize = settings.reanalize
        self.num_stacked_obs = settings.stacked_frame
        self.support_size = settings.support_size
        self.action_space = settings.action_space
        self.with_per = with_per
        self.alpha = settings.alpha
        self.beta = settings.beta

        self.buffer = []
        self.games_priority = []

        # variables for loading games from files
        self.current_game_folder = settings.current_game_folder
        self.max_nb_of_g_per_folder = settings.max_nb_of_g_per_folder
        self.current_number_of_g = 0

    def save_game(self, game, model=None):
        # Si ça bug faut passer le model
        if self.with_per:
            for i, root_value in enumerate(game.root_values):
                # pas sur que ca soit la root value, dans le papier ils disent search value
                priority = np.abs(root_value - self.compute_value(game, i, model)) ** self.alpha
                game.priorities.append(priority[0][0])
        else:
            game.priorities = [1.0 for _ in range(len(game.root_values))]

        game.priorities = np.array(game.priorities, dtype=np.float32)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self.games_priority.pop(0)
        self.buffer.append(game)
        self.games_priority.append(np.mean(game.priorities))

    def load_games_from_folder(self, load_from_seen=False):
        if not os.path.exists('games'):
            print('No folder where to find games')
            exit(1)
        else:
            if not os.path.exists('games/seen'):
                os.mkdir("games/seen")

            if not os.path.exists("games/seen/" + self.current_game_folder):
                os.mkdir("games/seen/" + self.current_game_folder)

            if load_from_seen:
                for batch in  os.listdir("games/seen/."):
                    games_files = glob.glob("games/seen/"+batch+"/*.game")
                    #if games_files:
                    #    for g in games_files:
                    #        if self.current_number_of_g == self.max_nb_of_g_per_folder:
                    #            new_id = int(self.current_game_folder.split("_")[-1]) + 1
                    #            self.current_game_folder = "games_batch_" + str(new_id)
                    #            os.mkdir("games/seen/"+self.current_game_folder)
                    #            self.current_number_of_g = 0
                    #
                    #        #name = g.split('/')[-1]
                    #        #shutil.move(g, "games/seen/"+self.current_game_folder + "/" + name)
                    #        self.current_number_of_g += 1
                    #
                    #game_to_load = glob.glob("games/seen/""/*.game")
                    # self.current_number_of_g = 0
                    if games_files:
                        for g in games_files:
                            # self.current_number_of_g += 1
                            with open(g, 'rb') as f:
                                game = pickle.load(f)
                            self.save_game(game)

            game_to_load = glob.glob("games/*.game")
            if game_to_load:
                for g in game_to_load:
                    if self.current_number_of_g == self.max_nb_of_g_per_folder:
                        new_id = int(self.current_game_folder.split("_")[-1]) + 1
                        self.current_game_folder = "games_batch_" + str(new_id)
                        os.mkdir("games/seen/"+self.current_game_folder)
                        self.current_number_of_g = 0
                    try:
                        with open(g, 'rb') as f:
                            game = pickle.load(f)
                        self.save_game(game)
                        name = g.split('/')[-1]
                        shutil.move(g, "games/seen/"+self.current_game_folder + "/" + name)
                        self.current_number_of_g += 1
                    except:
                        pass

    def get_batch(self, model):
        batch = {"observation_batch": [],
                 "target_values": [],
                 "target_rewards": [],
                 "target_policies": [],
                 "actions": [],
                 "mask": [],
                 "dynamic_mask": []}
        games, g_prob = [], []
        for _ in range(self.batch_size):
            game_and_prob = self.get_game()
            games.append(game_and_prob[0])
            g_prob.append(game_and_prob[1])

        game_pos = [(g, self.get_pos_in_g(g)) for g in games]
        pos_prob = [p for (g, (i, p)) in game_pos]
        n = len(self.buffer)
        weight_batch = np.array([(n*g_prob[i]*pos_prob[i])**(-self.beta) for i in range(len(pos_prob))])
        actions, target_value, target_reward, target_policy = [], [], [], []

        game_data = [(g.get_stacked_observations(i, self.num_stacked_obs), g.actions_history[i:i+self.num_unroll_steps], self.make_target(g, i, model)) for (g, (i, _)) in game_pos]
        image_batch, actions_time_batch, targets_batch = zip(*game_data)
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=None))

        # Building batch of valid actions and a dynamic mask for hidden representations during BPTT
        mask_time_batch = []
        dynamic_mask_time_batch = []
        last_mask = [True] * len(image_batch)
        
        for i, actions_batch in enumerate(actions_time_batch):
            mask = list(map(lambda a: (a is not None), actions_batch))
            dynamic_mask = [now for last, now in zip(last_mask, mask) if last]
            mask_time_batch.append(mask)
            dynamic_mask_time_batch.append(dynamic_mask)
            last_mask = mask
            actions_time_batch[i] = [action for action in actions_batch if (action is not None)]

        batch = image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch, weight_batch
        return batch

    def get_game(self):
        # g_id = np.random.choice(len(self.buffer))
        p = np.array(self.games_priority, dtype=np.float32)
        p = p/np.sum(p)
        g_id = np.random.choice(len(self.buffer), p=p)
        return self.buffer[g_id], p[g_id]

    def get_pos_in_g(self, game):
        # pos = np.random.choice(len(game.actions_history))
        p = game.priorities/np.sum(game.priorities)
        pos = np.random.choice(len(game.actions_history), p=p)

        return pos, p[pos]

    def make_target(self, game, start_index, model):
        targets = []#, target_rewards, target_policies, actions = [], [], [], []
        for idx in range(start_index, start_index + self.num_unroll_steps + 1):
            value = self.compute_value(game, idx, model)
            if idx < len(game.root_values):
                targets.append((value, game.rewards_history[idx], game.child_visits[idx]))
            else:
                # States past the end of games are treated as absorbing states
                targets.append((0., 0., []))

        return targets

    def compute_value(self, game, idx, model):
        bootstrap_index = idx + self.td_steps
        if bootstrap_index < len(game.root_values):
            if self.allow_reanalize:
                observation = game.get_stacked_observations(bootstrap_index, self.num_stacked_obs)
                observation = np.expand_dims(observation,0)
                last_step_val = muzero_model.support_to_scalar(model.initial_inference(observation, training=False)[0], self.support_size).numpy().item()
                value = last_step_val * game.discount ** self.td_steps
            else:
                value = game.root_values[bootstrap_index].numpy().item() * game.discount ** self.td_steps
        else:
            value = 0

        for i, reward in enumerate(game.rewards_history[idx:bootstrap_index]):
            value += reward * game.discount ** i
        return value

    def update_prio(self):
        pass

    def last_n_games_stat(self, n):
        last_n = self.buffer[-n:]
        steps = sum([len(last_n[i].rewards_history) for i in range(n)]) / n
        reward = sum([sum(last_n[i].rewards_history) for i in range(n)]) / n
        obs_histo = last_n[-1].observation_history
        return steps, reward, obs_histo
