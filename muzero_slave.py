import utils
import argparse
import sys
from environment_new import *
from game import Game
import muzero_model
from muzero_mcts import simulate_and_select#MCTS, select_action, select_temperature
import datetime
import tensorflow as tf
import os

def run(stg):
    settings = utils.Obj(stg)
    env = SnakeEnv(settings.env)
    nb_episodes = settings.nb_episodes
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
              tf.config.experimental.set_memory_growth(gpu, True)
    agent = muzero_model.MuZero(settings.model)
    agent.build()
    steps = settings.steps
    last = datetime.datetime.now().minute
    action = 0
    for ep in range(nb_episodes):
        game = Game(settings.game, env.action_space)
        observation = env.reset()

        for step in range(steps):
            game.observation_history.append(observation)
            stacked_observations = game.get_stacked_observations(-1, settings.model.stacked_frame)

            #root = MCTS(settings.mcts).run(agent, stacked_observations, [0, 1, 2, 3, 4], 0, True)
            action, root = simulate_and_select(settings.mcts, agent, stacked_observations, ep, action, training=True)
            game.store_MTCS_stat(root)
            new_obs, reward, terminal = env.step(action)
            game.actions_history.append(action)
            game.rewards_history.append(reward)

            if terminal:
                game.dump()
                break

            observation = new_obs

        if (datetime.datetime.now().minute - last) % 60 > 1:
            if os.path.exists(settings.model.model_path):
                agent.load_weights()
                last = datetime.datetime.now().minute
                print('new weights loaded')

if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)
    run(stg_obj)
