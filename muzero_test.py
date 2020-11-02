import utils
import argparse
import sys
from environment_new import *
from game import Game
import muzero_model
from muzero_mcts import MCTS, select_action, select_temperature
import datetime
import tensorflow as tf
import os


def test(stg):
    settings = utils.Obj(stg)
    env = SnakeEnv(settings.env)

    agent = muzero_model.MuZero(settings.model)
    agent.build()
    agent.load_weights()
    game = Game(settings.game, env.action_space)
    observation = env.reset()
    temperature = 0

    while True:
        env.render()

        game.observation_history.append(observation)
        stacked_observations = game.get_stacked_observations(-1, settings.model.stacked_frame)

        root, tree_depth = MCTS(settings.mcts).run(agent, stacked_observations, [0, 1, 2, 3, 4], 0, True)
        game.store_MTCS_stat(root)
        action = select_action(root, temperature)

        new_obs, _, terminal = env.step(action)

        if terminal:
            break

        observation = new_obs



if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)
    test(stg_obj)