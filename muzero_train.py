import utils
import argparse
import sys
from environment_new import *
from game import Game
from replay_buffer import ReplayBuffer
import muzero_model
from muzero_mcts import MCTS, select_action, select_temperature

def main(stg):
    settings = utils.Obj(stg)
    env = SnakeEnv(settings.env)
    nb_episodes = settings.nb_episodes
    steps = settings.steps
    replay_buffer = ReplayBuffer(settings.buffer)

    agent = muzero_model.MuZero(settings.model)
    agent.build()

    temperature_threshold = None

    for ep in range(nb_episodes):
        game = Game(settings.game, env.action_space)
        observation = env.reset()
        temperature = select_temperature(ep)

        for step in range(steps):
            # visual rendering
            env.render()

            game.observation_history.append(observation)
            stacked_observations = game.get_stacked_observations(-1, settings.model.stacked_frame)

            root, tree_depth = MCTS(settings.mcts).run(agent, stacked_observations, [0, 1, 2, 3, 4], 0, True)
            game.store_MTCS_stat(root)

            action = select_action(root, temperature)

            new_obs, reward, terminal = env.step(action)

            game.actions_history.append(action)
            game.rewards_history.append(reward)

            if terminal:
                replay_buffer.save_game(game)
                break

            observation = new_obs
    return replay_buffer


if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)
    a = main(stg_obj)
