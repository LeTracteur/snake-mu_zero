import utils
import argparse
import sys
from environment_new import *
from game import Game


def main(stg):
    settings = utils.Obj(stg)
    env = SnakeEnv(settings.env)
    nb_episodes = settings.nb_episodes
    steps = settings.steps

    for ep in nb_episodes:
        game = Game(settings.game, env.action_space)
        observation = env.reset()
        game.observation_history.append(observation)

        for step in range(steps):
            # visual rendering
            env.render()

            # stacked_observations = game.get_stacked_observations(-1, settings.model.stacked_frame)

            # waiting for monte carlo to be working
            action = np.random.randint(env.action_space)

            new_obs, reward, terminal = env.step(action)

            game.observation_history.append(observation)
            game.actions_history.append(action)
            game.rewards_history.append(reward)

            if terminal:
                break

            observation = new_obs

if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)
    a = main(stg_obj)
