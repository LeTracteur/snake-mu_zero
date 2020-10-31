import utils
import argparse
import sys
from environment_new import *
from game import Game
from replay_buffer import ReplayBuffer
import muzero_model
from muzero_mcts import MCTS, select_action, select_temperature
import datetime
import tensorflow as tf


def main(stg):
    settings = utils.Obj(stg)
    env = SnakeEnv(settings.env)
    nb_episodes = settings.nb_episodes
    steps = settings.steps
    train_for = settings.train_for
    train_every = settings.train_every
    replay_buffer = ReplayBuffer(settings.buffer)

    agent = muzero_model.MuZero(settings.model)
    agent.build()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/tensorboard/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for ep in range(nb_episodes):
        game = Game(settings.game, env.action_space)
        observation = env.reset()
        temperature = select_temperature(ep)

        for step in range(steps):
            # visual rendering
            # env.render()

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

        if ep%train_every == 0:
            value_loss, reward_loss, policy_loss, total_loss = 0, 0, 0, 0
            for _ in range(train_for):
                data_batch = replay_buffer.get_batch(agent)
                v_loss, r_loss, p_loss, t_loss = agent.train(data_batch)
                value_loss += v_loss
                reward_loss += r_loss
                policy_loss += p_loss
                total_loss += t_loss

            with train_summary_writer.as_default():
                tf.summary.scalar('value_loss', value_loss / train_for, step=ep)
                tf.summary.scalar('reward_loss', reward_loss / train_for, step=ep)
                tf.summary.scalar('policy_loss', policy_loss / train_for, step=ep)
                tf.summary.scalar('total_loss', total_loss / train_for, step=ep)
                tf.summary.scalar('steps', step, step=ep)
                tf.summary.scalar('reward', np.sum(game.rewards_history), step=ep)
                tf.summary.scalar('score', env.score, step=ep)
                env.draw_game(game, 'train_gif', ep)

    return replay_buffer, agent


if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)

    rp, a = main(stg_obj)
