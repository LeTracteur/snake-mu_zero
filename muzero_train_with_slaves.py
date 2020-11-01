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

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    agent = muzero_model.MuZero(settings.model)
    agent.build()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/tensorboard/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    while True:
        replay_buffer.load_games_from_folder()
        ep = len(replay_buffer.buffer)
        value_loss, reward_loss, policy_loss, total_loss = 0, 0, 0, 0
        for _ in range(train_for):
            data_batch = replay_buffer.get_batch(agent)
            v_loss, r_loss, p_loss, t_loss = agent.train(data_batch)
            value_loss += v_loss
            reward_loss += r_loss
            policy_loss += p_loss
            total_loss += t_loss

        agent.save_weights()

        with train_summary_writer.as_default():
            tf.summary.scalar('value_loss', value_loss / train_for, step=ep)
            tf.summary.scalar('reward_loss', reward_loss / train_for, step=ep)
            tf.summary.scalar('policy_loss', policy_loss / train_for, step=ep)
            tf.summary.scalar('total_loss', total_loss / train_for, step=ep)
            #tf.summary.scalar('steps', step, step=ep)
            #tf.summary.scalar('reward', np.sum(game.rewards_history), step=ep)
            #tf.summary.scalar('score', env.score, step=ep)
            #env.draw_game(game, 'train_gif', ep)


if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)

    rp, a = main(stg_obj)
