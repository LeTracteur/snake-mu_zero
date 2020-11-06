import utils
import argparse
import sys
from environment_new import *
from game import Game
from replay_buffer import ReplayBuffer
import muzero_model
#from muzero_mcts import MCTS, select_action, select_temperature
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
    #agent.load_weights()
    try:
        agent.load_weights()
    except:
        pass

    last_ep = 0
    replay_buffer.load_games_from_folder(load_from_seen = True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/tensorboard/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    while True:
        replay_buffer.load_games_from_folder()
        ep = len(replay_buffer.buffer)
        step, reward, obs_hist = replay_buffer.last_n_games_stat(10)
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
            tf.summary.scalar('steps', step, step=ep)
            tf.summary.scalar('reward', reward, step=ep)
            tf.summary.scalar('training_steps', agent.training_step, step=ep)
            env.draw_game(obs_hist, 'train_gif', ep)

        if ep-last_ep > 25:
            last_ep = ep
            print('ep:'+str(ep)+', value_loss:'+str(value_loss.numpy().item()/train_for)+', reward_loss:'+str(reward_loss.numpy().item()/train_for)+', policy_loss:'+str(policy_loss.numpy().item()/train_for)+', total_loss:'+str(total_loss.numpy().item()/train_for)+', steps:'+str(step)+', reward:'+str(reward)+', training_steps:'+str(agent.training_step)+'.')



if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)
    main(stg_obj)
