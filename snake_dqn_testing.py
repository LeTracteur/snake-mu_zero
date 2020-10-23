# from old.environment import *
from environment import *
from agents import *
import os
import datetime



##############################
# Game and Training parameters
##############################
wall_size = 7

screen_width = 70 + 2*wall_size
screen_height = 70 + 2*wall_size
# reshape = (84, 84)
snake_size = 7
log_freq = 50
nb_episodes = 30000
steps = 2000

counter = 0

c2 = 500

###############
# global param
###############

reward_list = []
score_list = []
steps_list = []
t_r = 0
tot_reward = []

best_score = 0
eps_val = 0.0
###############

env = SnakeEnvironment(screen_width, screen_height, snake_size, wall_size)
agent = DQNagent(4, env.states_space.shape, 100000, 64)

agent.model_policy.summary()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/tensorboard/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

loss = 0
os.makedirs("model/best",exist_ok=True)
for ep in range(nb_episodes):
    if ep < 1000:
        cd = True
        while cd:
            env, epd, disp, step = simulate(env, agent, log_freq, ep, steps)
            cd = env.score < 2
    else:
        env, epd, disp, step = simulate(env, agent, log_freq, ep, steps)

    epd = propagate_reward(epd)
    save_ep(agent, epd)

    # TRAIN HERE
    if ep == 1000:
        for i in range(20000):
            loss += agent.optimize()
	
    if (ep > agent.batch_size) and (ep > 1000):
        loss = 0
        for i in range(50):
            loss += agent.optimize_per()

    if ep%log_freq == 0:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss/50, step = ep)
            tf.summary.scalar('steps', step, step = ep)
            tf.summary.scalar('reward', np.sum(epd['reward']), step = ep)
            tf.summary.scalar('score', env.score, step = ep)
            video = np.expand_dims(np.array(disp),0)
            video_summary('dummy_snake', video, step=ep)


    agent.update_weights()

    printProgressBar(ep * steps + step, steps * nb_episodes, agent.current_eps, np.sum(epd['reward']), env.score, '%.4f' % (loss/50), ep,
                         prefix="Progress:")


agent.plot_loss()

plt.plot(tot_reward)
plt.title('total reward evolution')
plt.ylabel('total reward')
plt.xlabel('episodes')
plt.show()

plt.plot(steps_list)
plt.title('steps per ep')
plt.ylabel('steps')
plt.xlabel('ep')
plt.show()

plt.plot(score_list)
plt.title('score evolution')
plt.ylabel('final score')
plt.xlabel('game')
plt.show()

agent.save()
