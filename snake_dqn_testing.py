# from old.environment import *
from environment import *
from agents import *
import os
import datetime

import tensorflow.compat.v1 as tf1


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out

def video_summary(name, video, step=None, fps=5):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    #try:
    frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    summary = tf1.Summary()
    image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps)
    summary.value.add(tag=name + '/gif', image=image)
    tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    #print('GIF summaries require ffmpeg in $PATH.', e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name + '/grid', frames, step)

def image_to_color(grid):
    img =  np.zeros((grid.shape[0], grid.shape[1],3))
    img[grid == GREEN_ID] = np.array(GREEN)
    img[grid == HEAD_YELLOW_ID] = np.array(HEAD_YELLOW)
    img[grid == BODY_YELLOW_ID] = np.array(BODY_YELLOW)
    img[grid == BLACK_ID] = np.array(BLACK)
    img[grid == WALL_ID] = np.array(WALL)
    return img
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
#test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# state = env.observation()
# agent.next_state_buffer.append(state)
loss = 0
os.makedirs("model/best",exist_ok=True)
for ep in range(nb_episodes):
    # GENERATES 1000 HIGH VALUE EPISODES
    if ep < 1000:
        cd = True
        while cd:
            state = env.reset()
            disp = []
            for _ in range(agent.tau):
                agent.state_buffer.append(state)
                agent.next_state_buffer.append(state)
            agent.update_epsilon(0.9999, 1000.0, 4000.0, ep, 1.0, 0.5, 1.0)

            epd = {}
            epd['obs'] = []
            epd['reward'] = []
            epd['action'] = []
            epd['term'] = []
            for step in range(steps):
                if ep%log_freq == 0:
                    disp.append(image_to_color(env.board_status))
                action = agent.act(state, env.snake_list)
                new_state, reward, terminal = env.step(action)
                epd['obs'].append(state)
                epd['reward'].append(reward)
                epd['action'].append(action)
                epd['term'].append(terminal)

                agent.next_state_buffer.append(new_state)

                if terminal:
                    score_list.append(env.score)
                    steps_list.append(step)
                    break

                state = new_state
            cd = env.score < 2
    else:
        state = env.reset()
        disp = []
        for _ in range(agent.tau):
            agent.state_buffer.append(state)
            agent.next_state_buffer.append(state)
        agent.update_epsilon(0.9995, 1000.0, 4000.0, ep, 1.0, 0.5, 1.0)
        epd = {}
        epd['obs'] = []
        epd['reward'] = []
        epd['action'] = []
        epd['term'] = []
        
        for step in range(steps):
            if ep%log_freq == 0:
                disp.append(image_to_color(env.board_status))
            action = agent.act(state, env.snake_list)
            new_state, reward, terminal = env.step(action)
            epd['obs'].append(state)
            epd['reward'].append(reward)
            epd['action'].append(action)
            epd['term'].append(terminal)

            agent.next_state_buffer.append(new_state)

            if terminal:
                score_list.append(env.score)
                steps_list.append(step)
                break

            state = new_state

    # TO FUNC LAMBDA RETURN UGLY
    r = np.array(epd['reward'])
    idxs = np.argwhere(r > 0)
    for idx in idxs:
        idx = idx[0]
        for i in range(1,10):
            if idx-i >= 0:
                r[idx-i] += r[idx]*1./(2**i)
            else:
                break
    epd['reward'] = r
    # TO FUNC REBUILD DATA
    for i in range(len(epd['obs'])-1):
        if i == 0:
            for j in range(agent.tau):
                agent.state_buffer.append(epd['obs'][i])
                if j == (agent.tau-1):
                    agent.next_state_buffer.append(epd['obs'][i+1])
                else:
                    agent.next_state_buffer.append(epd['obs'][i])
        else:
            agent.state_buffer.append(epd['obs'][i])
            agent.next_state_buffer.append(epd['obs'][i+1])
        if epd['reward'][i] != 0:
            agent.add_to_memory(deepcopy(agent.state_buffer), epd['action'][i], epd['reward'][i], deepcopy(agent.next_state_buffer), epd['term'][i])

    # TRAIN HERE
    if ep == 1000:
        for i in range(20000):
            loss += agent.optimize()
	
    if (ep > agent.batch_size) and (ep > 1000):
        loss = 0
        for i in range(50):
            loss += agent.optimize()

    if ep%log_freq == 0:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss/50, step = ep)
            tf.summary.scalar('steps', step, step = ep)
            tf.summary.scalar('reward', np.sum(r), step = ep)
            tf.summary.scalar('score', env.score, step = ep)
            video = np.expand_dims(np.array(disp),0)
            video_summary('dummy_snake', video, step=ep)


    agent.update_weights()

    #if env.score > best_score:
    #    best_score = env.score
    #    agent.save('best')
    #    eps_val = agent.current_eps
    printProgressBar(ep * steps + step, steps * nb_episodes, agent.current_eps, np.sum(r), env.score, '%.4f' % (loss/50), ep,
                         prefix="Progress:")


agent.plot_loss()

plt.plot(tot_reward)
plt.title('total reward evolution')
plt.ylabel('total reward')
plt.xlabel('episodes')
plt.show()

# plt.plot(reward_list)
# plt.title('episode reward evolution')
# plt.ylabel('reward per ep')
# plt.show()

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
