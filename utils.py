import pygame
import numpy as np
import random
from copy import deepcopy
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import os
import yaml


#########
# COLORS
#########

GREEN = (0, 255, 0)
HEAD_YELLOW = (255, 255, 0)
BODY_YELLOW = (255, 255, 102)
WALL = (255, 0, 0)
BLACK = (0, 0, 0)

GREEN_ID = 3
HEAD_YELLOW_ID = 1
BODY_YELLOW_ID = 2
WALL_ID = 4
BLACK_ID = 0

#############################


class Obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [Obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, Obj(b) if isinstance(b, dict) else b)


def load_stgs(path_to_setting):
    assert os.path.exists(path_to_setting), "Path not found"
    with open(path_to_setting) as file:
        stg = yaml.load(file, Loader=yaml.FullLoader)
    return stg


def display_snake(screen, head_color, body_color, snake_vol, snake_list):
    pygame.draw.rect(screen, head_color, [snake_list[-1][0], snake_list[-1][1], snake_vol, snake_vol])
    for b in snake_list[:-1]:
        pygame.draw.rect(screen, body_color, [b[0], b[1], snake_vol, snake_vol])


def add_food(possible_x, possible_y, snake_list):
    c_possible_x, c_possible_y = deepcopy(possible_x), deepcopy(possible_y)
    all_compo = []
    for x in c_possible_x:
        for y in c_possible_y:
            all_compo.append((x, y))
    for b in snake_list:
        tuple_c = (b[0], b[1])
        if tuple_c in all_compo:
            all_compo.remove(tuple_c)
    pos_x, pos_y = random.choice(all_compo)
    return pos_x, pos_y


def printProgressBar(iteration, total, eps, tt_reward, score, loss, ep, prefix='', suffix='', decimals=1, length=100,
                     fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    eps = round(eps, 3)
    print('\r%s |%s| %s%% %s | loss %s | eps %s | total reward %s | ep %s | score %s' % (
    prefix, bar, percent, suffix, loss, eps, tt_reward, ep, score), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def scale_lumininance(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


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
    # try:
    frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    summary = tf1.Summary()
    image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps)
    summary.value.add(tag=name + '/gif', image=image)
    tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    # print('GIF summaries require ffmpeg in $PATH.', e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name + '/grid', frames, step)


def image_to_color(grid):
    img = np.zeros((grid.shape[0], grid.shape[1], 3))
    img[grid == GREEN_ID] = np.array(GREEN)
    img[grid == HEAD_YELLOW_ID] = np.array(HEAD_YELLOW)
    img[grid == BODY_YELLOW_ID] = np.array(BODY_YELLOW)
    img[grid == BLACK_ID] = np.array(BLACK)
    img[grid == WALL_ID] = np.array(WALL)
    return img


def init_ep():
    epd = {}
    epd['obs'] = []
    epd['reward'] = []
    epd['action'] = []
    epd['term'] = []
    return epd


def update_ep(epd, state, reward, action, terminal):
    epd['obs'].append(state)
    epd['reward'].append(reward)
    epd['action'].append(action)
    epd['term'].append(terminal)
    return epd


def simulate(env, agent, log_freq, ep, steps):
    state = env.reset()
    disp = []
    for _ in range(agent.tau):
        agent.state_buffer.append(state)
        agent.next_state_buffer.append(state)
    agent.update_epsilon(0.9999, 1000.0, 4000.0, ep, 1.0, 0.5, 1.0)
    epd = init_ep()
    for step in range(steps):
        if ep % log_freq == 0:
            disp.append(image_to_color(env.board_status))
        action = agent.act(state, env.snake_list)
        new_state, reward, terminal = env.step(action)
        epd = update_ep(epd, state, reward, action, terminal)
        agent.next_state_buffer.append(new_state)

        if terminal:
            #    score_list.append(env.score)
            #    steps_list.append(step)
            break

        state = new_state
    return env, epd, disp, step


def propagate_reward(epd):
    r = np.array(epd['reward'])
    idxs = np.argwhere(r > 0)
    for idx in idxs:
        idx = idx[0]
        for i in range(1, 10):
            if idx - i >= 0:
                r[idx - i] += r[idx] * 1. / (2 ** i)
            else:
                break
    epd['reward'] = r
    return epd


def save_ep(agent, epd):
    # TO FUNC REBUILD DATA
    for i in range(len(epd['obs']) - 1):
        if i == 0:
            for j in range(agent.tau):
                agent.state_buffer.append(epd['obs'][i])
                if j == (agent.tau - 1):
                    agent.next_state_buffer.append(epd['obs'][i + 1])
                else:
                    agent.next_state_buffer.append(epd['obs'][i])
        else:
            agent.state_buffer.append(epd['obs'][i])
            agent.next_state_buffer.append(epd['obs'][i + 1])
        if epd['reward'][i] != 0:
            agent.add_to_memory(deepcopy(agent.state_buffer), epd['action'][i], epd['reward'][i],
                                deepcopy(agent.next_state_buffer), epd['term'][i])
