import numpy as np

class ReplayBuffer:
    def __init__(self, settings):
        self.buffer_size = int(eval(settings.buffer_size))
        self.batch_size = settings.batch_size
        self.unroll_steps = settings.unroll_steps
        self.td_steps = settings.td_steps

        self.buffer = []

    def save_episode(self, episode):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def get_batch(self):
        episodes = [self.get_episode() for _ in range(self.batch_size)]
        episode_sp = [(e, self.get_pos_in_ep(e)) for e in episodes]

        return [(g.make_image(i), g.history[i:i + num_unroll_steps], g.make_target(i, num_unroll_steps, td_steps, g.to_play())) for (g, i) in game_pos]


    def get_episode(self):
        ep_id = np.random.choice(len(self.buffer))
        return ep_id, self.buffer[ep_id]

    def get_pos_in_ep(self, ep_history):
        pos = np.choice(len(ep_history['reward']))
        return pos
