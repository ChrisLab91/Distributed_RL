import numpy as np

# Based on https://arxiv.org/pdf/1511.05952.pdf
# "Prioritized Experience Replay"

class ReplayBuffer(object):
    def __init__(self, max_len=100, alpha=0.2):
        self.max_len = max_len
        self.alpha = alpha
        self.buffer = {}
        # weight is not normalized
        self.priorities = np.zeros(self.max_len)
        self.cur_size = 0

    def add(self, episodes):
        new_eps = 0
        new_index = []
        while self.cur_size < self.max_len and new_eps < len(episodes):
            self.buffer[self.cur_size] = episodes[new_eps]
            new_index.append(self.cur_size)
            new_eps += 1
            self.cur_size += 1

        if len(episodes) > new_eps:
            remove_index = self.remove_n(len(episodes) - new_eps)
            for remove_ind in remove_index:
                self.buffer[remove_ind] = episodes[new_eps]
                new_index.append(remove_ind)
                new_eps += 1

        priorities_reward = [np.sum(ep["rewards"]) for ep in episodes]
        self.priorities[new_index] = priorities_reward

    # Sample distribution retrieve episodes from the replay buffer
    def sampling_distribution(self):
        p = self.priorities[:self.cur_size]
        p = np.exp(self.alpha * (p - np.max(p)))
        norm = np.sum(p)
        if norm > 0:
            uniform = 0.0
            p = p / norm * (1 - uniform) + 1.0 / self.cur_size * uniform
        else:
            p = np.ones(self.cur_size) / self.cur_size

        return p

    def sample(self, episode_count = 1):
        # Get relevant probabilitie weights
        p = self.sampling_distribution()
        if len(self.buffer) < episode_count:
            episode_count = 1
        # Draw episodes from replay buffer
        idxs = np.random.choice(self.cur_size, size = int(episode_count), replace= False, p=p)
        sampled_episodes = []
        return [self.buffer[idx] for idx in idxs]

    def remove_n(self, n):
        # remove lowest-priority indices
        idxs = np.argpartition(self.priorities, n)[:n]
        return idxs

    @property
    def trainable(self):
        if len(self.buffer) > 32:
            return True
        else:
            return False