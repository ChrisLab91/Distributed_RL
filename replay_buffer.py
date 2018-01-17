import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_len=10000, alpha=1):
        self.max_len = max_len
        self.alpha = alpha
        self.buffer = []
        # weight is not normalized
        self.weight = np.array([])

    def add(self, episodes):
        for index in range(len(episodes)):
            episode = episodes[index]
            self.buffer.append(episode)
            self.weight = np.append(self.weight, np.exp(self.alpha*episode['rewards'].sum()))
            if len(self.buffer) > self.max_len:
                delete_ind = np.random.randint(len(self.buffer))
                del self.buffer[delete_ind]
                self.weight = np.delete(self.weight, delete_ind)

    def sample(self, episode_count = 1):
        sampled_episodes = []
        if len(self.buffer) < episode_count:
            episode_count = 1
        for i in range(episode_count):
            sampled_episodes.append(np.random.choice(self.buffer, p=self.weight/self.weight.sum()))
        return sampled_episodes

    @property
    def trainable(self):
        if len(self.buffer) > 32:
            return True
        else:
            return False