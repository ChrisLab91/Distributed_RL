import gym
import numpy as np
import random

def get_env(env_str):
  return gym.make(env_str)

import numpy as np
class spaces(object):
  discrete = 0
  box = 1


def get_space(space):
  if hasattr(space, 'n'):
    return space.n, spaces.discrete, None
  elif hasattr(space, 'shape'):
    return np.prod(space.shape), spaces.box, (space.low, space.high)


def get_spaces(spaces):
  if hasattr(spaces, 'spaces'):
    return zip(*[get_space(space) for space in spaces.spaces])
  else:
    return [(ret,) for ret in get_space(spaces)]


class GymWrapper(object):

  def __init__(self, env_str, count=1, seeds=None):

    self.count = count
    self.total = self.count
    self.seeds = seeds or [random.randint(0, 1e12)
                           for _ in range(self.total)]

    self.envs = []
    for seed in self.seeds:
        env = get_env(env_str)
        env.seed(seed)
        self.envs.append(env)

    self.dones = [True] * self.total
    self.num_episodes_played = 0

    one_env = self.get_one()
    self.use_action_list = hasattr(one_env.action_space, 'spaces')

    # figure out observation space
    self.obs_space = one_env.observation_space
    self.obs_dims, self.obs_types, self.obs_info = get_spaces(self.obs_space)

    # figure out action space
    self.act_space = one_env.action_space
    self.act_dims, self.act_types, self.act_info = get_spaces(self.act_space)

    #self.env_spec = env_spec.EnvSpec(self.get_one())

  def get_seeds(self):
    return self.seeds

  def reset(self):
    self.dones = [False] * self.total
    self.num_episodes_played += len(self.envs)

    # reset seeds to be synchronized
    self.seeds = [random.randint(0, 1e12) for _ in range(self.total)]

    for counter, seed in zip(range(self.total), self.seeds):
        self.envs[counter].seed(seed)

    return [self.convert_obs_to_list(env.reset())
            for env in self.envs]

  def reset_if(self, predicate=None):
    if predicate is None:
      predicate = self.dones
    if self.count != 1:
      assert np.all(predicate)
      return self.reset()
    self.num_episodes_played += sum(predicate)
    output = [self.env_spec.convert_obs_to_list(env.reset())
              if pred else None
              for env, pred in zip(self.envs, predicate)]
    for i, pred in enumerate(predicate):
      if pred:
        self.dones[i] = False
    return output

  def all_done(self):
    return all(self.dones)

  def step(self, actions):

    def env_step(action, env):
      obs, reward, done, tt = env.step(action)
      obs = self.convert_obs_to_list(obs)
      return obs, reward, done, tt

    #actions = zip(*actions)
    outputs = [env_step(action, env)
               if not done else (self.initial_obs(None), 0, True, None)
               for action, env, done in zip(actions, self.envs, self.dones)]
    for i, (_, _, done, _) in enumerate(outputs):
      self.dones[i] = self.dones[i] or done
    obs, reward, done, tt = zip(*outputs)
    #obs = [list(oo) for oo in zip(*obs)]
    return [obs, reward, done, tt]

  def get_one(self):
    return random.choice(self.envs)

  def __len__(self):
    return len(self.envs)

  def initial_obs(self, batch_size):
      batched = batch_size is not None
      batch_size = batch_size or 1

      obs = []
      for dim, typ in zip(self.obs_dims, self.obs_types):
          if typ == spaces.discrete:
              obs.append(np.zeros(batch_size))
          elif typ == spaces.box:
              obs.append(np.zeros([batch_size, dim]))

      if batched:
          return obs
      else:
          return obs[0]

  def convert_obs_to_list(self, obs):
      if len(self.obs_dims) == 1:
          return [obs]
      else:
          return list(obs)