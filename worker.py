import numpy as np
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
import os
import threading
import multiprocessing
import tensorflow as tf
import random
import gym_wrapper
import util as U

from ac_network import AC_Network
from replay_buffer import ReplayBuffer


REWARD_FACTOR = 1


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Weighted random selection returns n_picks random indexes.
# the chance to pick the index i is give by the weight weights[i].
def weighted_pick(weights,n_picks):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t,np.random.rand(n_picks)*s)

# Discounting function used to calculate discounted returns.
def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Normalization of inputs and outputs
def norm(x, upper, lower=0.):
    return (x-lower)/max((upper-lower), 1e-12)

# Unpack given episodes that are saved into a dict
def unpack_episode(sampled_eps):

    # Get max len of rollout episodes to perform padding
    lens_seqs = [ep["path_length"] for ep in sampled_eps]
    max_len = max(lens_seqs)
    min_len = min(lens_seqs)


    states_start = [np.pad(ep["states"],
                           [(0, 0), (0, max_len - np.shape(ep["states"])[1]), (0, 0)], mode='constant')
                    for ep in sampled_eps]
    states_start = np.vstack(states_start)

    actions_start = [np.pad(ep["actions"],
                            [(0, 0), (0, max_len - np.shape(ep["actions"])[1]), (0, 0)], mode='constant')
                     for ep in sampled_eps]
    actions_start = np.vstack(actions_start)

    values_start = [np.pad(ep["values"], [(0, 0), (0, max_len - np.shape(ep["values"])[1])],
                           mode='constant')
                    for ep in sampled_eps]
    values_start = np.vstack(values_start)

    rewards_start = [np.pad(ep["rewards"], [(0, 0), (0, max_len - np.shape(ep["rewards"])[1])],
                            mode='constant')
                     for ep in sampled_eps]
    rewards_start = np.vstack(rewards_start)

    return actions_start, states_start, rewards_start, values_start, min_len, lens_seqs


class Worker():
    def __init__(self, name, s_size, a_size, network_config, learning_rate, global_episodes,
                 env_name, number_envs = 1,
                 tau = 0.0, rollout = 10, method = "A3C", update_learning_rate_ = True, preprocessing_config = None,
                 gae_lambda = 1.0):

        self.name = "worker_" + str(name)
        self.method = method
        print(self.method)
        self.number = name
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        # A3C specific setting -> GAE Lambda
        # https://arxiv.org/abs/1506.02438
        self.gae_lambda = gae_lambda

        # Going to be memory buffer in case we are using PCL
        if self.method == "PCL":
            self.replay_buffer = ReplayBuffer()

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, network_config, learning_rate= learning_rate,
                                   tau = tau, rollout= rollout, method= self.method)
        self.update_local_ops = update_target_graph('global', self.name)

        # Set gym environment
        self.env = gym_wrapper.GymWrapper(env_name, count = number_envs)
        self.a_size = a_size
        self.s_size = s_size

        if preprocessing_config is not None:
            self.preprocessing_state = True
            self.preprocessing_config = preprocessing_config
        else:
            self.preprocessing_state = False
            self.preprocessing_config = None

        # Get information if RNN is used
        if "RNN" in network_config["shared_config"]["kind"]:
            self.rnn_network = True
        else:
            self.rnn_network = False

        # Get Noisy Net Information if applied
        self.noisy_policy = network_config["policy_config"]["noise_dist"]
        self.noisy_value = network_config["value_config"]["noise_dist"]

        # Desired KL-Divergence to update learning rate
        self.desired_kl = 0.002
        self.update_learning_rate_ = update_learning_rate_

    def train(self, states, rewards, actions, values, terminal, sess, gamma, r, merged_summary):

        # Get length of different rollouts --> Since given e.g. 10 envs maybe 5 terminated earlier
        lengths_rollouts = [int(-1 * sum(done - 1)) for done in terminal]

        # Get batch size
        batch_size = len(states)

        # Resize final values in case we have only one env
        r = np.reshape(r, [batch_size])

        # Get max_len
        max_len = len(states[0])
        for i in range(len(lengths_rollouts)):
            if lengths_rollouts[i] != max_len and lengths_rollouts[i] != 0:
                lengths_rollouts[i] = lengths_rollouts[i] + 1

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = []
        for i in range(batch_size):
            if not self.env.dones[i]:
                rewards_list.append(np.asarray(rewards[i].tolist()[:lengths_rollouts[i]] + [r[i]]) * REWARD_FACTOR)
            else:
                rewards_list.append(np.asarray(rewards[i].tolist()[:lengths_rollouts[i]] + [0]) * REWARD_FACTOR)
        discounted_rewards = [discounting(rewards_list[i], gamma)[:-1] for i in range(batch_size)]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = []
        # Check if episodes have been terminated, if so add 0 in the end otherwise the last value
        for i in range(batch_size):
            if not self.env.dones[i]:
                values_list.append(np.asarray(values[i].tolist()[:lengths_rollouts[i]] + [r[i]]) * REWARD_FACTOR)
            else:
                values_list.append(np.asarray(values[i].tolist()[:lengths_rollouts[i]] + [0]) * REWARD_FACTOR)

        # Compute TD residual of V with discount gamma --> can be considered as the advantage of the action a_t
        advantages = [rewards[i][:lengths_rollouts[i]] + gamma * values_list[i][1:] - values_list[i][:-1]
                      for i in range(batch_size)]
        discounted_advantages = [discounting(advantages[i], gamma *  self.gae_lambda)
                                 for i in range(batch_size)]

        # Since discounted_rewards and discounted_advantages have different lengths for all episodes
        # they need to be zero padded
        padded_discounted_advantages = [np.pad(discounted_advantages[i], [(0, max_len - lengths_rollouts[i])], mode="constant")
                                        for i in range(batch_size)]
        padded_discounted_advantages = np.stack(padded_discounted_advantages)

        # Test with classic future reward
        padded_discounted_rewards = [np.pad(discounted_rewards[i], [(0, max_len - lengths_rollouts[i])], mode="constant")
                                     for i in range(batch_size)]
        padded_discounted_rewards = np.stack(padded_discounted_rewards)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save

        feed_dict = {self.local_AC.target_v: padded_discounted_rewards,
                     self.local_AC.inputs: states,
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: padded_discounted_advantages,
                     self.local_AC.rewards: rewards,
                     self.local_AC.lengths_episodes: lengths_rollouts}

        if self.rnn_network:
            # Set initial rnn state based on number of episodes
            c_init = np.zeros((batch_size, self.local_AC.cell_units), np.float32)
            h_init = np.zeros((batch_size, self.local_AC.cell_units), np.float32)
            rnn_state = np.array([c_init, h_init])
            feed_dict[self.local_AC.state_in[0]] = rnn_state[0]
            feed_dict[self.local_AC.state_in[1]] = rnn_state[1]

        summary = None
        if self.name == "worker_0":
            summary, v_l, p_l, e_l, g_n, v_n, _, logits, val = sess.run([merged_summary, self.local_AC.value_loss,
                                                                   self.local_AC.policy_loss,
                                                                   self.local_AC.entropy,
                                                                   self.local_AC.grad_norms,
                                                                   self.local_AC.var_norms,
                                                                   self.local_AC.apply_grads,
                                                                   self.local_AC.policy,
                                                                   self.local_AC.value],
                                                                  feed_dict=feed_dict)
            """
            print(np.shape(states))
            print(np.count_nonzero(np.sum(states[0], 1)))
            print(np.count_nonzero(val[0]))
            print(terminal[0])
            print(sum(-1 * (terminal[0]-1)))
            print(len(discounted_rewards[0]))
            print(lengths_rollouts[0])
            print(np.shape(val))
            print(np.shape(discounted_rewards))
            """
        else:
            v_l, p_l, e_l, g_n, v_n, _, logits = sess.run([self.local_AC.value_loss,
                                                            self.local_AC.policy_loss,
                                                            self.local_AC.entropy,
                                                            self.local_AC.grad_norms,
                                                            self.local_AC.var_norms,
                                                            self.local_AC.apply_grads,
                                                            self.local_AC.policy],
                                                           feed_dict=feed_dict)

        return v_l , p_l , e_l , g_n, v_n, summary, logits

    # Training operations PCL
    def train_pcl(self, episodes, gamma, sess, merged_summary, rollout):

        # Train on sampled episodes
        a_ep, s_ep, r_ep, v_ep, min_len, lens_seqs = unpack_episode(episodes)

        """
        rollout = self.local_AC.rollout_pcl
        while rollout > min_len:
            rollout = int(rollout / 2)
        """

        # Get required discounting multipliers
        discount = np.array([gamma ** i for i in range(rollout)], dtype=np.float32)

        feed_dict_ = {
            self.local_AC.inputs: s_ep,
            self.local_AC.actions: a_ep,
            self.local_AC.rewards: r_ep,
            self.local_AC.discount: discount,
            self.local_AC.rollout: rollout,
            self.local_AC.lengths_episodes: lens_seqs
        }

        if self.rnn_network:
            # Set rnn_state based on the amount of episodes
            # Dynamic state initialization
            batch_size = len(episodes)
            c_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))
            h_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))
            feed_dict_[self.local_AC.state_in[0]] = c_init
            feed_dict_[self.local_AC.state_in[1]] = h_init

        summary = None
        if self.name == "worker_0":
            # Perform training on one episode batch
            summary, v_l, p_l, total_loss, _, _, logits = sess.run([merged_summary, self.local_AC.value_loss,
                                                         self.local_AC.policy_loss,
                                                         self.local_AC.loss,
                                                         self.local_AC.apply_grads_pol,
                                                         self.local_AC.apply_grads_val,
                                                         self.local_AC.policy],
                                                        feed_dict=feed_dict_)
        else:
            # Perform training on one episode batch
            v_l, p_l, total_loss, _, _, logits = sess.run([self.local_AC.value_loss,
                                                self.local_AC.policy_loss,
                                                self.local_AC.loss,
                                                self.local_AC.apply_grads_pol,
                                                self.local_AC.apply_grads_val,
                                                self.local_AC.policy],
                                               feed_dict=feed_dict_)

        return r_ep, v_ep, summary, logits

    # Calculate KL Divergence in order to update the learning rate
    def calculate_kl_divergence(self, old_logits, states, sess, dones=None):

        if self.method == "A3C":
            batch_size = len(states)
            lens_seqs = [int(-1 * sum(done - 1)) for done in dones]
            s_ep = states
        elif self.method == "PCL":
            batch_size = len(states)
            # Get states to obtain logits of updated policy
            _, s_ep, _, _, _, lens_seqs = unpack_episode(states)

        # Set rnn_state based on the amount of episodes
        # Dynamic state initialization

        feed_dict_ = {
            self.local_AC.oldPolicy: old_logits,
            self.local_AC.inputs: s_ep,
            self.local_AC.lengths_episodes: lens_seqs
        }

        if self.rnn_network:
            c_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))
            h_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))

            feed_dict_[self.local_AC.state_in[0]] = c_init
            feed_dict_[self.local_AC.state_in[1]] = h_init

        # Run Tensorgraph with old and new logits as input in order to compute the KL-Divergence
        kl_divergence = sess.run(self.local_AC.kl_divergence, feed_dict_)

        return kl_divergence

    # Function to update learning rate based on KL-Divergence
    def update_learning_rate(self, kl_divergence, sess):

        max_lr = 0.1
        min_lr = 0.000001

        act_lr = sess.run(self.local_AC.lr)
        if kl_divergence < self.desired_kl / 4:
            new_lr = min(max_lr, act_lr * 1.5)
            sess.run(self.local_AC.lr_update, feed_dict={self.local_AC.new_learning_rate: new_lr})
            #print(sess.run(self.local_AC.lr))
        elif kl_divergence > self.desired_kl * 4:
            new_lr = max(min_lr, act_lr / 1.5)
            sess.run(self.local_AC.lr_update, feed_dict={self.local_AC.new_learning_rate: new_lr})
            #print(sess.run(self.local_AC.lr))


    # Act on current policy
    def act(self, state, rnn_state, lens_seqs, sess):

        feed_dict_ = {self.local_AC.inputs: state,
                      self.local_AC.lengths_episodes: lens_seqs}

        if self.rnn_network:
            # If rnn (LSTM) cell is used supply current internal states
            feed_dict_[self.local_AC.state_in[0]] = rnn_state[0]
            feed_dict_[self.local_AC.state_in[1]] = rnn_state[1]
            action_dist, value, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                 feed_dict_)

        else:
            action_dist, value = sess.run(
                [self.local_AC.policy, self.local_AC.value],
                feed_dict_)
            rnn_state = None

        # Chose action with largest probability
        a0 = [weighted_pick(a,1) for a in action_dist]  # Use stochastic distribution sampling
        action_one_hot = np.eye(self.a_size, dtype=np.int32)[np.array(a0)]

        return action_one_hot, value, rnn_state, action_dist

    def getRewards(self):
        return self.episode_rewards

    # Add sampled states, rewards, actions, values and terminal information
    # to current A3C minibatch
    def add_to_batch(self, states, rewards, actions, values, terminals):

        self.episode_states_train = np.concatenate((self.episode_states_train, states), 1)
        self.episode_reward_train = np.concatenate((self.episode_reward_train, np.expand_dims(rewards, 1)), 1)
        self.episode_actions_train = np.concatenate((self.episode_actions_train, actions), 1)
        self.episode_values_train = np.concatenate((self.episode_values_train, np.reshape(values, [np.shape(values)[0], 1])), 1)
        self.episode_done_train = np.concatenate((self.episode_done_train, np.expand_dims(terminals, 1)), 1)

    # Clear current A3C minibatch
    def reset_batch(self):

        self.episode_states_train = np.array([], dtype=np.float32).reshape(len(self.env), 0, self.s_size)
        self.episode_reward_train = np.array([], dtype=np.float32).reshape(len(self.env), 0)
        self.episode_actions_train = np.array([], dtype=np.float32).reshape(len(self.env), 0, self.a_size)
        self.episode_values_train = np.array([], dtype=np.float32).reshape(len(self.env), 0)
        self.episode_done_train = np.array([], dtype=np.float32).reshape(len(self.env), 0)


    def rolloutPCL(self, sess, initial_state, rnn_state_init, max_path_length=None, episode_count = 1):

        # ToDo: Do not loop over "episode_count" but perform only one sess.run per step
        # Perform rollout of given environment
        if max_path_length is None:
            max_path_length = self.env.envs[0].spec.tags.get(
                'wrapper_config.TimeLimit.max_episode_steps')

        # Reset rnn_state for every iteration
        s = initial_state
        rnn_state = rnn_state_init
        path_length = np.zeros(len(self.env))

        # Sample one episode
        while all(path_length) < max_path_length and not self.env.all_done():
            dummy_lengths = np.ones(len(self.env))
            a, v, rnn_state, _ = self.act(s, rnn_state, dummy_lengths, sess)
            # Get action for every environment
            act_ = [np.argmax(a_) for a_ in a]
            # Sample new state and reward from environment
            s2, r, terminal, info = self.env.step(act_)
            if self.preprocessing_state:
                s2 = U.process_frame(s2, self.preprocessing_config)
            # Add states, rewards, actions, values and terminal information to PCL episode batch
            self.add_to_batch(s, r, a, v, terminal)
            for i in range(len(self.env)):
                if not self.env.dones[i]:
                    path_length[i] = path_length[i] + 1
            s=s2

        episodes = []
        for i in range(len(self.env)):
            path_length_temp = int(path_length[i]) + 1
            episodes.append(dict(
                states=np.expand_dims(self.episode_states_train[i][:path_length_temp],0),
                actions=np.expand_dims(self.episode_actions_train[i][:path_length_temp],0),
                rewards=np.expand_dims(self.episode_reward_train[i][:path_length_temp], 0),
                values=np.expand_dims(self.episode_values_train[i][:path_length_temp], 0),
                path_length=path_length_temp
            ))


        return episodes