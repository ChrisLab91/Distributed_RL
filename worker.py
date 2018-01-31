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

#WORKER
# PARAMETERS
# Clipping ratio for gradients

# Size of mini batches to run training on
MINI_BATCH = 40
REWARD_FACTOR = 0.001
EPISODE_RUNS = 1000

# Gym environment
ENV_NAME = 'CartPole-v0'  # Discrete (4, 2)

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
                 tau = 0.0, rollout = 10, method = "A3C", update_learning_rate_ = True, preprocessing_state = False,
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
        self.preprocessing_state = preprocessing_state

        # Get Noisy Net Information if applied
        self.noisy_policy = network_config["policy_config"]["noise_dist"]
        self.policy_layers = len(network_config["policy_config"]["layers"])
        self.noisy_value = network_config["value_config"]["noise_dist"]
        self.value_layers = len(network_config["value_config"]["layers"])

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

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = [np.asarray(rewards[i].tolist()[:lengths_rollouts[i]]) * REWARD_FACTOR for i in range(batch_size)]
        discounted_rewards = [discounting(rewards_list[i], gamma) for i in range(batch_size)]

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

        padded_discounted_rewards = [np.pad(discounted_rewards[i], [(0, max_len - lengths_rollouts[i])], mode="constant")
                                     for i in range(batch_size)]
        padded_discounted_rewards = np.stack(padded_discounted_rewards)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)
        # Set initial rnn state based on number of episodes
        c_init = np.zeros((batch_size, self.local_AC.cell_units), np.float32)
        h_init = np.zeros((batch_size, self.local_AC.cell_units), np.float32)
        rnn_state = np.array([c_init, h_init])

        feed_dict = {self.local_AC.target_v: padded_discounted_rewards,
                     self.local_AC.inputs: states,
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: padded_discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1],
                     self.local_AC.rewards: rewards,
                     self.local_AC.lengths_episodes: lengths_rollouts}

        summary = None
        if self.name == "worker_0":
            summary, v_l, p_l, e_l, g_n, v_n, _, logits = sess.run([merged_summary, self.local_AC.value_loss,
                                                                   self.local_AC.policy_loss,
                                                                   self.local_AC.entropy,
                                                                   self.local_AC.grad_norms,
                                                                   self.local_AC.var_norms,
                                                                   self.local_AC.apply_grads,
                                                                   self.local_AC.policy],
                                                                  feed_dict=feed_dict)
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
    def train_pcl(self, episodes, gamma, sess, merged_summary):

        # Train on sampled episodes

        a_ep, s_ep, r_ep, v_ep, min_len, lens_seqs = unpack_episode(episodes)

        rollout = self.local_AC.rollout_pcl
        while rollout > min_len:
            rollout = int(rollout / 2)

        discount = np.array([gamma ** i for i in range(rollout)], dtype=np.float32)

        # Set rnn_state based on the amount of episodes
        # Dynamic state initialization
        batch_size = len(episodes)
        c_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))
        h_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))

        # Perform padding based on the longest episode
        # Elements to be --> function pad_episodes given a list of sampled episodes

        summary = None
        if self.name == "worker_0":
            # Perform training on one episode batch
            feed_dict_ = {
                self.local_AC.inputs: s_ep,
                self.local_AC.actions: a_ep,
                self.local_AC.rewards: r_ep,
                self.local_AC.discount: discount,
                self.local_AC.rollout: rollout,
                self.local_AC.state_in[0]: c_init,
                self.local_AC.state_in[1]: h_init,
                self.local_AC.lengths_episodes: lens_seqs
            }
            summary, v_l, p_l, total_loss, _, _, logits = sess.run([merged_summary, self.local_AC.value_loss,
                                                         self.local_AC.policy_loss,
                                                         self.local_AC.loss,
                                                         self.local_AC.apply_grads_pol,
                                                         self.local_AC.apply_grads_val,
                                                         self.local_AC.policy],
                                                        feed_dict=feed_dict_)
        else:
            # Perform training on one episode batch
            feed_dict_ = {
                self.local_AC.inputs: s_ep,
                self.local_AC.actions: a_ep,
                self.local_AC.rewards: r_ep,
                self.local_AC.discount: discount,
                self.local_AC.rollout: rollout,
                self.local_AC.state_in[0]: c_init,
                self.local_AC.state_in[1]: h_init,
                self.local_AC.lengths_episodes: lens_seqs
            }

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
        c_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))
        h_init = np.zeros(shape=(batch_size, self.local_AC.cell_units))

        feed_dict_ = {
            self.local_AC.oldPolicy: old_logits,
            self.local_AC.inputs: s_ep,
            self.local_AC.state_in[0]: c_init,
            self.local_AC.state_in[1]: h_init,
            self.local_AC.lengths_episodes: lens_seqs
        }

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


        action_dist, value, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                        feed_dict={self.local_AC.inputs: state,
                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                   self.local_AC.state_in[1]: rnn_state[1],
                                                   self.local_AC.lengths_episodes: lens_seqs})

        # Chose action with largest probability
        a0 = [weighted_pick(a,1) for a in action_dist]  # Use stochastic distribution sampling
        action_one_hot = np.eye(self.a_size, dtype=np.int32)[np.array(a0)]

        return action_one_hot, value, rnn_state, action_dist

    def work(self, gamma, sess, coord, merged_summary, writer_summary):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        train_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                self.episode_values = []
                self.episode_reward = []

                #if self.method == "A3C":
                # Objects to hold the bacth used to update the Agent
                self.episode_states_train = np.array([], dtype=np.float32).reshape(len(self.env),0,self.s_size)
                self.episode_reward_train = np.array([], dtype=np.float32).reshape(len(self.env),0)
                self.episode_actions_train = np.array([], dtype=np.float32).reshape(len(self.env),0, self.a_size)
                self.episode_values_train = np.array([], dtype=np.float32).reshape(len(self.env),0)
                self.episode_done_train = np.array([], dtype=np.float32).reshape(len(self.env), 0)

                # Used by PCL
                # Hold reward and value function mean value of sampled episodes from replay buffer
                episode_reward_offline = 0
                episode_value_offline = 0
                episode_step_count = 0

                # Restart environment
                s = self.env.reset()

                # Set initial rnn state based on number of episodes
                c_init = np.zeros((len(self.env), self.local_AC.cell_units), np.float32)
                h_init = np.zeros((len(self.env), self.local_AC.cell_units), np.float32)
                rnn_state = np.array([c_init, h_init])

                # sample new noisy parameters in fully connected layers if
                # noisy net is used
                if episode_count % 15 == 0:
                    if self.noisy_policy is not None:
                        with tf.variable_scope(self.name):
                            with tf.variable_scope("policy_net"):
                                # Based on layers set scopes
                                scopes = []
                                for i in range(self.policy_layers):
                                    scopes.append(self.name + "/policy_net/" + "noise_action_" + str(i) + "/")
                                sample_new_weights(scopes, sess)

                    if self.noisy_value is not None:
                        with tf.variable_scope(self.name):
                            with tf.variable_scope("value_net"):
                                scopes = []
                                for i in range(self.value_layers):
                                    scopes.append(self.name + "/value_net/" + "noise_value_" + str(i) + "/")
                                sample_new_weights(scopes, sess)

                if self.method == "PCL":

                    # Perform a rollout of the chosen environment
                    episodes = self.rolloutPCL(sess, s, rnn_state, max_path_length = 1000, episode_count = len(self.env))

                    # Add sampled episode to replay buffer
                    self.replay_buffer.add(episodes)

                    # Get rewards and value estimates of current sample
                    _, _, r_ep, v_ep, _ = unpack_episode(episodes)

                    episode_values = np.mean(np.sum(v_ep, axis =1))
                    episode_reward = np.mean(np.sum(r_ep, axis =1))

                    # Train on online episode if applicable
                    train_online = False
                    train_offline = True

                    if train_online:

                        # Train PCL agent
                        _, _, summary = self.train_pcl(episodes, gamma, sess, merged_summary)

                        # Update summary information
                        train_steps = train_steps + 1
                        if self.name == "worker_0":
                            writer_summary.add_summary(summary, train_steps)


                    if train_offline:

                        # Sample len(envs) many episodes from the replay buffer
                        sampled_episodes = self.replay_buffer.sample(episode_count = len(self.env))

                        # Train PCL agent
                        r_ep, v_ep, summary = self.train_pcl(sampled_episodes, gamma, sess, merged_summary)

                        # Update summary information
                        train_steps = train_steps + 1
                        if self.name == "worker_0":
                            writer_summary.add_summary(summary, train_steps)

                        # Write add. summary information
                        episode_reward_offline = np.mean(np.sum(r_ep, axis = 1))
                        episode_value_offline = np.mean(np.sum(v_ep, axis = 1))

                elif self.method == "A3C":
                    # Run an episode
                    while not self.env.all_done():

                        # Get preferred action distribution
                        a, v, rnn_state, _ = self.act(s, rnn_state, sess)

                        # Get action for every environment
                        act_ = [np.argmax(a_) for a_ in a]
                        # Sample new state and reward from environment
                        s2, r, terminal, info = self.env.step(act_)

                        # Add states, rewards, actions, values and terminal information to A3C minibatch
                        self.add_to_batch(s,r,a,v,terminal)

                        # Get episode information for tracking the training process
                        self.episode_values.append(v)
                        self.episode_reward.append(r)

                        # Train on mini batches from episode
                        if (episode_step_count % MINI_BATCH == 0 and episode_step_count > 0) or self.env.all_done():
                            v1 = sess.run([self.local_AC.value],
                                          feed_dict={self.local_AC.inputs: s2,
                                                        self.local_AC.state_in[0]: rnn_state[0],
                                                        self.local_AC.state_in[1]: rnn_state[1]})

                            v_l, p_l, e_l, g_n, v_n, summary = self.train(self.episode_states_train, self.episode_reward_train,
                                                                          self.episode_actions_train, self.episode_values_train,
                                                                          self.episode_done_train,
                                                                          sess, gamma, np.squeeze(v1), merged_summary)
                            train_steps = train_steps + 1

                            # Update summary information
                            if self.name == "worker_0":
                                writer_summary.add_summary(summary, train_steps)

                            # Reset A3C minibatch after it has been used to update the model
                            self.reset_batch()

                        # Set previous state for next step
                        s = s2
                        total_steps += 1
                        episode_step_count += 1

                    episode_values = np.mean(np.sum(self.episode_values, axis = 0))
                    episode_reward = np.mean(np.sum(self.episode_reward, axis = 0))

                if episode_count % 20 == 0:
                    print("Reward: " + str(episode_reward), " | Episode", episode_count, " of " + self.name)
                    if self.method == "PCL":
                        print("Reward Offline: " + str(episode_reward_offline), " | Episode", episode_count, " of " + self.name)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(episode_values)

                sess.run(self.increment) # Next global episode

                episode_count += 1

                if episode_count == EPISODE_RUNS:
                    print("Worker stops because max episode runs are reached")
                    coord.request_stop()

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
                s2 = U.process_frame(s2, 84)
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