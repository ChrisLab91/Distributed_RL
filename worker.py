import numpy as np
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
import os
import threading
import multiprocessing
import tensorflow as tf

from ac_network import AC_Network
from replay_buffer import ReplayBuffer

#WORKER
# PARAMETERS
# Clipping ratio for gradients

# Size of mini batches to run training on
MINI_BATCH = 40
REWARD_FACTOR = 0.001
EPISODE_RUNS = 500

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

# Sample new weights if noisy network
def sample_new_weights(scopes, sess):
    # Update variables
    #print(scopes)
    for scope_ in scopes:
        #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_))
        assig_ops = []
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_):
            assig_ops.append(i.assign(np.random.normal(size=i.get_shape())))  # i.name if you want just a name
        sess.run(assig_ops)

# Unpack a given episode that is saved into a dict
def unpack_episode(episode):
    a_ep = episode["actions"]
    s_ep = episode["states"]
    r_ep = episode["rewards"]
    v_ep = episode["values"]
    ag_info_ep = episode["agent_infos"]
    enf_info_ep = episode["env_infos"]

    return a_ep, s_ep, r_ep, v_ep, ag_info_ep, enf_info_ep


class Worker():
    def __init__(self, name, s_size, a_size, network_config, trainer, global_episodes, env_name, seed, tau = 0.0, rollout = 10, method = "A3C"):
        self.name = "worker_" + str(name)
        self.method = method
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        # Going to be memory buffer in case we are using PCL
        if self.method == "PCL":
            self.replay_buffer = ReplayBuffer()

        self.episode_buffer = []

        self.a_size = a_size

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, network_config, tau, rollout, self.method)
        self.update_local_ops = update_target_graph('global', self.name)

        # Set gym environment
        self.env = gym.make(env_name)
        self.env.seed(seed)

        # Get Noisy Net Information if applied
        self.noisy_policy = network_config["policy_config"]["noise_dist"]
        self.policy_layers = len(network_config["policy_config"]["layers"])
        self.noisy_value = network_config["value_config"]["noise_dist"]
        self.value_layers = len(network_config["value_config"]["layers"])

    def get_env(self):
        return self.env

    def train(self, rollout, sess, gamma, r, merged_summary):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist()+[r])*REWARD_FACTOR
        discounted_rewards = discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist()+[r])*REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = discounting(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(states),
                     self.local_AC.actions: np.vstack(actions),
                     self.local_AC.advantages: discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}

        summary = None
        if self.name == "worker_0":
            summary, v_l, p_l, e_l, g_n, v_n, _ = sess.run([merged_summary, self.local_AC.value_loss,
                                                   self.local_AC.policy_loss,
                                                   self.local_AC.entropy,
                                                   self.local_AC.grad_norms,
                                                   self.local_AC.var_norms,
                                                   self.local_AC.apply_grads],
                                                  feed_dict=feed_dict)
        else:
            v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                            self.local_AC.policy_loss,
                                                            self.local_AC.entropy,
                                                            self.local_AC.grad_norms,
                                                            self.local_AC.var_norms,
                                                            self.local_AC.apply_grads],
                                                           feed_dict=feed_dict)

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, summary

    def act(self, state, rnn_state, sess):

        action_dist, value, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                        feed_dict={self.local_AC.inputs: [state],
                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                   self.local_AC.state_in[1]: rnn_state[1]})

        # Chose action with largest probability
        a0 = weighted_pick(action_dist[0], 1)  # Use stochastic distribution sampling
        action_one_hot = np.zeros(self.a_size)
        action_one_hot[a0] = 1

        return action_one_hot, value, rnn_state, action_dist

    def work(self, gamma, sess, coord, merged_summary, writer_summary):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        train_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_mini_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                # Used by PCL
                episode_reward_online = 0
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()

                rnn_state = self.local_AC.state_init

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
                    episodes = self.rolloutPCL(sess, s, rnn_state, episode_count = 1)

                    # Add sampled episode to replay buffer
                    self.replay_buffer.add(episodes)
                    _, _, r_ep, v_ep,_, _ = unpack_episode(episodes[0])
                    episode_values = v_ep
                    episode_reward = np.sum(r_ep)

                    # Train on online episode if applicable
                    train_online = False
                    train_offline = True

                    if train_online:
                        # Train PCL on current episode

                        # Get action, states, rollout length and reward information
                        a_ep, s_ep, r_ep, _,  _, _ = unpack_episode(episodes[0])
                        episodes_len = len(a_ep)

                        # Get action array
                        action_array = np.eye(self.a_size, dtype=np.int32)[a_ep]

                        rollout = self.local_AC.rollout_pcl
                        while rollout > episodes_len:
                            rollout = int(rollout/2)

                        discount = np.array([gamma**i for i in range(rollout)], dtype=np.float32)

                        # Perform training on one episode batch
                        feed_dict_ = {
                            self.local_AC.inputs: s_ep,
                            self.local_AC.actions: action_array,
                            self.local_AC.rewards: r_ep,
                            self.local_AC.discount: discount,
                            self.local_AC.rollout: rollout,
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]
                        }

                        v_l, p_l, total_loss, _ = sess.run([self.local_AC.value_loss,
                                                         self.local_AC.policy_loss,
                                                         self.local_AC.loss,
                                                         self.local_AC.apply_grads],
                                                        feed_dict = feed_dict_)


                    if train_offline:
                        sampled_episodes = self.replay_buffer.sample(episode_count = 1)

                        # Train on sampled episodes

                        a_ep, s_ep, r_ep, _,  _, _ = unpack_episode(sampled_episodes[0])
                        episodes_len = len(a_ep)

                        # Get action array
                        action_array = np.eye(self.a_size, dtype=np.int32)[a_ep]

                        rollout = self.local_AC.rollout_pcl
                        while rollout > episodes_len:
                            rollout = int(rollout / 2)

                        discount = np.array([gamma ** i for i in range(rollout)], dtype=np.float32)

                        # Set rnn_state based on the amount of episodes
                        # Dynamic state initialization
                        batch_size = len(sampled_episodes)
                        max_len = max([len(ep["actions"]) for ep in sampled_episodes])
                        c_init = np.zeros(shape=(batch_size, max_len))
                        h_init = np.zeros(shape=(batch_size, max_len))

                        # Perform padding based on the longest episode
                        # Elements to be --> function pad_episodes given a list of sampled episodes

                        if self.name == "worker_0":
                            # Perform training on one episode batch
                            feed_dict_ = {
                                self.local_AC.inputs: s_ep,
                                self.local_AC.actions: action_array,
                                self.local_AC.rewards: r_ep,
                                self.local_AC.discount: discount,
                                self.local_AC.rollout: rollout,
                                self.local_AC.state_in[0]: rnn_state[0],
                                self.local_AC.state_in[1]: rnn_state[1]
                            }
                            summary, v_l, p_l, total_loss , _ = sess.run([merged_summary,self.local_AC.value_loss,
                                                             self.local_AC.policy_loss,
                                                             self.local_AC.loss,
                                                             self.local_AC.apply_grads],
                                                            feed_dict=feed_dict_)

                            train_steps = train_steps + 1
                            writer_summary.add_summary(summary, train_steps)

                        else:
                            # Perform training on one episode batch
                            feed_dict_ = {
                                self.local_AC.inputs: s_ep,
                                self.local_AC.actions: action_array,
                                self.local_AC.rewards: r_ep,
                                self.local_AC.discount: discount,
                                self.local_AC.rollout: rollout,
                                self.local_AC.state_in[0]: rnn_state[0],
                                self.local_AC.state_in[1]: rnn_state[1]
                            }

                            v_l, p_l, total_loss, _ = sess.run([self.local_AC.value_loss,
                                                                 self.local_AC.policy_loss,
                                                                 self.local_AC.loss,
                                                                 self.local_AC.apply_grads],
                                                                feed_dict=feed_dict_)


                        episode_reward_online = np.sum(r_ep)

                elif self.method == "A3C":
                    # Run an episode
                    while not terminal:
                        episode_states.append(s)


                        # Get preferred action distribution
                        a, v, rnn_state, _ = self.act(s, rnn_state, sess)

                        s2, r, terminal, info = self.env.step(np.argmax(a))

                        episode_reward += r

                        episode_buffer.append([s, a, r, s2, terminal, v[0, 0]])
                        episode_mini_buffer.append([s, a, r, s2, terminal, v[0, 0]])

                        episode_values.append(v[0, 0])

                        # Train on mini batches from episode
                        if len(episode_mini_buffer) == MINI_BATCH:
                            v1 = sess.run([self.local_AC.value],
                                          feed_dict={self.local_AC.inputs: [s],
                                                        self.local_AC.state_in[0]: rnn_state[0],
                                                        self.local_AC.state_in[1]: rnn_state[1]})

                            v_l, p_l, e_l, g_n, v_n, summary = self.train(episode_mini_buffer, sess, gamma, v1[0][0], merged_summary)
                            train_steps = train_steps + 1

                            # Update summary information
                            if self.name == "worker_0":
                                writer_summary.add_summary(summary, train_steps)
                            # Reser episode batch
                            episode_mini_buffer = []

                        # Set previous state for next step
                        s = s2
                        total_steps += 1
                        episode_step_count += 1



                if episode_count % 20 == 0:
                    print("Reward: " + str(episode_reward), " | Episode", episode_count, " of " + self.name)
                    if self.method == "PCL":
                        print("Reward Online: " + str(episode_reward_online), " | Episode", episode_count, " of " + self.name)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                sess.run(self.increment) # Next global episode

                episode_count += 1

                if episode_count == EPISODE_RUNS:
                    self.episode_buffer = episode_buffer
                    print("Worker stops because max episode runs are reached")
                    coord.request_stop()

    def getRewards(self):
        return self.episode_rewards

    def getEpisodeBuffer(self):
        return self.episode_buffer

    def rolloutPCL(self, sess, initial_state, rnn_state, max_path_length=None, episode_count = 1):

        # Perform rollout of given environment
        if max_path_length is None:
            max_path_length = self.env.spec.tags.get(
                'wrapper_config.TimeLimit.max_episode_steps')

        # Define list so save episodes
        episodes = []

        for i in range(episode_count):

            if i > 0:
                # First initial state is supplied
                initial_state = self.env.reset()
            states = []
            actions = []
            rewards = []
            agent_infos = []
            env_infos = []
            values = []
            s = initial_state
            path_length = 0

            # Sample one episode
            while path_length < max_path_length:
                a, v, rnn_state, agent_info = self.act(s, rnn_state, sess)
                next_s, r, d, env_info = self.env.step(np.argmax(a))
                states.append(s)
                rewards.append(r)
                values.append(v)
                actions.append(np.argmax(a))
                agent_infos.append(agent_info)
                env_infos.append(env_info)
                path_length += 1
                if d:
                    break
                s = next_s
            # Append sampled episode
            # action_array = np.eye(self.a_size, dtype=np.int32)[np.array(actions)]
            episodes.append(dict(
                                states=np.array(states),
                                actions=np.array(actions),
                                rewards=np.array(rewards),
                                values = np.reshape(np.array(values), newshape= (len(np.array(values)))),
                                agent_infos=np.array(agent_infos),
                                env_infos=np.array(env_infos)
                            ))
        return episodes

