import numpy as np
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
import os
import threading
import multiprocessing
import tensorflow as tf

from ac_network import AC_Network

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

# Sample new weights if noisy network
def sample_new_weights(scopes, sess):
    # Update variables
    for scope_ in scopes:
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_):
            assign_op = i.assign(np.random.normal(size=i.get_shape()))   # i.name if you want just a name
            sess.run(assign_op)


class Worker():
    def __init__(self, name, s_size, a_size, network_config, trainer, global_episodes, env_name, seed):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.a_size = a_size

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, network_config)
        self.update_local_ops = update_target_graph('global', self.name)

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
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()

                rnn_state = self.local_AC.state_init

                # sample new noisy parameters in fully connected layers if
                # noisy net is used
                if self.noisy_policy is not None:
                    with tf.variable_scope(self.name):
                        with tf.variable_scope("policy_net"):
                            # Based on layers set scopes
                            scopes = []
                            for i in range(self.policy_layers):
                                scopes.append("noise_action_" + str(i))
                            sample_new_weights(scopes, sess)

                if self.noisy_value is not None:
                    with tf.variable_scope(self.name):
                        with tf.variable_scope("value_net"):
                            scopes = []
                            for i in range(self.value_layers):
                                scopes.append("noise_value_" + str(i))
                            sample_new_weights(scopes, sess)


                # Run an episode
                while not terminal:
                    episode_states.append(s)


                    # Get preferred action distribution
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                         feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})

                    a0 = weighted_pick(a_dist[0], 1) # Use stochastic distribution sampling
                    a = np.zeros(self.a_size)
                    a[a0] = 1

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

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                sess.run(self.increment) # Next global episode

                episode_count += 1

                if episode_count == EPISODE_RUNS:
                    print("Worker stops because max episode runs are reached")
                    coord.request_stop()

    def getRewards(self):
        return self.episode_rewards