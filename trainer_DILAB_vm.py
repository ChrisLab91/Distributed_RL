import argparse
import sys
import numpy as np
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
import os
import threading
import multiprocessing
import tensorflow as tf

import util as U

from ac_network import AC_Network
from worker import Worker
from worker import sample_new_weights, update_target_graph, weighted_pick, discounting, norm, unpack_episode

# PARAMETERS

# Run python train.py ps $i --worker_num $workers --env $env --ps_num $ps & --> for PS Server
# python train.py worker $i --mode $mode --env $env --worker_num $workers --ps_num $ps & --> For Worker Server

def main(job, task, worker_num, ps_num, initport, ps_hosts, worker_hosts):

    PS_HOST = ps_hosts.split(",")
    WORKER_HOSTS = worker_hosts.split(",")

    INITPORT = initport
    CLUSTER = dict()

    """
    workers = []
    ps_ = []
    for i in range(ps_num):
        ps_.append('localhost:{}'.format(INITPORT + i))
    for i in range(worker_num):
        workers.append("localhost:{}".format(i + ps_num + INITPORT))

    CLUSTER['worker'] = workers
    CLUSTER['ps'] = ps_
        """
    # Infer the amount of workers and ps servers


    cluster = tf.train.ClusterSpec({"ps": PS_HOST, "worker": WORKER_HOSTS})
    print({"ps": PS_HOST, "worker": WORKER_HOSTS})

    num_ps, num_workers = len(PS_HOST), len(WORKER_HOSTS)
    #  Get the Cluster Spec
    # cluster = tf.train.ClusterSpec(CLUSTER)

    # Get the current server element
    TASK_ID = task
    JOB = job
    server = tf.train.Server(cluster, job_name=JOB, task_index=TASK_ID)

    # Check if we have a worker or ps node running
    if JOB == 'ps':
        server.join()
    else:

        # Get all required Paramters

        # Gym environment
        ENV_NAME = 'CartPole-v0'  # Discrete (4, 2)
        STATE_DIM = 4
        ACTION_DIM = 2

        # Network configuration
        network_config = dict(shared=True,
                              shared_config=dict(kind='RNN',
                                                 Cell_Units=16),
                              policy_config=dict(layers=[ACTION_DIM],
                                                 noise_dist=None),
                              value_config=dict(layers=[1],
                                                noise_dist=None))

        # Learning rate
        LEARNING_RATE = 0.0005
        # Discount rate for advantage estimation and reward discounting
        GAMMA = 0.99

        # Summary LOGDIR
        LOG_DIR = '~/A3C/MyDistTest/'

        # Choose RL method (A3C, PCL)
        METHOD = "A3C"
        print("Run method: " + METHOD)

        # PCL variables
        TAU = 0.2
        ROLLOUT = 10
        # Define the global network and get relevant worker_device
        worker_device = '/job:worker/task:{}/cpu:0'.format(TASK_ID)

        with tf.device(tf.train.replica_device_setter(cluster=cluster,  # Makes sure global variables defined in
                                                      worker_device=worker_device,
                                                      # this contexts are synced across processes
                                                      ps_strategy=U.greedy_ps_strategy(ps_tasks=num_ps))):

            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None, network_config, tau=TAU, rollout=ROLLOUT,
                                        method=METHOD)  # Generate global network

        with tf.device(worker_device):
            worker = Worker(TASK_ID, STATE_DIM, ACTION_DIM, network_config, trainer, global_episodes,
                                  ENV_NAME, tau = TAU, rollout= ROLLOUT, method=METHOD)

        local_init_op = tf.global_variables_initializer()
        with tf.Session(server.target) as sess:
            sess.run(local_init_op)

        with tf.train.MonitoredTrainingSession(master=server.target) as sess:


            # Define input to worker.work( gamma, sess, coord, merged_summary, writer_summary)
            gamma = GAMMA

            MINI_BATCH = 40
            REWARD_FACTOR = 0.001
            EPISODE_RUNS = 1000

            #episode_count = sess.run(worker.global_episodes)
            episode_count = 0
            total_steps = 0
            train_steps = 0
            print("Starting worker " + str(TASK_ID))

            while not sess.should_stop():

                sess.run(worker.update_local_ops)
                worker.episode_values = []
                worker.episode_reward = []

                if worker.method == "A3C":
                    # Objects to hold the bacth used to update the Agent
                    worker.episode_states_train = np.array([], dtype=np.float32).reshape(len(worker.env), 0, worker.s_size)
                    worker.episode_reward_train = np.array([], dtype=np.float32).reshape(len(worker.env), 0)
                    worker.episode_actions_train = np.array([], dtype=np.float32).reshape(len(worker.env), 0, worker.a_size)
                    worker.episode_values_train = np.array([], dtype=np.float32).reshape(len(worker.env), 0)
                    worker.episode_done_train = np.array([], dtype=np.float32).reshape(len(worker.env), 0)

                # Used by PCL
                # Hold reward and value function mean value of sampled episodes from replay buffer
                episode_reward_offline = 0
                episode_value_offline = 0
                episode_step_count = 0

                # Restart environment
                s = worker.env.reset()

                # Set initial rnn state based on number of episodes
                c_init = np.zeros((len(worker.env), worker.local_AC.cell_units), np.float32)
                h_init = np.zeros((len(worker.env), worker.local_AC.cell_units), np.float32)
                rnn_state = np.array([c_init, h_init])

                # sample new noisy parameters in fully connected layers if
                # noisy net is used
                if worker.noisy_policy is not None or worker.noisy_value is not None:
                    sess.run(worker.local_AC.noisy_sampling)

                if worker.method == "PCL":

                    # Perform a rollout of the chosen environment
                    episodes = worker.rolloutPCL(sess, s, rnn_state, max_path_length=1000, episode_count=len(worker.env))

                    # Add sampled episode to replay buffer
                    worker.replay_buffer.add(episodes)

                    # Get rewards and value estimates of current sample
                    _, _, r_ep, v_ep, _ = unpack_episode(episodes)

                    episode_values = np.mean(np.sum(v_ep, axis=1))
                    episode_reward = np.mean(np.sum(r_ep, axis=1))

                    # Train on online episode if applicable
                    train_online = False
                    train_offline = True

                    if train_online:

                        # Train PCL agent
                        _, _, summary = worker.train_pcl(episodes, gamma, sess)

                        # Update summary information
                        train_steps = train_steps + 1
                        #if worker.name == "worker_0":
                        #    writer_summary.add_summary(summary, train_steps)

                    if train_offline:

                        # Sample len(envs) many episodes from the replay buffer
                        sampled_episodes = worker.replay_buffer.sample(episode_count=len(worker.env))

                        # Train PCL agent
                        r_ep, v_ep, _ = worker.train_pcl(sampled_episodes, gamma, sess)

                        # Update summary information
                        train_steps = train_steps + 1
                        #if worker.name == "worker_0":
                        #    writer_summary.add_summary(summary, train_steps)

                        # Write add. summary information
                        episode_reward_offline = np.mean(np.sum(r_ep, axis=1))
                        episode_value_offline = np.mean(np.sum(v_ep, axis=1))

                elif worker.method == "A3C":
                    # Run an episode
                    while not worker.env.all_done():

                        # Get preferred action distribution
                        a, v, rnn_state, _ = worker.act(s, rnn_state, sess)

                        # Get action for every environment
                        act_ = [np.argmax(a_) for a_ in a]
                        # Sample new state and reward from environment
                        s2, r, terminal, info = worker.env.step(act_)

                        # Add states, rewards, actions, values and terminal information to A3C minibatch
                        worker.add_to_batch(s, r, a, v, terminal)

                        # Get episode information for tracking the training process
                        worker.episode_values.append(v)
                        worker.episode_reward.append(r)

                        # Train on mini batches from episode
                        if (episode_step_count % MINI_BATCH == 0 and episode_step_count > 0) or worker.env.all_done():
                            v1 = sess.run([worker.local_AC.value],
                                          feed_dict={worker.local_AC.inputs: s2,
                                                     worker.local_AC.state_in[0]: rnn_state[0],
                                                     worker.local_AC.state_in[1]: rnn_state[1]})

                            v_l, p_l, e_l, g_n, v_n, summary = worker.train(worker.episode_states_train,
                                                                          worker.episode_reward_train,
                                                                          worker.episode_actions_train,
                                                                          worker.episode_values_train,
                                                                          worker.episode_done_train,
                                                                          sess, gamma, np.squeeze(v1))
                            train_steps = train_steps + 1

                            # Update summary information
                            #if worker.name == "worker_0":
                            #    writer_summary.add_summary(summary, train_steps)

                            # Reset A3C minibatch after it has been used to update the model
                            worker.reset_batch()

                        # Set previous state for next step
                        s = s2
                        total_steps += 1
                        episode_step_count += 1

                    episode_values = np.mean(np.sum(worker.episode_values, axis=0))
                    episode_reward = np.mean(np.sum(worker.episode_reward, axis=0))

                if episode_count % 20 == 0:
                    print("Reward: " + str(episode_reward), " | Episode", episode_count, " of " + worker.name)
                    if worker.method == "PCL":
                        print("Reward Offline: " + str(episode_reward_offline), " | Episode", episode_count,
                              " of " + worker.name)

                worker.episode_rewards.append(episode_reward)
                worker.episode_lengths.append(episode_step_count)
                worker.episode_mean_values.append(episode_values)

                sess.run(worker.increment)  # Next global episode

                episode_count += 1

                if episode_count == EPISODE_RUNS:
                    print("Worker stops because max episode runs are reached")
                    sess.request_stop()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("job", choices=["ps", "worker"])
    parser.add_argument("task", type=int)
    #parser.add_argument("--animate", default=False, action='store_true')
    #parser.add_argument("--env", default='Pendulum-v0')
    #parser.add_argument("--seed", default=12321, type=int)
    parser.add_argument("--tboard", default=False)
    parser.add_argument("--worker_num", default=2, type=int)  # worker jobs
    parser.add_argument("--ps_num", default=1, type=int)  # ps jobs
    parser.add_argument("--initport", default=2849, type=int)  # starting ports for cluster
    #parser.add_argument("--stdout_freq", default=20, type=int)
    #parser.add_argument("--save_every", default=600, type=int)  # save frequency
    #parser.add_argument("--outdir", default=os.path.join('tmp', 'logs'))  # file for the statistics of training
    parser.add_argument("--checkpoint_dir", default=os.path.join('tmp', 'checkpoints'))  # where to save checkpoint
    parser.add_argument("--frames", default=1, type=int)  # how many recent frames to send to model
    #parser.add_argument("--mode", choices=["train", "debug-light", "debug-full"],
    #                   default="train")  # how verbose to print to stdout
    #parser.add_argument("--desired_kl", default=0.002, type=float)
    parser.add_argument("--ps_hosts", default="", type=str)
    parser.add_argument("--worker_hosts", default="", type=str)

    args = parser.parse_args()

    main(args.job, args.task, args.worker_num, args.ps_num, args.initport, args.ps_hosts, args.worker_hosts)