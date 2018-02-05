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
from worker import update_target_graph, weighted_pick, discounting, norm, unpack_episode
from gym_wrapper import  GymWrapper

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

        # Running Paramters
        TOTAL_GLOBAL_EPISODES = 2050

        # Gym environment

        ENV_NAME = 'CartPole-v0'  # MsPacman CartPole
        NUM_ENVS = 5
        PREPROCESSING = False
        IMAGE_SIZE_PREPROCESSED = 35

        PREPROCESSING_CONFIG = [
            {
                "type": "image_resize",
                "width": IMAGE_SIZE_PREPROCESSED,
                "height": IMAGE_SIZE_PREPROCESSED
            }, {
                "type": "grayscale"
            }

            #     {
            #         "type": "sequence",         # TO-DO: sequence not supported
            #         "length": 2
            #     }
        ]

        # Get env parameters

        gw = GymWrapper(ENV_NAME)
        ACTION_DIM = gw.act_space.n

        if PREPROCESSING:
            STATE_DIM = IMAGE_SIZE_PREPROCESSED * IMAGE_SIZE_PREPROCESSED

            types_of_preprocess = []
            for operation in PREPROCESSING_CONFIG:
                types_of_preprocess.append(operation['type'])
                if operation['type'] == "sequence":
                    length_sequence = operation['length']

            print("Do following preprocessing steps: {0}".format(types_of_preprocess))

        else:
            PREPROCESSING_CONFIG = None
            STATE_DIM = gw.obs_space.shape[0]

        # Network configuration
        network_config = dict(shared=True,
                              shared_config=dict(kind=["Dense"],
                                                 cnn_input_size=IMAGE_SIZE_PREPROCESSED,
                                                 cnn_output_size=8,
                                                 dense_layers=[2],
                                                 lstm_cell_units=16),
                              policy_config=dict(layers=[ACTION_DIM],
                                                 noise_dist=None),
                              value_config=dict(layers=[1],
                                                noise_dist=None))

        # Learning rate
        LEARNING_RATE = 0.005
        UPDATE_LEARNING_RATE = True
        # Discount rate for advantage estimation and reward discounting
        GAMMA = 0.99

        # Summary LOGDIR
        # LOG_DIR = '~/A3C/MyDistTest/'
        LOG_DIR = os.getcwd() + '_tensorflowlogs'
        LOG_DIR_CHECKPOINT = os.getcwd() + "_modelcheckpoints"

        # Print latest checkpoint
        checkpoint_sync = True

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

            global_episodes = tf.train.get_or_create_global_step()
            master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', network_config, learning_rate=None, tau=TAU,
                                        rollout=ROLLOUT,
                                        method=METHOD)  # Generate global network

            with tf.device(worker_device):
                worker = Worker(TASK_ID, STATE_DIM, ACTION_DIM, network_config, LEARNING_RATE, global_episodes,
                                ENV_NAME, number_envs=NUM_ENVS, tau=TAU, rollout=ROLLOUT, method=METHOD,
                                update_learning_rate_=UPDATE_LEARNING_RATE, preprocessing_config=PREPROCESSING_CONFIG)

        # Get summary information
        if worker.name == "worker_0":
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
        else:
            merged_summary = None

        local_init_op = tf.global_variables_initializer()

        with tf.Session(server.target) as sess:
            sess.run(local_init_op)

        # Setup monitoring
        is_chief = (TASK_ID == 0)

        # Setup hooks required to coordinate training
        stopHook = tf.train.StopAtStepHook(num_steps=TOTAL_GLOBAL_EPISODES)
        saver = tf.train.Saver(max_to_keep=3,
                               var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global'))
        saverHook = tf.train.CheckpointSaverHook(checkpoint_dir=LOG_DIR_CHECKPOINT,
                                                 save_steps=10,
                                                 checkpoint_basename=worker.method,
                                                 saver=saver)

        # Start Training
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               chief_only_hooks=[saverHook],
                                               hooks=[stopHook]) as sess:

            # Reload global model from chief
            if is_chief:
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR_CHECKPOINT, latest_filename=None))
                except ValueError:
                    print("No Model Checkpoint available")

            # If Checkpoint is loaded make sure all workers start after
            # the variables have been reloaded in order to avoid "bad" updates
            if checkpoint_sync:
                while sess.run(worker.global_episodes) == 0:
                    print(worker.name + " Waiting for Sync of Checkpoint loaded by worker_0")
                    time.sleep(1)

            # Update from global
            sess.run(worker.update_local_ops)

            # Define input to worker.work( gaxmma, sess, coord, merged_summary, writer_summary)
            gamma = GAMMA

            MINI_BATCH = 40
            REWARD_FACTOR = 0.001
            EPISODE_RUNS = 1000

            episode_count = 0
            total_steps = 0
            train_steps = 0
            print("Starting worker " + str(TASK_ID))

            while not sess.should_stop():

                worker.episode_values = []
                worker.episode_reward = []

                # Objects to hold the bacth used to update the Agent
                worker.reset_batch()

                # Used by PCL
                # Hold reward and value function mean value of sampled episodes from replay buffer
                episode_reward_offline = 0
                episode_value_offline = 0
                episode_step_count = 0

                # Restart environment
                s = worker.env.reset()
                if worker.preprocessing_state:
                    s = U.process_frame(s, worker.preprocessing_config)

                if worker.rnn_network:
                    # Set initial rnn state based on number of episodes
                    c_init = np.zeros((len(worker.env), worker.local_AC.cell_units), np.float32)
                    h_init = np.zeros((len(worker.env), worker.local_AC.cell_units), np.float32)
                    rnn_state = np.array([c_init, h_init])
                else:
                    rnn_state = None

                # sample new noisy parameters in fully connected layers if
                # noisy net is used
                # if episode_count % 15 == 0:
                if worker.noisy_policy is not None or worker.noisy_value is not None:
                    sess.run(worker.local_AC.noisy_sampling)

                if worker.method == "PCL":

                    # Perform a rollout of the chosen environment
                    episodes = worker.rolloutPCL(sess, s, rnn_state, max_path_length=1000,
                                                 episode_count=len(worker.env))

                    # Add sampled episode to replay buffer
                    worker.replay_buffer.add(episodes)

                    # Get rewards and value estimates of current sample
                    _, _, r_ep, v_ep, _, _ = unpack_episode(episodes)

                    episode_values = np.mean(np.sum(v_ep, axis=1))
                    episode_reward = np.mean(np.sum(r_ep, axis=1))

                    # Train on online episode if applicable
                    train_online = False
                    train_offline = True

                    if train_online:
                        # Train PCL agent
                        _, _, summary = worker.train_pcl(episodes, gamma, sess, merged_summary)

                        # Update summary information
                        train_steps = train_steps + 1
                        # if worker.name == "worker_0":
                        #    writer_summary.add_summary(summary, train_steps)

                    if train_offline:

                        # Sample len(envs) many episodes from the replay buffer
                        sampled_episodes = worker.replay_buffer.sample(episode_count=len(worker.env))

                        # Train PCL agent
                        r_ep, v_ep, summary, logits = worker.train_pcl(sampled_episodes, gamma, sess, merged_summary)
                        # Update global network
                        sess.run(worker.update_local_ops)

                        # Update learning rate based on calculated KL Divergence
                        if worker.update_learning_rate_:
                            # Calculate KL-Divergence of updated policy and policy before update
                            kl_divergence = worker.calculate_kl_divergence(logits, sampled_episodes, sess)
                            # Perform learning rate update based on KL-Divergence
                            worker.update_learning_rate(kl_divergence, sess)

                        # Update summary information
                        train_steps = train_steps + 1
                        if worker.name == "worker_0":
                            writer.add_summary(summary, train_steps)

                        # Write add. summary information
                        episode_reward_offline = np.mean(np.sum(r_ep, axis=1))
                        episode_value_offline = np.mean(np.sum(v_ep, axis=1))

                elif worker.method == "A3C":
                    # Run an episode
                    while not worker.env.all_done():

                        # Get preferred action distribution
                        dummy_lengths = np.ones(len(worker.env))
                        a, v, rnn_state, _ = worker.act(s, rnn_state, dummy_lengths, sess)

                        # Get action for every environment
                        act_ = [np.argmax(a_) for a_ in a]
                        # Sample new state and reward from environment
                        s2, r, terminal, info = worker.env.step(act_)
                        if worker.preprocessing_state:
                            s2 = U.process_frame(s2, worker.preprocessing_config)

                        # Add states, rewards, actions, values and terminal information to A3C minibatch
                        worker.add_to_batch(s, r, a, v, terminal)

                        # Get episode information for tracking the training process
                        worker.episode_values.append(v)
                        worker.episode_reward.append(r)

                        # Train on mini batches from episode
                        if (episode_step_count % MINI_BATCH == 0 and episode_step_count > 0) or worker.env.all_done():

                            feed_dict_ = {worker.local_AC.inputs: s2,
                                          worker.local_AC.lengths_episodes: dummy_lengths}

                            if worker.rnn_network:
                                feed_dict_[worker.local_AC.state_in[0]] = rnn_state[0]
                                feed_dict_[worker.local_AC.state_in[1]] = rnn_state[1]

                            v1 = sess.run([worker.local_AC.value], feed_dict_)

                            v_l, p_l, e_l, g_n, v_n, summary, logits = worker.train(worker.episode_states_train,
                                                                                    worker.episode_reward_train,
                                                                                    worker.episode_actions_train,
                                                                                    worker.episode_values_train,
                                                                                    worker.episode_done_train,
                                                                                    sess, gamma, np.squeeze(v1),
                                                                                    merged_summary)

                            if worker.env.all_done():
                                # Update global network
                                sess.run(worker.update_local_ops)

                                # Update learning rate based on calculated KL Divergence
                                if worker.update_learning_rate_:
                                    # Calculate KL-Divergence of updated policy and policy before update
                                    kl_divergence = worker.calculate_kl_divergence(logits, worker.episode_states_train,
                                                                                   sess, worker.episode_done_train)
                                    # Perform learning rate update based on KL-Divergence
                                    if not np.isnan(kl_divergence):
                                        worker.update_learning_rate(kl_divergence, sess)

                            train_steps = train_steps + 1

                            # Update summary information
                            if worker.name == "worker_0":
                                writer.add_summary(summary, train_steps)

                            # Reset A3C minibatch after it has been used to update the model
                            worker.reset_batch()

                        # Set previous state for next step
                        s = s2
                        total_steps += 1
                        episode_step_count += 1

                    episode_values = np.mean(np.sum(worker.episode_values, axis=0))
                    episode_reward = np.mean(np.sum(worker.episode_reward, axis=0))

                if episode_count % 20 == 0:
                    print("Reward: " + str(episode_reward), " | Episode", episode_count, " of " + worker.name,
                          " | Global Episode", str(sess.run(worker.global_episodes)))
                    if worker.method == "PCL":
                        print("Reward Offline: " + str(episode_reward_offline), " | Episode", episode_count,
                              " of " + worker.name)

                worker.episode_rewards.append(episode_reward)
                worker.episode_lengths.append(episode_step_count)
                worker.episode_mean_values.append(episode_values)

                sess.run(worker.increment)  # Next global episode

                episode_count += 1

        # Ask for all the services to stop.
        print("Worker stops because max episode runs are reached")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("job", choices=["ps", "worker"])
    parser.add_argument("task", type=int)
    # parser.add_argument("--animate", default=False, action='store_true')
    # parser.add_argument("--env", default='Pendulum-v0')
    # parser.add_argument("--seed", default=12321, type=int)
    parser.add_argument("--tboard", default=False)
    parser.add_argument("--worker_num", default=2, type=int)  # worker jobs
    parser.add_argument("--ps_num", default=1, type=int)  # ps jobs
    parser.add_argument("--initport", default=2849, type=int)  # starting ports for cluster
    # parser.add_argument("--stdout_freq", default=20, type=int)
    # parser.add_argument("--save_every", default=600, type=int)  # save frequency
    # parser.add_argument("--outdir", default=os.path.join('tmp', 'logs'))  # file for the statistics of training
    parser.add_argument("--checkpoint_dir", default=os.path.join('tmp', 'checkpoints'))  # where to save checkpoint
    parser.add_argument("--frames", default=1, type=int)  # how many recent frames to send to model
    # parser.add_argument("--mode", choices=["train", "debug-light", "debug-full"],
    #                   default="train")  # how verbose to print to stdout
    # parser.add_argument("--desired_kl", default=0.002, type=float)
    parser.add_argument("--ps_hosts", default="", type=str)
    parser.add_argument("--worker_hosts", default="", type=str)

    args = parser.parse_args()

    main(args.job, args.task, args.worker_num, args.ps_num, args.initport, args.ps_hosts, args.worker_hosts)