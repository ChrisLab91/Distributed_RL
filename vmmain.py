import numpy as np
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
import os
import threading
import multiprocessing
import tensorflow as tf

from ac_network import AC_Network
from worker import Worker

# PARAMETERS

# Gym environment
ENV_NAME = 'CartPole-v0'  # Discrete (4, 2)
STATE_DIM = 4
ACTION_DIM = 2

# Network configuration
network_config = dict(shared = True,
                      shared_config = dict(kind = 'RNN',
                                           Cell_Units = 16),
                      policy_config = dict(layers = [ACTION_DIM],
                                           noise_dist = "independent"),
                      value_config = dict(layers = [1],
                                          noise_dist = "independent"))

# Learning rate
LEARNING_RATE = 0.0005
# Discount rate for advantage estimation and reward discounting
GAMMA = 0.99

# Summary LOGDIR
LOG_DIR = '~/A3C/MyDistTest/'

# Choose RL method (A3C, PCL)
METHOD = "PCL"
print("Run method: " + METHOD)

# PCL variables
TAU = 0.2
ROLLOUT = 10

#MAIN
def main(_):
    global master_network
    global global_episodes

    tf.reset_default_graph()

    with tf.device('/job:ps/task:0/device:CPU:0'): #Parameter server adress
        RANDOM_SEED = 1234
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None, network_config, tau = TAU, rollout = ROLLOUT, method = METHOD)  # Generate global network
        num_workers = 1  # Number of workers

        workers = []
        # Create worker classes
        for i in range(num_workers):
            with tf.device('/job:worker/task:%d/device:CPU:0' % i): #Worker server adresses
                workers.append(Worker(i, STATE_DIM, ACTION_DIM, network_config, trainer, global_episodes,
                                  ENV_NAME, RANDOM_SEED, TAU, ROLLOUT, METHOD))

    with tf.Session("grpc://10.155.209.25:2222") as sess:
        coord = tf.train.Coordinator()
        # Prepare summary information
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            with tf.device('/job:worker/task:%d/device:CPU:0' % i):
                worker_work = lambda: worker.work(GAMMA, sess, coord, merged, train_writer)
                t = threading.Thread(target=(worker_work))
                t.start()
            worker_threads.append(t)
        print("Start")
        coord.join(worker_threads)

tf.app.run()