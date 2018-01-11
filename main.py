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
LOG_DIR = '/home/adrian/Schreibtisch/Uni/Data-Innovation-Lab/DILAB/tensorflowlogs'

#MAIN
def main(_):
    global master_network
    global global_episodes

    tf.reset_default_graph()

    with tf.device('/job:local/task:0/device:CPU:0'): #Parameter server adress
        RANDOM_SEED = 1234
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None, network_config)  # Generate global network
        num_workers = 2  # Number of workers

        workers = []
        # Create worker classes
        with tf.device('/job:local/task:1/device:CPU:0'): #Worker server adresses
            workers.append(Worker(0, STATE_DIM, ACTION_DIM, network_config, trainer, global_episodes,
                                  ENV_NAME, RANDOM_SEED))
        with tf.device('/job:local/task:1/device:CPU:0'):
            workers.append(Worker(1, STATE_DIM, ACTION_DIM, network_config, trainer, global_episodes,
                                  ENV_NAME, RANDOM_SEED))

    with tf.Session("grpc://localhost:2222") as sess:
        coord = tf.train.Coordinator()
        # Prepare summary information
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        worker=workers[0]
        with tf.device('/job:local/task:1/device:CPU:0'):
            worker_work = lambda: worker.work(GAMMA, sess, coord, merged, train_writer)
            t = threading.Thread(target=(worker_work))
            t.start()
        worker_threads.append(t)
        worker=workers[1]
        with tf.device('/job:local/task:1/device:CPU:0'):
            worker_work = lambda: worker.work(GAMMA, sess, coord, merged, train_writer)
            t = threading.Thread(target=(worker_work))
            t.start()
        worker_threads.append(t)
        print("Start")
        coord.join(worker_threads)

tf.app.run()
