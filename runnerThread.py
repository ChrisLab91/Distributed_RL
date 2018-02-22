from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
from worker_queue import env_runner

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self, worker):

        self.states = np.array([], dtype=np.float32).reshape(len(worker.env), 0, worker.s_size)
        self.actions = np.array([], dtype=np.float32).reshape(len(worker.env), 0)
        self.rewards = np.array([], dtype=np.float32).reshape(len(worker.env), 0, worker.a_size)
        self.values = np.array([], dtype=np.float32).reshape(len(worker.env), 0)
        self.terminal = np.array([], dtype=np.float32).reshape(len(worker.env), 0)
        self.done = False

        #self.states = []
        #self.actions = []
        #self.rewards = []
        #self.values = []
        self.r = np.zeros(len(worker.env))
        #self.terminal = False
        #self.features = []

    def add(self, state, action, reward, value, terminal):

        self.states = np.concatenate((self.states, state), 1)
        self.rewards = np.concatenate((self.rewards, np.expand_dims(reward, 1)), 1)
        self.actions = np.concatenate((self.actions, action), 1)
        self.values = np.concatenate((self.values, np.reshape(value, [np.shape(value)[0], 1])), 1)
        self.terminal = np.concatenate((self.terminal, np.expand_dims(terminal, 1)), 1)

        #self.states += [state]
        #self.actions += [action]
        #self.rewards += [reward]
        #self.values += [value]
        #self.terminal = terminal
        #self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, worker, num_local_steps):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = worker.env
        self.last_features = None
        self.worker = worker
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        #self.visualise = visualise

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.worker, self.num_local_steps, self.summary_writer, self.sess)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)

