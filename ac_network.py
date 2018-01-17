import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

# PARAMETERS

# Cell units
ENTROPY_REGULARIZATION = True
ENTROPY_REGULARIZATION_LAMBDA = 0.1

####
#NETWORK
#Used to initialize weights for policy and value output layers
####

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer, network_config, tau = 0.0, rollout = 10, method = "A3C"):

        # Read network configurations
        self.shared = network_config["shared"]
        self.shared_config = network_config["shared_config"]
        self.policy_config = network_config["policy_config"]
        self.value_config = network_config["value_config"]

        # Paramters required by PCL
        self.tau_pcl = tau
        self.rollout_pcl = rollout

        with tf.variable_scope(scope):

            # Input
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            # Shared Recurrent network for temporal dependencies
            shared_output = self.build_shared_network(add_summaries=False)

            # Build policy network
            self.policy = self.build_policy_network(shared_output)

            # Build value network
            self.value = self.build_value_network(shared_output)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.setup_loss_gradients_summary(scope, trainer, a_size, method)


    def build_shared_network(self, add_summaries=False):

        """
        Builds the shared network part of the Actor-Critic-Network
        in the A3C paper. This network is shared by both the policy and value net.
        Args:
        inputs: Inputs
        add_summaries: If true, add layer summaries to Tensorboard.
        Returns:
        Final layer activations.
        """
        shared_network_kind = self.shared_config["kind"]

        if shared_network_kind == "RNN":

            cell_units = self.shared_config["Cell_Units"]
            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(cell_units, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = [c_in, h_in]
            rnn_in = tf.expand_dims(self.inputs, [0])
            #rnn_in = self.inputs
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in,
                initial_state=state_in,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, cell_units])

        else:
            rnn_out = None

        return rnn_out

    # Function to create and specify the value-function head of the shared
    # Actor-Critic Model
    def build_value_network(self, inputs_shared):

        noise_dist = self.value_config["noise_dist"]
        layers = self.value_config["layers"]

        with tf.variable_scope("value_net"):
            if noise_dist is None:
                value = slim.fully_connected(inputs_shared, layers[0],
                                             activation_fn=None,
                                             weights_initializer=normalized_columns_initializer(1.0),
                                             biases_initializer=None)
                for units in layers[1:]:
                    value = slim.fully_connected(value, units,
                                                 activation_fn=None,
                                                 weights_initializer=normalized_columns_initializer(1.0),
                                                 biases_initializer=None)

            else:
                value = noisy_dense(inputs_shared, layers[0],
                                    name="noise_value_" + str(0),
                                    bias=True, activation_fn=tf.identity,
                                    noise_dist=noise_dist)
                for units, i in zip(layers[1:], range(len(layers[1:]))):
                    value = noisy_dense(value, units,
                                        name="noise_value_" + str(i + 1),
                                        bias=True, activation_fn=tf.identity,
                                        noise_dist=noise_dist)

        return value

    # Function to create and specify the policy-function head of the shared
    # Actor-Critic Model
    def build_policy_network(self, inputs_shared):

        noise_dist = self.policy_config["noise_dist"]
        layers = self.policy_config["layers"]

        with tf.variable_scope("policy_net"):
            if noise_dist is None:
                policy = slim.fully_connected(inputs_shared, layers[0],
                                              activation_fn=tf.nn.softmax,
                                              weights_initializer=normalized_columns_initializer(0.01),
                                              biases_initializer=None)
                for units in layers[:1]:
                    policy = slim.fully_connected(policy, units,
                                                  activation_fn=tf.nn.softmax,
                                                  weights_initializer=normalized_columns_initializer(0.01),
                                                  biases_initializer=None)
            else:
                policy = noisy_dense(inputs_shared, layers[0],
                                     name="noise_action_" + str(0),
                                     bias=True, activation_fn=tf.nn.softmax,
                                     noise_dist=noise_dist)
                for units, i in zip(layers[1:], range(len(layers[1:]))):
                    policy = noisy_dense(policy, units,
                                         name="noise_action_" + str(i + 1),
                                         bias=True, activation_fn=tf.nn.softmax,
                                         noise_dist=noise_dist)

        return policy

    # Function defining the tensorflow graph of the loss functions
    # and the global and worker gradients
    def setup_loss_gradients_summary(self, scope, trainer, a_size, method = "A3C"):


        if method == "A3C":
            with tf.variable_scope("loss"):
                self.actions = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions, [1])

                with tf.variable_scope("value_loss"):
                    # Value loss function
                    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                with tf.variable_scope("policy_loss"):
                    # Softmax policy loss function
                    self.policy_loss = -tf.reduce_sum(tf.log(tf.maximum(self.responsible_outputs, 1e-12)) * self.advantages)

                # Softmax entropy function
                with tf.variable_scope("entropy"):
                    self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, 1e-12)))

                # If noisy net is active one can disable entropy regularization
                if ENTROPY_REGULARIZATION:
                    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * ENTROPY_REGULARIZATION_LAMBDA
                else:
                    self.loss = 0.5 * self.value_loss + self.policy_loss

            if scope == "worker_0":
                tf.summary.scalar('policy_loss_' + scope,
                                  self.policy_loss)
                tf.summary.scalar('value_loss_' + scope,
                                  self.value_loss)
                tf.summary.scalar('loss_' + scope,
                                  self.loss)
                tf.summary.scalar('advantages_' + scope,
                                  tf.reduce_mean(self.advantages))

        elif method == "PCL":

            with tf.variable_scope("loss"):
                self.actions = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
                self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
                self.discount = tf.placeholder(shape=[None], dtype=tf.float32)
                self.rollout = tf.placeholder(tf.int32, shape=())

                # Get first and last values of episode sub-batches
                length_episode = tf.shape(self.value)[0]

                # Compute first part of loss function
                values_first = self.value[:(length_episode - self.rollout + 1)]
                values_last = self.value[-(length_episode - self.rollout + 1):]

                with tf.variable_scope("value_loss"):
                    self.value_loss = -1 * values_first + self.discount[self.rollout - 1] * values_last

                # Compute second part of loss function
                # Get log probs
                with tf.variable_scope("policy_loss"):
                    self.log_probs = tf.log(tf.reduce_sum(self.policy * self.actions, axis=1))
                    rolling_log_probs = tf.map_fn(lambda i: self.log_probs[i:(i + self.rollout - 1)],
                                                  tf.range(length_episode - self.rollout + 1), dtype=tf.float32)
                    rolling_rewards = tf.map_fn(lambda i: self.rewards[i:(i + self.rollout - 1)],
                                                tf.range(length_episode - self.rollout + 1),dtype=tf.float32)
                    self.policy_loss = tf.reduce_sum((rolling_rewards - self.tau_pcl * rolling_log_probs) * self.discount[:(self.rollout-1)], axis=1)

                # Combine both loss parts
                self.loss = tf.reduce_mean(0.5 * tf.square(self.value_loss + self.policy_loss))

            if scope == "worker_0":
                tf.summary.scalar('policy_loss_' + scope,
                                  tf.reduce_mean(self.policy_loss))
                tf.summary.scalar('value_loss_' + scope,
                                  tf.reduce_mean(self.value_loss))
                tf.summary.histogram('log_probs' + scope,
                                     self.log_probs)
                tf.summary.scalar('loss_' + scope,
                                  self.loss)

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

        # Apply local gradients to global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


# Noisy layers implementation is based on
# https://arxiv.org/abs/1706.10295
def noisy_dense(x, size, name, bias=True, activation_fn=tf.identity, noise_dist = 'factorized'):

    # Create noise variables depending on chosen noise distribution
    with tf.variable_scope(name):
        if noise_dist == 'factorized':
            noise_input = tf.get_variable("noise_input_layer",
                                            shape=[x.get_shape().as_list()[1], 1],
                                            initializer = tf.random_normal_initializer,
                                            trainable= False)

            noise_output = tf.get_variable("noise_output_layer",
                                            shape=[1, size],
                                            initializer = tf.random_normal_initializer,
                                           trainable=False)

            # Initializer of \mu and \sigma in case of factorized noise distribution
            mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),
                                                    maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
            sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))

            def f(x):
                return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

            f_p = f(noise_input)
            f_q = f(noise_output)
            w_epsilon = f_p * f_q

            b_epsilon = tf.squeeze(f_q)

        if noise_dist == 'independent':
            noise_input = tf.get_variable("noise_input_layer",
                                        shape=[x.get_shape().as_list()[1], size],
                                        initializer = tf.random_normal_initializer,
                                        trainable=False)
            noise_output = tf.get_variable("noise_output_layer",
                                            shape=[1, size],
                                            initializer = tf.random_normal_initializer,
                                           trainable=False)

            # Initializer of \mu and \sigma in case of independent noise distribution
            mu_init = tf.random_uniform_initializer(minval=-np.power(3 / x.get_shape().as_list()[1], 0.5),
                                                    maxval= np.power(3 / x.get_shape().as_list()[1], 0.5))
            sigma_init = tf.constant_initializer(0.017)

            w_epsilon = tf.identity(noise_input)
            b_epsilon = tf.squeeze(noise_output)

    with tf.variable_scope(name + "_trainable"):
        # w = w_mu + w_sigma*w_epsilon
        w_mu = tf.get_variable("w_mu", [x.get_shape()[1], size], initializer=mu_init)
        w_sigma = tf.get_variable("w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
        w = w_mu + tf.multiply(w_sigma, w_epsilon)
        ret = tf.matmul(x, w)
        if bias:
            # b = b_mu + b_sigma*b_epsilon
            b_mu = tf.get_variable("b_mu", [size], initializer=mu_init)
            b_sigma = tf.get_variable("b_sigma", [size], initializer=sigma_init)
            b = b_mu + tf.multiply(b_sigma, b_epsilon)
            return(activation_fn(ret + b))
        else:
            return(activation_fn(ret))