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

            # Input --> Shape [batch_size, time_length, feature_size]
            self.inputs = tf.placeholder(shape=[None, None, s_size], dtype=tf.float32)

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

            def length(sequence):
                used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
                length = tf.reduce_sum(used, 1)
                length = tf.cast(length, tf.int32)
                return length

            self.lengths_episodes = length(self.inputs)

            self.cell_units = self.shared_config["Cell_Units"]
            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_units, state_is_tuple=True)
            #c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            #h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            #self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [None, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [None, lstm_cell.state_size.h])
            self.state_in = [c_in, h_in]
            #rnn_in = tf.expand_dims(self.inputs, [0])
            rnn_in = self.inputs
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in,
                initial_state=state_in,
                time_major=False,
                sequence_length=length(self.inputs))
            #lstm_c, lstm_h = lstm_state
            #self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.state_out = lstm_state
            #rnn_out = tf.reshape(lstm_outputs, [-1, cell_units])
            self.rnn_out = lstm_outputs

        else:
            self.rnn_out = None

        return self.rnn_out

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
                value = self.noisy_dense(inputs_shared, layers[0],
                                    name="noise_value_" + str(0),
                                    bias=True, activation_fn=tf.identity,
                                    noise_dist=noise_dist)
                for units, i in zip(layers[1:], range(len(layers[1:]))):
                    value = self.noisy_dense(value, units,
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
                policy = self.noisy_dense(inputs_shared, layers[0],
                                     name="noise_action_" + str(0),
                                     bias=True, activation_fn=tf.nn.softmax,
                                     noise_dist=noise_dist)
                for units, i in zip(layers[1:], range(len(layers[1:]))):
                    policy = self.noisy_dense(policy, units,
                                         name="noise_action_" + str(i + 1),
                                         bias=True, activation_fn=tf.nn.softmax,
                                         noise_dist=noise_dist)

        return policy

    # Function defining the tensorflow graph of the loss functions
    # and the global and worker gradients
    def setup_loss_gradients_summary(self, scope, trainer, a_size, method = "A3C"):


        if method == "A3C":
            with tf.variable_scope("loss"):
                self.actions = tf.placeholder(shape=[None, None, a_size], dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None, None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None, None], dtype=tf.float32)

                # Get first and last values of episode sub-batches
                length_episode = tf.shape(self.value)[1]
                length_batch = tf.shape(self.value)[0]

                with tf.variable_scope("value_loss"):

                    # Tensorarray to keep value loss for all episodes
                    value_loss_array_init = tf.TensorArray(tf.float32, size=length_batch, clear_after_read=False)
                    # Function to compute the value loss for all episodes
                    def compute_value_loss(i, ta):
                        value_loss_temp = 0.5 * tf.reduce_sum(
                            tf.square(self.target_v[i][:self.lengths_episodes[i]] - self.value[i][:self.lengths_episodes[i]]))
                        return i + 1, ta.write(i, value_loss_temp)

                    # Loop over all episodes
                    _, value_loss_array = tf.while_loop(
                        lambda i, ta: i < length_batch, compute_value_loss, [0, value_loss_array_init])

                    # Value loss function
                    self.value_loss = tf.reduce_mean(tf.map_fn(lambda i: value_loss_array.gather([i]),
                                                                tf.range(length_batch),
                                                                dtype=tf.float32))
                with tf.variable_scope("policy_loss"):

                    # Tensorarray to keep value loss for all episodes
                    policy_loss_array_init = tf.TensorArray(tf.float32, size=length_batch, clear_after_read=False)

                    # Function to compute the value loss for all episodes
                    def compute_policy_loss(i, ta):
                        responsible_outputs_temp = tf.reduce_sum(self.policy[i][:self.lengths_episodes[i]] * self.actions[i][:self.lengths_episodes[i]], [1])
                        policy_loss_temp = -tf.reduce_sum(tf.log(tf.maximum(responsible_outputs_temp, 1e-12)) * self.advantages[i][:self.lengths_episodes[i]])
                        return i+1, ta.write(i, policy_loss_temp)

                     # Loop over all episodes
                    _, policy_loss_array = tf.while_loop(
                            lambda i, ta: i < length_batch, compute_policy_loss, [0, policy_loss_array_init])

                    self.policy_loss = tf.reduce_mean(tf.map_fn(lambda i: policy_loss_array.gather([i]),
                                                                tf.range(length_batch),
                                                                dtype = tf.float32))

                # Softmax entropy function
                with tf.variable_scope("entropy"):

                    # Tensorarray to keep entropy for all episodes
                    entropy_array_init = tf.TensorArray(tf.float32, size= length_batch, clear_after_read=False)

                    # Compute entropy
                    def compute_entropy(i, ta):
                        entropy_temp = tf.reduce_sum(self.policy[i][:self.lengths_episodes[i]] * tf.log(tf.maximum(self.policy[i][:self.lengths_episodes[i]], 1e-12)))
                        return i + 1, ta.write(i, entropy_temp)

                    _, entropy_array = tf.while_loop(
                            lambda i, ta: i < length_batch, compute_entropy, [0, entropy_array_init])

                    self.entropy = tf.reduce_mean(tf.map_fn(lambda i: entropy_array.gather([i]),
                                                                tf.range(length_batch),
                                                                dtype = tf.float32))

                # If noisy net is active one can disable entropy regularization
                if ENTROPY_REGULARIZATION:
                    loss_map = tf.map_fn(lambda i: 0.5 * value_loss_array.gather([i]) + policy_loss_array.gather([i]) - entropy_array.gather([i]) * ENTROPY_REGULARIZATION_LAMBDA,
                                          tf.range(length_batch),
                                          dtype = tf.float32)
                else:
                    loss_map = tf.map_fn(lambda i: 0.5 * value_loss_array.gather([i]) + policy_loss_array.gather([i]),
                                          tf.range(length_batch),
                                          dtype = tf.float32)

                self.loss = tf.reduce_mean(loss_map)



            if scope == "worker_0":
                tf.summary.scalar('policy_loss_' + scope,
                                  self.policy_loss)
                tf.summary.scalar('value_loss_' + scope,
                                  self.value_loss)
                tf.summary.scalar('loss_' + scope,
                                  self.loss)
                #tf.summary.scalar('advantages_' + scope,
                #                  tf.reduce_mean(self.advantages))

        elif method == "PCL":

            with tf.variable_scope("loss"):
                self.actions = tf.placeholder(shape=[None, None, a_size], dtype=tf.float32)
                self.rewards = tf.placeholder(shape=[None, None], dtype=tf.float32)
                self.discount = tf.placeholder(shape=[None], dtype=tf.float32)
                self.rollout = tf.placeholder(tf.int32, shape=())

                # Get first and last values of episode sub-batches
                length_episode = tf.shape(self.value)[1]
                length_batch = tf.shape(self.value)[0]

                # Compute first part of loss function --> Value loss
                with tf.variable_scope("value_loss"):

                    # value_loss_array holds for every episode -value_first + gamma ** rollout * value_last
                    value_loss_array_init = tf.TensorArray(tf.float32, size=length_batch)
                    # Tensorarray to hold mean of value loss per episodes rollouts
                    value_loss_array_mean_init = tf.TensorArray(tf.float32, size=length_batch)

                    def compute_value_loss(i, ta, ta_mean):
                        value_loss = -1 * self.value[i][:(self.lengths_episodes[i] - self.rollout + 1)] + self.discount[self.rollout - 1] * self.value[i][-(self.lengths_episodes[i] + (length_episode - self.lengths_episodes[i]) - self.rollout + 1):self.lengths_episodes[i]]
                        value_loss_mean = tf.reduce_mean(value_loss)
                        return i + 1, ta.write(i, value_loss), ta_mean.write(i, value_loss_mean)

                    # Loop over all episodes
                    _, value_loss_array, value_loss_mean_array = tf.while_loop(
                        lambda i, ta, ta_mean: i < length_batch, compute_value_loss, [0, value_loss_array_init, value_loss_array_mean_init])

                # Compute second part of loss function --> Policy loss
                with tf.variable_scope("policy_loss"):

                    # Compute based on episode lengths relevant log probs
                    policy_loss_array_init = tf.TensorArray(tf.float32, size=length_batch)
                    # Tensorarray to hold mean of policy loss per episodes rollouts
                    policy_loss_mean_array_init = tf.TensorArray(tf.float32, size=length_batch)

                    def compute_policy_loss(i, ta, ta_mean):
                        # Compute log probs
                        log_probs = tf.log(
                            tf.reduce_sum(self.actions[i][:self.lengths_episodes[i]] * self.policy[i][:self.lengths_episodes[i]], axis=1) + 0.0001)
                        # Compute rolling log_probs
                        rolling_log_probs = tf.map_fn(lambda j: log_probs[j:(j + self.rollout - 1)],
                                                      tf.range(self.lengths_episodes[i] - self.rollout + 1), dtype=tf.float32)
                        # Compute rolling rewards
                        rolling_rewards = tf.map_fn(lambda j: self.rewards[i][j:(j + self.rollout - 1)],
                                                    tf.range(self.lengths_episodes[i] - self.rollout + 1), dtype=tf.float32)

                        # Compute policy loss
                        policy_loss = tf.reduce_sum((rolling_rewards - self.tau_pcl *
                                                     rolling_log_probs) * self.discount[:(self.rollout - 1)], axis=1)

                        # Compute mean of policy loss per episode
                        policy_loss_mean = tf.reduce_mean(policy_loss)

                        return i + 1, ta.write(i, policy_loss), ta_mean.write(i, policy_loss_mean)

                    # Loop over all episodes
                    _, policy_loss_array, policy_loss_array_mean = tf.while_loop(
                        lambda i, ta, ta_mean: i < length_batch, compute_policy_loss, [0, policy_loss_array_init, policy_loss_mean_array_init])


                # Combine both loss parts
                tf_loss_map = tf.map_fn(lambda i: tf.reduce_mean(
                                                    0.5 * tf.square(value_loss_array.gather([i]) + policy_loss_array.gather([i]))),
                                        tf.range(length_batch), dtype=tf.float32)

                # Total loss via weighted sum depending on the length of the different episodes
                self.loss = tf.reduce_mean(tf_loss_map * tf.cast(self.lengths_episodes, dtype=tf.float32)) / tf.cast(
                    tf.reduce_sum(self.lengths_episodes), dtype=tf.float32)

                # Mean value of value loss
                self.value_loss = tf.reduce_sum(tf.map_fn(lambda i: value_loss_mean_array.gather([i]),
                                                          tf.range(length_batch),
                                                          dtype=tf.float32) * tf.cast(self.lengths_episodes,
                                                                                      dtype=tf.float32)) / tf.cast(
                    tf.reduce_sum(self.lengths_episodes), dtype=tf.float32)

                # Mean value of policy loss
                self.policy_loss = tf.reduce_sum(tf.map_fn(lambda i: policy_loss_array_mean.gather([i]),
                                                           tf.range(length_batch),
                                                           dtype=tf.float32) * tf.cast(self.lengths_episodes,
                                                                                       dtype=tf.float32)) / tf.cast(
                    tf.reduce_sum(self.lengths_episodes), dtype=tf.float32)

            if scope == "worker_0":

                tf.summary.scalar('policy_loss_' + scope,
                                  self.policy_loss)
                tf.summary.scalar('value_loss_' + scope,
                                  self.value_loss)
                #tf.summary.histogram('log_probs' + scope,
                #                     self.log_probs)
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
    def noisy_dense(self, x_batch, size, name, bias=True, activation_fn=tf.identity, noise_dist = 'factorized'):

        # We assume batched input [batch_size, time_steps, dimension]
        # Reshape x_batch into the form [batch_size * time_steps, dimension]
        # ToDo: infer correct size of previous layer (at the moment hard coded)
        x = tf.reshape(x_batch, [-1, self.shared_config["Cell_Units"]])

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
                result_ = activation_fn(ret + b)
                # Reshape back to [batch_size, time_steps, size]
                result_reshape = tf.reshape(result_, [tf.shape(x_batch)[0], -1, size])
                return (result_reshape)
            else:
                result_ = activation_fn(ret)
                # Reshape back to [batch_size, time_steps, size]
                result_reshape = tf.reshape(result_, [tf.shape(x_batch)[0], -1, size])
                return (result_reshape)
