#!/usr/bin/python3

import tensorflow as tf

from .controller import Controller
from .heads import ReadHead, WriteHead

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class NTM(Model):

    def __init__(self, controller_size=100, memory_locations=128, memory_vector_size=20, maximum_shifts=3,
                 output_size=8, learn_r_bias=False, learn_w_bias=False, learn_m_bias=False):
        super(NTM, self).__init__()

        self.memory_locations = memory_locations  # N locations
        self.memory_vector_size = memory_vector_size  # M size memory vectors
        self.maximum_shifts = maximum_shifts
        
        self.controller = Controller(controller_size)
        self.read_head = ReadHead(self.memory_locations, self.memory_vector_size, self.maximum_shifts)
        self.write_head = WriteHead(self.memory_locations, self.memory_vector_size, self.maximum_shifts)

        self.final_fc = Dense(units=output_size, activation=tf.nn.sigmoid, name="final_fc",
                              kernel_initializer='glorot_uniform', bias_initializer='glorot_normal')

        # The bias vector (These bias vectors can be learned or initialized to the same value)
        self.r_bias = tf.Variable(tf.random.truncated_normal([1, self.memory_vector_size], mean=0.0, stddev=0.5), trainable=learn_r_bias)
        self.w_bias = tf.Variable(tf.nn.softmax(tf.random.normal([1, self.memory_locations])), trainable=learn_w_bias)
        self.M_bias = tf.Variable(tf.ones([1, self.memory_locations, self.memory_vector_size]) * 1e-6, trainable=learn_m_bias)

        # States of the NTM
        self.r_t_1 = None  # Previous read vector variable [Batch size, M]
        self.w_t_1 = None  # Previous weights over the memory matrix [Batch size, N]
        self.M_t = None  # The memory matrix [Batch size, N, M]

        # Extra outputs that are tracked
        self.e_t = None
        self.a_t = None

        # For visualizing the NTM working (Must be used only during predictions)
        self.reads = []
        self.adds = []
        self.read_weights = []
        self.write_weights = []

    def debug_ntm(self):  # Function to debug the NTM working
        # Reversed to correct for the time step order (t_0 at the bottom)
        self.reads.reverse()
        self.adds.reverse()
        self.read_weights.reverse()
        self.write_weights.reverse()

        rt = tf.stack(self.reads)
        at = tf.stack(self.adds)
        r_wt = tf.stack(self.read_weights)
        w_wt = tf.stack(self.write_weights)
        Mt = self.M_t[0]

        return rt.numpy(), r_wt.numpy(), at.numpy(), w_wt.numpy(), Mt.numpy()

    def reset_debug_vars(self):  # This is called if the NTM is not stateful
        self.reads = []
        self.adds = []
        self.read_weights = []
        self.write_weights = []

    def reset_ntm_state(self, batch_size):  # Creates a new NTM state
        # This has to be manually called if stateful is set to true
        self.r_t_1 = tf.tile(self.r_bias, [batch_size, 1])
        self.w_t_1 = tf.tile(self.w_bias, [batch_size, 1])
        self.M_t = tf.tile(self.M_bias, [batch_size, 1, 1])

    def call(self, inputs, stateful=False, training=False):
        # Convert from [Batch, Timesteps, Features] to [Timesteps, Batch, Features]
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = tf.TensorArray(dtype=inputs.dtype, size=inputs.shape[0])
        
        if not stateful:  # A new state will not be created at the start of each new batch
            self.reset_ntm_state(inputs.shape[1])
            self.reset_debug_vars()

        for i in tf.range(inputs.shape[0]):
            # Concatenated input and previous reads [Batch, Features + N]
            controller_inputs = tf.concat([inputs[i], self.r_t_1], axis=1)
            controller_outputs = self.controller(controller_inputs)  # [Batch size, Controller size]

            # [Batch size, M, N], [Batch size, M], [Batch size, M], [Batch size, N]
            self.M_t, self.e_t, self.a_t, self.w_t_1 = self.write_head(controller_outputs, self.w_t_1, self.M_t)
            if not training:
                self.adds.append(self.a_t[0])
                self.write_weights.append(self.w_t_1[0])

            # [Batch size, M], [Batch size, N]
            self.r_t_1, self.w_t_1 = self.read_head(controller_outputs, self.w_t_1, self.M_t)
            if not training:
                self.reads.append(self.r_t_1[0])
                self.read_weights.append(self.w_t_1[0])

            fc_input = tf.concat([controller_outputs, self.r_t_1], axis=1)  # [Batch size, Controller size + M],
            output_t = self.final_fc(fc_input)  # [Batch size, Output size]
            outputs = outputs.write(i, output_t)  # Write it to an array

        outputs = tf.transpose(outputs.stack(), [1, 0, 2])  # [Batch size, Timesteps, Output size]
        return outputs
