#!/usr/bin/python3

import tensorflow as tf
import numpy as np

from controller import Controller
from heads import ReadHead, WriteHead

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class NTM(Model):

    def __init__(self, controller_size=100, memory_locations=128, memory_vector_size=20, maximum_shifts=3, output_size=8):
        super(NTM, self).__init__()

        self.memory_locations = memory_locations  # N locations
        self.memory_vector_size = memory_vector_size  # M size memory vectors
        self.maximum_shifts = maximum_shifts
        
        self.controller = Controller(controller_size)
        self.read_head = ReadHead(self.memory_locations, self.memory_vector_size, self.maximum_shifts)
        self.write_head = WriteHead(self.memory_locations, self.memory_vector_size, self.maximum_shifts)

        self.final_fc = Dense(units=output_size, activation=tf.nn.sigmoid, name="final_fc",
                              kernel_initializer='glorot_uniform', bias_initializer='glorot_normal')

        self.stddev = 1.0 / (np.sqrt(self.memory_locations + self.memory_vector_size))

        # The learned bias vector
        self.r_bias = tf.constant(tf.random.normal([1, self.memory_vector_size]) * 0.01)  # Bias for previous reads
        self.M_bias = tf.constant(tf.random.uniform([1, self.memory_locations, self.memory_vector_size],
                                                    minval=-self.stddev, maxval=self.stddev))  # Bias for Memory matrix

        # States of the NTM
        self.r_t_1 = None  # Previous read vector variable [Batch size, M]
        self.w_t_1 = None  # Previous weights over the memory matrix [Batch size, N]
        self.M_t = None  # The memory matrix [Batch size, N, M]

        # Extra outputs that are tracked
        self.e_t = None
        self.a_t = None

    def create_new_state(self, batch_size):  # Creates a new NTM state
        # This has to be manually called if stateful is set to true
        if self.r_t_1 is None:
            self.r_t_1 = tf.Variable(tf.tile(self.r_bias, [batch_size, 1]), trainable=False)
        else:
            self.r_t_1.assign(tf.tile(self.r_bias, [batch_size, 1]))

        if self.w_t_1 is None:
            self.w_t_1 = tf.Variable(tf.zeros([batch_size, self.memory_locations]), trainable=False)
        else:
            self.w_t_1.assign(tf.zeros([batch_size, self.memory_locations]))

        if self.M_t is None:
            self.M_t = tf.Variable(tf.tile(self.M_bias, [batch_size, 1, 1]), trainable=False)
        else:
            self.M_t.assign(tf.tile(self.M_bias, [batch_size, 1, 1]))
    
    def call(self, inputs, stateful=False):
        # Convert from [Batch, Timesteps, Features] to [Timesteps, Batch, Features]
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = tf.TensorArray(dtype=inputs.dtype, size=inputs.shape[0])
        
        if not stateful:  # A new state will not be created at the start of each new batch
            self.create_new_state(inputs.shape[1])

        for i in range(inputs.shape[0]):
            # Concatenated input and previous reads [Batch, Features + N]
            controller_inputs = tf.concat([inputs[i], self.r_t_1], axis=1)
            controller_outputs = self.controller(controller_inputs)  # [Batch size, Controller size]

            r_t, w_t = self.read_head(controller_outputs, tf.identity(self.w_t_1), tf.identity(self.M_t))  # [Batch size, M], [Batch size, N]
            self.r_t_1.assign(r_t)
            self.w_t_1.assign(w_t)

            # [Batch size, M, N], [Batch size, M], [Batch size, M], [Batch size, N]
            M_t, self.e_t, self.a_t, w_t = self.write_head(controller_outputs, tf.identity(self.w_t_1), tf.identity(self.M_t))
            self.M_t.assign(M_t)
            self.w_t_1.assign(w_t)

            fc_input = tf.concat([controller_outputs, self.r_t_1], axis=1)  # [Batch size, Controller size + M],
            output_t = self.final_fc(fc_input)  # [Batch size, Output size]
            outputs.write(i, output_t)  # Write it to an array

        outputs = tf.transpose(outputs.stack(), [1, 0, 2])  # [Batch size, Timesteps, Output size]
        return outputs


# ntm = NTM(controller_size=100, memory_locations=10, memory_vector_size=5, output_size=3)

# # [Batch, Timesteps, Features]
# inp = tf.Variable(tf.reshape(tf.range(0.0,4.0,0.1),[2,5,4]))
# out = ntm(inp)
# print(out)
