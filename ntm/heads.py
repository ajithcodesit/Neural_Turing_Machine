#!/usr/bin/python3

import tensorflow as tf

from tensorflow.keras import Model

from .addressing import Addressing


class ReadHead(Model):

    def __init__(self, memory_locations=128, memory_vector_size=20, maximum_shifts=3):
        super(ReadHead, self).__init__()

        self.addr_mech = Addressing(memory_locations, memory_vector_size, maximum_shifts, reading=True)
    
    def call(self, controller_output, w_t_1, M_t):
        w_t = self.addr_mech(controller_output, w_t_1, M_t)
        r_t = tf.squeeze(tf.matmul(tf.expand_dims(w_t, axis=1), M_t), axis=1)
        return r_t, w_t


class WriteHead(Model):

    def __init__(self, memory_locations=128, memory_vector_size=20, maximum_shifts=3):
        super(WriteHead, self).__init__()
        
        self.memory_vector_size = memory_vector_size
        self.addr_mech = Addressing(memory_locations, memory_vector_size, maximum_shifts, reading=False)

    def call(self, controller_output, w_t_1, M_t_1):
        w_t, e_t, a_t = self.addr_mech(controller_output, w_t_1, M_t_1)
        w_t = tf.expand_dims(w_t, axis=1)

        # Erase
        e_t = tf.expand_dims(e_t, axis=1)
        M_tidle_t = tf.multiply(M_t_1, (1.0 - tf.matmul(w_t, e_t, transpose_a=True)))

        # Add
        a_t = tf.expand_dims(a_t, axis=1)
        M_t = M_tidle_t + tf.matmul(w_t, a_t, transpose_a=True)

        return M_t, tf.squeeze(e_t, axis=1), tf.squeeze(a_t, axis=1), tf.squeeze(w_t, axis=1)

