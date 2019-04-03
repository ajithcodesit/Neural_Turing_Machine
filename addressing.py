#!/usr/bin/python3

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class Addressing(Model):

    def __init__(self, memory_locations=128, memory_vector_size=20, maximum_shifts=3, reading=True):
        super(Addressing, self).__init__()

        self.memory_locations = memory_locations  # N locations
        self.memory_vector_size = memory_vector_size  # M vector size
        self.maximum_shifts = maximum_shifts
        self.reading = reading

        self.read_split = [self.memory_vector_size, 1, 1, self.maximum_shifts, 1]
        self.write_split = [self.memory_vector_size, 1, 1, self.maximum_shifts, 1,
                            self.memory_vector_size, self.memory_vector_size]

        if self.reading:
            self.emit_len = np.sum(self.read_split)
        else:
            self.emit_len = np.sum(self.write_split)

        self.fc_addr = Dense(units=self.emit_len, activation=tf.nn.tanh, name="emit_params",
                             kernel_initializer='glorot_uniform', bias_initializer='glorot_normal')
        
        self.k_t = None
        self.beta_t = None
        self.g_t = None
        self.s_t = None
        self.gamma_t = None

        self.e_t = None
        self.a_t = None
        
        # All of the below are the weights over N locations produced by the addressing mechanism
        # [Batch size, N]
        self.w_c_t = None
        self.w_g_t = None
        self.w_tidle_t = None
        self.w_t = None

    def emit_addressing_params(self, k_t, beta_t, g_t, s_t, gamma_t):
        self.k_t = k_t  # Key vector
        self.beta_t = tf.nn.softplus(beta_t)  # Key strength
        self.g_t = tf.nn.sigmoid(g_t)  # Interpolation gate
        self.s_t = tf.nn.softmax(s_t, axis=-1)  # Shift weighting
        self.gamma_t = 1.0 + tf.nn.softplus(gamma_t)  # Sharpen

    def emit_head_params(self, fc_output):

        if self.reading:
            k_t, beta_t, g_t, s_t, gamma_t = tf.split(fc_output, self.read_split, axis=-1)
            self.emit_addressing_params(k_t, beta_t, g_t, s_t, gamma_t)

        else:
            k_t, beta_t, g_t, s_t, gamma_t, e_t, a_t = tf.split(fc_output, self.write_split, axis=-1)
            self.emit_addressing_params(k_t, beta_t, g_t, s_t, gamma_t)
            self.e_t = tf.nn.sigmoid(e_t)  # Erase vector
            self.a_t = a_t  # Add vector

    @staticmethod
    def cosine_similarity(k, m):
        k_mag = tf.sqrt(tf.reduce_sum(tf.square(k), axis=-1))
        m_mag = tf.sqrt(tf.reduce_sum(tf.square(m), axis=-1))
        mag_prod = tf.multiply(k_mag, m_mag)
        dot = tf.squeeze(tf.keras.layers.dot([k, m], axes=(-1, -1)), axis=1)
        return tf.divide(dot, mag_prod)
    
    @staticmethod
    def circular_convolution(w, s):
        kernels=tf.TensorArray(dtype=s.dtype, size=s.shape[0])
        
        for i in range(0, s.shape[0]):
            kernels.write(i, tf.roll(w, shift=i-(s.shape[0]//2), axis=0))

        w_circ_conv = tf.transpose(kernels.stack())
        return tf.reduce_sum(w_circ_conv*s, axis=1)
    
    def content_addressing(self, M_t):
        k_t = tf.expand_dims(self.k_t, axis=1)
        self.w_c_t = tf.nn.softmax(self.beta_t * self.cosine_similarity(k_t, M_t), axis=-1)
    
    def interpolation(self, w_t_prev):
        self.w_g_t = (self.g_t * self.w_c_t) + ((1 - self.g_t)*w_t_prev)
    
    def convolutional_shift(self):
        convolved_weights = tf.TensorArray(dtype=self.w_g_t.dtype, size=self.w_g_t.shape[0])
        
        for i in range(self.s_t.shape[0]):
            cc = self.circular_convolution(self.w_g_t[i], self.s_t[i])
            convolved_weights.write(i, cc)
        
        self.w_tidle_t = convolved_weights.stack()

    def sharpening(self):
        w_raised = tf.pow(self.w_tidle_t, self.gamma_t)
        self.w_t = tf.divide(w_raised, tf.reduce_sum(w_raised, axis=-1, keepdims=True))

    def call(self, controller_output, w_t_prev, M_t):
        # Controller outputs used for addressing
        self.emit_head_params(self.fc_addr(controller_output))

        # Addressing mechanism
        self.content_addressing(M_t)
        self.interpolation(w_t_prev)
        self.convolutional_shift()
        self.sharpening()

        if self.reading:
            return self.w_t  # The new weight over the N locations of the memory matrix, and
        else:
            return self.w_t, self.e_t, self.a_t
