#!/usr/bin/python3

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class Controller(Model):

    def __init__(self, controller_size=100):
        super(Controller, self).__init__()

        # 3-layer feedforward controller
        self.d1 = Dense(units=controller_size, activation=tf.nn.tanh, name="controller_d1",
                        kernel_initializer='glorot_uniform', bias_initializer='glorot_normal')
        self.d2 = Dense(units=controller_size, activation=tf.nn.tanh, name="controller_d2",
                        kernel_initializer='glorot_uniform', bias_initializer='glorot_normal')
        self.d3 = Dense(units=controller_size, activation=tf.nn.tanh, name="controller_d3",
                        kernel_initializer='glorot_uniform', bias_initializer='glorot_normal')

    def call(self, controller_input):
        out = self.d1(controller_input)
        out = self.d2(out)
        return self.d3(out)

