#!/usr/bin/env python3
"""Module that defines an RNN cell"""

import numpy as np


class RNNCell:
    """Represents a simple RNN cell"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        i (int): dimensionality of the data
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Calculates the softmax of a matrix

        Parameters:
        x (numpy.ndarray): input data

        Returns:
        numpy.ndarray: softmax output
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
        h_prev (numpy.ndarray): previous hidden state (m, h)
        x_t (numpy.ndarray): input data (m, i)

        Returns:
        h_next, y:
            h_next (numpy.ndarray): next hidden state
            y (numpy.ndarray): output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
