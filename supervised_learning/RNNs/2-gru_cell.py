#!/usr/bin/env python3
"""GRU Cell implementation using numpy"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        i (int): dimensionality of the data
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        Sigmoid activation function

        Parameters:
        x (numpy.ndarray): input array

        Returns:
        numpy.ndarray: sigmoid activated output
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Softmax activation function

        Parameters:
        x (numpy.ndarray): input array

        Returns:
        numpy.ndarray: softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
        h_prev (numpy.ndarray): previous hidden state (m, h)
        x_t (numpy.ndarray): input data (m, i)

        Returns:
        tuple: h_next (next hidden state), y (output)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_hat

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
