#!/usr/bin/env python3
"""Bidirectional RNN cell forward propagation"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        i (int): dimensionality of input data
        h (int): dimensionality of hidden states
        o (int): dimensionality of outputs
        """
        # forward hidden weights + bias
        self.Whf = np.random.randn(h + i, h)
        self.bhf = np.zeros((1, h))

        # backward hidden weights + bias
        self.Whb = np.random.randn(h + i, h)
        self.bhb = np.zeros((1, h))

        # output weights + bias
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step (forward direction)

        Parameters:
        h_prev (numpy.ndarray): previous hidden state (m, h)
        x_t (numpy.ndarray): input data at time step (m, i)

        Returns:
        h_next (numpy.ndarray): next hidden state (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)

        return h_next
