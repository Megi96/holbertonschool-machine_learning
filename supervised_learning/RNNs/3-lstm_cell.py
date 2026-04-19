#!/usr/bin/env python3
"""LSTM Cell implementation"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        i (int): dimensionality of the data
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
        h_prev (numpy.ndarray): previous hidden state (m, h)
        c_prev (numpy.ndarray): previous cell state (m, h)
        x_t (numpy.ndarray): input data (m, i)

        Returns:
        h_next (numpy.ndarray): next hidden state
        c_next (numpy.ndarray): next cell state
        y (numpy.ndarray): output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        ft = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        ut = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        cct = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        ot = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        c_next = ft * c_prev + ut * cct
        h_next = ot * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
