#!/usr/bin/env python3
"""Module for Bidirectional RNN Cell with output method"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction
        for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing
                    the previous hidden state
            x_t: numpy.ndarray of shape (m, i) containing
                 the data input

        Returns:
            h_next: the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        for one time step

        Args:
            h_next: numpy.ndarray of shape (m, h) containing
                    the next hidden state
            x_t: numpy.ndarray of shape (m, i) containing
                 the data input

        Returns:
            h_prev: the previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Args:
            H: numpy.ndarray of shape (t, m, 2 * h) containing
               the concatenated hidden states from both directions,
               excluding their initialized states
               t is the number of time steps
               m is the batch size
               h is the dimensionality of the hidden states

        Returns:
            Y: the outputs
        """
        scores = np.dot(H, self.Wy) + self.by
        exp_s = np.exp(scores)
        Y = exp_s / exp_s.sum(axis=2, keepdims=True)
        return Y
