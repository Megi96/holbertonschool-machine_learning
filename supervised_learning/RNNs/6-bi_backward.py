#!/usr/bin/env python3
"""Module for Bidirectional RNN Cell with backward pass"""
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
        # Forward direction weights and biases
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Backward direction weights and biases
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data input

        Returns:
            h_next: the next hidden state
        """
        # Concatenate h_prev and x_t along the feature axis
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Compute next hidden state using tanh activation
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Args:
            h_next: numpy.ndarray of shape (m, h) containing the next
                    hidden state (in backward direction, this is the state
                    at t+1)
            x_t: numpy.ndarray of shape (m, i) containing the data input

        Returns:
            h_prev: the previous hidden state (at t-1 in backward direction)
        """
        # Concatenate h_next and x_t along the feature axis
        concat = np.concatenate((h_next, x_t), axis=1)
        # Compute previous hidden state using backward weights
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev
