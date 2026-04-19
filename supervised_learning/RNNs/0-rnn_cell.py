#!/usr/bin/env python3
import numpy as np


class RNNCell:
    """Represents a simple RNN cell"""

    def __init__(self, i, h, o):
        """
        i: input dimensionality
        h: hidden state dimensionality
        o: output dimensionality
        """

        # Weights for hidden state (concatenated [h_prev, x_t])
        self.Wh = np.random.randn(i + h, h)

        # Weights for output
        self.Wy = np.random.randn(h, o)

        # Biases
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Numerically stable softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        h_prev: shape (m, h)
        x_t: shape (m, i)
        """

        # Step 1: concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Step 2: compute next hidden state (tanh activation)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Step 3: compute output (softmax)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
