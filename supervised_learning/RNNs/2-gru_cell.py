#!/usr/bin/env python3
import numpy as np


class GRUCell:
    """Represents a GRU cell"""

    def __init__(self, i, h, o):
        """Initialize parameters"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Forward propagation for one time step"""
        
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)

        # Reset gate
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        # Candidate hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_hat

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
