#!/usr/bin/env python3
"""Module that performs forward propagation for a simple RNN"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Parameters:
    rnn_cell: instance of RNNCell
    X (numpy.ndarray): data of shape (t, m, i)
    h_0 (numpy.ndarray): initial hidden state (m, h)

    Returns:
    H, Y:
        H (numpy.ndarray): all hidden states
        Y (numpy.ndarray): all outputs
    """

    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Initialize outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Iterate over time steps
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])

        H[step + 1] = h_next
        Y[step] = y

    return H, Y
