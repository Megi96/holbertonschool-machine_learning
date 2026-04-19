#!/usr/bin/env python3
"""Deep RNN forward propagation"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
    rnn_cells (list): list of RNNCell instances (length l)
    X (numpy.ndarray): input data of shape (t, m, i)
    h_0 (numpy.ndarray): initial hidden states (l, m, h)

    Returns:
    H (numpy.ndarray): all hidden states (t+1, l, m, h)
    Y (numpy.ndarray): all outputs (t, m, o)
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    # Initialize storage
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Forward propagation
    for step in range(t):
        for layer in range(l):
            if layer == 0:
                x_t = X[step]
            else:
                x_t = H[step + 1, layer - 1]

            h_prev = H[step, layer]

            h_next, y = rnn_cells[layer].forward(h_prev, x_t)

            H[step + 1, layer] = h_next

        # Output only from last layer
        Y[step] = y

    return H, Y
