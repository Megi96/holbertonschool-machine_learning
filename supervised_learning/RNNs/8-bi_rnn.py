#!/usr/bin/env python3
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_T):
    """
    Bidirectional RNN forward propagation
    """

    t, m, i = X.shape
    h = h_0.shape[1]

    # hidden states for forward and backward
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    # OUTPUT hidden states (concatenated)
    H = np.zeros((t, m, 2 * h))

    # forward initial state
    h_prev = h_0

    # forward pass
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev

    # backward initial state
    h_next = h_T

    # backward pass (reverse time)
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next

    # concatenate forward + backward
    for step in range(t):
        H[step] = np.concatenate((Hf[step], Hb[step]), axis=1)

    # output layer
    Y = bi_cell.output(H)

    return H, Y
