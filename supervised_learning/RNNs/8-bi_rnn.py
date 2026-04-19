#!/usr/bin/env python3
"""Module for bidirectional RNN forward propagation"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Args:
        bi_cell: instance of BidirectionalCell used for forward propagation
        X: numpy.ndarray of shape (t, m, i) containing the data
           t is the maximum number of time steps
           m is the batch size
           i is the dimensionality of the data
        h_0: numpy.ndarray of shape (m, h) containing the initial hidden
             state in the forward direction
        h_t: numpy.ndarray of shape (m, h) containing the initial hidden
             state in the backward direction

    Returns:
        H: numpy.ndarray containing all concatenated hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_prev

    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_next

    H = np.concatenate((H_forward, H_backward), axis=2)
    Y = bi_cell.output(H)

    return H, Y
