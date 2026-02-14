#!/usr/bin/env python3
"""3-mini_batch.py"""

import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from X and Y for mini-batch gradient descent.

    Args:
        X (np.ndarray): Input data of shape (m, nx)
        Y (np.ndarray): Labels of shape (m, ny)
        batch_size (int): Number of examples per mini-batch

    Returns:
        list of tuples: Each tuple is (X_batch, Y_batch)
    """
    m = X.shape[0]
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
