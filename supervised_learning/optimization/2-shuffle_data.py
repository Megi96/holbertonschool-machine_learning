#!/usr/bin/env python3
"""
2-shuffle_data
Shuffles two datasets in the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features
        Y (numpy.ndarray): Matrix of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features

    Returns:
        tuple: (X_shuffled, Y_shuffled)
            X_shuffled is the shuffled X matrix
            Y_shuffled is the shuffled Y matrix
    """
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
