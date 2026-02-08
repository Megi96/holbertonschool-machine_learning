#!/usr/bin/env python3
"""
0-norm_constants
Calculates the normalization constants of a dataset
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) to normalize
            m is the number of data points
            nx is the number of features

    Returns:
        tuple: (mean, std)
            mean is a numpy.ndarray of shape (nx,) containing
            the mean of each feature
            std is a numpy.ndarray of shape (nx,) containing
            the standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
