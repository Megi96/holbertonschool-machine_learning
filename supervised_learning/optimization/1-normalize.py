#!/usr/bin/env python3
"""
1-normalize
Normalizes a dataset using provided mean and standard deviation
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
        X (numpy.ndarray): Matrix of shape (d, nx) to normalize
            d is the number of data points
            nx is the number of features
        m (numpy.ndarray): Mean of each feature, shape (nx,)
        s (numpy.ndarray): Standard deviation of each feature, shape (nx,)

    Returns:
        numpy.ndarray: The normalized matrix X
    """
    return (X - m) / s
