#!/usr/bin/env python3
"""24-one_hot_encode.py"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): shape (m,) numeric class labels
        classes (int): number of classes

    Returns:
        numpy.ndarray: one-hot encoded matrix of shape (classes, m)
                       or None on failure
    """
    try:
        # Ensure integer type
        Y = Y.astype(int)
        m = Y.shape[0]

        # Initialize zero matrix (classes x m)
        one_hot = np.zeros((classes, m))

        # Set 1 in the correct positions
        one_hot[Y, np.arange(m)] = 1

        return one_hot
    except Exception:
        return None
