#!/usr/bin/env python3
"""25-one_hot_decode.py"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a numeric label vector.

    Args:
        one_hot (numpy.ndarray): one-hot encoded matrix with shape
                                 (classes, m)

    Returns:
        numpy.ndarray: vector of labels with shape (m,) or None on failure
    """
    # ----- INPUT VALIDATION -----
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2:
        return None
    if one_hot.shape[0] == 0 or one_hot.shape[1] == 0:
        return None

    # ----- DECODE -----
    # For each column (sample), find the index of the max value
    labels = np.argmax(one_hot, axis=0)
    return labels
