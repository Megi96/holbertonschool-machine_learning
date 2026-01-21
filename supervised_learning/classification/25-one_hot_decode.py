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
    try:
        # For each column (sample), get the index of the 1
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
