#!/usr/bin/env python3
"""
3-one_hot.py
Converts a label vector into a one-hot encoded matrix.
"""

import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels: np.ndarray, shape (m,), label vector with integers
    - classes: int, optional, number of classes. If None, inferred from labels.

    Returns:
    - one_hot_matrix: np.ndarray, shape (m, classes)
    """
    labels = np.array(labels, dtype=int)  # ensure integer type
    m = labels.shape[0]

    if classes is None:
        classes = np.max(labels) + 1  # infer number of classes

    one_hot_matrix = np.zeros((m, classes), dtype=float)
    one_hot_matrix[np.arange(m), labels] = 1.0

    return one_hot_matrix
