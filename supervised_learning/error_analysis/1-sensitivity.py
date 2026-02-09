#!/usr/bin/env python3
"""Calculate sensitivity for each class."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)

    Returns:
        numpy.ndarray: Sensitivity of each class (shape (classes,))
    """
    true_pos = np.diag(confusion)
    actual = np.sum(confusion, axis=1)
    return true_pos / actual
