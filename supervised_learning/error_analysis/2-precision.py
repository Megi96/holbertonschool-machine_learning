#!/usr/bin/env python3
"""Calculate precision for each class."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)

    Returns:
        numpy.ndarray: Precision of each class (shape (classes,))
    """
    true_pos = np.diag(confusion)
    predicted = np.sum(confusion, axis=0)
    return true_pos / predicted
