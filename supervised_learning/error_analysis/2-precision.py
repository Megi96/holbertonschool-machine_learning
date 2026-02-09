#!/usr/bin/env python3
"""Calculate precision for each class"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class.

    Parameters:
    -----------
    confusion : numpy.ndarray of shape (classes, classes)
        Confusion matrix where:
        - rows represent actual (true) classes
        - columns represent predicted classes

    Returns:
    --------
    numpy.ndarray of shape (classes,)
        Precision for each class
    """
    # True Positives = diagonal elements
    true_positives = np.diag(confusion)

    # Predicted Positives = total number of times the model predicted each class
    # = sum over rows (axis=0) = column sums
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP) = TP / column_sum
    prec = true_positives / predicted_positives

    return prec

