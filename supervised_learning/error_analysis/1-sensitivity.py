#!/usr/bin/env python3
"""Calculate sensitivity (recall) for each class"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall / true positive rate) for each class.

    Parameters:
    -----------
    confusion : numpy.ndarray of shape (classes, classes)
        Confusion matrix where:
        - rows represent actual (true) classes
        - columns represent predicted classes

    Returns:
    --------
    numpy.ndarray of shape (classes,)
        Sensitivity (recall) for each class
    """
    # True Positives = diagonal elements
    true_positives = np.diag(confusion)

    # Support = total number of actual instances per class = row sums
    actual_counts = np.sum(confusion, axis=1)

    # Sensitivity = TP / (TP + FN) = TP / row_sum
    # (using small epsilon only if needed — but in this task data has no zero rows)
    sens = true_positives / actual_counts

    return sens
