#!/usr/bin/env python3
"""Create confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters:
    -----------
    labels : numpy.ndarray of shape (m, classes)
        One-hot encoded true labels
    logits : numpy.ndarray of shape (m, classes)
        One-hot encoded predicted labels

    Returns:
    --------
    numpy.ndarray of shape (classes, classes)
        Confusion matrix where:
        - rows = true classes
        - columns = predicted classes
    """
    # Get the true class indices (argmax over one-hot)
    true_classes = np.argmax(labels, axis=1)

    # Get the predicted class indices (argmax over one-hot)
    pred_classes = np.argmax(logits, axis=1)

    # Number of classes = number of columns in labels/logits
    num_classes = labels.shape[1]

    # Create confusion matrix (rows = true, columns = predicted)
    confusion = np.zeros((num_classes, num_classes), dtype=float)

    # Fill the confusion matrix
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
