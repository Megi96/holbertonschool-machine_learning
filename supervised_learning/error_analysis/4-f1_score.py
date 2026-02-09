#!/usr/bin/env python3
import numpy as np

def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,) containing specificity per class
    """
    classes = confusion.shape[0]
    spec = np.zeros(classes)

    total = np.sum(confusion)

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = total - (TP + FP + FN)

        spec[i] = TN / (TN + FP)


    return spec
