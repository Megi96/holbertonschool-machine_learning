#!/usr/bin/env python3
"""
3-one_hot.py
Converts a label vector into a one-hot encoded matrix using Keras.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels: list or array-like of integers, shape (m,)
    - classes: int, optional, number of classes. If None, inferred.

    Returns:
    - one_hot_matrix: Keras tensor, shape (m, classes)
    """
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes,
                                            dtype='float32')
    return one_hot_matrix
