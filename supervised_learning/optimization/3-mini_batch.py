#!/usr/bin/env python3
"""
3-mini_batch
Creates mini-batches for mini-batch gradient descent
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from input data and labels.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx)
            m is the number of data points
            nx is the number of features
        Y (numpy.ndarray): Labels of shape (m, ny)
            m is the same number of data points as in X
            ny is the number of classes
        batch_size (int): Number of data points per batch

    Returns:
        list: List of tuples (X_batch, Y_batch)
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X_shuffled.shape[0]
    mini_batches_
holbertonschool-machine_learning