#!/usr/bin/env python3
"""
4-train.py
Trains a neural network using mini-batch gradient descent.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of the batch for gradient descent
        epochs: number of passes through the data
        verbose: boolean to determine if output is printed during training
        shuffle: boolean to determine whether to shuffle the batches

    Returns:
        The History object generated after training the model
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
