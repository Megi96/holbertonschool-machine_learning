#!/usr/bin/env python3
"""
4-train.py
Module that contains the train_model function to train a Keras model
using mini-batch gradient descent.
"""

def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Parameters
    ----------
    network : keras.Model
        The compiled Keras model to train.
    data : numpy.ndarray of shape (m, nx)
        Input data for training.
    labels : numpy.ndarray of shape (m, classes)
        One-hot encoded labels for the training data.
    batch_size : int
        The number of samples per batch of computation.
    epochs : int
        Number of epochs (full passes through the data).
    verbose : bool, optional
        Whether to print training progress (default is True).
    shuffle : bool, optional
        Whether to shuffle the training data before each epoch
        (default is False, for reproducibility).

    Returns
    -------
    History
        A Keras History object generated after training the model.
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
