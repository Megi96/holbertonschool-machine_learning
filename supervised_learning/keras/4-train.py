#!/usr/bin/env python3
"""
4-train.py
"""

def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    network: the model to train
    data: numpy.ndarray of shape (m, nx) with input data
    labels: one-hot numpy.ndarray of shape (m, classes)
    batch_size: size of the batch
    epochs: number of epochs
    verbose: boolean, whether to print output
    shuffle: boolean, whether to shuffle data every epoch

    Returns: the History object generated after training
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
