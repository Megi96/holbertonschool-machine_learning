#!/usr/bin/env python3
"""
7-train.py
Trains a neural network using mini-batch gradient descent with
optional validation, early stopping, and learning rate decay.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of the batch for gradient descent
        epochs: number of passes through the data
        validation_data: data to validate the model with
        early_stopping: boolean to determine whether early stopping is used
        patience: patience used for early stopping
        learning_rate_decay: boolean to determine whether LR decay is used
        alpha: initial learning rate
        decay_rate: decay rate
        verbose: boolean to determine if output is printed during training
        shuffle: boolean to determine whether to shuffle the batches

    Returns:
        The History object generated after training the model
    """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(
                K.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience
                )
            )

        if learning_rate_decay:
            def scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(
                K.callbacks.LearningRateScheduler(
                    scheduler,
                    verbose=1
                )
            )

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
