#!/usr/bin/env python3
"""
8-train.py
Trains a neural network using mini-batch gradient descent with
optional validation, early stopping, learning rate decay, and
saving the best model.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx)
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of the batch
        epochs: number of passes through the data
        validation_data: data to validate the model with
        early_stopping: boolean to enable early stopping
        patience: patience used for early stopping
        learning_rate_decay: boolean to enable learning rate decay
        alpha: initial learning rate
        decay_rate: decay rate
        save_best: boolean to save best model
        filepath: path to save the model
        verbose: boolean to print output
        shuffle: boolean to shuffle batches

    Returns:
        The History object generated after training
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

        if save_best and filepath is not None:
            callbacks.append(
                K.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    monitor='val_loss',
                    save_best_only=True
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
