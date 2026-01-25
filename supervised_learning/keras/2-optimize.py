#!/usr/bin/env python3
"""
2-optimize.py
Sets up Adam optimization for a Keras model with categorical crossentropy
loss and accuracy metrics.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures a Keras model for training using the Adam optimizer.

    Parameters:
    - network: Keras model to optimize
    - alpha: float, learning rate
    - beta1: float, first Adam moment
    - beta2: float, second Adam moment

    Returns:
    - None
    """
    # Create Adam optimizer with given hyperparameters
    adam = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)

    # Compile the model with categorical crossentropy loss and accuracy metric
    network.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
