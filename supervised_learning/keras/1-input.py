#!/usr/bin/env python3
"""
1-input.py
Builds a neural network using the Keras Functional API.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras Functional API.

    Parameters:
    - nx: int, number of input features to the network
    - layers: list of ints, number of nodes in each layer
    - activations: list of strings, activation functions for each layer
    - lambtha: float, L2 regularization parameter
    - keep_prob: float, probability that a node will be kept (dropout)

    Returns:
    - model: Keras Model, the built neural network
    """
    # Define input layer
    inputs = K.Input(shape=(nx,))
    x = inputs

    # Loop through all layers
    for i in range(len(layers)):
        # Dense layer with L2 regularization
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        # Dropout only between layers (not after last layer)
        if i < len(layers) - 1 and keep_prob < 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=x)

    return model
