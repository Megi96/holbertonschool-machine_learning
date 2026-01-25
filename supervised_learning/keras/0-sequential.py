#!/usr/bin/env python3
"""
0-sequential.py
Builds a neural network using Keras Sequential API.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using the Keras Sequential API.

    Parameters:
    - nx: number of input features
    - layers: list containing the number of nodes in each layer
    - activations: list containing activation functions for each layer
    - lambtha: L2 regularization parameter
    - keep_prob: probability of keeping a node for dropout

    Returns:
    - The Keras model
    """
    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_dim=nx
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if keep_prob < 1 and i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
