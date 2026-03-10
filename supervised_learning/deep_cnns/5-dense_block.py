#!/usr/bin/env python3
"""
Dense Block module for DenseNet architectures
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a Dense Block as described in DenseNet-B.

    A dense block consists of multiple convolutional layers where
    each layer receives as input the concatenation of all previous
    feature maps.

    Each layer follows the bottleneck design:
        BatchNorm -> ReLU -> 1x1 Conv
        BatchNorm -> ReLU -> 3x3 Conv

    Parameters
    ----------
    X : tensor
        Output tensor from the previous layer.
    nb_filters : int
        Number of filters in X.
    growth_rate : int
        Growth rate of the dense block.
    layers : int
        Number of layers inside the dense block.

    Returns
    -------
    tuple
        (output_tensor, nb_filters)

        output_tensor: concatenated output of the dense block
        nb_filters: updated number of filters
    """

    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):

        # Bottleneck layer
        BN1 = K.layers.BatchNormalization()(X)
        A1 = K.layers.Activation('relu')(BN1)
        C1 = K.layers.Conv2D(
            4 * growth_rate,
            kernel_size=1,
            padding='same',
            kernel_initializer=init
        )(A1)

        # 3x3 convolution
        BN2 = K.layers.BatchNormalization()(C1)
        A2 = K.layers.Activation('relu')(BN2)
        C2 = K.layers.Conv2D(
            growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=init
        )(A2)

        # Concatenate input with output
        X = K.layers.Concatenate()([X, C2])

        # Update number of filters
        nb_filters += growth_rate

    return X, nb_filters
