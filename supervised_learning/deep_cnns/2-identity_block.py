#!/usr/bin/env python3
"""Identity block for ResNet"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in ResNet.
    
    A_prev: output from the previous layer
    filters: tuple/list containing F11, F3, F12
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # First component
    X = K.layers.Conv2D(
        F11, (1, 1), padding='same', kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component
    X = K.layers.Conv2D(
        F3, (3, 3), padding='same', kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component
    X = K.layers.Conv2D(
        F12, (1, 1), padding='same', kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut connection
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
