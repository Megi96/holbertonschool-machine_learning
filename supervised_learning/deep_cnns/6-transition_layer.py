#!/usr/bin/env python3
"""Transition Layer module"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a DenseNet transition layer

    Parameters:
    X -- output tensor from the previous layer
    nb_filters -- integer, number of filters in X
    compression -- compression factor for the transition layer

    Returns:
    X -- output tensor of the transition layer
    nb_filters -- number of filters in the output
    """

    initializer = K.initializers.he_normal(seed=0)

    # Compute compressed number of filters
    nb_filters = int(nb_filters * compression)

    # BatchNorm -> ReLU
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)

    # 1x1 convolution (compression)
    conv = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer
    )(activation)

    # Average pooling
    X = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2
    )(conv)

    return X, nb_filters
