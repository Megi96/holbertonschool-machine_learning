#!/usr/bin/env python3
"""Inception Block Module"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block.

    Parameters
    ----------
    A_prev : keras layer
        Output from the previous layer.
    filters : tuple or list
        Contains F1, F3R, F3, F5R, F5, FPP:
        F1: filters for 1x1 convolution
        F3R: filters for 1x1 convolution before 3x3 convolution
        F3: filters for 3x3 convolution
        F5R: filters for 1x1 convolution before 5x5 convolution
        F5: filters for 5x5 convolution
        FPP: filters for 1x1 convolution after max pooling

    Returns
    -------
    keras layer
        Concatenated output of the inception block.
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3_reduce)

    conv5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5_reduce)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    pool_conv = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(pool)

    output = K.layers.Concatenate(axis=-1)(
        [conv1, conv3, conv5, pool_conv]
    )

    return output
