#!/usr/bin/env python3
"""Creates a batch normalization layer using TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    prev: the activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function used on the output of the layer

    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )

    Z = dense(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer="ones",
        beta_initializer="zeros",
        epsilon=1e-7
    )

    Z_norm = batch_norm(Z, training=True)

    if activation is not None:
        return activation(Z_norm)
    return Z_norm
