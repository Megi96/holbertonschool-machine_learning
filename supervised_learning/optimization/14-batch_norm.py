#!/usr/bin/env python3
"""Creates a batch normalization layer using TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Parameters:
    -----------
    prev : tf.Tensor
        The activated output of the previous layer
    n : int
        The number of nodes in the layer to be created
    activation : callable
        The activation function that should be used on the output of the layer

    Returns:
    --------
    tf.Tensor
        The activated output for the layer (after Dense → BatchNorm → Activation)
    """
    # Kernel initializer as specified: VarianceScaling with mode='fan_avg'
    initializer = tf.keras.initializers.VarianceScaling(
        mode='fan_avg'
    )

    # Dense layer WITHOUT bias (BatchNormalization will provide its own beta)
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )

    # Apply dense transformation first
    Z = dense(prev)

    # Batch Normalization layer
    # gamma initialized to 1, beta to 0, epsilon=1e-7
    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer='ones',
        beta_initializer='zeros',
        epsilon=1e-7
    )

    # Apply batch normalization
    Z_norm = batch_norm(Z, training=True)

    # Apply the given activation (if None, just return normalized Z)
    if activation is not None:
        return activation(Z_norm)
    return Z_norm

