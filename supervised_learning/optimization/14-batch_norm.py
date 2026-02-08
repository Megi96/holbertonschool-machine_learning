#!/usr/bin/env python3
"""
14-batch_norm
Creates a batch normalization layer for a neural network using TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer in TensorFlow.

    Args:
        prev (tf.Tensor): Activated output from the previous layer
        n (int): Number of nodes in the current layer
        activation (function): Activation function to apply

    Returns:
        tf.Tensor: Activated output of the batch-normalized layer
    """
    # Dense layer with VarianceScaling initializer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=None,  # no activation yet
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)

    # Batch normalization with trainable gamma and beta
    batch_norm_layer = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,   # adds beta
        scale=True     # adds gamma
    )(layer)

    # Apply activation
    output = activation(batch_norm_layer)

    return output
