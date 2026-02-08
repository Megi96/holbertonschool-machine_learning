#!/usr/bin/env python3
"""Batch Normalization Layer for TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Args:
        prev (tf.Tensor): Activated output of the previous layer
        n (int): Number of nodes in the new layer
        activation (callable): Activation function to apply

    Returns:
        tf.Tensor: Activated output of the new layer
    """
    # Set TensorFlow seed for reproducibility
    tf.random.set_seed(0)

    # Dense layer with variance scaling initializer
    dense_layer = tf.keras.layers.Dense(
        n,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=0
        ),
        use_bias=False
    )(prev)

    # Batch normalization with gamma=1, beta=0, epsilon=1e-7
    bn_layer = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(dense_layer, training=True)

    # Apply activation function
    return activation(bn_layer)
