#!/usr/bin/env python3
"""Batch Normalization Layer for TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network."""
    tf.random.set_seed(0)

    dense_layer = tf.keras.layers.Dense(
        n,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=0
        ),
        use_bias=False
    )(prev)

    bn_layer = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(dense_layer)

    return activation(bn_layer)
