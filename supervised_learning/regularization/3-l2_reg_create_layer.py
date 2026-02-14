#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a Dense layer with L2 regularization"""
    initializer = tf.keras.initializers.HeNormal()
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
