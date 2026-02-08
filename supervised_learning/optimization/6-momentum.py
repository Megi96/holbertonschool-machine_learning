#!/usr/bin/env python3
"""
6-momentum
Creates a TensorFlow optimizer using gradient descent with momentum
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Args:
        alpha (float): Learning rate
        beta1 (float): Momentum weight

    Returns:
        tf.keras.optimizers.Optimizer: Momentum optimizer
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
