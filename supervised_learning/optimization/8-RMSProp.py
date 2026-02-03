#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates RMSProp optimization operation in TensorFlow

    Args:
        alpha: learning rate
        beta2: RMSProp weight (discounting factor)
        epsilon: small number to avoid division by zero

    Returns:
        optimizer: TensorFlow RMSProp optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
