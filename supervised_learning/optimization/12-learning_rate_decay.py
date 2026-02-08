#!/usr/bin/env python3
"""
12-learning_rate_decay
Creates a learning rate decay operation in TensorFlow using
stepwise inverse time decay.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a TensorFlow learning rate schedule using inverse time decay
    in a stepwise fashion.

    Args:
        alpha (float): Original learning rate
        decay_rate (float): Weight used to determine rate of decay
        decay_step (int): Number of steps before learning rate decays

    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule:
            A callable learning rate schedule
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
