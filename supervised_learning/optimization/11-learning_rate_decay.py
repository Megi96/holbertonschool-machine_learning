#!/usr/bin/env python3
"""
11-learning_rate_decay
Updates the learning rate using stepwise inverse time decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Args:
        alpha (float): Original learning rate
        decay_rate (float): Weight used to determine the rate of decay
        global_step (int): Number of passes of gradient descent elapsed
        decay_step (int): Number of passes before alpha is decayed further

    Returns:
        float: Updated learning rate
    """
    # Calculate how many decay steps have passed
    step_count = global_step // decay_step
    # Apply inverse time decay
    alpha_updated = alpha / (1 + decay_rate * step_count)
    return alpha_updated
