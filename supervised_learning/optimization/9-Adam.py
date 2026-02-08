#!/usr/bin/env python3
"""
9-Adam
Updates a variable using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): Learning rate
        beta1 (float): Weight for the first moment (momentum)
        beta2 (float): Weight for the second moment (RMS)
        epsilon (float): Small number to avoid division by zero
        var (numpy.ndarray): Variable to be updated
        grad (numpy.ndarray): Gradient of var
        v (numpy.ndarray): Previous first moment of var
        s (numpy.ndarray): Previous second moment of var
        t (int): Time step for bias correction

    Returns:
        tuple: (updated variable, new first moment, new second moment)
    """
    # Update biased first moment estimate (Momentum)
    v = beta1 * v + (1 - beta1) * grad
    # Update biased second raw moment estimate (RMSProp)
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first moment
    v_corrected = v / (1 - beta1 ** t)
    # Compute bias-corrected second moment
    s_corrected = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)

    return var, v, s
