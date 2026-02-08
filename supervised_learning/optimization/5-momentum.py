#!/usr/bin/env python3
"""
5-momentum
Updates variables using gradient descent with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum
    optimization algorithm.

    Args:
        alpha (float): Learning rate
        beta1 (float): Momentum weight
        var (numpy.ndarray): Variable to be updated
        grad (numpy.ndarray): Gradient of the variable
        v (numpy.ndarray): Previous first moment of the variable

    Returns:
        tuple: (updated variable, new first moment)
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
