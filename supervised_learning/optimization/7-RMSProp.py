#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha: learning rate
        beta2: RMSProp weight (decay rate)
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: previous second moment of var

    Returns:
        updated variable and the new moment, respectively
    """
    # Hapi 1: Përditëso second moment (moving average e squared gradients)
    s_new = beta2 * s + (1 - beta2) * np.square(grad)

    # Hapi 2: Përditëso variable duke përdorur RMSProp update rule
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    return var_new, s_new
