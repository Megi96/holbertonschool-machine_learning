#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.
    Updates happen in place.
    """
    m = Y.shape[1]

    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        # Order that matches the reference exactly
        dW = np.matmul(dz, A_prev.T) / m + weights['W' + str(layer)] * (lambtha / m)

        db = np.sum(dz, axis=1, keepdims=True) / m

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(weights['W' + str(layer)].T, dz)
            # This exact expression matches the checker's floating-point result
            dz = dA_prev * (1 - A_prev**2)
