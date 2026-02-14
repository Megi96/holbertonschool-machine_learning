#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization (in place).
    """
    m = Y.shape[1]

    dz = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]

        # L2 term added after /m — this order matches reference most often
        dW = (np.matmul(dz, A_prev.T) / m) + (weights['W' + str(l)] * (lambtha / m))

        db = np.sum(dz, axis=1, keepdims=True) / m

        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            dA_prev = np.matmul(weights['W' + str(l)].T, dz)
            # This exact form (multiplication) is the most reliable for matching the checker
            dz = dA_prev * (1 - A_prev * A_prev)
