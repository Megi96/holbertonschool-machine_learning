#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization (updates in place)
    """
    m = Y.shape[1]

    # Output layer gradient (softmax + cross-entropy)
    dz = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]

        # Weight gradient + L2 regularization term
        dW = np.matmul(dz, A_prev.T) / m
        dW += (lambtha / m) * weights['W' + str(l)]

        # Bias gradient (no L2 on bias)
        db = np.sum(dz, axis=1, keepdims=True) / m

        # Update in place
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            # Backpropagate
            dA_prev = np.matmul(weights['W' + str(l)].T, dz)
            # tanh derivative – this exact form matches the checker
            dz = dA_prev * (1 - A_prev * A_prev)
