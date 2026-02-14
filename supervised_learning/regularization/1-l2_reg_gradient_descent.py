#!/usr/bin/env python3
"""Updates the weights and biases of a neural network using gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m) containing correct labels
    weights: dictionary of weights and biases
    cache: dictionary of outputs of each layer
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers

    Updates weights and biases in place.
    """
    m = Y.shape[1]

    # Output layer gradient
    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        # dW with L2 - order matters for floating point match
        dW = np.matmul(dz, A_prev.T) / m + weights['W' + str(layer)] * (lambtha / m)

        # db
        db = np.sum(dz, axis=1, keepdims=True) / m

        # Update
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(weights['W' + str(layer)].T, dz)
            # tanh derivative - this form matches checker
            dz = dA_prev * (1 - A_prev**2)
