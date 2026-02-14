#!/usr/bin/env python3
"""Gradient descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network using gradient descent with Dropout
    """

    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A{}'.format(i - 1)]
        W = weights['W{}'.format(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights in place
        weights['W{}'.format(i)] = W - alpha * dW
        weights['b{}'.format(i)] = weights['b{}'.format(i)] - alpha * db

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)

            # Apply dropout mask
            D_prev = cache['D{}'.format(i - 1)]
            dA_prev = dA_prev * D_prev
            dA_prev = dA_prev / keep_prob

            A_prev = cache['A{}'.format(i - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)
