#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    -----------
    Y : numpy.ndarray of shape (classes, m)
        one-hot encoded correct labels
    weights : dict
        contains W{l} and b{l} for l=1..L
    cache : dict
        contains A{l} for l=0..L (A0 = input data)
    alpha : float
        learning rate
    lambtha : float
        L2 regularization parameter
    L : int
        number of layers

    Returns:
    --------
    None (updates weights in place)
    """
    m = Y.shape[1]

    # Output layer gradient (softmax + cross-entropy → very simple)
    dZ = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]

        # Weight gradient with L2 regularization term
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        dW += (lambtha / m) * weights['W' + str(l)]

        # Bias gradient (no regularization on bias)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases in place
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            # Backprop to previous layer
            dA_prev = np.matmul(weights['W' + str(l)].T, dZ)
            # tanh derivative
            dZ = dA_prev * (1 - np.power(A_prev, 2))


            dZ = np.matmul(W.T, dZ) * (1 - np.square(A))
