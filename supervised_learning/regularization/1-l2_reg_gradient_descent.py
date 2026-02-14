#!/usr/bin/env python3
"""L2 Regularization Gradient Descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Y: one-hot labels (classes, m)
    weights: dictionary of weights and biases
    cache: dictionary of layer activations
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers
    """

    m = Y.shape[1]

    # Output layer gradient (softmax + cross-entropy)
    dZ = cache["A" + str(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]

        # Gradient with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases in place
        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db

        # Compute dZ for next layer (if not input layer)
        if layer > 1:
            A_prev = cache["A" + str(layer - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
