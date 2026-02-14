#!/usr/bin/env python3
"""Write a function that updates the weights and
biases of a neural network using gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update weights and biases using
      gradient descent with L2 regularization"""

    m = Y.shape[1]

    dZ = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W{}".format(i)] = W - alpha * dW
        weights["b{}".format(i)] = weights["b{}".format(i)] - alpha * db

        if i > 1:
            A_prev_layer = cache["A{}".format(i - 1)]
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.square(A_prev_layer))
