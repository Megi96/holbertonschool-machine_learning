#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2"""
    
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]

        # L2 regularized dW
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update parameters in place
        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

        # Compute dZ for next layer (if not first layer)
        if i > 1:
            A = cache["A" + str(i - 1)]

            dZ = np.matmul(W.T, dZ) * (1 - np.square(A))
