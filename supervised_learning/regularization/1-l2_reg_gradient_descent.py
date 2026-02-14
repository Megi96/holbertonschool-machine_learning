#!/usr/bin/env python3
""" L2 regularized gradient descent – one step """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights & biases using gradient descent + L2 reg (in place)
    """
    m = Y.shape[1]

    # Start from output layer (softmax + cross-entropy)
    dZ = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):

        A_prev = cache['A' + str(l - 1)]

        # ────────────────────────────────────────────────
        # Gradients
        # ────────────────────────────────────────────────
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        dW += (lambtha / m) * weights['W' + str(l)]          # ← L2 term here

        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # ────────────────────────────────────────────────
        # Updates – in place
        # ────────────────────────────────────────────────
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        # ────────────────────────────────────────────────
        # Propagate to previous layer (only if not input)
        # ────────────────────────────────────────────────
        if l > 1:
            dA_prev = np.matmul(weights['W' + str(l)].T, dZ)
            # tanh derivative: 1 - tanh(z)² = 1 - A²
            dZ = dA_prev * (1 - np.square(A_prev))           # ← correct line

    # No return – updates happen in-place

