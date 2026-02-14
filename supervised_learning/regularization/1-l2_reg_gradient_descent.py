#!/usr/bin/env python3
"""1-l2_reg_gradient_descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    m = Y.shape[1]

    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        dW = (1 / m) * np.matmul(dz, A_prev.T)
        dW += (lambtha / m) * weights['W' + str(layer)]

        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(weights['W' + str(layer)].T, dz)
            dz = dA_prev * (1.0 - A_prev**2)  # ← this exact line is key
