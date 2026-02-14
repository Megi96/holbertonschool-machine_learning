#!/usr/bin/env python3
"""1-l2_reg_gradient_descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    m = Y.shape[1]

    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        dW = np.matmul(dz, A_prev.T) / m
        dW += weights['W' + str(layer)] * (lambtha / m)  # this order often wins

        db = np.sum(dz, axis=1, keepdims=True) / m

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(weights['W' + str(layer)].T, dz)
            dz = dA_prev * (1 - np.square(A_prev))  # ← change to this if not already
