#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Performs forward propagation with dropout"""
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        Z = np.matmul(W, A_prev) + b

        if l != L:
            # Hidden layer: tanh + dropout
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(l)] = D
        else:
            # Output layer: softmax
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(l)] = A

    return cache
