#!/usr/bin/env python3
"""Module that performs gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) containing
           the correct labels for the data
        weights: dictionary of the weights and biases of the network
        cache: dictionary of the outputs of each layer of the network
        alpha: the learning rate
        lambtha: the L2 regularization parameter
        L: the number of layers of the network

    The network uses tanh activations on each layer except the last,
    which uses softmax.

    Updates weights and biases in place.
    """
    m = Y.shape[1]

    # Start from the output layer (softmax + cross-entropy)
    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        # Compute weight gradient with L2 regularization
        dW = (1 / m) * np.matmul(dz, A_prev.T)
        dW += (lambtha / m) * weights['W' + str(layer)]

        # Compute bias gradient (no regularization on bias)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # Update weights and biases in place
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            # Backpropagate to previous layer
            dA_prev = np.matmul(weights['W' + str(layer)].T, dz)

            # tanh derivative: 1 - tanh(z)^2 = 1 - A^2
            dz = dA_prev * (1 - np.power(A_prev, 2))
