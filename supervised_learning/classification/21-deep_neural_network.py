#!/usr/bin/env python3
"""
21-deep_neural_network.py
Defines a DeepNeuralNetwork class performing binary classification
with forward propagation, cost calculation, evaluation, and gradient descent.
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(self.__L):
            layer_size = layers[layer_idx]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if layer_idx == 0 else layers[layer_idx - 1]
            w_key = "W" + str(layer_idx + 1)
            b_key = "b" + str(layer_idx + 1)

            self.__weights[w_key] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights[b_key] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Performs forward propagation using sigmoid activation."""
        self.__cache['A0'] = X
        for layer_idx in range(self.__L):
            w_key = "W" + str(layer_idx + 1)
            b_key = "b" + str(layer_idx + 1)
            a_prev = self.__cache["A" + str(layer_idx)]
            Z = np.dot(self.__weights[w_key], a_prev) + self.__weights[b_key]
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(layer_idx + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression."""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the network’s predictions and cost."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
    """Performs one pass of gradient descent on the network."""
    m = Y.shape[1]
    L = self.__L
    dZ = None

    # Loop over layers in reverse order
    for layer_idx in reversed(range(L)):
        A = cache["A" + str(layer_idx + 1)]
        A_prev = cache["A" + str(layer_idx)]
        W = self.__weights["W" + str(layer_idx + 1)]

        # Last layer
        if layer_idx == L - 1:
            dZ = A - Y
        else:
            dZ = np.dot(self.__weights["W" + str(layer_idx + 2)].T, dZ) * (A * (1 - A))

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        self.__weights["W" + str(layer_idx + 1)] -= alpha * dW
        self.__weights["b" + str(layer_idx + 1)] -= alpha * d
    
    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent on the network."""
        m = Y.shape[1]
        L = self.__L
        weights_copy = self.__weights.copy()
        dZ = None

        for layer_idx in reversed(range(L)):
            A = cache["A" + str(layer_idx + 1)]
            A_prev = cache["A" + str(layer_idx)]
            W = weights_copy["W" + str(layer_idx + 1)]

            if layer_idx == L - 1:
                dZ = A - Y
            else:
                dZ = np.dot(weights_copy["W" + str(layer_idx + 2)].T, dZ) * (A * (1 - A))

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W" + str(layer_idx + 1)] -= alpha * dW
            self.__weights["b" + str(layer_idx + 1)] -= alpha * db
