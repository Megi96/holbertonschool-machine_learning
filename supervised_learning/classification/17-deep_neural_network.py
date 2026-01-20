#!/usr/bin/env python3
"""Deep Neural Network for binary classification with only one loop."""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        nx: number of input features
        layers: list of nodes per layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Only loop allowed: validate layers and initialize weights/biases
        for layer_idx in range(self.L):
            layer_size = layers[layer_idx]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )
            prev_size = nx if layer_idx == 0 else layers[layer_idx - 1]
            w_key = 'W' + str(layer_idx + 1)
            b_key = 'b' + str(layer_idx + 1)
            # He initialization
            self.weights[w_key] = (
                np.random.randn(layer_size, prev_size) *
                np.sqrt(2 / prev_size)
            )
            self.weights[b_key] = np.zeros((layer_size, 1))

    def sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Performs forward propagation through all layers."""
        self.cache['A0'] = X
        for layer_idx in range(self.L):
            W = self.weights['W' + str(layer_idx + 1)]
            b = self.weights['b' + str(layer_idx + 1)]
            A_prev = self.cache['A' + str(layer_idx)]
            Z = W @ A_prev + b  # matrix multiplication
            A = self.sigmoid(Z)
            self.cache['A' + str(layer_idx + 1)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Computes binary cross-entropy cost."""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions for given data."""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost
