#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[i], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(
                        layers[i], layers[i - 1]
                    ) * np.sqrt(2 / layers[i - 1])
                )

            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Cached values"""
        return self.__cache

    @property
    def weights(self):
        """Weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions"""
        A, _ = self.forward_prop(X)
        predictions = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent"""
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for i in reversed(range(1, self.__L + 1)):
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dZ = np.matmul(W.T, dZ) * (
                    A_prev * (1 - A_prev)
                )

            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        return self.evaluate(X, Y)
