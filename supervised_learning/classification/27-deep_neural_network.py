#!/usr/bin/env python3
"""27-deep_neural_network.py
DeepNeuralNetwork class for multiclass classification
"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network performing multiclass classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if (not isinstance(layers, list) or len(layers) == 0 or
                not all(isinstance(x, int) and x > 0 for x in layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(1, self.__L + 1):
            layer_size = layers[layer_idx - 1]
            prev_size = nx if layer_idx == 1 else layers[layer_idx - 2]

            self.__weights["W{}".format(layer_idx)] = (
                np.random.randn(layer_size, prev_size) *
                np.sqrt(1 / prev_size)
            )
            self.__weights["b{}".format(layer_idx)] = np.zeros(
                (layer_size, 1)
            )

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
        self.__cache["A0"] = X

        for layer_idx in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer_idx)]
            b = self.__weights["b{}".format(layer_idx)]
            A_prev = self.__cache["A{}".format(layer_idx - 1)]

            Z = np.matmul(W, A_prev) + b

            if layer_idx != self.__L:
                A = 1 / (1 + np.exp(-Z))
            else:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            self.__cache["A{}".format(layer_idx)] = A

        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A + 1e-8)) / m

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for layer_idx in reversed(range(1, self.__L + 1)):
            A_prev = cache["A{}".format(layer_idx - 1)]
            W = self.__weights["W{}".format(layer_idx)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_idx > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights["W{}".format(layer_idx)] -= alpha * dW
            self.__weights["b{}".format(layer_idx)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              step=100, graph=True):
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

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
