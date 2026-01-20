#!/usr/bin/env python3
"""DeepNeuralNetwork class performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """Class constructor
        nx: number of input features
        layers: list of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Single loop: validate layers & initialize weights/biases
        for idx in range(self.__L):
            layer_size = layers[idx]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if idx == 0 else layers[idx - 1]
            w_key = "W" + str(idx + 1)
            b_key = "b" + str(idx + 1)

            # He initialization
            self.__weights[w_key] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights[b_key] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary of intermediate values"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary of weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Performs forward propagation"""
        self.__cache["A0"] = X
        for idx in range(self.__L):
            W = self.__weights["W" + str(idx + 1)]
            b = self.__weights["b" + str(idx + 1)]
            A_prev = self.__cache["A" + str(idx)]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache["A" + str(idx + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates logistic regression cost"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the network’s predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        L = self.__L
        dZ = None

        # Loop over layers in reverse
        for idx in reversed(range(L)):
            A = cache["A" + str(idx + 1)]
            A_prev = cache["A" + str(idx)]
            W = self.__weights["W" + str(idx + 1)]

            if idx == L - 1:
                dZ = A - Y
            else:
                dZ = np.dot(self.__weights["W" + str(idx + 2)].T, dZ) * (A * (1 - A))

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W" + str(idx + 1)] -= alpha * dW
            self.__weights["b" + str(idx + 1)] -= alpha * db