#!/usr/bin/env python3
"""DeepNeuralNetwork class performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor
        nx: number of input features
        layers: list of number of nodes per layer
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

        # Initialize weights/biases using single loop
        for l in range(self.__L):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")
            w_key = "W" + str(l + 1)
            b_key = "b" + str(l + 1)
            input_size = nx if l == 0 else layers[l - 1]
            # He et al. initialization
            self.__weights[w_key] = np.random.randn(layers[l], input_size) * np.sqrt(2 / input_size)
            self.__weights[b_key] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary of all intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary of weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation using sigmoid"""
        self.__cache["A0"] = X
        for l in range(self.__L):
            W = self.__weights["W" + str(l + 1)]
            b = self.__weights["b" + str(l + 1)]
            A_prev = self.__cache["A" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(l + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Logistic regression cost"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One step of gradient descent"""
        m = Y.shape[1]
        L = self.__L
        dZ = None

        for l in reversed(range(L)):
            A = cache["A" + str(l + 1)]
            A_prev = cache["A" + str(l)]
            W = self.__weights["W" + str(l + 1)]
            if l == L - 1:
                dZ = A - Y
            else:
                dZ = np.dot(self.__weights["W" + str(l + 2)].T, dZ) * (A * (1 - A))
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W" + str(l + 1)] -= alpha * dW
            self.__weights["b" + str(l + 1)] -= alpha * db
