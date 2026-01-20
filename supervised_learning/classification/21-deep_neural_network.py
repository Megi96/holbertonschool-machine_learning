#!/usr/bin/env python3
"""
Defines a deep neural network for binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Constructor

        nx (int): number of input features
        layers (list): number of nodes in each layer
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

        for index, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            layer_key = index + 1

            if layer_key == 1:
                self.__weights["W1"] = (
                    np.random.randn(nodes, nx) * np.sqrt(2 / nx)
                )
            else:
                prev_nodes = layers[index - 1]
                self.__weights["W" + str(layer_key)] = (
                    np.random.randn(nodes, prev_nodes)
                    * np.sqrt(2 / prev_nodes)
                )

            self.__weights["b" + str(layer_key)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation

        X (numpy.ndarray): input data of shape (nx, m)

        Returns:
        A (numpy.ndarray): output of the network
        cache (dict): intermediary values
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W" + str(layer)]
            b = self.__weights["b" + str(layer)]
            A_prev = self.__cache["A" + str(layer - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A" + str(layer)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression

        Y (numpy.ndarray): correct labels
        A (numpy.ndarray): predicted labels
        """
        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        ) / m
        return cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent

        Y (numpy.ndarray): correct labels
        cache (dict): intermediary values
        alpha (float): learning rate
        """
        m = Y.shape[1]
        layers = self.__L

        dZ = cache["A" + str(layers)] - Y

        for layer in range(layers, 0, -1):
            A_prev = cache["A" + str(layer - 1)]
            W = self.__weights["W" + str(layer)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights["W" + str(layer)] = W - alpha * dW
            self.__weights["b" + str(layer)] -= alpha * db

            if layer > 1:
                A = cache["A" + str(layer - 1)]
                dZ = np.matmul(W.T, dZ) * A * (1 - A)
