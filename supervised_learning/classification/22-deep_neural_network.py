#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network with multiple layers performing binary classification.

    Attributes
    ----------
    __L : int
        Number of layers in the neural network
    __cache : dict
        Dictionary to hold all intermediary values of the network
    __weights : dict
        Dictionary to hold all weights and biases of the network
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network

        Parameters
        ----------
        nx : int
            Number of input features
        layers : list
            List representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or \
           not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            if l == 0:
                weight_shape = (layers[l], nx)
            else:
                weight_shape = (layers[l], layers[l - 1])

            self.__weights['W' + str(l + 1)] = np.random.randn(*weight_shape) * np.sqrt(2 / weight_shape[1])
            self.__weights['b' + str(l + 1)] = np.zeros((layers[l], 1))

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
        """
        Performs forward propagation

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (nx, m)

        Returns
        -------
        A : numpy.ndarray
            Output of the neural network
        cache : dict
            Updated cache dictionary
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]
            A_prev = self.__cache['A' + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Computes the cost using logistic regression

        Parameters
        ----------
        Y : numpy.ndarray
            True labels
        A : numpy.ndarray
            Predicted output

        Returns
        -------
        cost : float
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network’s predictions

        Parameters
        ----------
        X : numpy.ndarray
            Input data
        Y : numpy.ndarray
            True labels

        Returns
        -------
        prediction : numpy.ndarray
        cost : float
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, alpha=0.05):
        """
        Performs one pass of gradient descent

        Parameters
        ----------
        Y : numpy.ndarray
            True labels
        alpha : float
            Learning rate
        """
        m = Y.shape[1]
        L = self.__L
        weights_copy = self.__weights.copy()
        dZ = self.__cache['A' + str(L)] - Y

        for l in range(L, 0, -1):
            A_prev = self.__cache['A' + str(l - 1)]
            W = weights_copy['W' + str(l)]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dA_prev = np.dot(W.T, dZ)
                dZ = dA_prev * self.__cache['A' + str(l - 1)] * (1 - self.__cache['A' + str(l - 1)])

            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network

        Parameters
        ----------
        X : numpy.ndarray
            Input data
        Y : numpy.ndarray
            True labels
        iterations : int
            Number of iterations to train
        alpha : float
            Learning rate

        Returns
        -------
        prediction : numpy.ndarray
        cost : float
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, alpha)

        return self.evaluate(X, Y)
